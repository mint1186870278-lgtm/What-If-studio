"""Project management API routes"""

import hashlib
import logging
from uuid import UUID
from typing import List, Optional
from datetime import datetime
import json

from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from src.db import get_db
from src.models import Project, Asset, Session as DBSession, VideoJob, UserPreference, ScriptVector
from src.schemas import (
    ProjectCreate, ProjectUpdate, ProjectResponse,
    OutputSelectRequest, StoryboardGenerateResponse,
    StoryboardConfirmRequest, StoryboardConfirmResponse,
)
from src.core.memory_service import memory_service, detect_script_tone
from src.core.model_router import model_router, ModelProvider

# Prefer LangGraph; fall back to AutoGen
try:
    from src.agents import run_langgraph_discussion_stream as _discuss_stream
except Exception:
    from src.agents import run_autogen_discussion_stream as _discuss_stream  # type: ignore[assignment]

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/projects", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    project_create: ProjectCreate,
    db: Session = Depends(get_db),
):
    """Create a new project"""
    try:
        project = Project(
            name=project_create.name,
            description=project_create.description,
            prompt=project_create.prompt or "",
            style_preference=project_create.style_preference or "auto",
            discussion_history=[],
            discussion_status="idle",
            metadata_={},
            output_type="script_only",
        )
        db.add(project)
        db.commit()
        db.refresh(project)
        logger.info(f"✅ Project created: {project.id} - {project.name}")
        return project
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Failed to create project: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create project: {str(e)}",
        )


@router.get("/projects/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: UUID,
    db: Session = Depends(get_db),
):
    """Get project by ID"""
    project_id_str = str(project_id)
    project = db.query(Project).filter(Project.id == project_id_str).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )
    project.last_opened_at = datetime.utcnow()
    db.commit()
    db.refresh(project)
    return project


@router.get("/projects", response_model=List[ProjectResponse])
async def list_projects(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """List all projects with pagination"""
    projects = (
        db.query(Project)
        .order_by(Project.last_opened_at.desc().nullslast(), Project.updated_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return projects


@router.put("/projects/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: UUID,
    project_update: ProjectUpdate,
    db: Session = Depends(get_db),
):
    """Update project"""
    project_id_str = str(project_id)
    project = db.query(Project).filter(Project.id == project_id_str).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    try:
        if project_update.name is not None:
            project.name = project_update.name
        if project_update.description is not None:
            project.description = project_update.description
        if project_update.prompt is not None:
            project.prompt = project_update.prompt
        if project_update.style_preference is not None:
            project.style_preference = project_update.style_preference

        db.commit()
        db.refresh(project)
        logger.info(f"✅ Project updated: {project.id}")
        return project
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Failed to update project: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update project: {str(e)}",
        )


@router.delete("/projects/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: UUID,
    db: Session = Depends(get_db),
):
    """Delete project and all associated data"""
    try:
        project_id_str = str(project_id)
        project = db.query(Project).filter(Project.id == project_id_str).first()
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found",
            )

        # Delete associated sessions and assets
        db.query(Asset).filter(Asset.project_id == project_id_str).delete()
        db.query(DBSession).filter(DBSession.project_id == project_id_str).delete()

        # Delete project
        db.delete(project)
        db.commit()

        logger.info(f"✅ Project deleted: {project_id}")
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Failed to delete project: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete project: {str(e)}",
        )


async def _generate_project_discussion_stream(project: Project, db: Session, user_id: Optional[str] = None):
    turns = []
    script = ""
    project.discussion_status = "running"
    db.commit()
    yield f"data: {json.dumps({'type': 'system', 'content': 'discussion_started'}, ensure_ascii=False)}\n\n"
    try:
        user_request = f"{project.name}：{project.prompt or ''}" if project.name else (project.prompt or "")
        # Use project_id as session_id so WebSocket intervention can match
        sid = str(project.id)
        async for event in _discuss_stream(
            user_request=user_request,
            style=project.style_preference or "auto",
            user_id=user_id,
            session_id=sid,
        ):
            # Map type → event for frontend compatibility
            if "type" in event and "event" not in event:
                event["event"] = event["type"]
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            if event.get("type") == "turn":
                turns.append(event)
            elif event.get("type") == "script":
                script = str(event.get("script", ""))

        project.discussion_history = turns
        project.script = script
        project.discussion_status = "completed"
        project.last_opened_at = datetime.utcnow()
        db.commit()

        # --- Memory persistence: auto-save scripts ---
        if script:
            metadata = {
                "project_id": str(project.id),
                "style": project.style_preference or "auto",
                "user_request": user_request[:256],
                "timestamp": datetime.utcnow().isoformat(),
            }
            if user_id:
                metadata["user_id"] = user_id
            try:
                chroma_id = await memory_service.store_script(script, metadata)
                sv = ScriptVector(
                    chroma_doc_id=chroma_id,
                    project_id=str(project.id),
                    user_id=user_id or "anonymous",
                    style=project.style_preference or "auto",
                    prompt_hash=hashlib.sha256(user_request.encode()).hexdigest()[:64],
                )
                db.add(sv)
                db.commit()
                logger.info("Script saved to memory: chroma_id=%s project=%s", chroma_id, project.id)
            except Exception as mem_exc:
                logger.warning("Failed to save script to memory: %s", mem_exc)

        # --- Auto-learn user preferences ---
        if user_id and script:
            try:
                tone = detect_script_tone(script)
                for pref_key, pref_val in [
                    ("style_preference", project.style_preference or "auto"),
                    ("last_script_type", tone),
                ]:
                    await memory_service.store_user_preference(user_id, {
                        "key": pref_key,
                        "value": pref_val,
                        "source": "inferred",
                        "confidence": 70 if pref_key == "style_preference" else 60,
                    })
                    db.add(UserPreference(
                        user_id=user_id,
                        preference_key=pref_key,
                        preference_value=pref_val,
                        confidence=70 if pref_key == "style_preference" else 60,
                        source="inferred",
                    ))
                db.commit()
                logger.info("User preferences inferred: user=%s style=%s tone=%s", user_id, project.style_preference, tone)
            except Exception as pref_exc:
                db.rollback()
                logger.warning("Failed to store inferred preferences: %s", pref_exc)

    except Exception as exc:
        project.discussion_status = "failed"
        db.commit()
        logger.exception("Project discussion failed: project_id=%s", project.id)
        yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"


@router.post("/projects/{project_id}/script/stream")
async def generate_project_script_stream(
    project_id: UUID,
    db: Session = Depends(get_db),
    user_id: Optional[str] = Query(None, description="Optional user identifier for memory features"),
):
    project_id_str = str(project_id)
    project = db.query(Project).filter(Project.id == project_id_str).first()
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    if not (project.prompt or "").strip():
        raise HTTPException(status_code=400, detail="Project prompt is empty")
    return StreamingResponse(
        _generate_project_discussion_stream(project, db, user_id=user_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/projects/{project_id}/video-jobs")
async def create_project_video_job(
    project_id: UUID,
    db: Session = Depends(get_db),
):
    project_id_str = str(project_id)
    project = db.query(Project).filter(Project.id == project_id_str).first()
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    if not (project.script or "").strip():
        raise HTTPException(status_code=409, detail="Project script not generated yet")

    session = db.query(DBSession).filter(DBSession.project_id == project_id_str).order_by(DBSession.created_at.desc()).first()
    if not session:
        session = DBSession(
            project_id=project_id_str,
            prompt=project.prompt or "",
            style_preference=project.style_preference or "auto",
            status="completed",
            script=project.script or "",
            discussion_history=project.discussion_history or [],
        )
        db.add(session)
        db.commit()
        db.refresh(session)
    else:
        session.prompt = project.prompt or ""
        session.style_preference = project.style_preference or "auto"
        session.script = project.script or ""
        session.discussion_history = project.discussion_history or []
        session.status = "completed"
        db.commit()
        db.refresh(session)

    job = VideoJob(
        session_id=str(session.id),
        phase="collect",
        status="pending",
        script=project.script or "",
        output_path=None,
    )
    db.add(job)
    project.last_opened_at = datetime.utcnow()
    db.commit()
    db.refresh(job)
    return {"job": {"id": str(job.id), "jobId": str(job.id), "session_id": str(session.id), "status": job.status, "phase": job.phase}}


# ---------------------------------------------------------------------------
# Output format selection
# ---------------------------------------------------------------------------

@router.put("/projects/{project_id}/output/select", response_model=ProjectResponse)
async def select_output_format(
    project_id: UUID,
    req: OutputSelectRequest,
    db: Session = Depends(get_db),
):
    """Set the project's output format preference after discussion."""
    project_id_str = str(project_id)
    project = db.query(Project).filter(Project.id == project_id_str).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )
    project.output_type = req.output_type
    project.last_opened_at = datetime.utcnow()
    db.commit()
    db.refresh(project)
    logger.info("Project %s output_type set to %s", project_id_str, req.output_type)
    return project


# ---------------------------------------------------------------------------
# Storyboard generation and confirmation
# ---------------------------------------------------------------------------

@router.post("/projects/{project_id}/storyboard/generate", response_model=StoryboardGenerateResponse)
async def generate_storyboard(
    project_id: UUID,
    db: Session = Depends(get_db),
):
    """Generate a storyboard/preview from the project's script using a text LLM."""
    project_id_str = str(project_id)
    project = db.query(Project).filter(Project.id == project_id_str).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )
    if not (project.script or "").strip():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Project script not generated yet. Run the discussion first.",
        )

    try:
        storyboard = await model_router.video.generate_storyboard(
            project.script,
        )
    except Exception as e:
        logger.exception("Storyboard generation failed for project %s", project_id_str)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Storyboard generation failed: {str(e)}",
        )

    project.storyboard = storyboard
    project.last_opened_at = datetime.utcnow()
    db.commit()

    logger.info("Storyboard generated for project %s: %d frames", project_id_str, len(storyboard.get("frames", [])))
    return StoryboardGenerateResponse(
        project_id=project_id_str,
        frames=storyboard.get("frames", []),
        total_duration=str(storyboard.get("total_duration", "")),
        generated_at=datetime.utcnow(),
    )


@router.post("/projects/{project_id}/storyboard/confirm", response_model=StoryboardConfirmResponse)
async def confirm_storyboard(
    project_id: UUID,
    req: StoryboardConfirmRequest,
    db: Session = Depends(get_db),
):
    """Confirm a storyboard (kicks off video job) or request regeneration."""
    project_id_str = str(project_id)
    project = db.query(Project).filter(Project.id == project_id_str).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    if req.confirmed:
        if not (project.script or "").strip():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Project script not generated yet",
            )

        session = db.query(DBSession).filter(
            DBSession.project_id == project_id_str
        ).order_by(DBSession.created_at.desc()).first()

        if not session:
            session = DBSession(
                project_id=project_id_str,
                prompt=project.prompt or "",
                style_preference=project.style_preference or "auto",
                status="completed",
                script=project.script or "",
                discussion_history=project.discussion_history or [],
            )
            db.add(session)
            db.commit()
            db.refresh(session)
        else:
            session.prompt = project.prompt or ""
            session.style_preference = project.style_preference or "auto"
            session.script = project.script or ""
            session.discussion_history = project.discussion_history or []
            session.status = "completed"
            db.commit()
            db.refresh(session)

        job = VideoJob(
            session_id=str(session.id),
            phase="collect",
            status="pending",
            script=project.script or "",
            output_path=None,
        )
        db.add(job)
        project.last_opened_at = datetime.utcnow()
        db.commit()
        db.refresh(job)

        logger.info("Video job %s created via storyboard confirm for project %s", job.id, project_id_str)
        return StoryboardConfirmResponse(
            status="video_started",
            message="Video generation job created",
            job={
                "id": str(job.id),
                "jobId": str(job.id),
                "session_id": str(session.id),
                "status": job.status,
                "phase": job.phase,
            },
        )
    else:
        feedback = (req.feedback or "").strip()
        if not feedback:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Feedback is required when storyboard is not confirmed",
            )

        try:
            adjusted_script = f"{project.script}\n\n[导演调整意见]: {feedback}"
            storyboard = await model_router.video.generate_storyboard(
                adjusted_script,
            )
        except Exception as e:
            logger.exception("Storyboard regeneration failed for project %s", project_id_str)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Storyboard regeneration failed: {str(e)}",
            )

        project.storyboard = storyboard
        project.last_opened_at = datetime.utcnow()
        db.commit()

        logger.info("Storyboard regenerated with feedback for project %s", project_id_str)
        return StoryboardConfirmResponse(
            status="storyboard_regenerated",
            message="Storyboard regenerated with feedback",
            storyboard=storyboard,
        )


# ---------------------------------------------------------------------------
# Script export
# ---------------------------------------------------------------------------

@router.get("/projects/{project_id}/script/export")
async def export_script(
    project_id: UUID,
    format: str = "markdown",
    db: Session = Depends(get_db),
):
    """Export the project script in the requested format (markdown, json, txt)."""
    project_id_str = str(project_id)
    project = db.query(Project).filter(Project.id == project_id_str).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )
    if not (project.script or "").strip():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Project script not generated yet",
        )

    script = project.script or ""

    safe_name = project_id_str[:8]  # ASCII-safe fallback

    if format == "json":
        import json as _json
        turns = [
            {
                "speaker": t.get("speaker", ""),
                "role": t.get("role", ""),
                "content": t.get("content", ""),
                "stage": t.get("stage", ""),
            }
            for t in (project.discussion_history or [])
        ]
        content = _json.dumps({
            "project_id": project_id_str,
            "name": project.name,
            "prompt": project.prompt,
            "script": script,
            "discussion_turns": turns,
            "storyboard": project.storyboard,
        }, ensure_ascii=False, indent=2)
        headers = {"Content-Disposition": f'attachment; filename="{safe_name}_script.json"'}

    elif format == "txt":
        content = script
        headers = {"Content-Disposition": f'attachment; filename="{safe_name}_script.txt"'}

    else:  # markdown
        lines = [
            f"# {project.name}",
            "",
            f"**Prompt:** {project.prompt or '(none)'}",
            f"**Style:** {project.style_preference or 'auto'}",
            f"**Generated:** {project.updated_at.isoformat() if project.updated_at else 'N/A'}",
            "",
            "---",
            "",
            "## Discussion",
            "",
        ]
        for turn in (project.discussion_history or []):
            speaker = turn.get("speaker", "Unknown")
            role = turn.get("role", "")
            content = turn.get("content", "")
            stage = turn.get("stage", "")
            lines.append(f"### {speaker} ({role})")
            lines.append(f"*Stage: {stage}*")
            lines.append("")
            lines.append(content)
            lines.append("")

        lines.extend([
            "---", "",
            "## Final Script", "",
            script, "",
        ])

        if project.storyboard:
            lines.extend([
                "---", "",
                "## Storyboard", "",
            ])
            for i, frame in enumerate(project.storyboard.get("frames", []), 1):
                lines.append(f"### Frame {i}")
                lines.append(f"- **Description:** {frame.get('description', '')}")
                lines.append(f"- **Timing:** {frame.get('timing', '')}")
                if frame.get("visual_prompt"):
                    lines.append(f"- **Visual Prompt:** {frame.get('visual_prompt', '')}")
                lines.append("")
            lines.append(f"**Total Duration:** {project.storyboard.get('total_duration', 'N/A')}")
            lines.append("")

        content = "\n".join(lines)
        headers = {"Content-Disposition": f'attachment; filename="{safe_name}_script.md"'}

    from fastapi.responses import Response

    if format == "json":
        mt = "application/json"
    elif format == "txt":
        mt = "text/plain; charset=utf-8"
    else:
        mt = "text/plain; charset=utf-8"
    return Response(content=content, media_type=mt, headers=headers)
