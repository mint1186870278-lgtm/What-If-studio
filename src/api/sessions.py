"""Session management API routes with SSE streaming"""

import hashlib
import json
import logging
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session as DBSession

from src.db import get_db
from src.models import Project, Session, VideoJob, UserPreference, ScriptVector
from src.schemas import SessionCreate, SessionResponse, DiscussionTurn, InterveneRequest
from src.core.memory_service import memory_service, detect_script_tone

# Prefer LangGraph; fall back to AutoGen
try:
    from src.agents import run_langgraph_discussion_stream as _discuss_stream
    _BACKEND = "langgraph"
except Exception:
    from src.agents import run_autogen_discussion_stream as _discuss_stream  # type: ignore[assignment]
    _BACKEND = "autogen"

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    session_create: SessionCreate,
    db: DBSession = Depends(get_db),
):
    """Create a new session"""
    project = db.query(Project).filter(Project.id == session_create.project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {session_create.project_id} not found",
        )

    try:
        session = Session(
            project_id=session_create.project_id,
            prompt=session_create.prompt,
            style_preference=session_create.style_preference,
            status="active",
            discussion_history=[],
            script="",
        )
        db.add(session)
        db.commit()
        db.refresh(session)

        logger.info(f"Session created: {session.id}")
        return session

    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create session: {str(e)}",
        )


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    db: DBSession = Depends(get_db),
):
    """Get session by ID"""
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )
    return session


async def generate_discussion_stream(
    session_id: str,
    db: DBSession,
    user_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """Generate SSE stream for discussion"""
    try:
        session = db.query(Session).filter(Session.id == session_id).first()
        if not session:
            yield f"data: {json.dumps({'error': 'Session not found'})}\n\n"
            return
        project = db.query(Project).filter(Project.id == session.project_id).first()

        resolved_user = user_id or (str(project.id) if project else None)
        turns = []
        script = ""

        async for event in _discuss_stream(
            user_request=session.prompt,
            style=session.style_preference,
            user_id=resolved_user,
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            if event.get("type") == "turn":
                turns.append(event)
            elif event.get("type") == "script":
                script = str(event.get("script", ""))

        if not script.strip():
            raise RuntimeError("Discussion completed without a valid script.")
        session.script = script
        session.discussion_history = turns
        session.status = "completed"
        if project is not None:
            project.script = script
            project.discussion_history = turns
            project.discussion_status = "completed"
        db.commit()

        # --- Memory persistence: auto-save scripts ---
        if script:
            metadata = {
                "project_id": session.project_id,
                "session_id": session_id,
                "style": session.style_preference or "auto",
                "user_request": session.prompt[:256],
                "timestamp": str(session.created_at) if session.created_at else "",
            }
            if resolved_user:
                metadata["user_id"] = resolved_user
            try:
                chroma_id = await memory_service.store_script(script, metadata)
                sv = ScriptVector(
                    chroma_doc_id=chroma_id,
                    project_id=session.project_id,
                    user_id=resolved_user or "anonymous",
                    style=session.style_preference or "auto",
                    prompt_hash=hashlib.sha256(session.prompt.encode()).hexdigest()[:64],
                )
                db.add(sv)
                db.commit()
                logger.info("Script saved to memory: chroma_id=%s session=%s", chroma_id, session_id)
            except Exception as mem_exc:
                logger.warning("Failed to save script to memory: %s", mem_exc)

        # --- Auto-learn user preferences ---
        if resolved_user and script:
            try:
                tone = detect_script_tone(script)
                for pref_key, pref_val in [
                    ("style_preference", session.style_preference or "auto"),
                    ("last_script_type", tone),
                ]:
                    await memory_service.store_user_preference(resolved_user, {
                        "key": pref_key,
                        "value": pref_val,
                        "source": "inferred",
                        "confidence": 70 if pref_key == "style_preference" else 60,
                    })
                    db.add(UserPreference(
                        user_id=resolved_user,
                        preference_key=pref_key,
                        preference_value=pref_val,
                        confidence=70 if pref_key == "style_preference" else 60,
                        source="inferred",
                    ))
                db.commit()
                logger.info("User preferences inferred: user=%s style=%s tone=%s", resolved_user, session.style_preference, tone)
            except Exception as pref_exc:
                db.rollback()
                logger.warning("Failed to store inferred preferences: %s", pref_exc)

        logger.info(f"Discussion stream completed for session {session_id}")

    except Exception as e:
        failed_session = db.query(Session).filter(Session.id == session_id).first()
        if failed_session:
            failed_session.status = "failed"
        if failed_session and failed_session.project_id:
            failed_project = db.query(Project).filter(Project.id == failed_session.project_id).first()
            if failed_project:
                failed_project.discussion_status = "failed"
        db.commit()
        logger.error(f"Discussion stream error: {e}")
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"


@router.get("/sessions/{session_id}/stream")
async def stream_discussion(
    session_id: str,
    db: DBSession = Depends(get_db),
    user_id: Optional[str] = Query(None, description="Optional user identifier for memory features"),
):
    """Stream discussion as Server-Sent Events"""
    return StreamingResponse(
        generate_discussion_stream(session_id, db, user_id=user_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/sessions/{session_id}/discuss/stream")
async def stream_discussion_legacy(
    session_id: str,
    db: DBSession = Depends(get_db),
    user_id: Optional[str] = Query(None, description="Optional user identifier for memory features"),
):
    """Stream discussion (legacy endpoint for compatibility)"""
    return StreamingResponse(
        generate_discussion_stream(session_id, db, user_id=user_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# REST intervention endpoint (fallback for WebSocket)
# ---------------------------------------------------------------------------

@router.post("/sessions/{session_id}/intervene")
async def intervene_session(
    session_id: str,
    body: InterveneRequest,
):
    """Inject user intervention into a running discussion via HTTP.

    This is a REST fallback for the WebSocket ``intervene`` action.
    """
    import asyncio as _asyncio

    from src.api.ws import intervention_flags, intervention_texts, pause_events

    text = body.text
    intervention_texts[session_id] = text

    flag = intervention_flags.get(session_id)
    if flag is None:
        flag = _asyncio.Event()
        intervention_flags[session_id] = flag
    flag.set()

    # Auto-resume if paused
    pause_evt = pause_events.get(session_id)
    if pause_evt is not None:
        pause_evt.set()

    logger.info("REST intervention for session %s: %s", session_id, text[:80])

    return {
        "status": "ok",
        "message": "Intervention queued -- will be injected at the next node boundary",
        "session_id": session_id,
    }
