"""Session management API routes with SSE streaming"""

import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session as DBSession

from src.db import get_db
from src.models import Project, Session, VideoJob
from src.schemas import SessionCreate, SessionResponse, DiscussionTurn

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
    # Verify project exists
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

        logger.info(f"✅ Session created: {session.id}")
        return session

    except Exception as e:
        db.rollback()
        logger.error(f"❌ Failed to create session: {e}")
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
) -> AsyncGenerator[str, None]:
    """Generate SSE stream for discussion"""
    try:
        # Get session from database
        session = db.query(Session).filter(Session.id == session_id).first()
        if not session:
            yield f"data: {json.dumps({'error': 'Session not found'})}\n\n"
            return
        project = db.query(Project).filter(Project.id == session.project_id).first()

        turns = []
        script = ""

        async for event in _discuss_stream(
            user_request=session.prompt,
            style=session.style_preference,
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

        logger.info(f"✅ Discussion stream completed for session {session_id}")

    except Exception as e:
        failed_session = db.query(Session).filter(Session.id == session_id).first()
        if failed_session:
            failed_session.status = "failed"
        if failed_session and failed_session.project_id:
            failed_project = db.query(Project).filter(Project.id == failed_session.project_id).first()
            if failed_project:
                failed_project.discussion_status = "failed"
        db.commit()
        logger.error(f"❌ Discussion stream error: {e}")
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"


@router.get("/sessions/{session_id}/stream")
async def stream_discussion(
    session_id: str,
    db: DBSession = Depends(get_db),
):
    """Stream discussion as Server-Sent Events"""
    return StreamingResponse(
        generate_discussion_stream(session_id, db),
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
):
    """Stream discussion (legacy endpoint for compatibility)"""
    return StreamingResponse(
        generate_discussion_stream(session_id, db),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
