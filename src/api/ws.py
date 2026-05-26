"""WebSocket API for human-in-the-loop discussion intervention."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Connection registry
# ---------------------------------------------------------------------------

active_connections: dict[str, WebSocket] = {}
pending_inputs: dict[str, asyncio.Event] = {}
pending_input_values: dict[str, str] = {}

# ---------------------------------------------------------------------------
# True real-time intervention -- signals graph nodes to pick up user input
# at any time, not just during the "awaiting_user" phase.
# ---------------------------------------------------------------------------

intervention_flags: dict[str, asyncio.Event] = {}
intervention_texts: dict[str, str] = {}

# Pause / resume at any node boundary
pause_events: dict[str, asyncio.Event] = {}


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@router.websocket("/ws/{session_id}")
async def discussion_websocket(websocket: WebSocket, session_id: str):
    """WebSocket for real-time user intervention during LangGraph discussion."""
    await websocket.accept()
    active_connections[session_id] = websocket
    logger.info("WebSocket connected for session %s", session_id)

    try:
        while True:
            data = await websocket.receive_text()
            try:
                message: dict[str, Any] = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            action = message.get("action", "")

            if action == "intervene":
                user_text = str(message.get("text", ""))

                # Legacy pathway: unblock wait_for_user_input (awaiting_user phase)
                pending_input_values[session_id] = user_text
                event = pending_inputs.get(session_id)
                if event:
                    event.set()

                # New pathway: real-time async intervention at any node boundary
                intervention_texts[session_id] = user_text
                flag = intervention_flags.get(session_id)
                if flag is None:
                    flag = asyncio.Event()
                    intervention_flags[session_id] = flag
                flag.set()

                # Auto-resume if currently paused
                pause_evt = pause_events.get(session_id)
                if pause_evt is not None:
                    pause_evt.set()

                await websocket.send_json({"type": "ack", "action": "intervene"})
                logger.info("User intervention for session %s: %s", session_id, user_text[:80])

            elif action == "pause_now":
                # Create a pause Event in the CLEARED state -- nodes will block on it
                if session_id not in pause_events:
                    pause_events[session_id] = asyncio.Event()
                await websocket.send_json({"type": "paused", "session_id": session_id})
                logger.info("Pause requested for session %s", session_id)

            elif action == "resume_now":
                evt = pause_events.get(session_id)
                if evt is not None:
                    evt.set()
                await websocket.send_json({"type": "resumed", "session_id": session_id})
                logger.info("Resume triggered for session %s", session_id)

            elif action == "ping":
                await websocket.send_json({"type": "pong"})

            else:
                await websocket.send_json({"type": "error", "message": f"Unknown action: {action}"})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for session %s", session_id)
    except Exception as exc:
        logger.error("WebSocket error for session %s: %s", session_id, exc)
    finally:
        active_connections.pop(session_id, None)
        pending_inputs.pop(session_id, None)
        pending_input_values.pop(session_id, None)
        intervention_flags.pop(session_id, None)
        intervention_texts.pop(session_id, None)
        pause_events.pop(session_id, None)


# ---------------------------------------------------------------------------
# Helper for LangGraph integration
# ---------------------------------------------------------------------------

async def wait_for_user_input(
    session_id: str,
    question: str,
    timeout: float = 300.0,
) -> str | None:
    """Wait for user input via WebSocket. Returns None on timeout."""
    ws = active_connections.get(session_id)
    if not ws:
        logger.warning("No active WebSocket for session %s", session_id)
        return None

    event = asyncio.Event()
    pending_inputs[session_id] = event

    try:
        await ws.send_json({
            "type": "question",
            "question": question,
            "timeout": timeout,
        })
        await asyncio.wait_for(event.wait(), timeout=timeout)
        return pending_input_values.get(session_id, "")
    except asyncio.TimeoutError:
        logger.info("User input timeout for session %s", session_id)
        try:
            await ws.send_json({"type": "timeout", "message": "等待用户输入超时，讨论将继续"})
        except Exception:
            pass
        return None
