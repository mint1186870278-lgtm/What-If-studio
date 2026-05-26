"""Feedback API routes for explicit user ratings and preference learning."""

from __future__ import annotations

import logging
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.db import get_db
from src.models import UserPreference, ScriptVector
from src.schemas import FeedbackRequest, FeedbackResponse
from src.core.memory_service import memory_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    req: FeedbackRequest,
    db: Session = Depends(get_db),
):
    """Submit explicit feedback on a discussion/script.

    Feedback is stored in Mem0 (for cross-session preference learning) and
    the local UserPreference table.
    """
    feedback_id = str(uuid4())
    user_id = req.user_id or (req.project_id or "anonymous")

    # Store via memory service
    await memory_service.record_feedback(
        user_id,
        feedback_id,
        {
            "rating": req.rating,
            "comments": req.comments or "",
            "liked": req.liked_aspects,
            "disliked": req.disliked_aspects,
        },
    )

    # Persist individual preferences to local DB
    preferences_saved = 0
    try:
        if req.rating >= 4:
            for liked in req.liked_aspects:
                db.add(UserPreference(
                    user_id=user_id,
                    preference_key="likes",
                    preference_value=liked,
                    confidence=80,
                    source="explicit",
                ))
                preferences_saved += 1
        if req.rating <= 2:
            for disliked in req.disliked_aspects:
                db.add(UserPreference(
                    user_id=user_id,
                    preference_key="dislikes",
                    preference_value=disliked,
                    confidence=80,
                    source="explicit",
                ))
                preferences_saved += 1

        # Store the rating as a preference
        db.add(UserPreference(
            user_id=user_id,
            preference_key="explicit_rating",
            preference_value=str(req.rating),
            confidence=90,
            source="explicit",
        ))
        preferences_saved += 1

        # Link to session if provided
        if req.session_id:
            db.add(UserPreference(
                user_id=user_id,
                preference_key=f"feedback_session_{req.session_id}",
                preference_value=str(req.rating),
                confidence=90,
                source="explicit",
            ))
            preferences_saved += 1

        db.commit()
    except Exception as exc:
        db.rollback()
        logger.warning("Failed to persist feedback preferences: %s", exc)

    logger.info(
        "Feedback %s recorded: user=%s rating=%d comments=%s",
        feedback_id, user_id, req.rating, (req.comments or "")[:60],
    )

    return FeedbackResponse(
        status="ok",
        message=f"Feedback recorded. {preferences_saved} preferences updated.",
        feedback_id=feedback_id,
    )
