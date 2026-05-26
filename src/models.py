"""SQLAlchemy ORM models"""

from datetime import datetime
from uuid import uuid4
import uuid
from sqlalchemy import Column, String, Integer, DateTime, Text, JSON, ForeignKey, Enum
from sqlalchemy.orm import relationship
import enum

from src.db import Base


class Project(Base):
    """Video editing project"""

    __tablename__ = "projects"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    prompt = Column(Text, nullable=True)
    style_preference = Column(String(50), default="auto", nullable=False)
    script = Column(Text, nullable=True)
    discussion_history = Column(JSON, default=list, nullable=False)
    discussion_status = Column(String(50), default="idle", nullable=False)
    output_type = Column(String(50), default="script_only", nullable=False)
    storyboard = Column(JSON, default=None, nullable=True)
    last_opened_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata_ = Column(JSON, default=dict, nullable=False)  # Custom fields

    assets = relationship("Asset", back_populates="project", cascade="all, delete-orphan")
    sessions = relationship("Session", back_populates="project", cascade="all, delete-orphan")


class Asset(Base):
    """Project asset (video, audio, image, text)"""

    __tablename__ = "assets"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False, index=True)
    file_type = Column(String(50), nullable=False)  # 'video', 'audio', 'image', 'text'
    file_name = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False, unique=True)
    file_size = Column(Integer, nullable=False)  # bytes
    metadata_ = Column(JSON, default=dict, nullable=False)  # resolution, duration, etc.
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    project = relationship("Project", back_populates="assets")


class Session(Base):
    """Creative session for a project"""

    __tablename__ = "sessions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False, index=True)
    prompt = Column(Text, nullable=False)  # User's input prompt
    style_preference = Column(String(50), default="auto", nullable=False)
    status = Column(String(50), default="active", nullable=False)  # 'active', 'completed', 'failed'
    script = Column(Text, nullable=True)  # Markdown format script (initially empty)
    discussion_history = Column(JSON, default=list, nullable=False)  # List of DiscussionTurn
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    project = relationship("Project", back_populates="sessions")
    video_jobs = relationship("VideoJob", back_populates="session", cascade="all, delete-orphan")


class VideoJob(Base):
    """Video generation task"""

    __tablename__ = "video_jobs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    session_id = Column(String(36), ForeignKey("sessions.id"), nullable=False, index=True)
    phase = Column(
        String(50),
        default="collect",
        nullable=False,
    )  # 'collect', 'analyze', 'discuss', 'edit', 'render', 'deliver'
    status = Column(String(50), default="pending", nullable=False)  # 'pending', 'running', 'done', 'failed'
    script = Column(Text, nullable=True)  # Markdown script
    output_path = Column(String(512), nullable=True)  # Final video file path
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    session = relationship("Session", back_populates="video_jobs")


class ANetInvocation(Base):
    """ANet agent service invocation log"""

    __tablename__ = "anet_invocations"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    service_name = Column(String(255), nullable=False, index=True)
    status = Column(String(50), nullable=False)  # 'pending', 'success', 'error'
    payload = Column(JSON, nullable=True)
    response = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)


# ---------------------------------------------------------------------------
# New models for refactored system (LangGraph + Memory)
# ---------------------------------------------------------------------------


class UserPreference(Base):
    """Cross-session user preference (Mem0-backed with local cache)."""

    __tablename__ = "user_preferences"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id = Column(String(255), nullable=False, index=True)
    preference_key = Column(String(255), nullable=False)
    preference_value = Column(Text, nullable=True)
    confidence = Column(Integer, default=50)
    source = Column(String(50), default="inferred")  # 'explicit', 'inferred', 'feedback'
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class ScriptVector(Base):
    """Metadata for scripts stored in Chroma vector DB."""

    __tablename__ = "script_vectors"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    chroma_doc_id = Column(String(255), nullable=False, unique=True, index=True)
    project_id = Column(String(36), nullable=True, index=True)
    user_id = Column(String(255), nullable=True, index=True)
    style = Column(String(50), nullable=True)
    prompt_hash = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class DiscussionCheckpoint(Base):
    """LangGraph discussion checkpoint metadata."""

    __tablename__ = "discussion_checkpoints"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    session_id = Column(String(36), nullable=False, index=True)
    thread_id = Column(String(255), nullable=False, unique=True, index=True)
    checkpoint_data = Column(JSON, nullable=False)  # Serialized LangGraph checkpoint
    status = Column(String(50), default="active")  # 'active', 'completed', 'interrupted'
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
