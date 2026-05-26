"""Pydantic schemas for request/response validation"""

from datetime import datetime
from typing import Optional, List, Any
from uuid import UUID
from pydantic import BaseModel, Field


# Project schemas
class ProjectCreate(BaseModel):
    """Create project request"""

    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    prompt: Optional[str] = None
    style_preference: Optional[str] = "auto"


class ProjectUpdate(BaseModel):
    """Update project request"""

    name: Optional[str] = None
    description: Optional[str] = None
    prompt: Optional[str] = None
    style_preference: Optional[str] = None
    output_type: Optional[str] = None


class ProjectResponse(BaseModel):
    """Project response"""

    id: UUID
    name: str
    description: Optional[str]
    prompt: Optional[str]
    style_preference: str
    script: Optional[str]
    discussion_history: List[Any]
    discussion_status: str
    output_type: str
    storyboard: Optional[dict] = None
    last_opened_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    metadata_: dict

    class Config:
        from_attributes = True


# Asset schemas
class AssetMetadata(BaseModel):
    """Asset metadata"""

    duration: Optional[float] = None  # For video/audio
    resolution: Optional[str] = None  # For video/image
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None


class AssetResponse(BaseModel):
    """Asset response"""

    id: UUID
    project_id: UUID
    file_type: str
    file_name: str
    file_path: str
    file_size: int
    metadata_: dict
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Session schemas
class DiscussionTurn(BaseModel):
    """Single turn in discussion"""

    speaker: str
    role: str  # 'guardian', 'director', 'crew'
    content: str
    stage: str  # 'briefing', 'topic-1', 'topic-2', 'topic-3', 'finalize'
    ts: int  # timestamp in ms


class SessionCreate(BaseModel):
    """Create session request"""

    project_id: str
    prompt: str = Field(..., min_length=1)
    style_preference: str = Field(default="auto")


class SessionResponse(BaseModel):
    """Session response"""

    id: UUID
    project_id: UUID
    prompt: str
    style_preference: str
    status: str
    script: Optional[str]
    discussion_history: List[DiscussionTurn]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# VideoJob schemas
class VideoJobCreate(BaseModel):
    """Create video job request"""

    session_id: str
    asset_ids: List[str] = Field(default_factory=list)


class VideoJobResponse(BaseModel):
    """Video job response"""

    id: UUID
    session_id: UUID
    phase: str
    status: str
    script: Optional[str]
    output_path: Optional[str]
    error: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Gateway schemas
class ANetInvocationResponse(BaseModel):
    """ANet invocation log response"""

    id: UUID
    service_name: str
    status: str
    payload: Optional[dict]
    response: Optional[dict]
    error: Optional[str]
    timestamp: datetime

    class Config:
        from_attributes = True


class GatewayCapability(BaseModel):
    """Gateway capability description"""

    name: str
    service_name: str
    description: str
    input_schema: dict
    output_schema: dict


class GatewayService(BaseModel):
    """Registered service in gateway"""

    name: str
    endpoint: str
    status: str  # 'active', 'inactive'
    tags: List[str]


# Feedback schemas
class FeedbackRequest(BaseModel):
    """Explicit user feedback on a discussion/script"""

    session_id: Optional[str] = None
    project_id: Optional[str] = None
    user_id: Optional[str] = None
    rating: int = Field(..., ge=1, le=5)
    comments: Optional[str] = None
    liked_aspects: List[str] = Field(default_factory=list)
    disliked_aspects: List[str] = Field(default_factory=list)


class FeedbackResponse(BaseModel):
    """Feedback acknowledgement"""

    status: str = "ok"
    message: str
    feedback_id: Optional[str] = None


# Output format selection
class OutputSelectRequest(BaseModel):
    """Request to set project output format preference"""

    output_type: str = Field(..., pattern=r"^(script_only|script_and_storyboard|script_and_video)$")


class StoryboardGenerateResponse(BaseModel):
    """Storyboard generation response"""

    project_id: str
    frames: list[dict]
    total_duration: str
    generated_at: Optional[datetime] = None


class StoryboardConfirmRequest(BaseModel):
    """Confirm or reject a storyboard"""

    confirmed: bool
    feedback: Optional[str] = None


class StoryboardConfirmResponse(BaseModel):
    """Response after storyboard confirmation"""

    status: str
    message: str
    job: Optional[dict] = None
    storyboard: Optional[dict] = None


class ScriptExportResponse(BaseModel):
    """Script export response"""

    project_id: str
    format: str
    content: str


class InterveneRequest(BaseModel):
    """Request to inject user intervention via REST"""

    text: str
