"""Pydantic schemas for API requests and responses.

All API responses follow a consistent structure:
{
    "success": true/false,
    "data": <response-specific data>,
    "error": "error message if failed",
    "meta": {"error_code": "CODE", "timestamp": "..."}
}
"""
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

from ..database.repository import Photo


# =============================================================================
# Error Codes
# =============================================================================

class ErrorCode(str, Enum):
    """Standardized error codes for API responses."""
    # General errors
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    INTERNAL_ERROR = "INTERNAL_ERROR"

    # Analysis errors
    MODEL_NOT_LOADED = "MODEL_NOT_LOADED"
    MODEL_INFERENCE_FAILED = "MODEL_INFERENCE_FAILED"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    FOLDER_NOT_FOUND = "FOLDER_NOT_FOUND"
    NOT_A_DIRECTORY = "NOT_A_DIRECTORY"

    # Job errors
    JOB_NOT_FOUND = "JOB_NOT_FOUND"
    JOB_NOT_COMPLETE = "JOB_NOT_COMPLETE"
    JOB_FAILED = "JOB_FAILED"

    # Action errors
    ACTION_NOT_FOUND = "ACTION_NOT_FOUND"
    ACTION_CANNOT_UNDO = "ACTION_CANNOT_UNDO"
    ACTION_ALREADY_UNDONE = "ACTION_ALREADY_UNDONE"
    TRASH_FAILED = "TRASH_FAILED"

    # Chat errors
    CONVERSATION_NOT_FOUND = "CONVERSATION_NOT_FOUND"
    SEARCH_FAILED = "SEARCH_FAILED"


# =============================================================================
# Response Meta
# =============================================================================

class ResponseMeta(BaseModel):
    """Metadata included in all API responses."""
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )
    error_code: ErrorCode | None = None

    # Optional pagination info
    total: int | None = None
    offset: int | None = None
    limit: int | None = None


# =============================================================================
# Generic API Response
# =============================================================================

T = TypeVar("T")


class APIResponse(BaseModel, Generic[T]):
    """Generic wrapper for all API responses.

    Usage:
        return APIResponse(success=True, data=MyData(...))
        return APIResponse.error("Something failed", ErrorCode.INTERNAL_ERROR)
    """
    success: bool
    data: T | None = None
    error: str | None = None
    meta: ResponseMeta = Field(default_factory=ResponseMeta)

    @classmethod
    def ok(cls, data: T, **meta_kwargs) -> "APIResponse[T]":
        """Create a successful response."""
        return cls(
            success=True,
            data=data,
            meta=ResponseMeta(**meta_kwargs)
        )

    @classmethod
    def fail(cls, error: str, error_code: ErrorCode = ErrorCode.INTERNAL_ERROR) -> "APIResponse[None]":
        """Create an error response."""
        return cls(
            success=False,
            error=error,
            meta=ResponseMeta(error_code=error_code)
        )


# =============================================================================
# Data Models (used in responses)
# =============================================================================

class PhotoData(BaseModel):
    """Photo information returned in API responses."""
    id: int
    path: str
    filename: str
    file_hash: str | None = None
    phash: str | None = None
    file_size: int | None = None
    width: int | None = None
    height: int | None = None


class ClassificationData(BaseModel):
    """Classification result for a single image."""
    filename: str
    category: str
    confidence: float
    contains_faces: bool
    is_screenshot: bool
    is_meme: bool
    description: str


class MetricsData(BaseModel):
    """Real-time job metrics."""
    job_id: str
    photos_processed: int
    total_photos: int
    progress_percent: float
    photos_per_second: float
    memory_mb: float
    memory_percent: float
    avg_inference_ms: float
    current_photo: str | None = None
    elapsed_seconds: float
    eta_seconds: float | None = None


class JobData(BaseModel):
    """Job status information."""
    job_id: str
    folder_path: str
    status: str
    total_photos: int
    processed_photos: int
    progress_percent: float
    error: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    metrics: MetricsData | None = None


class DuplicateGroupData(BaseModel):
    """Duplicate group information."""
    id: int
    group_type: str
    group_hash: str | None = None
    photo_count: int
    description: str | None = None
    photos: list[PhotoData]


class AnalysisResultsData(BaseModel):
    """Aggregated analysis results."""
    job_id: str
    total_photos: int
    categories: dict[str, int]
    blurry_count: int
    screenshot_count: int
    duplicate_groups: int
    metrics: dict[str, Any] | None = None


class PhotoListData(BaseModel):
    """List of photos with pagination info."""
    photos: list[PhotoData]


class DuplicateListData(BaseModel):
    """List of duplicate groups."""
    groups: list[DuplicateGroupData]


class ActionData(BaseModel):
    """Action result information."""
    action_id: int | None = None
    message: str


class ChatData(BaseModel):
    """Chat/search response data."""
    response: str
    photos: list[PhotoData]
    total_found: int
    conversation_id: str
    confidence: float
    clarifying_question: str | None = None
    suggested_actions: list[str] | None = None


class ExplainData(BaseModel):
    """Group explanation data."""
    explanation: str


# =============================================================================
# Request Models
# =============================================================================

class FolderAnalysisRequest(BaseModel):
    """Request to start folder analysis."""
    folder_path: str
    skip_lfm: bool = False


class TrashRequest(BaseModel):
    """Request to trash photos."""
    photo_ids: list[int]


class MoveRequest(BaseModel):
    """Request to move photos."""
    photo_ids: list[int]
    destination: str


class UndoRequest(BaseModel):
    """Request to undo an action."""
    action_id: int


class ChatRequest(BaseModel):
    """Request for natural language search."""
    message: str
    conversation_id: str | None = None


class ChatRefineRequest(BaseModel):
    """Request to refine previous search."""
    conversation_id: str
    feedback: str


class ExplainGroupRequest(BaseModel):
    """Request to explain a photo group."""
    photo_ids: list[int]


class AnalyzeUploadedRequest(BaseModel):
    """Request to analyze uploaded photos."""
    session_id: str
    skip_lfm: bool = False


# =============================================================================
# Upload Data Models
# =============================================================================

class UploadSessionData(BaseModel):
    """Upload session information."""
    session_id: str
    photo_count: int
    message: str | None = None


class UploadStatsData(BaseModel):
    """Upload statistics."""
    files_received: int
    files_saved: int
    files_skipped: int
    session_id: str


# =============================================================================
# Response Type Aliases (for cleaner route signatures)
# =============================================================================

HealthResponse = APIResponse[dict]
ClassifyResponse = APIResponse[ClassificationData]
JobStartResponse = APIResponse[JobData]
JobStatusResponse = APIResponse[JobData]
AnalysisResultsResponse = APIResponse[AnalysisResultsData]
PhotoListResponse = APIResponse[PhotoListData]
DuplicateListResponse = APIResponse[DuplicateListData]
ActionResponse = APIResponse[ActionData]
ChatResponse = APIResponse[ChatData]
ExplainResponse = APIResponse[ExplainData]
UploadSessionResponse = APIResponse[UploadSessionData]
UploadStatsResponse = APIResponse[UploadStatsData]


# =============================================================================
# Conversion Utilities
# =============================================================================

def photo_to_data(photo: Photo) -> PhotoData:
    """Convert a Photo database model to PhotoData response model.

    This utility ensures consistent conversion across all endpoints.
    """
    return PhotoData(
        id=photo.id,
        path=photo.path,
        filename=photo.filename,
        file_hash=photo.file_hash,
        phash=photo.phash,
        file_size=photo.file_size,
        width=photo.width,
        height=photo.height,
    )


def photos_to_data(photos: list[Photo]) -> list[PhotoData]:
    """Convert a list of Photo models to PhotoData response models."""
    return [photo_to_data(p) for p in photos]
