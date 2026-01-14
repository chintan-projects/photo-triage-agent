"""API route definitions.

All endpoints return consistent APIResponse[T] structure with error codes.
"""
import shutil
import tempfile
from pathlib import Path

import structlog
from fastapi import APIRouter, Depends, File, Query, Request, UploadFile

from ..database import (
    ActionRepository,
    DuplicateRepository,
    PhotoRepository,
)
from ..services import JobService
from ..services.model_service import classify_image, is_model_loaded
from ..utils.image import SUPPORTED_FORMATS
from .schemas import (
    APIResponse,
    ActionData,
    ActionResponse,
    AnalysisResultsData,
    AnalysisResultsResponse,
    ChatData,
    ChatRefineRequest,
    ChatRequest,
    ChatResponse,
    ClassificationData,
    ClassifyResponse,
    DuplicateGroupData,
    DuplicateListData,
    DuplicateListResponse,
    ErrorCode,
    ExplainData,
    ExplainGroupRequest,
    ExplainResponse,
    FolderAnalysisRequest,
    HealthResponse,
    JobData,
    JobStartResponse,
    JobStatusResponse,
    PhotoListData,
    PhotoListResponse,
    TrashRequest,
    UndoRequest,
    photo_to_data,
    photos_to_data,
)

logger = structlog.get_logger()
router = APIRouter()

# Global job service instance (initialized on first use)
_job_service: JobService | None = None


def get_db(request: Request):
    """Get database connection from app state."""
    return request.app.state.db


def get_job_service(request: Request) -> JobService:
    """Get or create job service instance."""
    global _job_service
    if _job_service is None:
        _job_service = JobService(request.app.state.db)
    return _job_service


# =============================================================================
# Health Check
# =============================================================================

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API is running and model is loaded."""
    return APIResponse.ok(
        data={
            "status": "healthy",
            "model_loaded": is_model_loaded(),
            "version": "0.1.0",
        }
    )


# =============================================================================
# Single Image Analysis
# =============================================================================

@router.post("/analyze/single", response_model=ClassifyResponse)
async def analyze_single(image: UploadFile = File(...)):
    """Analyze a single image.

    Upload an image file (JPEG, PNG, HEIC, WebP) and get classification results.
    """
    logger.info("analyze_single_request", filename=image.filename)

    # Validate file extension
    if image.filename:
        ext = Path(image.filename).suffix.lower()
        if ext not in SUPPORTED_FORMATS:
            return APIResponse.fail(
                f"Unsupported format: {ext}. Supported: {SUPPORTED_FORMATS}",
                ErrorCode.UNSUPPORTED_FORMAT
            )

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=Path(image.filename or ".jpg").suffix,
        ) as tmp:
            content = await image.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Classify the image
        result = classify_image(tmp_path)

        return APIResponse.ok(
            data=ClassificationData(
                filename=image.filename or "unknown",
                category=result.category.value,
                confidence=result.confidence,
                contains_faces=result.contains_faces,
                is_screenshot=result.is_screenshot,
                is_meme=result.is_meme,
                description=result.description,
            )
        )

    except FileNotFoundError as e:
        logger.error("analyze_single_file_error", error=str(e))
        return APIResponse.fail(str(e), ErrorCode.FILE_NOT_FOUND)

    except Exception as e:
        logger.error("analyze_single_error", error=str(e))
        return APIResponse.fail(
            f"Classification failed: {str(e)}",
            ErrorCode.MODEL_INFERENCE_FAILED
        )

    finally:
        # Clean up temp file
        if tmp_path:
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass


# =============================================================================
# Folder Analysis
# =============================================================================

@router.post("/analyze/folder", response_model=JobStartResponse)
async def analyze_folder(
    request: FolderAnalysisRequest,
    job_service: JobService = Depends(get_job_service),
):
    """Start analysis job for a folder.

    Returns a job_id that can be used to check status and get results.
    """
    folder_path = request.folder_path
    logger.info("analyze_folder_request", folder=folder_path)

    # Validate folder exists
    folder = Path(folder_path).expanduser().resolve()
    if not folder.exists():
        return APIResponse.fail(
            f"Folder not found: {folder_path}",
            ErrorCode.FOLDER_NOT_FOUND
        )

    if not folder.is_dir():
        return APIResponse.fail(
            f"Not a directory: {folder_path}",
            ErrorCode.NOT_A_DIRECTORY
        )

    # Start async job
    job_id = await job_service.start_analysis(
        str(folder), skip_lfm=request.skip_lfm
    )

    return APIResponse.ok(
        data=JobData(
            job_id=job_id,
            folder_path=str(folder),
            status="started",
            total_photos=0,
            processed_photos=0,
            progress_percent=0.0,
        )
    )


@router.get("/analyze/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    job_service: JobService = Depends(get_job_service),
):
    """Get current status of an analysis job."""
    status = await job_service.get_status(job_id)
    if not status:
        return APIResponse.fail(
            f"Job not found: {job_id}",
            ErrorCode.JOB_NOT_FOUND
        )

    return APIResponse.ok(data=JobData(**status))


@router.get("/analyze/results/{job_id}", response_model=AnalysisResultsResponse)
async def get_job_results(
    job_id: str,
    db=Depends(get_db),
    job_service: JobService = Depends(get_job_service),
):
    """Get analysis results for a completed job."""
    status = await job_service.get_status(job_id)
    if not status:
        return APIResponse.fail(
            f"Job not found: {job_id}",
            ErrorCode.JOB_NOT_FOUND
        )

    if status["status"] not in ("completed", "failed"):
        return APIResponse.fail(
            f"Job not complete. Status: {status['status']}",
            ErrorCode.JOB_NOT_COMPLETE
        )

    # Get aggregated results
    photo_repo = PhotoRepository(db)
    duplicate_repo = DuplicateRepository(db)

    categories = await photo_repo.get_category_counts()
    blurry = await photo_repo.get_blurry()
    screenshots = await photo_repo.get_screenshots()
    duplicates = await duplicate_repo.get_all_groups()

    return APIResponse.ok(
        data=AnalysisResultsData(
            job_id=job_id,
            total_photos=status["total_photos"],
            categories=categories,
            blurry_count=len(blurry),
            screenshot_count=len(screenshots),
            duplicate_groups=len(duplicates),
            metrics=status.get("metrics"),
        )
    )


# =============================================================================
# Photo Endpoints
# =============================================================================

@router.get("/photos", response_model=PhotoListResponse)
async def list_photos(
    db=Depends(get_db),
    category: str | None = Query(None, description="Filter by category"),
    is_blurry: bool | None = Query(None, description="Filter blurry photos"),
    is_screenshot: bool | None = Query(None, description="Filter screenshots"),
    limit: int = Query(50, ge=1, le=500, description="Results per page"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
):
    """List photos with optional filters."""
    photo_repo = PhotoRepository(db)

    if category:
        photos = await photo_repo.get_by_category(category)
        total = len(photos)
        photos = photos[offset : offset + limit]
    elif is_blurry:
        photos = await photo_repo.get_blurry()
        total = len(photos)
        photos = photos[offset : offset + limit]
    elif is_screenshot:
        photos = await photo_repo.get_screenshots()
        total = len(photos)
        photos = photos[offset : offset + limit]
    else:
        photos = await photo_repo.get_all(limit=limit, offset=offset)
        total = await photo_repo.count()

    return APIResponse.ok(
        data=PhotoListData(photos=photos_to_data(photos)),
        total=total,
        offset=offset,
        limit=limit,
    )


# =============================================================================
# Duplicate Endpoints
# =============================================================================

@router.get("/duplicates", response_model=DuplicateListResponse)
async def list_duplicates(db=Depends(get_db)):
    """Get all duplicate photo groups."""
    duplicate_repo = DuplicateRepository(db)
    groups = await duplicate_repo.get_all_groups()

    return APIResponse.ok(
        data=DuplicateListData(
            groups=[
                DuplicateGroupData(
                    id=g.id,
                    group_type=g.group_type,
                    group_hash=g.group_hash,
                    photo_count=g.photo_count,
                    description=g.description,
                    photos=photos_to_data(g.photos or []),
                )
                for g in groups
            ]
        ),
        total=len(groups),
    )


# =============================================================================
# File Action Endpoints
# =============================================================================

@router.post("/actions/trash", response_model=ActionResponse)
async def trash_photos(request: TrashRequest, db=Depends(get_db)):
    """Move photos to system trash (reversible)."""
    photo_repo = PhotoRepository(db)
    action_repo = ActionRepository(db)

    trashed = 0
    errors = []

    for photo_id in request.photo_ids:
        photo = await photo_repo.get_by_id(photo_id)
        if not photo:
            errors.append(f"Photo {photo_id} not found")
            continue

        photo_path = Path(photo.path)
        if not photo_path.exists():
            errors.append(f"File not found: {photo.path}")
            continue

        try:
            # Move to system trash using send2trash if available, else use temp trash
            trash_dir = Path.home() / ".Trash"
            if trash_dir.exists():
                trash_path = trash_dir / photo_path.name
                # Handle name collision
                counter = 1
                while trash_path.exists():
                    trash_path = trash_dir / f"{photo_path.stem}_{counter}{photo_path.suffix}"
                    counter += 1
                shutil.move(str(photo_path), str(trash_path))
            else:
                # Fallback: create app trash directory
                app_trash = Path.home() / ".photo_triage_trash"
                app_trash.mkdir(exist_ok=True)
                trash_path = app_trash / photo_path.name
                shutil.move(str(photo_path), str(trash_path))

            # Record action for undo
            await action_repo.record(
                action_type="trash",
                photo_id=photo_id,
                original_path=str(photo_path),
                new_path=str(trash_path),
            )
            trashed += 1

        except Exception as e:
            errors.append(f"Failed to trash {photo.path}: {e}")

    if errors:
        return APIResponse.fail(
            f"Trashed {trashed}/{len(request.photo_ids)} photos. Errors: {'; '.join(errors)}",
            ErrorCode.TRASH_FAILED
        )

    return APIResponse.ok(
        data=ActionData(message=f"Moved {trashed} photos to trash")
    )


@router.post("/actions/undo", response_model=ActionResponse)
async def undo_action(request: UndoRequest, db=Depends(get_db)):
    """Undo a previous action (restore from trash)."""
    action_repo = ActionRepository(db)

    action = await action_repo.get(request.action_id)
    if not action:
        return APIResponse.fail(
            f"Action not found: {request.action_id}",
            ErrorCode.ACTION_NOT_FOUND
        )

    if not action.can_undo:
        return APIResponse.fail(
            "Action cannot be undone",
            ErrorCode.ACTION_CANNOT_UNDO
        )

    if action.undone_at:
        return APIResponse.fail(
            "Action already undone",
            ErrorCode.ACTION_ALREADY_UNDONE
        )

    try:
        if action.action_type == "trash" and action.new_path:
            # Restore from trash
            trash_path = Path(action.new_path)
            original_path = Path(action.original_path)

            if not trash_path.exists():
                return APIResponse.fail(
                    f"Trash file not found: {trash_path}",
                    ErrorCode.FILE_NOT_FOUND
                )

            # Ensure original directory exists
            original_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(trash_path), str(original_path))

        # Mark action as undone
        await action_repo.mark_undone(action.id)

        return APIResponse.ok(
            data=ActionData(
                action_id=action.id,
                message=f"Undone action: {action.action_type}",
            )
        )

    except Exception as e:
        logger.error("undo_failed", action_id=action.id, error=str(e))
        return APIResponse.fail(
            f"Undo failed: {str(e)}",
            ErrorCode.INTERNAL_ERROR
        )


# =============================================================================
# Chat/Search Endpoints
# =============================================================================

# Global services (initialized on first use)
_conversation_service = None
_search_agent = None


def get_conversation_service(request: Request):
    """Get or create conversation service."""
    global _conversation_service
    if _conversation_service is None:
        from ..services import ConversationService

        _conversation_service = ConversationService(request.app.state.db)
    return _conversation_service


def get_search_agent(request: Request):
    """Get or create search agent."""
    global _search_agent
    if _search_agent is None:
        from ..services import PhotoSearchAgent

        # Pass model if available (initialized at startup)
        model = getattr(request.app.state, "model", None)
        _search_agent = PhotoSearchAgent(request.app.state.db, model_provider=model)
    return _search_agent


@router.post("/chat", response_model=ChatResponse)
async def chat_search(
    request: ChatRequest,
    http_request: Request,
):
    """Natural language photo search.

    Send a message like "Find photos from my trip to Japan" and get matching photos.
    """
    conversation_service = get_conversation_service(http_request)
    search_agent = get_search_agent(http_request)

    logger.info("chat_request", message=request.message[:100])

    try:
        # Get or create conversation
        conv = await conversation_service.get_or_create(request.conversation_id)

        # Add user message
        await conversation_service.add_message(conv.id, "user", request.message)

        # Search using the agent
        result = await search_agent.search(
            query=request.message,
            conversation_history=conv.get_history(),
        )

        # Add assistant response
        photo_ids = [p.id for p in result.photos]
        response_text = result.interpretation
        if result.clarifying_question:
            response_text += f"\n\n{result.clarifying_question}"

        await conversation_service.add_message(
            conv.id, "assistant", response_text, photo_ids
        )

        # Generate suggested actions based on results
        suggested_actions = []
        if result.total_found > 10:
            suggested_actions.append("Show only the best quality")
        if result.total_found > 0:
            suggested_actions.append("Filter by category")
            suggested_actions.append("Find similar photos")

        return APIResponse.ok(
            data=ChatData(
                response=result.interpretation,
                photos=photos_to_data(result.photos),
                total_found=result.total_found,
                conversation_id=conv.id,
                confidence=result.confidence,
                clarifying_question=result.clarifying_question,
                suggested_actions=suggested_actions if suggested_actions else None,
            )
        )

    except Exception as e:
        logger.error("chat_error", error=str(e))
        return APIResponse.fail(
            f"Search failed: {str(e)}",
            ErrorCode.SEARCH_FAILED
        )


@router.post("/chat/refine", response_model=ChatResponse)
async def chat_refine(
    request: ChatRefineRequest,
    http_request: Request,
):
    """Refine previous search results.

    Use this after a /chat request to narrow or expand results.
    """
    conversation_service = get_conversation_service(http_request)
    search_agent = get_search_agent(http_request)

    logger.info("chat_refine_request", feedback=request.feedback[:100])

    try:
        # Get conversation
        conv = await conversation_service.get(request.conversation_id)
        if not conv:
            return APIResponse.fail(
                f"Conversation not found: {request.conversation_id}",
                ErrorCode.CONVERSATION_NOT_FOUND
            )

        # Get previous search context from conversation
        history = conv.get_history()

        # Add user feedback
        await conversation_service.add_message(conv.id, "user", request.feedback)

        # Create a mock previous result from the last assistant message
        from ..services import SearchResult

        last_assistant = None
        for msg in reversed(history):
            if msg["role"] == "assistant":
                last_assistant = msg
                break

        previous_result = SearchResult(
            interpretation=last_assistant["content"] if last_assistant else "",
            photos=[],
            total_found=len(last_assistant.get("photo_ids", []))
            if last_assistant
            else 0,
            confidence=0.7,
        )

        # Refine search
        result = await search_agent.refine(
            feedback=request.feedback,
            previous_result=previous_result,
        )

        # Add assistant response
        photo_ids = [p.id for p in result.photos]
        await conversation_service.add_message(
            conv.id, "assistant", result.interpretation, photo_ids
        )

        return APIResponse.ok(
            data=ChatData(
                response=result.interpretation,
                photos=photos_to_data(result.photos),
                total_found=result.total_found,
                conversation_id=conv.id,
                confidence=result.confidence,
                clarifying_question=None,
                suggested_actions=None,
            )
        )

    except Exception as e:
        logger.error("chat_refine_error", error=str(e))
        return APIResponse.fail(
            f"Refine failed: {str(e)}",
            ErrorCode.SEARCH_FAILED
        )


@router.post("/chat/explain", response_model=ExplainResponse)
async def explain_photo_group(
    request: ExplainGroupRequest,
    http_request: Request,
):
    """Get LFM explanation of why photos are grouped together."""
    search_agent = get_search_agent(http_request)

    logger.info("explain_group_request", photo_count=len(request.photo_ids))

    try:
        explanation = await search_agent.explain_group(request.photo_ids)
        return APIResponse.ok(data=ExplainData(explanation=explanation))
    except Exception as e:
        logger.error("explain_group_error", error=str(e))
        return APIResponse.fail(
            f"Could not explain group: {str(e)}",
            ErrorCode.MODEL_INFERENCE_FAILED
        )


# =============================================================================
# Upload Endpoints
# =============================================================================

from ..services.upload_service import get_upload_service
from .schemas import (
    AnalyzeUploadedRequest,
    UploadSessionData,
    UploadSessionResponse,
    UploadStatsData,
    UploadStatsResponse,
)


@router.post("/upload/session", response_model=UploadSessionResponse)
async def create_upload_session():
    """Create a new upload session.

    Call this first, then upload photos to the returned session_id.
    """
    upload_service = get_upload_service()
    session = upload_service.create_session()

    return APIResponse.ok(
        data=UploadSessionData(
            session_id=session.session_id,
            photo_count=0,
            message="Session created. Upload photos using /upload/photos endpoint.",
        )
    )


@router.post("/upload/photos/{session_id}", response_model=UploadStatsResponse)
async def upload_photos(
    session_id: str,
    files: list[UploadFile] = File(...),
):
    """Upload photos to an existing session.

    Accepts multiple photo files. Non-image files are skipped.
    """
    upload_service = get_upload_service()
    session = upload_service.get_session(session_id)

    if not session:
        return APIResponse.fail(
            f"Session not found: {session_id}",
            ErrorCode.NOT_FOUND
        )

    files_saved = 0
    files_skipped = 0

    for file in files:
        if not file.filename:
            files_skipped += 1
            continue

        content = await file.read()
        result = await upload_service.add_photo(session_id, file.filename, content)

        if result:
            files_saved += 1
        else:
            files_skipped += 1

    logger.info(
        "photos_uploaded",
        session_id=session_id,
        received=len(files),
        saved=files_saved,
        skipped=files_skipped,
    )

    return APIResponse.ok(
        data=UploadStatsData(
            files_received=len(files),
            files_saved=files_saved,
            files_skipped=files_skipped,
            session_id=session_id,
        )
    )


@router.get("/upload/session/{session_id}", response_model=UploadSessionResponse)
async def get_upload_session(session_id: str):
    """Get current status of an upload session."""
    upload_service = get_upload_service()
    session = upload_service.get_session(session_id)

    if not session:
        return APIResponse.fail(
            f"Session not found: {session_id}",
            ErrorCode.NOT_FOUND
        )

    return APIResponse.ok(
        data=UploadSessionData(
            session_id=session.session_id,
            photo_count=session.photo_count,
        )
    )


@router.post("/analyze/uploaded", response_model=JobStartResponse)
async def analyze_uploaded(
    request: AnalyzeUploadedRequest,
    job_service: JobService = Depends(get_job_service),
):
    """Start analysis on uploaded photos.

    Use this after uploading photos via /upload/photos.
    """
    upload_service = get_upload_service()
    folder_path = upload_service.get_session_folder(request.session_id)

    if not folder_path or not folder_path.exists():
        return APIResponse.fail(
            f"Upload session not found or empty: {request.session_id}",
            ErrorCode.NOT_FOUND
        )

    session = upload_service.get_session(request.session_id)
    if not session or session.photo_count == 0:
        return APIResponse.fail(
            "No photos in upload session",
            ErrorCode.VALIDATION_ERROR
        )

    logger.info(
        "analyze_uploaded_request",
        session_id=request.session_id,
        photo_count=session.photo_count,
    )

    # Start async job on the upload folder
    job_id = await job_service.start_analysis(
        str(folder_path), skip_lfm=request.skip_lfm
    )

    return APIResponse.ok(
        data=JobData(
            job_id=job_id,
            folder_path=str(folder_path),
            status="started",
            total_photos=session.photo_count,
            processed_photos=0,
            progress_percent=0.0,
        )
    )


@router.delete("/upload/session/{session_id}")
async def delete_upload_session(session_id: str):
    """Delete an upload session and its files."""
    upload_service = get_upload_service()

    if upload_service.cleanup_session(session_id):
        return APIResponse.ok(
            data={"message": f"Session {session_id} deleted"}
        )
    else:
        return APIResponse.fail(
            f"Session not found: {session_id}",
            ErrorCode.NOT_FOUND
        )
