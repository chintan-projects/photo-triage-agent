"""Web dashboard for Photo Triage Agent."""
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from ..database import DuplicateRepository, PhotoRepository

# Setup templates
TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

router = APIRouter()


def get_db(request: Request):
    """Get database connection from app state."""
    return request.app.state.db


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page - start analysis."""
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/progress/{job_id}", response_class=HTMLResponse)
async def progress_page(request: Request, job_id: str):
    """Progress page - show analysis progress."""
    return templates.TemplateResponse(
        "progress.html", {"request": request, "job_id": job_id}
    )


@router.get("/results/{job_id}", response_class=HTMLResponse)
async def results_page(request: Request, job_id: str, db=Depends(get_db)):
    """Results page - show analysis summary."""
    from ..services import JobService

    job_service = JobService(db)
    status = await job_service.get_status(job_id)

    if not status:
        raise HTTPException(status_code=404, detail="Job not found")

    # Get aggregated results
    photo_repo = PhotoRepository(db)
    duplicate_repo = DuplicateRepository(db)

    categories = await photo_repo.get_category_counts()
    blurry = await photo_repo.get_blurry()
    screenshots = await photo_repo.get_screenshots()
    duplicates = await duplicate_repo.get_all_groups()

    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "job_id": job_id,
            "total_photos": status["total_photos"],
            "processed_photos": status["processed_photos"],
            "categories": categories,
            "blurry_count": len(blurry),
            "screenshot_count": len(screenshots),
            "duplicate_groups": len(duplicates),
            "metrics": status.get("metrics"),
        },
    )


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(
    request: Request,
    category: str | None = None,
    is_blurry: bool = False,
    is_screenshot: bool = False,
    limit: int = 50,
    offset: int = 0,
    db=Depends(get_db),
):
    """Dashboard - browse photos with filters."""
    photo_repo = PhotoRepository(db)

    # Get filtered photos
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

    # Get available categories for filter
    categories = list((await photo_repo.get_category_counts()).keys())

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "photos": photos,
            "categories": categories,
            "current_category": category,
            "is_blurry": is_blurry,
            "is_screenshot": is_screenshot,
            "total": total,
            "limit": limit,
            "offset": offset,
        },
    )


@router.get("/duplicates-view", response_class=HTMLResponse)
async def duplicates_page(request: Request, db=Depends(get_db)):
    """Duplicates page - review duplicate groups."""
    duplicate_repo = DuplicateRepository(db)
    groups = await duplicate_repo.get_all_groups()

    return templates.TemplateResponse(
        "duplicates.html",
        {
            "request": request,
            "groups": groups,
        },
    )


@router.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Chat page - natural language search."""
    return templates.TemplateResponse("chat.html", {"request": request})


@router.get("/thumb/{photo_id}")
async def get_thumbnail(photo_id: int, db=Depends(get_db)):
    """Serve photo thumbnail (or original if small enough)."""
    photo_repo = PhotoRepository(db)
    photo = await photo_repo.get_by_id(photo_id)

    if not photo:
        raise HTTPException(status_code=404, detail="Photo not found")

    photo_path = Path(photo.path)
    if not photo_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # For now, serve original (in production, generate thumbnails)
    return FileResponse(
        photo_path,
        media_type="image/jpeg",
        headers={"Cache-Control": "max-age=3600"},
    )


@router.get("/static/{filename:path}")
async def serve_static(filename: str):
    """Serve static files."""
    file_path = STATIC_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)
