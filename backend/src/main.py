"""Photo Triage Agent - Backend API"""
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router as api_router
from .classifiers import LFMCliProvider
from .config import DATABASE_PATH, get_model_config, models_available
from .database import close_database, init_database
from .database.repository import JobRepository
from .web import router as web_router

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger()

# Global database connection
db = None

# Job recovery timeout (minutes) - jobs running longer than this are considered crashed
JOB_RECOVERY_TIMEOUT_MINUTES = 30


async def recover_stale_jobs(db_conn) -> int:
    """Recover jobs that were left in 'running' state from a previous crash.

    This is part of the robustness principle - ensure app can recover from crashes.
    """
    job_repo = JobRepository(db_conn)
    recovered = await job_repo.recover_stale_jobs(JOB_RECOVERY_TIMEOUT_MINUTES)
    if recovered > 0:
        logger.warning(
            "stale_jobs_recovered",
            count=recovered,
            timeout_minutes=JOB_RECOVERY_TIMEOUT_MINUTES
        )
    return recovered


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - database init, model loading, and cleanup."""
    global db
    logger.info("server_starting", version="0.1.0")

    # Initialize database
    db = await init_database(DATABASE_PATH)
    app.state.db = db

    # Recover any jobs that were running when the server crashed
    await recover_stale_jobs(db)

    # Initialize LFM model if available (optional - search works with fallback)
    app.state.model = None
    if models_available():
        try:
            config = get_model_config()
            app.state.model = LFMCliProvider(
                model_path=config["model_path"],
                mmproj_path=config["mmproj_path"],
                cli_path=config["cli_path"],
                n_ctx=config["n_ctx"],
                n_gpu_layers=config["n_gpu_layers"],
            )
            logger.info("lfm_model_initialized")
        except Exception as e:
            logger.warning("lfm_model_init_failed", error=str(e))
    else:
        logger.info("lfm_model_skipped", reason="model files not found")

    yield

    # Cleanup
    if db:
        await close_database(db)
    logger.info("server_stopping")


app = FastAPI(
    title="Photo Triage Agent",
    description="Local-first photo analysis API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(api_router)

# Web dashboard routes
app.include_router(web_router)
