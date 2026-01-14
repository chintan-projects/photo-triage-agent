"""Job service for managing async analysis jobs."""
import asyncio
import uuid
from collections.abc import Callable

import aiosqlite
import structlog

from ..database import JobRepository
from ..orchestrator import FolderOrchestrator
from .metrics import AnalysisMetrics

logger = structlog.get_logger()


class JobService:
    """Manages async analysis jobs with background processing."""

    def __init__(self, db: aiosqlite.Connection):
        self.db = db
        self.job_repo = JobRepository(db)
        self._active_jobs: dict[str, asyncio.Task] = {}
        self._orchestrators: dict[str, FolderOrchestrator] = {}
        self._progress_callbacks: dict[str, list[Callable[[AnalysisMetrics], None]]] = (
            {}
        )

    async def start_analysis(
        self,
        folder_path: str,
        skip_lfm: bool = False,
    ) -> str:
        """Start a new analysis job.

        Args:
            folder_path: Path to folder to analyze.
            skip_lfm: Skip LFM classification (faster, for testing).

        Returns:
            Job ID for tracking progress.
        """
        job_id = str(uuid.uuid4())

        # Create job record
        await self.job_repo.create(folder_path=folder_path, job_id=job_id)

        # Create orchestrator
        orchestrator = FolderOrchestrator(self.db)
        self._orchestrators[job_id] = orchestrator

        # Register any existing callbacks
        for callback in self._progress_callbacks.get(job_id, []):
            orchestrator.on_progress(callback)

        # Start background task
        task = asyncio.create_task(
            self._run_analysis(job_id, folder_path, orchestrator, skip_lfm)
        )
        self._active_jobs[job_id] = task

        logger.info("job_started", job_id=job_id, folder_path=folder_path)
        return job_id

    async def _run_analysis(
        self,
        job_id: str,
        folder_path: str,
        orchestrator: FolderOrchestrator,
        skip_lfm: bool,
    ) -> None:
        """Run analysis in background."""
        try:
            await orchestrator.process_folder(job_id, folder_path, skip_lfm=skip_lfm)
        except asyncio.CancelledError:
            logger.info("job_cancelled", job_id=job_id)
            await self.job_repo.update_status(job_id, "cancelled")
        except Exception as e:
            logger.error("job_failed", job_id=job_id, error=str(e))
            await self.job_repo.update_status(job_id, "failed", error=str(e))
        finally:
            # Cleanup
            self._active_jobs.pop(job_id, None)

    async def get_status(self, job_id: str) -> dict | None:
        """Get current status of a job.

        Returns:
            Job status dict or None if job not found.
        """
        job = await self.job_repo.get(job_id)
        if not job:
            return None

        # SQLite returns datetime as strings, so just use them directly
        started_at = job.started_at
        if started_at and hasattr(started_at, 'isoformat'):
            started_at = started_at.isoformat()

        completed_at = job.completed_at
        if completed_at and hasattr(completed_at, 'isoformat'):
            completed_at = completed_at.isoformat()

        status = {
            "job_id": job.id,
            "folder_path": job.folder_path,
            "status": job.status,
            "total_photos": job.total_photos,
            "processed_photos": job.processed_photos,
            "progress_percent": (
                round((job.processed_photos / job.total_photos) * 100, 1)
                if job.total_photos > 0
                else 0
            ),
            "error": job.error,
            "started_at": started_at,
            "completed_at": completed_at,
        }

        # Add live metrics if job is running
        if job_id in self._orchestrators:
            metrics = self._orchestrators[job_id].get_current_metrics()
            if metrics:
                status["metrics"] = metrics.to_dict()

        return status

    async def get_metrics(self, job_id: str) -> dict | None:
        """Get current metrics for a running job."""
        orchestrator = self._orchestrators.get(job_id)
        if orchestrator:
            metrics = orchestrator.get_current_metrics()
            if metrics:
                return metrics.to_dict()
        return None

    def on_progress(
        self, job_id: str, callback: Callable[[AnalysisMetrics], None]
    ) -> None:
        """Register callback for progress updates.

        Call this before or after starting the job.
        """
        if job_id not in self._progress_callbacks:
            self._progress_callbacks[job_id] = []
        self._progress_callbacks[job_id].append(callback)

        # If orchestrator already exists, register directly
        if job_id in self._orchestrators:
            self._orchestrators[job_id].on_progress(callback)

    async def cancel(self, job_id: str) -> bool:
        """Cancel a running job.

        Returns:
            True if job was cancelled, False if not found or already complete.
        """
        task = self._active_jobs.get(job_id)
        if task and not task.done():
            task.cancel()
            logger.info("job_cancellation_requested", job_id=job_id)
            return True
        return False

    async def list_jobs(
        self, status: str | None = None, limit: int = 20
    ) -> list[dict]:
        """List recent jobs.

        Args:
            status: Filter by status (pending, running, completed, failed).
            limit: Maximum number of jobs to return.

        Returns:
            List of job status dicts.
        """
        # Get all jobs and filter in memory for simplicity
        # In production, add proper filtering to JobRepository
        jobs = []
        # This is a simplified implementation - would need pagination in JobRepository
        job = await self.job_repo.get_latest()
        if job:
            if status is None or job.status == status:
                # SQLite returns datetime as strings
                created_at = job.created_at
                if created_at and hasattr(created_at, 'isoformat'):
                    created_at = created_at.isoformat()

                jobs.append(
                    {
                        "job_id": job.id,
                        "folder_path": job.folder_path,
                        "status": job.status,
                        "total_photos": job.total_photos,
                        "processed_photos": job.processed_photos,
                        "created_at": created_at,
                    }
                )
        return jobs[:limit]

    def is_running(self, job_id: str) -> bool:
        """Check if a job is currently running."""
        task = self._active_jobs.get(job_id)
        return task is not None and not task.done()
