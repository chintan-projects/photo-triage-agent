"""Real-time metrics tracking for analysis jobs."""
import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import psutil
import structlog

logger = structlog.get_logger()


@dataclass
class AnalysisMetrics:
    """Metrics snapshot for an analysis job."""

    job_id: str
    photos_processed: int = 0
    total_photos: int = 0
    photos_per_second: float = 0.0
    memory_mb: float = 0.0
    memory_percent: float = 0.0
    avg_inference_ms: float = 0.0
    current_photo: str | None = None
    started_at: float = field(default_factory=time.time)
    eta_seconds: float | None = None

    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        if self.total_photos == 0:
            return 0.0
        return (self.photos_processed / self.total_photos) * 100

    @property
    def elapsed_seconds(self) -> float:
        """Time elapsed since start."""
        return time.time() - self.started_at

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "job_id": self.job_id,
            "photos_processed": self.photos_processed,
            "total_photos": self.total_photos,
            "progress_percent": round(self.progress_percent, 1),
            "photos_per_second": round(self.photos_per_second, 2),
            "memory_mb": round(self.memory_mb, 1),
            "memory_percent": round(self.memory_percent, 1),
            "avg_inference_ms": round(self.avg_inference_ms, 1),
            "current_photo": self.current_photo,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "eta_seconds": round(self.eta_seconds, 1) if self.eta_seconds else None,
        }


class MetricsTracker:
    """Track and compute real-time metrics for analysis jobs."""

    def __init__(self, job_id: str, total_photos: int):
        self.metrics = AnalysisMetrics(job_id=job_id, total_photos=total_photos)
        self._inference_times: list[float] = []
        self._photo_timestamps: list[float] = []
        self._callbacks: list[Callable[[AnalysisMetrics], None]] = []
        self._lock = asyncio.Lock()

    def on_update(self, callback: Callable[[AnalysisMetrics], None]) -> None:
        """Register a callback for metrics updates."""
        self._callbacks.append(callback)

    async def record_photo_start(self, photo_path: str) -> None:
        """Record start of processing a photo."""
        async with self._lock:
            self.metrics.current_photo = photo_path

    async def record_photo_complete(self, inference_time_ms: float) -> None:
        """Record completion of a photo."""
        async with self._lock:
            now = time.time()
            self.metrics.photos_processed += 1
            self._inference_times.append(inference_time_ms)
            self._photo_timestamps.append(now)

            # Compute rolling average inference time
            recent_times = self._inference_times[-50:]  # Last 50 photos
            self.metrics.avg_inference_ms = sum(recent_times) / len(recent_times)

            # Compute photos per second (using last 30 seconds of data)
            cutoff = now - 30
            recent_timestamps = [t for t in self._photo_timestamps if t > cutoff]
            if len(recent_timestamps) >= 2:
                window_seconds = now - recent_timestamps[0]
                if window_seconds > 0:
                    self.metrics.photos_per_second = len(recent_timestamps) / window_seconds

            # Compute ETA
            if self.metrics.photos_per_second > 0:
                remaining = self.metrics.total_photos - self.metrics.photos_processed
                self.metrics.eta_seconds = remaining / self.metrics.photos_per_second
            else:
                self.metrics.eta_seconds = None

            # Update memory usage
            self._update_memory()

        # Notify callbacks
        await self._notify_callbacks()

    def _update_memory(self) -> None:
        """Update memory usage metrics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        self.metrics.memory_mb = memory_info.rss / (1024 * 1024)
        self.metrics.memory_percent = process.memory_percent()

    async def _notify_callbacks(self) -> None:
        """Notify all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(self.metrics)
            except Exception as e:
                logger.error("metrics_callback_error", error=str(e))

    def get_metrics(self) -> AnalysisMetrics:
        """Get current metrics snapshot."""
        self._update_memory()
        return self.metrics

    def get_summary(self) -> dict:
        """Get final summary statistics."""
        elapsed = self.metrics.elapsed_seconds
        total_inference = sum(self._inference_times)

        return {
            "job_id": self.metrics.job_id,
            "total_photos": self.metrics.total_photos,
            "photos_processed": self.metrics.photos_processed,
            "elapsed_seconds": round(elapsed, 1),
            "avg_photos_per_second": round(
                self.metrics.photos_processed / elapsed if elapsed > 0 else 0, 2
            ),
            "avg_inference_ms": round(
                total_inference / len(self._inference_times)
                if self._inference_times
                else 0,
                1,
            ),
            "total_inference_seconds": round(total_inference / 1000, 1),
            "peak_memory_mb": round(self.metrics.memory_mb, 1),
        }
