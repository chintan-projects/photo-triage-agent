"""Folder analysis orchestrator for Photo Triage Agent."""
import hashlib
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import aiosqlite
import structlog

from .analyzers import AnalyzerRegistry, find_duplicates
from .classifiers import LFMCliProvider, ModelProvider
from .config import get_model_config, models_available
from .database import (
    AnalysisRepository,
    DuplicateRepository,
    JobRepository,
    PhotoRepository,
)
from .services.metrics import AnalysisMetrics, MetricsTracker
from .utils.image import SUPPORTED_FORMATS

logger = structlog.get_logger()


def compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of file for change detection."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


async def enumerate_images(folder_path: Path) -> AsyncIterator[Path]:
    """Yield all supported image files in folder recursively."""
    for ext in SUPPORTED_FORMATS:
        for path in folder_path.rglob(f"*{ext}"):
            yield path
        # Also check uppercase extensions
        for path in folder_path.rglob(f"*{ext.upper()}"):
            yield path


class FolderOrchestrator:
    """Coordinates batch analysis of a photo folder."""

    def __init__(
        self,
        db: aiosqlite.Connection,
        model_provider: ModelProvider | None = None,
    ):
        self.db = db
        self.photo_repo = PhotoRepository(db)
        self.analysis_repo = AnalysisRepository(db)
        self.job_repo = JobRepository(db)
        self.duplicate_repo = DuplicateRepository(db)

        # Use provided model or create default
        self._model_provider = model_provider
        self._model_loaded = False

        # Metrics tracking
        self._metrics_tracker: MetricsTracker | None = None
        self._progress_callbacks: list[Callable[[AnalysisMetrics], None]] = []

    def on_progress(self, callback: Callable[[AnalysisMetrics], None]) -> None:
        """Register callback for progress updates."""
        self._progress_callbacks.append(callback)

    async def _ensure_model(self) -> ModelProvider | None:
        """Lazy-load model provider.

        Returns:
            ModelProvider if models available, None otherwise.
        """
        if self._model_provider is None and not self._model_loaded:
            if models_available():
                logger.info("loading_lfm_model")
                config = get_model_config()
                self._model_provider = LFMCliProvider(
                    model_path=config["model_path"],
                    mmproj_path=config["mmproj_path"],
                    cli_path=config["cli_path"],
                    n_ctx=config["n_ctx"],
                    n_gpu_layers=config["n_gpu_layers"],
                )
            else:
                logger.warning("lfm_model_not_available", reason="model files not found")
            self._model_loaded = True
        return self._model_provider

    async def process_folder(
        self,
        job_id: str,
        folder_path: str | Path,
        skip_lfm: bool = False,
        limit: int | None = None,
    ) -> dict:
        """Process all images in a folder.

        Args:
            job_id: Unique job identifier.
            folder_path: Path to folder containing images.
            skip_lfm: If True, skip LFM classification (faster, for testing).
            limit: Max NEW photos to process (already-analyzed photos don't count).
                   Useful for incremental analysis of large libraries.

        Returns:
            Summary of analysis results.
        """
        folder_path = Path(folder_path).expanduser().resolve()

        if not folder_path.exists():
            raise ValueError(f"Folder does not exist: {folder_path}")
        if not folder_path.is_dir():
            raise ValueError(f"Path is not a directory: {folder_path}")

        logger.info("analysis_starting", job_id=job_id, folder=str(folder_path))

        # Update job to running
        await self.job_repo.update_status(job_id, "running")

        try:
            # Enumerate all images
            image_paths = []
            try:
                async for path in enumerate_images(folder_path):
                    image_paths.append(path)
            except PermissionError as e:
                error_msg = (
                    f"Permission denied accessing {folder_path}. "
                    "For Photos Library, grant Full Disk Access to Terminal in "
                    "System Settings → Privacy & Security → Full Disk Access."
                )
                logger.error("permission_denied", job_id=job_id, folder=str(folder_path))
                raise PermissionError(error_msg) from e

            total = len(image_paths)
            logger.info("images_found", job_id=job_id, count=total)

            await self.job_repo.update_progress(job_id, processed=0, total=total)

            # Initialize metrics tracker
            self._metrics_tracker = MetricsTracker(job_id, total)
            for callback in self._progress_callbacks:
                self._metrics_tracker.on_update(callback)

            # Process each image
            results = {
                "processed": 0,
                "skipped": 0,
                "errors": 0,
                "categories": {},
                "blurry_count": 0,
                "screenshot_count": 0,
            }

            for i, image_path in enumerate(image_paths):
                # Check limit BEFORE processing next photo
                if limit and results["processed"] >= limit:
                    logger.info(
                        "batch_limit_reached",
                        job_id=job_id,
                        limit=limit,
                        processed=results["processed"],
                        skipped=results["skipped"],
                        remaining=total - i,
                    )
                    results["stopped_at_limit"] = True
                    results["remaining"] = total - i
                    break

                try:
                    await self._process_single_image(
                        image_path, job_id, skip_lfm, results
                    )
                except Exception as e:
                    logger.error(
                        "image_processing_error",
                        job_id=job_id,
                        path=str(image_path),
                        error=str(e),
                    )
                    results["errors"] += 1

                # Update progress
                await self.job_repo.update_progress(job_id, processed=i + 1)

                # Log progress every 50 photos
                if (i + 1) % 50 == 0:
                    logger.info(
                        "batch_progress",
                        job_id=job_id,
                        iteration=i + 1,
                        processed=results["processed"],
                        skipped=results["skipped"],
                        limit=limit,
                    )

                # Double-check limit AFTER processing (safety net)
                if limit and results["processed"] >= limit:
                    logger.info(
                        "batch_limit_reached",
                        job_id=job_id,
                        limit=limit,
                        processed=results["processed"],
                        skipped=results["skipped"],
                        remaining=total - i - 1,
                    )
                    results["stopped_at_limit"] = True
                    results["remaining"] = total - i - 1
                    break

            # Find duplicates after all images processed
            duplicates_found = await self._find_and_store_duplicates(job_id)
            results["duplicate_groups"] = duplicates_found

            # Mark job complete
            await self.job_repo.update_status(job_id, "completed")

            # Get final metrics
            if self._metrics_tracker:
                results["metrics"] = self._metrics_tracker.get_summary()

            logger.info(
                "analysis_complete",
                job_id=job_id,
                processed=results["processed"],
                skipped=results["skipped"],
                errors=results["errors"],
                duplicates=duplicates_found,
            )

            return results

        except Exception as e:
            logger.error("analysis_failed", job_id=job_id, error=str(e))
            await self.job_repo.update_status(job_id, "failed", error=str(e))
            raise

    async def _process_single_image(
        self,
        image_path: Path,
        job_id: str,
        skip_lfm: bool,
        results: dict,
    ) -> None:
        """Process a single image through all analyzers."""
        path_str = str(image_path)

        # Record start in metrics
        if self._metrics_tracker:
            await self._metrics_tracker.record_photo_start(image_path.name)

        start_time = time.time()

        # Check if already processed
        existing = await self.photo_repo.get_by_path(path_str)
        if existing:
            # Check if file changed
            current_hash = compute_file_hash(image_path)
            if existing.file_hash == current_hash:
                results["skipped"] += 1
                # Record with zero inference time for skipped photos
                if self._metrics_tracker:
                    await self._metrics_tracker.record_photo_complete(0)
                return

        # Create or update photo record
        file_hash = compute_file_hash(image_path)
        file_size = image_path.stat().st_size

        if existing:
            photo = await self.photo_repo.update(
                existing.id, file_hash=file_hash, file_size=file_size
            )
        else:
            photo = await self.photo_repo.create(
                path=path_str,
                filename=image_path.name,
                file_hash=file_hash,
                file_size=file_size,
            )

        # Run fast analyzers first (blur, screenshot, hasher)
        analyzer_results = AnalyzerRegistry.run_all(path_str)

        for analyzer_name, result in analyzer_results.items():
            if result.success and result.value:
                await self.analysis_repo.save(
                    photo_id=photo.id,
                    analyzer=analyzer_name,
                    result=result.value.metadata,
                    confidence=result.value.confidence,
                )

                # Track counts
                if analyzer_name == "blur" and result.value.metadata.get("is_blurry"):
                    results["blurry_count"] += 1
                elif analyzer_name == "screenshot" and result.value.metadata.get(
                    "is_screenshot"
                ):
                    results["screenshot_count"] += 1
                elif analyzer_name == "hasher" and result.value.metadata.get("phash"):
                    # Store phash on photo record for duplicate detection
                    await self.photo_repo.update(
                        photo.id, phash=result.value.metadata["phash"]
                    )

        # Run LFM classification (slower)
        if not skip_lfm:
            try:
                model = await self._ensure_model()
                if model is None:
                    # Model not available, skip classification
                    pass
                else:
                    classification = model.classify(path_str)

                    await self.analysis_repo.save(
                        photo_id=photo.id,
                        analyzer="lfm",
                        result={
                            "category": classification.category.value,
                            "description": classification.description,
                            "contains_faces": classification.contains_faces,
                            "is_screenshot": classification.is_screenshot,
                            "is_meme": classification.is_meme,
                        },
                        confidence=classification.confidence,
                    )

                    # Track category counts
                    category = classification.category.value
                    results["categories"][category] = (
                        results["categories"].get(category, 0) + 1
                    )
            except Exception as e:
                logger.warning(
                    "lfm_classification_failed", path=path_str, error=str(e)
                )

        results["processed"] += 1

        # Record completion with inference time
        inference_ms = (time.time() - start_time) * 1000
        if self._metrics_tracker:
            await self._metrics_tracker.record_photo_complete(inference_ms)

    async def _find_and_store_duplicates(self, job_id: str) -> int:
        """Find duplicate groups from perceptual hashes and store them."""
        logger.info("finding_duplicates", job_id=job_id)

        # Get all photos with phashes
        photos = await self.photo_repo.get_all(limit=100000)
        photos_with_hash = {p.id: p.phash for p in photos if p.phash}

        if len(photos_with_hash) < 2:
            return 0

        # Create path->hash mapping (using photo_id as "path" for find_duplicates)
        # find_duplicates expects dict[str, str] so convert ids to strings
        hash_dict = {str(photo_id): phash for photo_id, phash in photos_with_hash.items()}

        # Find duplicate groups using hasher utility
        duplicate_groups = find_duplicates(hash_dict, threshold=10)

        groups_created = 0

        for group_paths in duplicate_groups:
            if len(group_paths) < 2:
                continue

            # Convert string paths back to photo IDs
            photo_ids = [int(p) for p in group_paths]

            # Create duplicate group
            group = await self.duplicate_repo.create_group(
                group_type="exact",
                group_hash=photos_with_hash[photo_ids[0]],
                description=f"Group of {len(photo_ids)} visually identical photos",
            )

            # Add members (first one marked as best for now)
            for i, photo_id in enumerate(photo_ids):
                await self.duplicate_repo.add_member(
                    group.id, photo_id, is_best=(i == 0)
                )

            groups_created += 1

        logger.info("duplicates_found", job_id=job_id, groups=groups_created)
        return groups_created

    def get_current_metrics(self) -> AnalysisMetrics | None:
        """Get current metrics for active job."""
        if self._metrics_tracker:
            return self._metrics_tracker.get_metrics()
        return None
