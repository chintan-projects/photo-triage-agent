"""Tests for folder orchestrator."""
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from src.database import close_database, init_database
from src.orchestrator import (
    FolderOrchestrator,
    compute_file_hash,
    enumerate_images,
)
from src.utils.image import SUPPORTED_FORMATS


@pytest.fixture
async def db():
    """Create temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    db = await init_database(db_path)
    yield db
    await close_database(db)
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def temp_folder():
    """Create temporary folder with test images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        folder = Path(tmpdir)

        # Create test images
        for i in range(3):
            img = Image.new("RGB", (100, 100), color=(i * 50, i * 50, i * 50))
            img.save(folder / f"test_{i}.jpg")

        # Create a subfolder with more images
        subfolder = folder / "sub"
        subfolder.mkdir()
        img = Image.new("RGB", (200, 200), color=(100, 100, 100))
        img.save(subfolder / "nested.png")

        yield folder


@pytest.fixture
def single_image():
    """Create a single test image."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        img.save(f.name)
        yield f.name
    Path(f.name).unlink(missing_ok=True)


class TestComputeFileHash:
    def test_returns_consistent_hash(self, single_image):
        """Same file returns same hash."""
        hash1 = compute_file_hash(Path(single_image))
        hash2 = compute_file_hash(Path(single_image))
        assert hash1 == hash2

    def test_returns_sha256_format(self, single_image):
        """Returns valid SHA256 hex string."""
        file_hash = compute_file_hash(Path(single_image))
        assert len(file_hash) == 64  # SHA256 hex is 64 chars
        assert all(c in "0123456789abcdef" for c in file_hash)


class TestEnumerateImages:
    async def test_finds_images_in_folder(self, temp_folder):
        """Finds all supported image files."""
        images = []
        async for path in enumerate_images(temp_folder):
            images.append(path)

        assert len(images) == 4  # 3 jpgs + 1 png

    async def test_finds_images_recursively(self, temp_folder):
        """Finds images in subfolders."""
        images = []
        async for path in enumerate_images(temp_folder):
            images.append(path)

        nested = [p for p in images if "nested" in p.name]
        assert len(nested) == 1

    async def test_empty_folder_returns_nothing(self):
        """Empty folder yields no images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            images = []
            async for path in enumerate_images(Path(tmpdir)):
                images.append(path)
            assert len(images) == 0


class TestSupportedFormats:
    def test_common_formats_supported(self):
        """Common image formats are supported."""
        assert ".jpg" in SUPPORTED_FORMATS
        assert ".jpeg" in SUPPORTED_FORMATS
        assert ".png" in SUPPORTED_FORMATS
        assert ".heic" in SUPPORTED_FORMATS


class TestFolderOrchestrator:
    async def test_process_folder_creates_records(self, db, temp_folder):
        """Processing folder creates photo records."""
        orchestrator = FolderOrchestrator(db)

        # Create job first
        from src.database import JobRepository

        job_repo = JobRepository(db)
        job = await job_repo.create(folder_path=str(temp_folder))

        # Process folder (skip LFM for speed)
        result = await orchestrator.process_folder(
            job.id, temp_folder, skip_lfm=True
        )

        assert result["processed"] == 4
        assert result["errors"] == 0

    async def test_process_folder_stores_analysis(self, db, temp_folder):
        """Processing stores analysis results."""
        orchestrator = FolderOrchestrator(db)

        from src.database import JobRepository, PhotoRepository

        job_repo = JobRepository(db)
        photo_repo = PhotoRepository(db)
        job = await job_repo.create(folder_path=str(temp_folder))

        await orchestrator.process_folder(job.id, temp_folder, skip_lfm=True)

        # Check photos were created
        count = await photo_repo.count()
        assert count == 4

    async def test_process_folder_skips_unchanged(self, db, temp_folder):
        """Processing skips files already analyzed."""
        orchestrator = FolderOrchestrator(db)

        from src.database import JobRepository

        job_repo = JobRepository(db)

        # First run
        job1 = await job_repo.create(folder_path=str(temp_folder))
        result1 = await orchestrator.process_folder(
            job1.id, temp_folder, skip_lfm=True
        )
        assert result1["processed"] == 4

        # Second run - should skip all
        job2 = await job_repo.create(folder_path=str(temp_folder))
        result2 = await orchestrator.process_folder(
            job2.id, temp_folder, skip_lfm=True
        )
        assert result2["skipped"] == 4
        assert result2["processed"] == 0

    async def test_process_folder_tracks_metrics(self, db, temp_folder):
        """Processing tracks performance metrics."""
        orchestrator = FolderOrchestrator(db)

        from src.database import JobRepository

        job_repo = JobRepository(db)
        job = await job_repo.create(folder_path=str(temp_folder))

        result = await orchestrator.process_folder(
            job.id, temp_folder, skip_lfm=True
        )

        assert "metrics" in result
        metrics = result["metrics"]
        assert metrics["photos_processed"] == 4
        assert metrics["avg_inference_ms"] >= 0

    async def test_process_folder_updates_job_status(self, db, temp_folder):
        """Processing updates job status correctly."""
        orchestrator = FolderOrchestrator(db)

        from src.database import JobRepository

        job_repo = JobRepository(db)
        job = await job_repo.create(folder_path=str(temp_folder))

        await orchestrator.process_folder(job.id, temp_folder, skip_lfm=True)

        # Check job status
        updated_job = await job_repo.get(job.id)
        assert updated_job.status == "completed"
        assert updated_job.processed_photos == 4
        assert updated_job.total_photos == 4

    async def test_process_nonexistent_folder_raises(self, db):
        """Processing nonexistent folder raises error."""
        orchestrator = FolderOrchestrator(db)

        from src.database import JobRepository

        job_repo = JobRepository(db)
        job = await job_repo.create(folder_path="/nonexistent/folder")

        with pytest.raises(ValueError, match="does not exist"):
            await orchestrator.process_folder(job.id, "/nonexistent/folder")

    async def test_process_file_instead_of_folder_raises(self, db, single_image):
        """Processing a file path instead of folder raises error."""
        orchestrator = FolderOrchestrator(db)

        from src.database import JobRepository

        job_repo = JobRepository(db)
        job = await job_repo.create(folder_path=single_image)

        with pytest.raises(ValueError, match="not a directory"):
            await orchestrator.process_folder(job.id, single_image)


class TestMetricsTracker:
    async def test_metrics_snapshot(self, db, temp_folder):
        """Can get metrics during processing."""
        orchestrator = FolderOrchestrator(db)

        from src.database import JobRepository

        job_repo = JobRepository(db)
        job = await job_repo.create(folder_path=str(temp_folder))

        # Start processing
        await orchestrator.process_folder(job.id, temp_folder, skip_lfm=True)

        # After processing, should have final metrics
        # (During processing we'd use get_current_metrics())
        metrics = orchestrator.get_current_metrics()
        assert metrics is not None
        assert metrics.photos_processed == 4
