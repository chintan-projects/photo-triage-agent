"""Tests for database repository."""
import pytest
import tempfile
from pathlib import Path

from src.database import (
    ActionRepository,
    AnalysisRepository,
    DuplicateRepository,
    Job,
    JobRepository,
    Photo,
    PhotoRepository,
    init_database,
    close_database,
)


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
async def photo_repo(db):
    """Create photo repository."""
    return PhotoRepository(db)


@pytest.fixture
async def analysis_repo(db):
    """Create analysis repository."""
    return AnalysisRepository(db)


@pytest.fixture
async def job_repo(db):
    """Create job repository."""
    return JobRepository(db)


@pytest.fixture
async def duplicate_repo(db):
    """Create duplicate repository."""
    return DuplicateRepository(db)


@pytest.fixture
async def action_repo(db):
    """Create action repository."""
    return ActionRepository(db)


# Photo Repository Tests

class TestPhotoRepository:
    async def test_create_photo(self, photo_repo):
        """Can create a photo record."""
        photo = await photo_repo.create(
            path="/test/photo.jpg",
            filename="photo.jpg",
            file_hash="abc123",
            file_size=1024,
        )

        assert photo.id is not None
        assert photo.path == "/test/photo.jpg"
        assert photo.filename == "photo.jpg"
        assert photo.file_hash == "abc123"

    async def test_get_by_id(self, photo_repo):
        """Can get photo by ID."""
        created = await photo_repo.create(path="/test/get_by_id.jpg")
        fetched = await photo_repo.get_by_id(created.id)

        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.path == created.path

    async def test_get_by_path(self, photo_repo):
        """Can get photo by path."""
        await photo_repo.create(path="/test/unique_path.jpg")
        fetched = await photo_repo.get_by_path("/test/unique_path.jpg")

        assert fetched is not None
        assert fetched.path == "/test/unique_path.jpg"

    async def test_get_nonexistent_returns_none(self, photo_repo):
        """Nonexistent photo returns None."""
        fetched = await photo_repo.get_by_id(99999)
        assert fetched is None

    async def test_update_photo(self, photo_repo):
        """Can update photo fields."""
        photo = await photo_repo.create(path="/test/update.jpg")
        updated = await photo_repo.update(photo.id, phash="hash123", file_size=2048)

        assert updated.phash == "hash123"
        assert updated.file_size == 2048

    async def test_delete_photo(self, photo_repo):
        """Can delete photo."""
        photo = await photo_repo.create(path="/test/delete.jpg")
        success = await photo_repo.delete(photo.id)

        assert success
        assert await photo_repo.get_by_id(photo.id) is None

    async def test_count_photos(self, photo_repo):
        """Can count photos."""
        initial_count = await photo_repo.count()

        await photo_repo.create(path="/test/count1.jpg")
        await photo_repo.create(path="/test/count2.jpg")

        assert await photo_repo.count() == initial_count + 2

    async def test_get_all_with_pagination(self, photo_repo):
        """Can get photos with pagination."""
        for i in range(5):
            await photo_repo.create(path=f"/test/paginate_{i}.jpg")

        page1 = await photo_repo.get_all(limit=2, offset=0)
        page2 = await photo_repo.get_all(limit=2, offset=2)

        assert len(page1) == 2
        assert len(page2) == 2
        assert page1[0].id != page2[0].id


# Analysis Repository Tests

class TestAnalysisRepository:
    async def test_save_analysis(self, photo_repo, analysis_repo):
        """Can save analysis result."""
        photo = await photo_repo.create(path="/test/analysis.jpg")

        result = await analysis_repo.save(
            photo_id=photo.id,
            analyzer="blur",
            result={"is_blurry": True, "variance": 50.0},
            confidence=0.95,
        )

        assert result.photo_id == photo.id
        assert result.analyzer == "blur"
        assert result.result["is_blurry"] is True

    async def test_upsert_analysis(self, photo_repo, analysis_repo):
        """Saves update existing analysis."""
        photo = await photo_repo.create(path="/test/upsert.jpg")

        # First save
        await analysis_repo.save(
            photo_id=photo.id,
            analyzer="blur",
            result={"is_blurry": True},
        )

        # Update (upsert)
        updated = await analysis_repo.save(
            photo_id=photo.id,
            analyzer="blur",
            result={"is_blurry": False},
        )

        assert updated.result["is_blurry"] is False

    async def test_get_for_photo(self, photo_repo, analysis_repo):
        """Can get all analyses for a photo."""
        photo = await photo_repo.create(path="/test/multi_analysis.jpg")

        await analysis_repo.save(photo.id, "blur", {"is_blurry": False})
        await analysis_repo.save(photo.id, "screenshot", {"is_screenshot": True})
        await analysis_repo.save(photo.id, "lfm", {"category": "landscape"})

        results = await analysis_repo.get_for_photo(photo.id)
        assert len(results) == 3

        analyzers = {r.analyzer for r in results}
        assert analyzers == {"blur", "screenshot", "lfm"}


# Job Repository Tests

class TestJobRepository:
    async def test_create_job(self, job_repo):
        """Can create a job."""
        job = await job_repo.create(folder_path="/test/photos")

        assert job.id is not None
        assert job.folder_path == "/test/photos"
        assert job.status == "pending"

    async def test_update_job_status(self, job_repo):
        """Can update job status."""
        job = await job_repo.create(folder_path="/test/status")

        updated = await job_repo.update_status(job.id, "running")
        assert updated.status == "running"
        assert updated.started_at is not None

        completed = await job_repo.update_status(job.id, "completed")
        assert completed.status == "completed"
        assert completed.completed_at is not None

    async def test_update_job_progress(self, job_repo):
        """Can update job progress."""
        job = await job_repo.create(folder_path="/test/progress")

        await job_repo.update_progress(job.id, processed=0, total=100)
        job = await job_repo.get(job.id)
        assert job.total_photos == 100
        assert job.processed_photos == 0

        await job_repo.update_progress(job.id, processed=50)
        job = await job_repo.get(job.id)
        assert job.processed_photos == 50

    async def test_failed_job_saves_error(self, job_repo):
        """Failed job saves error message."""
        job = await job_repo.create(folder_path="/test/error")

        await job_repo.update_status(job.id, "failed", error="Something went wrong")
        job = await job_repo.get(job.id)

        assert job.status == "failed"
        assert job.error == "Something went wrong"


# Duplicate Repository Tests

class TestDuplicateRepository:
    async def test_create_duplicate_group(self, duplicate_repo):
        """Can create duplicate group."""
        group = await duplicate_repo.create_group(
            group_type="exact",
            group_hash="abc123",
            description="Identical photos",
        )

        assert group.id is not None
        assert group.group_type == "exact"

    async def test_add_members_to_group(self, photo_repo, duplicate_repo):
        """Can add photos to duplicate group."""
        photo1 = await photo_repo.create(path="/test/dup1.jpg")
        photo2 = await photo_repo.create(path="/test/dup2.jpg")

        group = await duplicate_repo.create_group(group_type="exact")

        await duplicate_repo.add_member(group.id, photo1.id, is_best=True)
        await duplicate_repo.add_member(group.id, photo2.id, is_best=False)

        fetched = await duplicate_repo.get_group(group.id)
        assert fetched.photo_count == 2
        assert len(fetched.photos) == 2

    async def test_get_all_groups(self, photo_repo, duplicate_repo):
        """Can get all duplicate groups."""
        # Create group with photos
        photo1 = await photo_repo.create(path="/test/all_dup1.jpg")
        photo2 = await photo_repo.create(path="/test/all_dup2.jpg")

        group = await duplicate_repo.create_group(group_type="semantic")
        await duplicate_repo.add_member(group.id, photo1.id)
        await duplicate_repo.add_member(group.id, photo2.id)

        groups = await duplicate_repo.get_all_groups()
        assert len(groups) >= 1
        assert any(g.group_type == "semantic" for g in groups)


# Action Repository Tests

class TestActionRepository:
    async def test_record_action(self, action_repo):
        """Can record an action."""
        action = await action_repo.record(
            action_type="trash",
            original_path="/test/trashed.jpg",
            new_path="/trash/trashed.jpg",
        )

        assert action.id is not None
        assert action.action_type == "trash"
        assert action.can_undo is True

    async def test_get_undoable_actions(self, action_repo):
        """Can get undoable actions."""
        await action_repo.record("trash", "/test/undo1.jpg")
        await action_repo.record("move", "/test/undo2.jpg")

        undoable = await action_repo.get_undoable()
        assert len(undoable) >= 2

    async def test_mark_undone(self, action_repo):
        """Can mark action as undone."""
        action = await action_repo.record("trash", "/test/mark_undone.jpg")

        await action_repo.mark_undone(action.id)

        # Should no longer appear in undoable list
        undoable = await action_repo.get_undoable()
        undoable_ids = [a.id for a in undoable]
        assert action.id not in undoable_ids
