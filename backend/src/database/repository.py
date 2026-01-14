"""Database repository for CRUD operations."""
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import aiosqlite
import structlog

logger = structlog.get_logger()


@dataclass
class Photo:
    """Photo record."""
    id: int
    path: str
    filename: str
    file_hash: str | None = None
    phash: str | None = None
    file_size: int | None = None
    width: int | None = None
    height: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class AnalysisResult:
    """Analysis result record."""
    id: int
    photo_id: int
    analyzer: str
    result: dict
    confidence: float | None = None
    created_at: datetime | None = None


@dataclass
class Job:
    """Processing job record."""
    id: str
    folder_path: str
    status: str = "pending"
    total_photos: int = 0
    processed_photos: int = 0
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime | None = None


@dataclass
class DuplicateGroup:
    """Duplicate group record."""
    id: int
    group_type: str
    group_hash: str | None = None
    photo_count: int = 0
    description: str | None = None
    photos: list[Photo] | None = None


@dataclass
class ActionRecord:
    """Action history record for undo."""
    id: int
    action_type: str
    photo_id: int | None
    original_path: str
    new_path: str | None = None
    can_undo: bool = True
    undone_at: datetime | None = None
    created_at: datetime | None = None


class PhotoRepository:
    """Repository for photo-related database operations."""

    def __init__(self, db: aiosqlite.Connection):
        self.db = db

    async def create(
        self,
        path: str,
        filename: str | None = None,
        file_hash: str | None = None,
        phash: str | None = None,
        file_size: int | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> Photo:
        """Create a new photo record."""
        if filename is None:
            filename = Path(path).name

        cursor = await self.db.execute(
            """
            INSERT INTO photos (path, filename, file_hash, phash, file_size, width, height)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (path, filename, file_hash, phash, file_size, width, height)
        )
        await self.db.commit()

        return Photo(
            id=cursor.lastrowid,
            path=path,
            filename=filename,
            file_hash=file_hash,
            phash=phash,
            file_size=file_size,
            width=width,
            height=height,
        )

    async def get_by_id(self, photo_id: int) -> Photo | None:
        """Get photo by ID."""
        cursor = await self.db.execute(
            "SELECT * FROM photos WHERE id = ?",
            (photo_id,)
        )
        row = await cursor.fetchone()
        return self._row_to_photo(row) if row else None

    async def get_by_path(self, path: str) -> Photo | None:
        """Get photo by file path."""
        cursor = await self.db.execute(
            "SELECT * FROM photos WHERE path = ?",
            (path,)
        )
        row = await cursor.fetchone()
        return self._row_to_photo(row) if row else None

    async def get_all(self, limit: int = 1000, offset: int = 0) -> list[Photo]:
        """Get all photos with pagination."""
        cursor = await self.db.execute(
            "SELECT * FROM photos ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset)
        )
        rows = await cursor.fetchall()
        return [self._row_to_photo(row) for row in rows]

    async def get_by_category(self, category: str) -> list[Photo]:
        """Get photos by LFM classification category."""
        cursor = await self.db.execute(
            """
            SELECT p.* FROM photos p
            JOIN analysis_results ar ON p.id = ar.photo_id
            WHERE ar.analyzer = 'lfm'
            AND json_extract(ar.result, '$.category') = ?
            ORDER BY p.created_at DESC
            """,
            (category,)
        )
        rows = await cursor.fetchall()
        return [self._row_to_photo(row) for row in rows]

    async def get_blurry(self, threshold: float = 100.0) -> list[Photo]:
        """Get photos marked as blurry."""
        cursor = await self.db.execute(
            """
            SELECT p.* FROM photos p
            JOIN analysis_results ar ON p.id = ar.photo_id
            WHERE ar.analyzer = 'blur'
            AND json_extract(ar.result, '$.is_blurry') = 1
            ORDER BY json_extract(ar.result, '$.laplacian_variance') ASC
            """
        )
        rows = await cursor.fetchall()
        return [self._row_to_photo(row) for row in rows]

    async def get_screenshots(self) -> list[Photo]:
        """Get photos detected as screenshots."""
        cursor = await self.db.execute(
            """
            SELECT p.* FROM photos p
            JOIN analysis_results ar ON p.id = ar.photo_id
            WHERE ar.analyzer = 'screenshot'
            AND json_extract(ar.result, '$.is_screenshot') = 1
            ORDER BY p.created_at DESC
            """
        )
        rows = await cursor.fetchall()
        return [self._row_to_photo(row) for row in rows]

    async def update(self, photo_id: int, **kwargs) -> Photo | None:
        """Update photo fields."""
        if not kwargs:
            return await self.get_by_id(photo_id)

        fields = ", ".join(f"{k} = ?" for k in kwargs.keys())
        values = list(kwargs.values()) + [photo_id]

        await self.db.execute(
            f"UPDATE photos SET {fields}, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            values
        )
        await self.db.commit()

        return await self.get_by_id(photo_id)

    async def delete(self, photo_id: int) -> bool:
        """Delete photo record."""
        cursor = await self.db.execute(
            "DELETE FROM photos WHERE id = ?",
            (photo_id,)
        )
        await self.db.commit()
        return cursor.rowcount > 0

    async def count(self) -> int:
        """Count total photos."""
        cursor = await self.db.execute("SELECT COUNT(*) FROM photos")
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def get_category_counts(self) -> dict[str, int]:
        """Get counts by category."""
        cursor = await self.db.execute(
            """
            SELECT json_extract(ar.result, '$.category') as category, COUNT(*) as count
            FROM analysis_results ar
            WHERE ar.analyzer = 'lfm'
            GROUP BY category
            ORDER BY count DESC
            """
        )
        rows = await cursor.fetchall()
        return {row[0]: row[1] for row in rows}

    def _row_to_photo(self, row: tuple) -> Photo:
        """Convert database row to Photo object."""
        return Photo(
            id=row[0],
            path=row[1],
            filename=row[2],
            file_hash=row[3],
            phash=row[4],
            file_size=row[5],
            width=row[6],
            height=row[7],
            created_at=row[8],
            updated_at=row[9],
        )


class AnalysisRepository:
    """Repository for analysis results."""

    def __init__(self, db: aiosqlite.Connection):
        self.db = db

    async def save(
        self,
        photo_id: int,
        analyzer: str,
        result: dict,
        confidence: float | None = None,
    ) -> AnalysisResult:
        """Save or update analysis result."""
        result_json = json.dumps(result)

        await self.db.execute(
            """
            INSERT INTO analysis_results (photo_id, analyzer, result, confidence)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(photo_id, analyzer) DO UPDATE SET
                result = excluded.result,
                confidence = excluded.confidence,
                created_at = CURRENT_TIMESTAMP
            """,
            (photo_id, analyzer, result_json, confidence)
        )
        await self.db.commit()

        cursor = await self.db.execute(
            "SELECT * FROM analysis_results WHERE photo_id = ? AND analyzer = ?",
            (photo_id, analyzer)
        )
        row = await cursor.fetchone()
        return self._row_to_result(row)

    async def get_for_photo(self, photo_id: int) -> list[AnalysisResult]:
        """Get all analysis results for a photo."""
        cursor = await self.db.execute(
            "SELECT * FROM analysis_results WHERE photo_id = ?",
            (photo_id,)
        )
        rows = await cursor.fetchall()
        return [self._row_to_result(row) for row in rows]

    async def get_by_analyzer(
        self,
        analyzer: str,
        limit: int = 1000
    ) -> list[AnalysisResult]:
        """Get results by analyzer type."""
        cursor = await self.db.execute(
            "SELECT * FROM analysis_results WHERE analyzer = ? LIMIT ?",
            (analyzer, limit)
        )
        rows = await cursor.fetchall()
        return [self._row_to_result(row) for row in rows]

    def _row_to_result(self, row: tuple) -> AnalysisResult:
        """Convert database row to AnalysisResult object."""
        return AnalysisResult(
            id=row[0],
            photo_id=row[1],
            analyzer=row[2],
            result=json.loads(row[3]),
            confidence=row[4],
            created_at=row[5],
        )


class JobRepository:
    """Repository for processing jobs."""

    def __init__(self, db: aiosqlite.Connection):
        self.db = db

    async def create(self, folder_path: str, job_id: str | None = None) -> Job:
        """Create a new job."""
        if job_id is None:
            job_id = str(uuid.uuid4())

        await self.db.execute(
            "INSERT INTO jobs (id, folder_path) VALUES (?, ?)",
            (job_id, folder_path)
        )
        await self.db.commit()

        return Job(id=job_id, folder_path=folder_path)

    async def get(self, job_id: str) -> Job | None:
        """Get job by ID."""
        cursor = await self.db.execute(
            "SELECT * FROM jobs WHERE id = ?",
            (job_id,)
        )
        row = await cursor.fetchone()
        return self._row_to_job(row) if row else None

    async def update_status(
        self,
        job_id: str,
        status: str,
        error: str | None = None
    ) -> Job | None:
        """Update job status."""
        if status == "running":
            await self.db.execute(
                "UPDATE jobs SET status = ?, started_at = CURRENT_TIMESTAMP WHERE id = ?",
                (status, job_id)
            )
        elif status in ("completed", "failed"):
            await self.db.execute(
                """
                UPDATE jobs SET status = ?, error = ?, completed_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (status, error, job_id)
            )
        else:
            await self.db.execute(
                "UPDATE jobs SET status = ? WHERE id = ?",
                (status, job_id)
            )

        await self.db.commit()
        return await self.get(job_id)

    async def update_progress(
        self,
        job_id: str,
        processed: int,
        total: int | None = None
    ) -> Job | None:
        """Update job progress."""
        if total is not None:
            await self.db.execute(
                "UPDATE jobs SET processed_photos = ?, total_photos = ? WHERE id = ?",
                (processed, total, job_id)
            )
        else:
            await self.db.execute(
                "UPDATE jobs SET processed_photos = ? WHERE id = ?",
                (processed, job_id)
            )
        await self.db.commit()
        return await self.get(job_id)

    async def save_metrics(
        self,
        job_id: str,
        photos_per_second: float,
        memory_mb: float,
        avg_inference_ms: float,
    ) -> None:
        """Save job performance metrics."""
        await self.db.execute(
            """
            INSERT INTO job_metrics (job_id, photos_per_second, memory_mb, avg_inference_ms)
            VALUES (?, ?, ?, ?)
            """,
            (job_id, photos_per_second, memory_mb, avg_inference_ms)
        )
        await self.db.commit()

    async def get_recent(self, limit: int = 10) -> list[Job]:
        """Get recent jobs."""
        cursor = await self.db.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
            (limit,)
        )
        rows = await cursor.fetchall()
        return [self._row_to_job(row) for row in rows]

    async def get_latest(self) -> Job | None:
        """Get the most recent job."""
        cursor = await self.db.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        return self._row_to_job(row) if row else None

    async def get_stale_running_jobs(self, max_age_minutes: int = 30) -> list[Job]:
        """Get jobs stuck in 'running' status beyond max_age_minutes.

        These are likely crashed jobs that need recovery.
        """
        cursor = await self.db.execute(
            """
            SELECT * FROM jobs
            WHERE status = 'running'
            AND started_at < datetime('now', ? || ' minutes')
            ORDER BY started_at ASC
            """,
            (f"-{max_age_minutes}",)
        )
        rows = await cursor.fetchall()
        return [self._row_to_job(row) for row in rows]

    async def recover_stale_jobs(self, max_age_minutes: int = 30) -> int:
        """Mark stale running jobs as failed.

        Returns the number of jobs recovered.
        """
        stale_jobs = await self.get_stale_running_jobs(max_age_minutes)

        for job in stale_jobs:
            await self.update_status(
                job.id,
                "failed",
                error=f"Job interrupted (recovered after {max_age_minutes} minutes)"
            )

        return len(stale_jobs)

    def _row_to_job(self, row: tuple) -> Job:
        """Convert database row to Job object."""
        return Job(
            id=row[0],
            folder_path=row[1],
            status=row[2],
            total_photos=row[3],
            processed_photos=row[4],
            error=row[5],
            started_at=row[6],
            completed_at=row[7],
            created_at=row[8],
        )


class DuplicateRepository:
    """Repository for duplicate groups."""

    def __init__(self, db: aiosqlite.Connection):
        self.db = db

    async def create_group(
        self,
        group_type: str = "exact",
        group_hash: str | None = None,
        description: str | None = None,
    ) -> DuplicateGroup:
        """Create a new duplicate group."""
        cursor = await self.db.execute(
            """
            INSERT INTO duplicate_groups (group_type, group_hash, description)
            VALUES (?, ?, ?)
            """,
            (group_type, group_hash, description)
        )
        await self.db.commit()

        return DuplicateGroup(
            id=cursor.lastrowid,
            group_type=group_type,
            group_hash=group_hash,
            description=description,
        )

    async def add_member(
        self,
        group_id: int,
        photo_id: int,
        is_best: bool = False,
        quality_score: float | None = None,
    ) -> None:
        """Add photo to duplicate group."""
        await self.db.execute(
            """
            INSERT INTO duplicate_members (group_id, photo_id, is_best, quality_score)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(group_id, photo_id) DO UPDATE SET
                is_best = excluded.is_best,
                quality_score = excluded.quality_score
            """,
            (group_id, photo_id, is_best, quality_score)
        )

        # Update photo count
        await self.db.execute(
            """
            UPDATE duplicate_groups SET photo_count = (
                SELECT COUNT(*) FROM duplicate_members WHERE group_id = ?
            ) WHERE id = ?
            """,
            (group_id, group_id)
        )
        await self.db.commit()

    async def get_all_groups(self) -> list[DuplicateGroup]:
        """Get all duplicate groups with photos."""
        cursor = await self.db.execute(
            "SELECT * FROM duplicate_groups WHERE photo_count > 1 ORDER BY photo_count DESC"
        )
        rows = await cursor.fetchall()

        groups = []
        for row in rows:
            group = DuplicateGroup(
                id=row[0],
                group_type=row[1],
                group_hash=row[2],
                photo_count=row[3],
                description=row[4],
            )
            group.photos = await self._get_group_photos(group.id)
            groups.append(group)

        return groups

    async def get_group(self, group_id: int) -> DuplicateGroup | None:
        """Get duplicate group with photos."""
        cursor = await self.db.execute(
            "SELECT * FROM duplicate_groups WHERE id = ?",
            (group_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return None

        group = DuplicateGroup(
            id=row[0],
            group_type=row[1],
            group_hash=row[2],
            photo_count=row[3],
            description=row[4],
        )
        group.photos = await self._get_group_photos(group.id)
        return group

    async def _get_group_photos(self, group_id: int) -> list[Photo]:
        """Get photos in a duplicate group."""
        cursor = await self.db.execute(
            """
            SELECT p.*, dm.is_best, dm.quality_score
            FROM photos p
            JOIN duplicate_members dm ON p.id = dm.photo_id
            WHERE dm.group_id = ?
            ORDER BY dm.is_best DESC, dm.quality_score DESC
            """,
            (group_id,)
        )
        rows = await cursor.fetchall()

        # Return just the photo data (first 10 columns)
        photos = []
        for row in rows:
            photos.append(Photo(
                id=row[0],
                path=row[1],
                filename=row[2],
                file_hash=row[3],
                phash=row[4],
                file_size=row[5],
                width=row[6],
                height=row[7],
                created_at=row[8],
                updated_at=row[9],
            ))
        return photos


class ActionRepository:
    """Repository for action history (undo support)."""

    def __init__(self, db: aiosqlite.Connection):
        self.db = db

    async def record(
        self,
        action_type: str,
        original_path: str,
        photo_id: int | None = None,
        new_path: str | None = None,
    ) -> ActionRecord:
        """Record an action for undo."""
        cursor = await self.db.execute(
            """
            INSERT INTO action_history (action_type, photo_id, original_path, new_path)
            VALUES (?, ?, ?, ?)
            """,
            (action_type, photo_id, original_path, new_path)
        )
        await self.db.commit()

        return ActionRecord(
            id=cursor.lastrowid,
            action_type=action_type,
            photo_id=photo_id,
            original_path=original_path,
            new_path=new_path,
        )

    async def get_undoable(self, limit: int = 20) -> list[ActionRecord]:
        """Get recent undoable actions."""
        cursor = await self.db.execute(
            """
            SELECT * FROM action_history
            WHERE can_undo = 1 AND undone_at IS NULL
            ORDER BY created_at DESC LIMIT ?
            """,
            (limit,)
        )
        rows = await cursor.fetchall()
        return [self._row_to_action(row) for row in rows]

    async def mark_undone(self, action_id: int) -> None:
        """Mark action as undone."""
        await self.db.execute(
            "UPDATE action_history SET undone_at = CURRENT_TIMESTAMP WHERE id = ?",
            (action_id,)
        )
        await self.db.commit()

    async def get(self, action_id: int) -> ActionRecord | None:
        """Get action by ID."""
        cursor = await self.db.execute(
            "SELECT * FROM action_history WHERE id = ?",
            (action_id,)
        )
        row = await cursor.fetchone()
        return self._row_to_action(row) if row else None

    def _row_to_action(self, row: tuple) -> ActionRecord:
        """Convert database row to ActionRecord object."""
        return ActionRecord(
            id=row[0],
            action_type=row[1],
            photo_id=row[2],
            original_path=row[3],
            new_path=row[4],
            can_undo=row[5],
            undone_at=row[6],
            created_at=row[7],
        )
