"""Upload service for handling photo uploads.

Handles temporary storage of uploaded photos for analysis.
Follows robustness principle - all temp files are cleaned up.
"""
import shutil
import uuid
from datetime import datetime
from pathlib import Path

import structlog

from ..utils.image import SUPPORTED_FORMATS

logger = structlog.get_logger()

# Base directory for uploaded photos (inside backend/data/)
UPLOAD_BASE_DIR = Path(__file__).parent.parent.parent / "data" / "uploads"


class UploadSession:
    """Represents a photo upload session."""

    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.upload_dir = UPLOAD_BASE_DIR / self.session_id
        self.photos: list[Path] = []
        self.created_at = datetime.now()

    @property
    def photo_count(self) -> int:
        return len(self.photos)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "photo_count": self.photo_count,
            "upload_dir": str(self.upload_dir),
            "photos": [str(p) for p in self.photos],
        }


class UploadService:
    """Manages photo upload sessions.

    Handles:
    - Creating upload sessions
    - Storing uploaded files
    - Tracking session state
    - Cleaning up old sessions
    """

    def __init__(self):
        self._sessions: dict[str, UploadSession] = {}
        # Ensure base upload directory exists
        UPLOAD_BASE_DIR.mkdir(parents=True, exist_ok=True)

    def create_session(self) -> UploadSession:
        """Create a new upload session."""
        session = UploadSession()
        session.upload_dir.mkdir(parents=True, exist_ok=True)
        self._sessions[session.session_id] = session
        logger.info("upload_session_created", session_id=session.session_id)
        return session

    def get_session(self, session_id: str) -> UploadSession | None:
        """Get an existing session."""
        return self._sessions.get(session_id)

    async def add_photo(
        self,
        session_id: str,
        filename: str,
        content: bytes,
    ) -> Path | None:
        """Add a photo to an upload session.

        Args:
            session_id: The upload session ID.
            filename: Original filename.
            content: File content bytes.

        Returns:
            Path to saved file, or None if invalid.
        """
        session = self.get_session(session_id)
        if not session:
            logger.warning("upload_to_unknown_session", session_id=session_id)
            return None

        # Validate file extension
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_FORMATS:
            logger.debug("skipping_unsupported_format", filename=filename, ext=ext)
            return None

        # Save file with unique name to avoid collisions
        safe_filename = f"{len(session.photos):05d}_{Path(filename).name}"
        file_path = session.upload_dir / safe_filename

        try:
            file_path.write_bytes(content)
            session.photos.append(file_path)
            return file_path
        except Exception as e:
            logger.error("failed_to_save_upload", filename=filename, error=str(e))
            return None

    def get_session_folder(self, session_id: str) -> Path | None:
        """Get the folder path for a session (for analysis)."""
        session = self.get_session(session_id)
        if session:
            return session.upload_dir
        return None

    def cleanup_session(self, session_id: str) -> bool:
        """Clean up a session and its files.

        Returns True if cleanup was successful.
        """
        session = self._sessions.pop(session_id, None)
        if not session:
            return False

        try:
            if session.upload_dir.exists():
                shutil.rmtree(session.upload_dir)
            logger.info(
                "upload_session_cleaned",
                session_id=session_id,
                photo_count=session.photo_count,
            )
            return True
        except Exception as e:
            logger.error(
                "cleanup_failed",
                session_id=session_id,
                error=str(e),
            )
            return False

    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """Clean up sessions older than max_age_hours.

        Returns number of sessions cleaned up.
        """
        now = datetime.now()
        old_sessions = []

        for session_id, session in self._sessions.items():
            age_hours = (now - session.created_at).total_seconds() / 3600
            if age_hours > max_age_hours:
                old_sessions.append(session_id)

        for session_id in old_sessions:
            self.cleanup_session(session_id)

        if old_sessions:
            logger.info("cleaned_old_sessions", count=len(old_sessions))

        return len(old_sessions)

    def get_stats(self) -> dict:
        """Get upload service statistics."""
        total_photos = sum(s.photo_count for s in self._sessions.values())
        return {
            "active_sessions": len(self._sessions),
            "total_photos": total_photos,
        }


# Global singleton instance
_upload_service: UploadService | None = None


def get_upload_service() -> UploadService:
    """Get the global upload service instance."""
    global _upload_service
    if _upload_service is None:
        _upload_service = UploadService()
    return _upload_service
