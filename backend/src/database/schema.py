"""SQLite database schema for Photo Triage Agent."""
from pathlib import Path

import aiosqlite
import structlog

logger = structlog.get_logger()

# Schema version for migrations
SCHEMA_VERSION = 1

SCHEMA_SQL = """
-- Photos table: Core photo metadata
CREATE TABLE IF NOT EXISTS photos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL,
    filename TEXT NOT NULL,
    file_hash TEXT,              -- SHA256 for change detection
    phash TEXT,                  -- Perceptual hash for duplicates
    file_size INTEGER,
    width INTEGER,
    height INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Analysis results: One row per analyzer per photo
CREATE TABLE IF NOT EXISTS analysis_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    analyzer TEXT NOT NULL,      -- 'lfm', 'blur', 'screenshot', 'hasher'
    result JSON NOT NULL,        -- Analyzer-specific result data
    confidence REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(photo_id, analyzer)
);

-- Duplicate groups: Groups of similar photos
CREATE TABLE IF NOT EXISTS duplicate_groups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    group_type TEXT NOT NULL DEFAULT 'exact',  -- 'exact', 'semantic', 'burst'
    group_hash TEXT,             -- Hash identifying the group
    photo_count INTEGER DEFAULT 0,
    description TEXT,            -- LFM-generated description of similarity
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Duplicate members: Photos belonging to duplicate groups
CREATE TABLE IF NOT EXISTS duplicate_members (
    group_id INTEGER NOT NULL REFERENCES duplicate_groups(id) ON DELETE CASCADE,
    photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    is_best BOOLEAN DEFAULT FALSE,  -- Recommended keeper
    quality_score REAL,             -- For ranking within group
    PRIMARY KEY (group_id, photo_id)
);

-- Jobs: Async processing jobs for folder analysis
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,         -- UUID
    folder_path TEXT NOT NULL,
    status TEXT DEFAULT 'pending',  -- pending, running, completed, failed, cancelled
    total_photos INTEGER DEFAULT 0,
    processed_photos INTEGER DEFAULT 0,
    error TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Job metrics: Performance tracking per job
CREATE TABLE IF NOT EXISTS job_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    photos_per_second REAL,
    memory_mb REAL,
    avg_inference_ms REAL,
    UNIQUE(job_id, timestamp)
);

-- Action history: For undo functionality
CREATE TABLE IF NOT EXISTS action_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action_type TEXT NOT NULL,   -- 'trash', 'move', 'restore'
    photo_id INTEGER REFERENCES photos(id),
    original_path TEXT NOT NULL,
    new_path TEXT,
    can_undo BOOLEAN DEFAULT TRUE,
    undone_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Conversations: Chat session state for conversational search
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,         -- UUID
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Conversation messages: Individual messages in a conversation
CREATE TABLE IF NOT EXISTS conversation_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,          -- 'user', 'assistant'
    content TEXT NOT NULL,
    photo_ids JSON,              -- Photos referenced in response
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Schema metadata
CREATE TABLE IF NOT EXISTS schema_info (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_photos_phash ON photos(phash);
CREATE INDEX IF NOT EXISTS idx_photos_path ON photos(path);
CREATE INDEX IF NOT EXISTS idx_analysis_photo_id ON analysis_results(photo_id);
CREATE INDEX IF NOT EXISTS idx_analysis_analyzer ON analysis_results(analyzer);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_action_history_photo ON action_history(photo_id);
CREATE INDEX IF NOT EXISTS idx_duplicate_members_photo ON duplicate_members(photo_id);
"""


async def init_database(db_path: str | Path) -> aiosqlite.Connection:
    """Initialize database with schema.

    Args:
        db_path: Path to SQLite database file.

    Returns:
        Database connection.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("initializing_database", path=str(db_path))

    db = await aiosqlite.connect(str(db_path))

    # Enable foreign keys
    await db.execute("PRAGMA foreign_keys = ON")

    # Execute schema
    await db.executescript(SCHEMA_SQL)

    # Set schema version
    await db.execute(
        "INSERT OR REPLACE INTO schema_info (key, value) VALUES (?, ?)",
        ("version", str(SCHEMA_VERSION))
    )

    await db.commit()

    logger.info("database_initialized", path=str(db_path), version=SCHEMA_VERSION)

    return db


async def get_schema_version(db: aiosqlite.Connection) -> int | None:
    """Get current schema version from database."""
    try:
        cursor = await db.execute(
            "SELECT value FROM schema_info WHERE key = ?",
            ("version",)
        )
        row = await cursor.fetchone()
        return int(row[0]) if row else None
    except aiosqlite.OperationalError:
        return None


async def close_database(db: aiosqlite.Connection) -> None:
    """Close database connection."""
    await db.close()
    logger.info("database_closed")
