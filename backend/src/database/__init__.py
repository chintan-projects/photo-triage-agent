"""Database module for Photo Triage Agent."""
from .repository import (
    ActionRecord,
    ActionRepository,
    AnalysisRepository,
    AnalysisResult,
    DuplicateGroup,
    DuplicateRepository,
    Job,
    JobRepository,
    Photo,
    PhotoRepository,
)
from .schema import (
    SCHEMA_VERSION,
    close_database,
    get_schema_version,
    init_database,
)

__all__ = [
    # Schema
    "SCHEMA_VERSION",
    "init_database",
    "close_database",
    "get_schema_version",
    # Data classes
    "Photo",
    "AnalysisResult",
    "Job",
    "DuplicateGroup",
    "ActionRecord",
    # Repositories
    "PhotoRepository",
    "AnalysisRepository",
    "JobRepository",
    "DuplicateRepository",
    "ActionRepository",
]
