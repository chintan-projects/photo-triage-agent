"""Service layer for business logic."""
from .metrics import AnalysisMetrics, MetricsTracker

__all__ = [
    "JobService",
    "MetricsTracker",
    "AnalysisMetrics",
    "ConversationService",
    "PhotoSearchAgent",
    "SearchResult",
]


def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name == "JobService":
        from .job_service import JobService

        return JobService
    if name == "ConversationService":
        from .conversation import ConversationService

        return ConversationService
    if name == "PhotoSearchAgent":
        from .search_agent import PhotoSearchAgent

        return PhotoSearchAgent
    if name == "SearchResult":
        from .search_agent import SearchResult

        return SearchResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
