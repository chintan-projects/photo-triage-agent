"""Image analyzers for blur, screenshot, and duplicate detection."""
from .base import AnalysisResult, BaseAnalyzer, Result
from .blur import BlurAnalyzer, compute_laplacian_variance
from .hasher import (
    PerceptualHasher,
    are_duplicates,
    compute_phash,
    find_duplicates,
    hamming_distance,
)
from .screenshot import ScreenshotAnalyzer

__all__ = [
    # Base classes
    "AnalysisResult",
    "BaseAnalyzer",
    "Result",
    # Analyzers
    "BlurAnalyzer",
    "PerceptualHasher",
    "ScreenshotAnalyzer",
    # Utility functions
    "are_duplicates",
    "compute_laplacian_variance",
    "compute_phash",
    "find_duplicates",
    "hamming_distance",
]


class AnalyzerRegistry:
    """Registry for all available analyzers."""

    _analyzers: dict[str, BaseAnalyzer] = {}

    @classmethod
    def register(cls, analyzer: BaseAnalyzer) -> None:
        """Register an analyzer instance."""
        cls._analyzers[analyzer.name] = analyzer

    @classmethod
    def get(cls, name: str) -> BaseAnalyzer | None:
        """Get analyzer by name."""
        return cls._analyzers.get(name)

    @classmethod
    def all(cls) -> dict[str, BaseAnalyzer]:
        """Get all registered analyzers."""
        return cls._analyzers.copy()

    @classmethod
    def run_all(cls, image_path: str) -> dict[str, Result[AnalysisResult]]:
        """Run all registered analyzers on an image."""
        return {name: a.analyze(image_path) for name, a in cls._analyzers.items()}


# Register default analyzers
AnalyzerRegistry.register(BlurAnalyzer())
AnalyzerRegistry.register(ScreenshotAnalyzer())
AnalyzerRegistry.register(PerceptualHasher())
