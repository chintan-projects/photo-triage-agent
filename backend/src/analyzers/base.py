"""Base classes for analyzers"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar('T')

@dataclass
class Result(Generic[T]):
    """Result type for operations that can fail."""
    success: bool
    value: T | None = None
    error: str | None = None

    @classmethod
    def ok(cls, value: T) -> "Result[T]":
        return cls(success=True, value=value)

    @classmethod
    def fail(cls, error: str) -> "Result[T]":
        return cls(success=False, error=error)

@dataclass
class AnalysisResult:
    """Base result from any analyzer."""
    analyzer_name: str
    confidence: float
    metadata: dict

class BaseAnalyzer(ABC):
    """Abstract base class for all analyzers."""

    name: str = "base"
    version: str = "1.0.0"

    @abstractmethod
    def analyze(self, image_path: str) -> Result[AnalysisResult]:
        """Analyze a single image."""
        pass

    def analyze_batch(self, paths: list[str]) -> list[Result[AnalysisResult]]:
        """Analyze multiple images. Override for efficiency."""
        return [self.analyze(p) for p in paths]
