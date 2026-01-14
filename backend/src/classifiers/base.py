"""Base classes for model providers"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class ImageCategory(Enum):
    PEOPLE = "people"
    LANDSCAPE = "landscape"
    FOOD = "food"
    DOCUMENT = "document"
    SCREENSHOT = "screenshot"
    MEME = "meme"
    OBJECT = "object"
    ANIMAL = "animal"
    OTHER = "other"

@dataclass
class ClassificationResult:
    """Result from image classification."""
    category: ImageCategory
    confidence: float
    contains_faces: bool
    is_screenshot: bool
    is_meme: bool
    description: str

class ModelProvider(ABC):
    """Abstract interface for vision model providers.

    Implement this to add new models. The orchestrator doesn't
    care which model is used—just that it returns ClassificationResult.
    """

    name: str = "base"

    @abstractmethod
    def classify(self, image_path: str) -> ClassificationResult:
        """Classify a single image."""
        pass

    @abstractmethod
    def classify_batch(self, image_paths: list[str]) -> list[ClassificationResult]:
        """Classify multiple images."""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Verify model is loaded and ready."""
        pass
