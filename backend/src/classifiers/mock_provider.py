"""Mock model provider for development and testing."""
import random
from pathlib import Path

import structlog

from .base import ClassificationResult, ImageCategory, ModelProvider

logger = structlog.get_logger()


class MockProvider(ModelProvider):
    """Mock provider that returns random classifications.

    Useful for:
    - Development without loading real models
    - Testing the pipeline
    - CI/CD environments
    """

    name = "mock"

    def __init__(self, default_category: ImageCategory | None = None):
        """Initialize mock provider.

        Args:
            default_category: If set, always return this category.
                             If None, return random categories.
        """
        self.default_category = default_category
        logger.info("mock_provider_initialized")

    def classify(self, image_path: str) -> ClassificationResult:
        """Return a mock classification result."""
        path = Path(image_path)

        # Use filename hints for semi-realistic results
        filename_lower = path.name.lower()

        if self.default_category:
            category = self.default_category
        elif "screenshot" in filename_lower:
            category = ImageCategory.SCREENSHOT
        elif "meme" in filename_lower:
            category = ImageCategory.MEME
        elif any(x in filename_lower for x in ["person", "people", "selfie", "portrait"]):
            category = ImageCategory.PEOPLE
        elif any(x in filename_lower for x in ["food", "meal", "dinner", "lunch"]):
            category = ImageCategory.FOOD
        elif any(x in filename_lower for x in ["landscape", "nature", "mountain", "beach"]):
            category = ImageCategory.LANDSCAPE
        elif any(x in filename_lower for x in ["doc", "document", "receipt", "scan"]):
            category = ImageCategory.DOCUMENT
        else:
            category = random.choice(list(ImageCategory))

        return ClassificationResult(
            category=category,
            confidence=random.uniform(0.7, 0.95),
            contains_faces=category == ImageCategory.PEOPLE or random.random() > 0.7,
            is_screenshot=category == ImageCategory.SCREENSHOT,
            is_meme=category == ImageCategory.MEME,
            description=f"Mock classification of {path.name}",
        )

    def classify_batch(self, image_paths: list[str]) -> list[ClassificationResult]:
        """Classify multiple images."""
        return [self.classify(p) for p in image_paths]

    def health_check(self) -> bool:
        """Always healthy."""
        return True
