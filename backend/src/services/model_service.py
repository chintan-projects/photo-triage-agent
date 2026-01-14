"""Model provider service - manages singleton model instance."""

import structlog

from ..classifiers import LFMCliProvider, MockProvider, ModelProvider
from ..classifiers.base import ClassificationResult
from ..config import LFM_MMPROJ_PATH, LFM_MODEL_PATH

logger = structlog.get_logger()

# Singleton model provider instance
_provider: ModelProvider | None = None


def get_provider() -> ModelProvider:
    """Get or create the model provider singleton.

    Uses LFMCliProvider if model files exist, otherwise MockProvider.
    """
    global _provider

    if _provider is not None:
        return _provider

    if LFM_MODEL_PATH.exists() and LFM_MMPROJ_PATH.exists():
        logger.info(
            "initializing_lfm_provider",
            model_path=str(LFM_MODEL_PATH),
        )
        _provider = LFMCliProvider(
            model_path=str(LFM_MODEL_PATH),
            mmproj_path=str(LFM_MMPROJ_PATH),
        )
    else:
        logger.warning(
            "model_not_found_using_mock",
            expected_model=str(LFM_MODEL_PATH),
        )
        _provider = MockProvider()

    return _provider


def is_model_loaded() -> bool:
    """Check if a real model (not mock) is loaded."""
    provider = get_provider()
    return provider.name != "mock" and provider.health_check()


def classify_image(image_path: str) -> ClassificationResult:
    """Classify a single image using the model provider."""
    provider = get_provider()
    return provider.classify(image_path)


def classify_images(image_paths: list[str]) -> list[ClassificationResult]:
    """Classify multiple images."""
    provider = get_provider()
    return provider.classify_batch(image_paths)
