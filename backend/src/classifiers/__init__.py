"""Model providers for image classification."""
from .base import ClassificationResult, ImageCategory, ModelProvider
from .lfm_cli_provider import LFMCliProvider
from .mock_provider import MockProvider

# Note: LFMProvider (Python bindings) doesn't work yet due to
# https://github.com/abetlen/llama-cpp-python/issues/2105
# Use LFMCliProvider instead which uses llama-mtmd-cli subprocess.

__all__ = [
    "ModelProvider",
    "ClassificationResult",
    "ImageCategory",
    "LFMCliProvider",
    "MockProvider",
]
