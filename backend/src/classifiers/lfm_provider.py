"""LFM2.5-VL model provider using llama-cpp-python."""
import base64
import json
import re
from pathlib import Path

import structlog
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

from .base import ClassificationResult, ImageCategory, ModelProvider

logger = structlog.get_logger()

# Classification prompt that asks for structured output
CLASSIFICATION_PROMPT = """Analyze this image and classify it. Respond with JSON only.

Categories: people, landscape, food, document, screenshot, meme, object, animal, other

JSON format:
{
  "category": "<one of the categories above>",
  "contains_faces": <true/false>,
  "is_screenshot": <true/false>,
  "is_meme": <true/false>,
  "description": "<brief 1-sentence description>"
}

Respond with only the JSON, nothing else."""


def image_to_data_uri(image_path: str) -> str:
    """Convert image file to base64 data URI."""
    path = Path(image_path)
    suffix = path.suffix.lower()

    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".heic": "image/heic",
    }

    mime_type = mime_types.get(suffix, "image/jpeg")

    with open(path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{image_data}"


def parse_category(category_str: str) -> ImageCategory:
    """Parse category string to enum, with fallback."""
    category_map = {
        "people": ImageCategory.PEOPLE,
        "landscape": ImageCategory.LANDSCAPE,
        "food": ImageCategory.FOOD,
        "document": ImageCategory.DOCUMENT,
        "screenshot": ImageCategory.SCREENSHOT,
        "meme": ImageCategory.MEME,
        "object": ImageCategory.OBJECT,
        "animal": ImageCategory.ANIMAL,
        "other": ImageCategory.OTHER,
    }
    return category_map.get(category_str.lower().strip(), ImageCategory.OTHER)


def extract_json_from_response(response: str) -> dict:
    """Extract JSON object from model response, handling markdown blocks."""
    # Try to find JSON in markdown code block
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))

    # Try to find raw JSON object
    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(0))

    raise ValueError(f"No JSON found in response: {response[:200]}")


class LFMProvider(ModelProvider):
    """LFM2.5-VL model provider for image classification."""

    name = "lfm2.5-vl"

    def __init__(
        self,
        model_path: str,
        mmproj_path: str,
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,
        verbose: bool = False,
    ):
        """Initialize the LFM provider.

        Args:
            model_path: Path to the main GGUF model file
            mmproj_path: Path to the mmproj (vision) GGUF file
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU (-1 for all)
            verbose: Enable verbose logging from llama.cpp
        """
        self.model_path = Path(model_path)
        self.mmproj_path = Path(mmproj_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not self.mmproj_path.exists():
            raise FileNotFoundError(f"mmproj not found: {mmproj_path}")

        logger.info(
            "loading_lfm_model",
            model_path=str(self.model_path),
            mmproj_path=str(self.mmproj_path),
        )

        self.chat_handler = Llava15ChatHandler(
            clip_model_path=str(self.mmproj_path),
            verbose=verbose,
        )

        self.llm = Llama(
            model_path=str(self.model_path),
            chat_handler=self.chat_handler,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )

        logger.info("lfm_model_loaded", model=self.name)

    def classify(self, image_path: str) -> ClassificationResult:
        """Classify a single image.

        Args:
            image_path: Path to the image file

        Returns:
            ClassificationResult with category and metadata
        """
        logger.debug("classifying_image", image_path=image_path)

        # Convert image to data URI for the model
        image_uri = image_to_data_uri(image_path)

        # Create chat message with image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_uri}},
                    {"type": "text", "text": CLASSIFICATION_PROMPT},
                ],
            }
        ]

        # Get model response
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=256,
            temperature=0.1,  # Low temperature for consistent classification
        )

        response_text = response["choices"][0]["message"]["content"]
        logger.debug("model_response", response=response_text[:200])

        # Parse the JSON response
        try:
            result = extract_json_from_response(response_text)

            return ClassificationResult(
                category=parse_category(result.get("category", "other")),
                confidence=0.8,  # LFM doesn't provide confidence scores
                contains_faces=result.get("contains_faces", False),
                is_screenshot=result.get("is_screenshot", False),
                is_meme=result.get("is_meme", False),
                description=result.get("description", ""),
            )
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(
                "classification_parse_error",
                error=str(e),
                response=response_text[:200],
            )
            # Return a default result on parse failure
            return ClassificationResult(
                category=ImageCategory.OTHER,
                confidence=0.0,
                contains_faces=False,
                is_screenshot=False,
                is_meme=False,
                description=f"Parse error: {response_text[:100]}",
            )

    def classify_batch(self, image_paths: list[str]) -> list[ClassificationResult]:
        """Classify multiple images.

        Currently processes sequentially. Could be optimized with batching
        if the model supports it.
        """
        results = []
        for path in image_paths:
            try:
                result = self.classify(path)
                results.append(result)
            except Exception as e:
                logger.error("batch_classification_error", path=path, error=str(e))
                results.append(
                    ClassificationResult(
                        category=ImageCategory.OTHER,
                        confidence=0.0,
                        contains_faces=False,
                        is_screenshot=False,
                        is_meme=False,
                        description=f"Error: {str(e)}",
                    )
                )
        return results

    def health_check(self) -> bool:
        """Verify model is loaded and ready."""
        try:
            # Simple check - model object exists and has expected attributes
            return self.llm is not None and self.chat_handler is not None
        except Exception:
            return False
