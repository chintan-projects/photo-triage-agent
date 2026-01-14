"""LFM2.5-VL provider using llama-mtmd-cli subprocess.

This provider uses the llama-mtmd-cli command-line tool for vision inference,
which has full support for LFM2.5-VL models. This is a workaround until
llama-cpp-python has native handler support for LFM2-VL.

Requires: llama-mtmd-cli installed via `brew install llama.cpp`
"""
import json
import re
import subprocess
from pathlib import Path

import structlog

from .base import ClassificationResult, ImageCategory, ModelProvider

logger = structlog.get_logger()

CLASSIFICATION_PROMPT = """Analyze this image and classify it. Respond with JSON only.

Categories: people, landscape, food, document, screenshot, meme, object, animal, other

JSON format:
{
  "category": "<category>",
  "contains_faces": true/false,
  "is_screenshot": true/false,
  "is_meme": true/false,
  "description": "<brief description>"
}

JSON only, no other text:"""


def parse_category(category_str: str) -> ImageCategory:
    """Parse category string to enum."""
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
    """Extract JSON object from CLI output."""
    # Try to find JSON in markdown code block
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))

    # Try to find raw JSON object
    json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(0))

    raise ValueError(f"No JSON found in response: {response[:200]}")


class LFMCliProvider(ModelProvider):
    """LFM2.5-VL provider using llama-mtmd-cli subprocess."""

    name = "lfm2.5-vl-cli"

    def __init__(
        self,
        model_path: str,
        mmproj_path: str,
        cli_path: str = "llama-mtmd-cli",
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,
    ):
        """Initialize the CLI-based provider.

        Args:
            model_path: Path to the main GGUF model file
            mmproj_path: Path to the mmproj GGUF file
            cli_path: Path to llama-mtmd-cli executable
            n_ctx: Context window size
            n_gpu_layers: GPU layers (-1 for all)
        """
        self.model_path = Path(model_path)
        self.mmproj_path = Path(mmproj_path)
        self.cli_path = cli_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not self.mmproj_path.exists():
            raise FileNotFoundError(f"mmproj not found: {mmproj_path}")

        # Verify CLI is available
        try:
            subprocess.run([cli_path, "--version"], capture_output=True, check=False)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"llama-mtmd-cli not found at {cli_path}. "
                "Install with: brew install llama.cpp"
            )

        logger.info(
            "lfm_cli_provider_initialized",
            model_path=str(self.model_path),
            mmproj_path=str(self.mmproj_path),
        )

    def classify(self, image_path: str) -> ClassificationResult:
        """Classify a single image using CLI subprocess."""
        image_path = Path(image_path).resolve()

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        logger.debug("classifying_image_cli", image_path=str(image_path))

        # Build command - use -mm for mmproj (short form)
        cmd = [
            self.cli_path,
            "-m", str(self.model_path),
            "-mm", str(self.mmproj_path),
            "--image", str(image_path),
            "-p", CLASSIFICATION_PROMPT,
            "-n", "256",  # max tokens
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=120,  # 120 second timeout for model loading
            )
            # Decode with error handling for non-UTF-8 bytes in output
            stdout = result.stdout.decode("utf-8", errors="replace")
            stderr = result.stderr.decode("utf-8", errors="replace")

            # Check for actual errors (not just warnings in stderr)
            # The CLI outputs Metal init info to stderr even on success
            if result.returncode != 0 and not stdout.strip():
                logger.error(
                    "cli_classification_error",
                    stderr=stderr[-500:],
                    returncode=result.returncode,
                )
                return self._error_result(f"CLI error: {stderr[-200:]}")

            response_text = stdout.strip()
            logger.debug("cli_response", response=response_text[:200])

            # Parse JSON from response
            try:
                parsed = extract_json_from_response(response_text)
                return ClassificationResult(
                    category=parse_category(parsed.get("category", "other")),
                    confidence=0.8,
                    contains_faces=parsed.get("contains_faces", False),
                    is_screenshot=parsed.get("is_screenshot", False),
                    is_meme=parsed.get("is_meme", False),
                    description=parsed.get("description", ""),
                )
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    "cli_parse_error",
                    error=str(e),
                    response=response_text[:200],
                )
                return self._error_result(f"Parse error: {str(e)}")

        except subprocess.TimeoutExpired:
            logger.error("cli_timeout", image_path=str(image_path))
            return self._error_result("Classification timed out")

        except Exception as e:
            logger.error("cli_exception", error=str(e))
            return self._error_result(f"Exception: {str(e)}")

    def _error_result(self, description: str) -> ClassificationResult:
        """Return a default result for errors."""
        return ClassificationResult(
            category=ImageCategory.OTHER,
            confidence=0.0,
            contains_faces=False,
            is_screenshot=False,
            is_meme=False,
            description=description,
        )

    def classify_batch(self, image_paths: list[str]) -> list[ClassificationResult]:
        """Classify multiple images (sequential)."""
        results = []
        for path in image_paths:
            try:
                results.append(self.classify(path))
            except Exception as e:
                logger.error("batch_cli_error", path=path, error=str(e))
                results.append(self._error_result(f"Error: {str(e)}"))
        return results

    def health_check(self) -> bool:
        """Check if CLI and model files are accessible."""
        try:
            subprocess.run(
                [self.cli_path, "--version"],
                capture_output=True,
                check=True,
                timeout=5,
            )
            return self.model_path.exists() and self.mmproj_path.exists()
        except Exception:
            return False

    def reason(self, prompt: str, max_tokens: int = 512) -> str:
        """Run text-only reasoning without an image.

        Uses the LFM model for natural language reasoning tasks like
        search query interpretation or photo group explanation.

        Args:
            prompt: The reasoning prompt (no image context needed).
            max_tokens: Maximum tokens to generate.

        Returns:
            Model's text response.

        Raises:
            RuntimeError: If inference fails.
        """
        logger.debug("reason_prompt", prompt=prompt[:100])

        # Build command without --image flag for text-only inference
        cmd = [
            self.cli_path,
            "-m", str(self.model_path),
            "-mm", str(self.mmproj_path),
            "-p", prompt,
            "-n", str(max_tokens),
            "--temp", "0.7",  # Slightly creative for reasoning
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=120,
            )
            stdout = result.stdout.decode("utf-8", errors="replace")
            stderr = result.stderr.decode("utf-8", errors="replace")

            if result.returncode != 0 and not stdout.strip():
                logger.error(
                    "reason_cli_error",
                    stderr=stderr[-500:],
                    returncode=result.returncode,
                )
                raise RuntimeError(f"CLI error: {stderr[-200:]}")

            response = stdout.strip()
            logger.debug("reason_response", response=response[:200])
            return response

        except subprocess.TimeoutExpired:
            logger.error("reason_timeout")
            raise RuntimeError("Reasoning timed out")

        except RuntimeError:
            raise

        except Exception as e:
            logger.error("reason_exception", error=str(e))
            raise RuntimeError(f"Reasoning failed: {str(e)}")
