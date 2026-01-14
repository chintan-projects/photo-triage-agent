"""Blur detection analyzer using Laplacian variance."""
import cv2
import structlog

from ..utils.image import load_image_cv2
from .base import AnalysisResult, BaseAnalyzer, Result

logger = structlog.get_logger()

# Threshold below which an image is considered blurry
# This value is tuned empirically - lower values = more blur
DEFAULT_BLUR_THRESHOLD = 100.0


class BlurAnalyzer(BaseAnalyzer):
    """Detect blurry images using Laplacian variance.

    The Laplacian operator highlights edges. Sharp images have high
    variance in the Laplacian, blurry images have low variance.
    """

    name = "blur"
    version = "1.0.0"

    def __init__(self, threshold: float = DEFAULT_BLUR_THRESHOLD):
        """Initialize blur analyzer.

        Args:
            threshold: Laplacian variance below this is considered blurry.
                       Default is 100.0, adjust based on your dataset.
        """
        self.threshold = threshold

    def analyze(self, image_path: str) -> Result[AnalysisResult]:
        """Analyze an image for blur.

        Args:
            image_path: Path to the image file.

        Returns:
            Result with blur analysis data including:
            - is_blurry: bool
            - laplacian_variance: float (higher = sharper)
            - threshold_used: float
        """
        try:
            # Load image in grayscale for Laplacian
            img = load_image_cv2(image_path, mode="grayscale")

            if img is None:
                return Result.fail(f"Failed to load image: {image_path}")

            # Compute Laplacian and its variance
            laplacian = cv2.Laplacian(img, cv2.CV_64F)
            variance = float(laplacian.var())

            is_blurry = variance < self.threshold

            # Confidence is based on how far from threshold
            # 1.0 = definitely sharp/blurry, 0.5 = borderline
            if is_blurry:
                # More blurry = higher confidence in blur detection
                confidence = min(1.0, 1.0 - (variance / self.threshold))
            else:
                # More sharp = higher confidence in sharpness
                confidence = min(1.0, variance / (self.threshold * 2))

            logger.debug(
                "blur_analysis_complete",
                image_path=image_path,
                variance=variance,
                is_blurry=is_blurry,
            )

            return Result.ok(AnalysisResult(
                analyzer_name=self.name,
                confidence=confidence,
                metadata={
                    "is_blurry": is_blurry,
                    "laplacian_variance": variance,
                    "threshold_used": self.threshold,
                },
            ))

        except Exception as e:
            logger.error("blur_analysis_failed", image_path=image_path, error=str(e))
            return Result.fail(f"Blur analysis failed: {str(e)}")


def compute_laplacian_variance(image_path: str) -> float | None:
    """Utility function to get just the Laplacian variance.

    Useful for debugging or quick checks without full analysis.

    Returns:
        Laplacian variance value, or None if image couldn't be loaded.
    """
    img = load_image_cv2(image_path, mode="grayscale")
    if img is None:
        return None
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    return float(laplacian.var())
