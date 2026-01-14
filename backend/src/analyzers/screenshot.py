"""Screenshot detection analyzer using EXIF and heuristics."""
import re
from pathlib import Path

import structlog
from PIL import Image
from PIL.ExifTags import TAGS

from .base import AnalysisResult, BaseAnalyzer, Result

logger = structlog.get_logger()

# Common screenshot dimensions for various devices
# Format: (width, height) - we check both orientations
SCREENSHOT_DIMENSIONS = {
    # iPhone
    (1170, 2532),  # iPhone 12/13/14 Pro
    (1284, 2778),  # iPhone 12/13/14 Pro Max
    (1179, 2556),  # iPhone 14 Pro
    (1290, 2796),  # iPhone 14/15/16 Pro Max
    (1125, 2436),  # iPhone X/XS/11 Pro
    (1242, 2688),  # iPhone XS Max/11 Pro Max
    (750, 1334),   # iPhone 6/7/8
    (1080, 1920),  # iPhone 6/7/8 Plus
    (640, 1136),   # iPhone 5/5S/SE
    (1179, 2556),  # iPhone 15 Pro
    (1290, 2796),  # iPhone 15 Pro Max
    # iPad
    (2048, 2732),  # iPad Pro 12.9"
    (1668, 2388),  # iPad Pro 11"
    (2160, 1620),  # iPad 10.2"
    (2360, 1640),  # iPad Air
    # Mac common resolutions
    (2560, 1600),  # MacBook Pro 13"
    (2880, 1800),  # MacBook Pro 15"
    (3024, 1964),  # MacBook Pro 14"
    (3456, 2234),  # MacBook Pro 16"
    (1440, 900),   # MacBook Air
    (2560, 1440),  # iMac 27" / external displays
    (1920, 1080),  # Common external monitor
    (3840, 2160),  # 4K display
    (5120, 2880),  # 5K display
    # Android common
    (1080, 2340),  # Many Android phones
    (1440, 3040),  # Samsung Galaxy S10+
    (1440, 3200),  # Samsung Galaxy S20/S21
}

# Filename patterns that suggest screenshot
SCREENSHOT_PATTERNS = [
    r"(?i)screenshot",
    r"(?i)screen\s*shot",
    r"(?i)^IMG_\d+\.PNG$",  # iOS screenshots are PNG
    r"(?i)^Screen\s*Recording",
    r"(?i)simulator",
    r"(?i)^Capture",
    r"(?i)snip",
]

# EXIF tags that indicate a real camera
CAMERA_EXIF_TAGS = {"Make", "Model", "LensModel", "FocalLength", "ExposureTime", "FNumber"}


class ScreenshotAnalyzer(BaseAnalyzer):
    """Detect screenshots using EXIF metadata and heuristics."""

    name = "screenshot"
    version = "1.0.0"

    def analyze(self, image_path: str) -> Result[AnalysisResult]:
        """Analyze an image to determine if it's a screenshot.

        Uses multiple heuristics:
        1. EXIF data - screenshots lack camera metadata
        2. Dimensions - match known device resolutions
        3. Filename - matches common screenshot naming patterns
        4. File format - screenshots often PNG

        Returns:
            Result with screenshot analysis data including:
            - is_screenshot: bool
            - signals: list of detection signals that matched
            - exif_present: bool
            - has_camera_metadata: bool
        """
        try:
            path = Path(image_path)
            signals = []
            score = 0.0

            # Check filename patterns
            filename = path.name
            for pattern in SCREENSHOT_PATTERNS:
                if re.search(pattern, filename):
                    signals.append(f"filename_match:{pattern}")
                    score += 0.3
                    break  # One filename match is enough

            # PNG format is more common for screenshots
            if path.suffix.lower() == ".png":
                signals.append("png_format")
                score += 0.1

            # Load image for dimensions and EXIF
            with Image.open(image_path) as img:
                width, height = img.size

                # Check dimensions (check both orientations)
                dims_match = (
                    (width, height) in SCREENSHOT_DIMENSIONS or
                    (height, width) in SCREENSHOT_DIMENSIONS
                )
                if dims_match:
                    signals.append(f"dimensions_match:{width}x{height}")
                    score += 0.3

                # Check EXIF data
                exif = img.getexif()
                exif_present = bool(exif)

                if not exif_present:
                    signals.append("no_exif")
                    score += 0.2
                else:
                    # Get human-readable tag names
                    exif_tags = set()
                    for tag_id, value in exif.items():
                        tag_name = TAGS.get(tag_id, str(tag_id))
                        exif_tags.add(tag_name)

                    # Check for camera-specific tags
                    camera_tags_found = exif_tags.intersection(CAMERA_EXIF_TAGS)
                    has_camera_metadata = bool(camera_tags_found)

                    if not has_camera_metadata:
                        signals.append("no_camera_exif")
                        score += 0.3
                    else:
                        # Has camera metadata - unlikely to be screenshot
                        signals.append(f"camera_exif_found:{list(camera_tags_found)[:3]}")
                        score -= 0.5

            # Clamp score between 0 and 1
            score = max(0.0, min(1.0, score))
            is_screenshot = score >= 0.4  # Threshold for classification

            logger.debug(
                "screenshot_analysis_complete",
                image_path=image_path,
                is_screenshot=is_screenshot,
                score=score,
                signals=signals,
            )

            return Result.ok(AnalysisResult(
                analyzer_name=self.name,
                confidence=score if is_screenshot else (1.0 - score),
                metadata={
                    "is_screenshot": is_screenshot,
                    "score": score,
                    "signals": signals,
                    "dimensions": f"{width}x{height}",
                    "exif_present": exif_present,
                    "has_camera_metadata": has_camera_metadata if exif_present else False,
                },
            ))

        except Exception as e:
            logger.error("screenshot_analysis_failed", image_path=image_path, error=str(e))
            return Result.fail(f"Screenshot analysis failed: {str(e)}")
