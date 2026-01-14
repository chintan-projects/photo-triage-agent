"""Image loading and processing utilities.

This module provides shared image utilities used across all analyzers.
All image format constants and loading functions should be defined here.
"""
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import structlog
from PIL import Image

logger = structlog.get_logger()

# Canonical list of supported image formats - use this everywhere
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".gif", ".bmp", ".tiff", ".tif"}

# Formats that PIL can handle directly
PIL_FORMATS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp"}

# Formats requiring special handling (HEIC needs pillow-heif)
SPECIAL_FORMATS = {".heic"}

def load_image(path: str, mode: str = "RGB") -> Image.Image:
    """Load and normalize an image.

    Args:
        path: Path to the image file
        mode: Color mode to convert to (default RGB)

    Returns:
        PIL Image object

    Raises:
        FileNotFoundError: If image doesn't exist
        ValueError: If format not supported
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {path.suffix}")

    img = Image.open(path)

    if img.mode != mode:
        img = img.convert(mode)

    return img

def get_image_dimensions(path: str) -> tuple[int, int]:
    """Get image dimensions without loading full image."""
    with Image.open(path) as img:
        return img.size

def is_supported_format(path: str) -> bool:
    """Check if file format is supported."""
    return Path(path).suffix.lower() in SUPPORTED_FORMATS


def load_image_cv2(
    path: str,
    mode: Literal["color", "grayscale"] = "color"
) -> np.ndarray | None:
    """Load image using OpenCV.

    Use this for operations requiring numpy arrays (blur detection, etc.).

    Args:
        path: Path to the image file.
        mode: "color" for BGR, "grayscale" for single channel.

    Returns:
        numpy array of image data, or None if loading failed.
    """
    flag = cv2.IMREAD_COLOR if mode == "color" else cv2.IMREAD_GRAYSCALE
    img = cv2.imread(path, flag)
    return img
