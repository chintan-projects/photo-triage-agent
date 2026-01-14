"""Tests for blur detection analyzer."""
import cv2
import numpy as np
import pytest
from pathlib import Path
import tempfile

from src.analyzers.blur import BlurAnalyzer, compute_laplacian_variance


@pytest.fixture
def blur_analyzer():
    """Create blur analyzer with default settings."""
    return BlurAnalyzer()


@pytest.fixture
def sharp_image_path():
    """Create a sharp test image (high frequency content)."""
    # Create image with sharp edges
    img = np.zeros((100, 100), dtype=np.uint8)
    # Add checkerboard pattern (very sharp)
    for i in range(0, 100, 10):
        for j in range(0, 100, 10):
            if (i // 10 + j // 10) % 2 == 0:
                img[i:i+10, j:j+10] = 255

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        cv2.imwrite(f.name, img)
        yield f.name
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def blurry_image_path():
    """Create a blurry test image (low frequency content)."""
    # Create uniform gray image (no edges = very blurry detection)
    img = np.full((100, 100), 128, dtype=np.uint8)
    # Add slight noise to avoid divide-by-zero
    img = cv2.GaussianBlur(img, (21, 21), 0)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        cv2.imwrite(f.name, img)
        yield f.name
    Path(f.name).unlink(missing_ok=True)


def test_blur_analyzer_creation():
    """Test blur analyzer can be created with custom threshold."""
    analyzer = BlurAnalyzer(threshold=50.0)
    assert analyzer.name == "blur"
    assert analyzer.threshold == 50.0


def test_blur_analyzer_detects_sharp_image(blur_analyzer, sharp_image_path):
    """Sharp images should not be detected as blurry."""
    result = blur_analyzer.analyze(sharp_image_path)

    assert result.success
    assert result.value is not None
    assert result.value.analyzer_name == "blur"
    assert result.value.metadata["is_blurry"] is False
    assert result.value.metadata["laplacian_variance"] > blur_analyzer.threshold


def test_blur_analyzer_detects_blurry_image(blur_analyzer, blurry_image_path):
    """Blurry images should be detected."""
    result = blur_analyzer.analyze(blurry_image_path)

    assert result.success
    assert result.value is not None
    assert result.value.metadata["is_blurry"] is True
    assert result.value.metadata["laplacian_variance"] < blur_analyzer.threshold


def test_blur_analyzer_returns_error_for_missing_file(blur_analyzer):
    """Missing files should return error result."""
    result = blur_analyzer.analyze("/nonexistent/path/image.jpg")

    assert not result.success
    assert result.error is not None
    assert "Failed to load" in result.error


def test_compute_laplacian_variance(sharp_image_path, blurry_image_path):
    """Test standalone variance computation utility."""
    sharp_variance = compute_laplacian_variance(sharp_image_path)
    blurry_variance = compute_laplacian_variance(blurry_image_path)

    assert sharp_variance is not None
    assert blurry_variance is not None
    assert sharp_variance > blurry_variance


def test_compute_laplacian_variance_missing_file():
    """Missing file returns None."""
    variance = compute_laplacian_variance("/nonexistent/image.jpg")
    assert variance is None


def test_blur_analyzer_confidence_calculation(blur_analyzer, sharp_image_path, blurry_image_path):
    """Confidence should be between 0 and 1."""
    sharp_result = blur_analyzer.analyze(sharp_image_path)
    blurry_result = blur_analyzer.analyze(blurry_image_path)

    assert sharp_result.value.confidence >= 0.0
    assert sharp_result.value.confidence <= 1.0
    assert blurry_result.value.confidence >= 0.0
    assert blurry_result.value.confidence <= 1.0


def test_blur_analyzer_batch(blur_analyzer, sharp_image_path, blurry_image_path):
    """Test batch analysis."""
    results = blur_analyzer.analyze_batch([sharp_image_path, blurry_image_path])

    assert len(results) == 2
    assert results[0].success
    assert results[1].success
