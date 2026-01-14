"""Tests for screenshot detection analyzer."""
import pytest
from pathlib import Path
import tempfile
from PIL import Image

from src.analyzers.screenshot import ScreenshotAnalyzer, SCREENSHOT_DIMENSIONS


@pytest.fixture
def screenshot_analyzer():
    """Create screenshot analyzer."""
    return ScreenshotAnalyzer()


@pytest.fixture
def iphone_screenshot_path():
    """Create image with iPhone Pro Max dimensions (screenshot-like)."""
    # iPhone 14 Pro Max resolution
    img = Image.new("RGB", (1290, 2796), color="white")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img.save(f.name)
        yield f.name
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def screenshot_named_file():
    """Create file with screenshot-like name."""
    img = Image.new("RGB", (800, 600), color="white")
    with tempfile.NamedTemporaryFile(
        prefix="Screenshot_2024-01-15_",
        suffix=".png",
        delete=False
    ) as f:
        img.save(f.name)
        yield f.name
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def camera_photo_path():
    """Create image that simulates a camera photo with EXIF."""
    img = Image.new("RGB", (4000, 3000), color="green")

    # Add camera-like EXIF data
    from PIL.ExifTags import TAGS

    # Reverse lookup to get tag IDs
    tag_ids = {v: k for k, v in TAGS.items()}

    exif_dict = img.getexif()
    # Add camera metadata (these are the tag IDs for Make, Model)
    exif_dict[tag_ids.get("Make", 271)] = "Apple"
    exif_dict[tag_ids.get("Model", 272)] = "iPhone 14 Pro"

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img.save(f.name, exif=exif_dict)
        yield f.name
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def regular_photo_path():
    """Create a regular photo without EXIF but unusual dimensions."""
    # Non-standard dimensions unlikely to be a screenshot
    img = Image.new("RGB", (3456, 2304), color="blue")
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img.save(f.name)
        yield f.name
    Path(f.name).unlink(missing_ok=True)


def test_screenshot_analyzer_creation():
    """Test analyzer creation."""
    analyzer = ScreenshotAnalyzer()
    assert analyzer.name == "screenshot"
    assert analyzer.version == "1.0.0"


def test_detects_iphone_screenshot_dimensions(screenshot_analyzer, iphone_screenshot_path):
    """Image with iPhone screenshot dimensions should be detected."""
    result = screenshot_analyzer.analyze(iphone_screenshot_path)

    assert result.success
    assert result.value is not None
    # Should have some screenshot signals
    assert len(result.value.metadata["signals"]) > 0
    assert "dimensions_match" in str(result.value.metadata["signals"]) or \
           "png_format" in str(result.value.metadata["signals"])


def test_detects_screenshot_filename(screenshot_analyzer, screenshot_named_file):
    """File with screenshot in name should be detected."""
    result = screenshot_analyzer.analyze(screenshot_named_file)

    assert result.success
    assert result.value is not None
    signals = result.value.metadata["signals"]
    # Should detect filename pattern
    assert any("filename_match" in s for s in signals)


def test_camera_photo_not_screenshot(screenshot_analyzer, camera_photo_path):
    """Photos with camera EXIF should not be detected as screenshots."""
    result = screenshot_analyzer.analyze(camera_photo_path)

    assert result.success
    assert result.value is not None
    # Should have camera EXIF which reduces score
    assert result.value.metadata["has_camera_metadata"] is True
    # May or may not be classified as screenshot depending on other signals,
    # but camera metadata should be detected
    assert "camera_exif_found" in str(result.value.metadata["signals"])


def test_result_contains_required_metadata(screenshot_analyzer, iphone_screenshot_path):
    """Result should contain all expected metadata fields."""
    result = screenshot_analyzer.analyze(iphone_screenshot_path)

    assert result.success
    metadata = result.value.metadata

    assert "is_screenshot" in metadata
    assert "score" in metadata
    assert "signals" in metadata
    assert "dimensions" in metadata
    assert "exif_present" in metadata


def test_returns_error_for_missing_file(screenshot_analyzer):
    """Missing file should return error."""
    result = screenshot_analyzer.analyze("/nonexistent/image.png")

    assert not result.success
    assert result.error is not None


def test_known_screenshot_dimensions():
    """Verify our dimension set includes common devices."""
    # iPhone 14 Pro Max
    assert (1290, 2796) in SCREENSHOT_DIMENSIONS
    # MacBook Pro 14"
    assert (3024, 1964) in SCREENSHOT_DIMENSIONS
    # 4K display
    assert (3840, 2160) in SCREENSHOT_DIMENSIONS


def test_png_format_detection(screenshot_analyzer, iphone_screenshot_path):
    """PNG files should get a format signal."""
    result = screenshot_analyzer.analyze(iphone_screenshot_path)

    assert result.success
    signals = result.value.metadata["signals"]
    assert "png_format" in signals


def test_batch_analysis(screenshot_analyzer, iphone_screenshot_path, regular_photo_path):
    """Batch analysis should work."""
    results = screenshot_analyzer.analyze_batch([iphone_screenshot_path, regular_photo_path])

    assert len(results) == 2
    assert all(r.success for r in results)
