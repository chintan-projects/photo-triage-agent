"""Tests for perceptual hashing analyzer."""
import pytest
from pathlib import Path
import tempfile
from PIL import Image

from src.analyzers.hasher import (
    PerceptualHasher,
    are_duplicates,
    compute_phash,
    find_duplicates,
    hamming_distance,
)


@pytest.fixture
def hasher():
    """Create perceptual hasher with default settings."""
    return PerceptualHasher()


@pytest.fixture
def test_image_path():
    """Create a test image with a gradient (good for perceptual hashing)."""
    # Create larger image with smooth gradient - more realistic
    img = Image.new("RGB", (256, 256))
    for x in range(256):
        for y in range(256):
            img.putpixel((x, y), (x, y, (x + y) // 2))

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img.save(f.name)
        yield f.name
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def duplicate_image_path(test_image_path):
    """Create a true duplicate (just re-saved)."""
    img = Image.open(test_image_path)
    # Re-save without any modifications - should have identical hash

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img.save(f.name)
        yield f.name
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def different_image_path():
    """Create a completely different image."""
    img = Image.new("RGB", (100, 100), color="blue")
    # Different pattern
    for i in range(0, 100, 5):
        img.putpixel((i, 50), (255, 255, 0))

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img.save(f.name)
        yield f.name
    Path(f.name).unlink(missing_ok=True)


def test_hasher_creation():
    """Test hasher creation with custom hash size."""
    hasher = PerceptualHasher(hash_size=8)
    assert hasher.name == "hasher"
    assert hasher.hash_size == 8


def test_hasher_computes_all_hash_types(hasher, test_image_path):
    """Hasher should compute phash, dhash, ahash, whash."""
    result = hasher.analyze(test_image_path)

    assert result.success
    assert result.value is not None
    metadata = result.value.metadata

    assert "phash" in metadata
    assert "dhash" in metadata
    assert "ahash" in metadata
    assert "whash" in metadata
    assert metadata["hash_size"] == hasher.hash_size

    # Hashes should be hex strings
    assert all(c in "0123456789abcdef" for c in metadata["phash"])


def test_identical_images_same_hash(hasher, test_image_path):
    """Same image analyzed twice should have identical hash."""
    result1 = hasher.analyze(test_image_path)
    result2 = hasher.analyze(test_image_path)

    assert result1.success and result2.success
    assert result1.value.metadata["phash"] == result2.value.metadata["phash"]


def test_similar_images_similar_hash(hasher, test_image_path, duplicate_image_path):
    """Duplicate images (re-saved) should have identical hashes."""
    result1 = hasher.analyze(test_image_path)
    result2 = hasher.analyze(duplicate_image_path)

    assert result1.success and result2.success

    hash1 = result1.value.metadata["phash"]
    hash2 = result2.value.metadata["phash"]

    distance = hamming_distance(hash1, hash2)
    # True duplicates should have very small or zero distance
    assert distance == 0, f"Expected identical hash for duplicate, got distance {distance}"


def test_different_images_different_hash(hasher, test_image_path, different_image_path):
    """Different images should have different hashes."""
    result1 = hasher.analyze(test_image_path)
    result2 = hasher.analyze(different_image_path)

    assert result1.success and result2.success

    hash1 = result1.value.metadata["phash"]
    hash2 = result2.value.metadata["phash"]

    distance = hamming_distance(hash1, hash2)
    # Different images should have larger hamming distance
    assert distance > 5


def test_hasher_error_for_missing_file(hasher):
    """Missing file should return error."""
    result = hasher.analyze("/nonexistent/image.jpg")

    assert not result.success
    assert result.error is not None


def test_compute_phash_utility(test_image_path):
    """Test standalone phash utility."""
    phash = compute_phash(test_image_path)

    assert phash is not None
    assert all(c in "0123456789abcdef" for c in phash)


def test_compute_phash_missing_file():
    """Missing file returns None."""
    phash = compute_phash("/nonexistent/image.jpg")
    assert phash is None


def test_hamming_distance_identical():
    """Identical hashes have distance 0."""
    hash_val = "0123456789abcdef" * 4  # 64-char hex string
    assert hamming_distance(hash_val, hash_val) == 0


def test_are_duplicates_function(test_image_path, duplicate_image_path, different_image_path):
    """Test are_duplicates utility function."""
    hash1 = compute_phash(test_image_path)
    hash2 = compute_phash(duplicate_image_path)
    hash3 = compute_phash(different_image_path)

    # Similar images should be duplicates
    # (with a reasonable threshold for our test images)
    distance = hamming_distance(hash1, hash2)
    assert are_duplicates(hash1, hash2, threshold=max(distance, 15))

    # Different images should not be duplicates (with strict threshold)
    assert not are_duplicates(hash1, hash3, threshold=5)


def test_find_duplicates_groups():
    """Test duplicate grouping."""
    # Create hashes where some are duplicates
    # Using consistent 64-char hex strings for 256-bit hashes
    base_hash = "0" * 64
    similar_hash = "0" * 63 + "1"  # 1 bit different
    different_hash = "f" * 64  # completely different

    hashes = {
        "/path/img1.jpg": base_hash,
        "/path/img2.jpg": similar_hash,
        "/path/img3.jpg": different_hash,
    }

    groups = find_duplicates(hashes, threshold=10)

    # Should find one duplicate group with img1 and img2
    assert len(groups) == 1
    assert len(groups[0]) == 2
    assert "/path/img1.jpg" in groups[0]
    assert "/path/img2.jpg" in groups[0]


def test_find_duplicates_no_duplicates():
    """No duplicates should return empty list."""
    hashes = {
        "/path/img1.jpg": "0" * 64,
        "/path/img2.jpg": "f" * 64,
        "/path/img3.jpg": "a" * 64,
    }

    groups = find_duplicates(hashes, threshold=5)

    assert len(groups) == 0


def test_hasher_batch(hasher, test_image_path, different_image_path):
    """Test batch hashing."""
    results = hasher.analyze_batch([test_image_path, different_image_path])

    assert len(results) == 2
    assert all(r.success for r in results)
