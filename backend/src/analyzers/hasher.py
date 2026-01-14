"""Perceptual hashing for duplicate image detection."""
from dataclasses import dataclass

import imagehash
import structlog

from ..utils.image import load_image
from .base import AnalysisResult, BaseAnalyzer, Result

logger = structlog.get_logger()

# Hamming distance threshold for duplicate detection
# Lower = stricter matching (more likely to be exact duplicates)
DEFAULT_DUPLICATE_THRESHOLD = 10


@dataclass
class ImageHash:
    """Container for multiple hash types of an image."""
    phash: str          # Perceptual hash - best for duplicate detection
    dhash: str          # Difference hash - faster, catches resizes
    ahash: str          # Average hash - simplest, good baseline
    whash: str          # Wavelet hash - good for rotations


class PerceptualHasher(BaseAnalyzer):
    """Compute perceptual hashes for duplicate detection.

    Uses multiple hash algorithms for robust duplicate detection:
    - pHash: Best overall for near-duplicates
    - dHash: Fast, catches resized images
    - aHash: Simple baseline
    - wHash: Handles rotations better

    Hamming distance between hashes indicates similarity:
    - 0: Identical
    - 1-10: Near duplicates (crops, resizes, small edits)
    - 10+: Different images
    """

    name = "hasher"
    version = "1.0.0"

    def __init__(self, hash_size: int = 16):
        """Initialize hasher.

        Args:
            hash_size: Size of hash (larger = more precision but slower).
                       Default 16 produces 256-bit hashes.
        """
        self.hash_size = hash_size

    def analyze(self, image_path: str) -> Result[AnalysisResult]:
        """Compute perceptual hashes for an image.

        Args:
            image_path: Path to the image file.

        Returns:
            Result with hash data including:
            - phash, dhash, ahash, whash: hex string hashes
            - hash_size: size used for computation
        """
        try:
            img = load_image(image_path)

            # Compute all hash types
            phash = str(imagehash.phash(img, hash_size=self.hash_size))
            dhash = str(imagehash.dhash(img, hash_size=self.hash_size))
            ahash = str(imagehash.average_hash(img, hash_size=self.hash_size))
            whash = str(imagehash.whash(img, hash_size=self.hash_size))

            logger.debug(
                "hash_computation_complete",
                image_path=image_path,
                phash=phash[:16] + "...",  # Log truncated for readability
            )

            return Result.ok(AnalysisResult(
                analyzer_name=self.name,
                confidence=1.0,  # Hash computation is deterministic
                metadata={
                    "phash": phash,
                    "dhash": dhash,
                    "ahash": ahash,
                    "whash": whash,
                    "hash_size": self.hash_size,
                },
            ))

        except Exception as e:
            logger.error("hash_computation_failed", image_path=image_path, error=str(e))
            return Result.fail(f"Hash computation failed: {str(e)}")


def compute_phash(image_path: str, hash_size: int = 16) -> str | None:
    """Utility to get just the perceptual hash.

    Args:
        image_path: Path to image.
        hash_size: Size of hash.

    Returns:
        Hex string of pHash, or None if failed.
    """
    try:
        img = load_image(image_path)
        return str(imagehash.phash(img, hash_size=hash_size))
    except Exception:
        return None


def hamming_distance(hash1: str, hash2: str) -> int:
    """Compute Hamming distance between two hex hash strings.

    Args:
        hash1: First hash as hex string.
        hash2: Second hash as hex string.

    Returns:
        Hamming distance (number of differing bits).
    """
    # Convert hex to imagehash objects for comparison
    h1 = imagehash.hex_to_hash(hash1)
    h2 = imagehash.hex_to_hash(hash2)
    return h1 - h2  # imagehash overloads subtraction for Hamming distance


def are_duplicates(
    hash1: str,
    hash2: str,
    threshold: int = DEFAULT_DUPLICATE_THRESHOLD
) -> bool:
    """Check if two images are duplicates based on hash distance.

    Args:
        hash1: First perceptual hash.
        hash2: Second perceptual hash.
        threshold: Maximum Hamming distance to consider duplicates.

    Returns:
        True if images are likely duplicates.
    """
    return hamming_distance(hash1, hash2) <= threshold


def find_duplicates(
    hashes: dict[str, str],
    threshold: int = DEFAULT_DUPLICATE_THRESHOLD
) -> list[list[str]]:
    """Find groups of duplicate images from a dict of path->hash.

    Args:
        hashes: Dictionary mapping image paths to their pHash values.
        threshold: Maximum Hamming distance for duplicates.

    Returns:
        List of duplicate groups (each group is list of paths).
    """
    paths = list(hashes.keys())
    n = len(paths)

    # Union-find structure for grouping
    parent = {p: p for p in paths}

    def find(x: str) -> str:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: str, y: str):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Compare all pairs (O(n^2) - use spatial hashing for large datasets)
    for i in range(n):
        for j in range(i + 1, n):
            h1, h2 = hashes[paths[i]], hashes[paths[j]]
            if are_duplicates(h1, h2, threshold):
                union(paths[i], paths[j])

    # Group by parent
    groups: dict[str, list[str]] = {}
    for p in paths:
        root = find(p)
        if root not in groups:
            groups[root] = []
        groups[root].append(p)

    # Return only groups with more than one member
    return [g for g in groups.values() if len(g) > 1]
