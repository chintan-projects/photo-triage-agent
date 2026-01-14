"""Pytest configuration and fixtures"""
import pytest
from pathlib import Path

@pytest.fixture
def sample_images_dir():
    """Directory containing test images."""
    return Path(__file__).parent / "fixtures" / "images"

@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary directory for test outputs."""
    return tmp_path / "output"
