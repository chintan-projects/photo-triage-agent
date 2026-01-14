"""Tests for LFM provider."""
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.classifiers.base import ImageCategory, ClassificationResult
from src.classifiers.lfm_provider import (
    LFMProvider,
    parse_category,
    extract_json_from_response,
    image_to_data_uri,
)


class TestParseCategory:
    """Tests for category parsing."""

    def test_valid_categories(self):
        assert parse_category("people") == ImageCategory.PEOPLE
        assert parse_category("LANDSCAPE") == ImageCategory.LANDSCAPE
        assert parse_category("  food  ") == ImageCategory.FOOD
        assert parse_category("Document") == ImageCategory.DOCUMENT

    def test_unknown_category_returns_other(self):
        assert parse_category("unknown") == ImageCategory.OTHER
        assert parse_category("") == ImageCategory.OTHER
        assert parse_category("xyz") == ImageCategory.OTHER


class TestExtractJsonFromResponse:
    """Tests for JSON extraction from model responses."""

    def test_raw_json(self):
        response = '{"category": "people", "contains_faces": true}'
        result = extract_json_from_response(response)
        assert result["category"] == "people"
        assert result["contains_faces"] is True

    def test_json_in_markdown_block(self):
        response = '''Here is the analysis:
```json
{"category": "landscape", "contains_faces": false}
```
'''
        result = extract_json_from_response(response)
        assert result["category"] == "landscape"

    def test_json_in_plain_markdown_block(self):
        response = '''```
{"category": "food"}
```'''
        result = extract_json_from_response(response)
        assert result["category"] == "food"

    def test_json_with_surrounding_text(self):
        response = 'The image shows {"category": "animal", "description": "A dog"} as the result.'
        result = extract_json_from_response(response)
        assert result["category"] == "animal"

    def test_no_json_raises_error(self):
        with pytest.raises(ValueError, match="No JSON found"):
            extract_json_from_response("This has no JSON")


class TestImageToDataUri:
    """Tests for image to data URI conversion."""

    def test_jpeg_image(self, tmp_path):
        # Create a minimal JPEG file
        img_path = tmp_path / "test.jpg"
        img_path.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")

        uri = image_to_data_uri(str(img_path))
        assert uri.startswith("data:image/jpeg;base64,")

    def test_png_image(self, tmp_path):
        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n")

        uri = image_to_data_uri(str(img_path))
        assert uri.startswith("data:image/png;base64,")


class TestLFMProviderUnit:
    """Unit tests for LFMProvider (mocked model)."""

    @patch("src.classifiers.lfm_provider.Llama")
    @patch("src.classifiers.lfm_provider.Llava15ChatHandler")
    def test_health_check(self, mock_handler, mock_llama):
        """Test health check with mocked model."""
        mock_handler.return_value = MagicMock()
        mock_llama.return_value = MagicMock()

        # Create provider with mocked paths
        with patch.object(Path, "exists", return_value=True):
            provider = LFMProvider(
                model_path="/fake/model.gguf",
                mmproj_path="/fake/mmproj.gguf",
            )

        assert provider.health_check() is True

    @patch("src.classifiers.lfm_provider.Llama")
    @patch("src.classifiers.lfm_provider.Llava15ChatHandler")
    def test_classify_parses_response(self, mock_handler, mock_llama, tmp_path):
        """Test classification with mocked model response."""
        mock_handler.return_value = MagicMock()

        # Mock the model response
        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"category": "people", "contains_faces": true, "is_screenshot": false, "is_meme": false, "description": "A photo of people"}'
                    }
                }
            ]
        }
        mock_llama.return_value = mock_llm

        # Create test image
        img_path = tmp_path / "test.jpg"
        img_path.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")

        with patch.object(Path, "exists", return_value=True):
            provider = LFMProvider(
                model_path="/fake/model.gguf",
                mmproj_path="/fake/mmproj.gguf",
            )

        result = provider.classify(str(img_path))

        assert result.category == ImageCategory.PEOPLE
        assert result.contains_faces is True
        assert result.is_screenshot is False
        assert "people" in result.description.lower()


@pytest.mark.integration
class TestLFMProviderIntegration:
    """Integration tests that require actual model files.

    Run with: pytest -m integration
    Skip by default in CI.
    """

    @pytest.fixture
    def provider(self):
        """Create provider with real model if available."""
        from src.config import LFM_MODEL_PATH, LFM_MMPROJ_PATH

        if not LFM_MODEL_PATH.exists():
            pytest.skip("LFM model not downloaded")

        return LFMProvider(
            model_path=str(LFM_MODEL_PATH),
            mmproj_path=str(LFM_MMPROJ_PATH),
        )

    def test_classify_real_image(self, provider, tmp_path):
        """Test classification with real model.

        This test requires:
        1. Model files downloaded
        2. A test image
        """
        # This would use a real test image
        pytest.skip("Need real test image for integration test")
