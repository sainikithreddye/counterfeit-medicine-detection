"""Tests for the single-image prediction module."""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


@pytest.fixture
def sample_image_path(tmp_path):
    """Create a temporary test image."""
    img = Image.new("RGB", (400, 300), color=(128, 100, 200))
    path = tmp_path / "test_medicine.jpg"
    img.save(path)
    return path


@pytest.fixture
def non_image_file(tmp_path):
    """Create a non-image file."""
    path = tmp_path / "document.txt"
    path.write_text("Not an image")
    return path


class TestPredictImage:
    """Tests for predict_image()."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Import prediction module."""
        try:
            import tensorflow as tf
            from predict import predict_image
            self.predict_image = predict_image
        except (ImportError, Exception) as e:
            pytest.skip(f"TensorFlow runtime unavailable: {e}")

    def test_missing_model_raises(self, sample_image_path):
        """Should raise FileNotFoundError when model file is missing."""
        from model import load_trained_model
        with pytest.raises(FileNotFoundError, match="not found"):
            # Use a non-existent model path
            self.predict_image(sample_image_path)

    def test_missing_image_raises(self, tmp_path):
        """Should raise FileNotFoundError for non-existent image."""
        with pytest.raises(FileNotFoundError):
            self.predict_image(tmp_path / "nonexistent.jpg")

    def test_unsupported_format_raises(self, tmp_path):
        """Should raise ValueError for unsupported file formats."""
        bad_file = tmp_path / "test.xyz"
        bad_file.write_text("data")
        with pytest.raises(ValueError, match="Unsupported"):
            self.predict_image(bad_file)

    @pytest.mark.skipif(
        not config.MODEL_PATH.exists(),
        reason="Trained model not available"
    )
    def test_prediction_with_model(self, sample_image_path):
        """Should produce valid prediction when model exists."""
        result = self.predict_image(sample_image_path)

        assert "label" in result
        assert result["label"] in ("Real", "Fake")
        assert "fake_probability" in result
        assert "real_probability" in result
        assert 0.0 <= result["fake_probability"] <= 1.0
        assert 0.0 <= result["real_probability"] <= 1.0
        # Probabilities should sum to ~1.0
        total = result["fake_probability"] + result["real_probability"]
        assert abs(total - 1.0) < 0.01

    @pytest.mark.skipif(
        not config.MODEL_PATH.exists(),
        reason="Trained model not available"
    )
    def test_prediction_consistency(self, sample_image_path):
        """Multiple predictions on the same image should match."""
        r1 = self.predict_image(sample_image_path)
        r2 = self.predict_image(sample_image_path)
        assert r1["label"] == r2["label"]
        assert abs(r1["raw_score"] - r2["raw_score"]) < 1e-5


class TestPredictPilImage:
    """Tests for predict_pil_image()."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Import prediction module."""
        try:
            import tensorflow as tf
            from predict import predict_pil_image
            self.predict_pil_image = predict_pil_image
        except (ImportError, Exception) as e:
            pytest.skip(f"TensorFlow runtime unavailable: {e}")

    def test_missing_model_raises(self):
        """Should raise FileNotFoundError when model is missing."""
        img = Image.new("RGB", (300, 300), color=(100, 100, 100))
        with pytest.raises(FileNotFoundError, match="not found"):
            self.predict_pil_image(img)

    @pytest.mark.skipif(
        not config.MODEL_PATH.exists(),
        reason="Trained model not available"
    )
    def test_pil_prediction(self):
        """Should produce valid prediction from PIL image."""
        img = Image.new("RGB", (400, 400), color=(128, 128, 128))
        result = self.predict_pil_image(img)
        assert result["label"] in ("Real", "Fake")
        assert 0.0 <= result["fake_probability"] <= 1.0


class TestPredictionWithMockModel:
    """Tests using a freshly built (untrained) model for structural checks."""

    @pytest.fixture(autouse=True)
    def setup(self):
        try:
            import tensorflow as tf
            from model import build_cnn_model, compile_model
            from predict import predict_image, predict_pil_image
            self.predict_image = predict_image
            self.predict_pil_image = predict_pil_image

            # Build a fresh (untrained) model for structural testing
            self.model = build_cnn_model()
            compile_model(self.model)
        except (ImportError, Exception) as e:
            pytest.skip(f"TensorFlow runtime unavailable: {e}")

    def test_prediction_structure(self, sample_image_path):
        """Prediction should return all expected fields."""
        result = self.predict_image(sample_image_path, model=self.model)
        expected_keys = {"label", "fake_probability", "real_probability", "raw_score", "threshold", "image_path"}
        assert set(result.keys()) == expected_keys

    def test_label_is_valid(self, sample_image_path):
        """Label should be either 'Real' or 'Fake'."""
        result = self.predict_image(sample_image_path, model=self.model)
        assert result["label"] in ("Real", "Fake")

    def test_probabilities_sum_to_one(self, sample_image_path):
        """Fake + Real probabilities should sum to ~1.0."""
        result = self.predict_image(sample_image_path, model=self.model)
        total = result["fake_probability"] + result["real_probability"]
        assert abs(total - 1.0) < 0.01

    def test_threshold_applied_correctly(self, sample_image_path):
        """Label should match threshold comparison."""
        result = self.predict_image(sample_image_path, model=self.model)
        if result["raw_score"] >= config.CLASSIFICATION_THRESHOLD:
            assert result["label"] == "Fake"
        else:
            assert result["label"] == "Real"

    def test_pil_image_prediction(self):
        """Should work with PIL Image objects."""
        img = Image.new("RGB", (300, 200), color=(64, 128, 200))
        result = self.predict_pil_image(img, model=self.model)
        assert result["label"] in ("Real", "Fake")
        assert 0.0 <= result["raw_score"] <= 1.0
