"""Tests for the image preprocessing module."""

import numpy as np
import pytest
from pathlib import Path
from PIL import Image
import tempfile
import os
import sys

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from preprocessing import (
    load_and_preprocess_image,
    preprocess_for_prediction,
    preprocess_pil_image,
)


@pytest.fixture
def rgb_image_path(tmp_path):
    """Create a temporary RGB test image."""
    img = Image.new("RGB", (640, 480), color=(128, 64, 200))
    path = tmp_path / "test_rgb.jpg"
    img.save(path)
    return path


@pytest.fixture
def grayscale_image_path(tmp_path):
    """Create a temporary grayscale test image."""
    img = Image.new("L", (320, 240), color=128)
    path = tmp_path / "test_gray.png"
    img.save(path)
    return path


@pytest.fixture
def rgba_image_path(tmp_path):
    """Create a temporary RGBA (transparent) test image."""
    img = Image.new("RGBA", (400, 300), color=(100, 150, 200, 128))
    path = tmp_path / "test_rgba.png"
    img.save(path)
    return path


@pytest.fixture
def corrupted_file_path(tmp_path):
    """Create a file that is not a valid image."""
    path = tmp_path / "corrupted.jpg"
    path.write_text("This is not an image file!")
    return path


class TestLoadAndPreprocessImage:
    """Tests for load_and_preprocess_image()."""

    def test_valid_rgb_image_resized(self, rgb_image_path):
        """A valid RGB image should be resized to 224x224x3."""
        result = load_and_preprocess_image(rgb_image_path)
        assert result.shape == (224, 224, 3)

    def test_pixel_values_normalized(self, rgb_image_path):
        """Pixel values should be in [0, 1] range."""
        result = load_and_preprocess_image(rgb_image_path)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_output_dtype_float(self, rgb_image_path):
        """Output should be float32."""
        result = load_and_preprocess_image(rgb_image_path)
        assert result.dtype == np.float32

    def test_grayscale_converted_to_rgb(self, grayscale_image_path):
        """Grayscale images should be converted to 3-channel RGB."""
        result = load_and_preprocess_image(grayscale_image_path)
        assert result.shape == (224, 224, 3)

    def test_rgba_converted_to_rgb(self, rgba_image_path):
        """RGBA images should be converted to 3-channel RGB (no alpha)."""
        result = load_and_preprocess_image(rgba_image_path)
        assert result.shape == (224, 224, 3)

    def test_file_not_found_raises(self, tmp_path):
        """Non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_and_preprocess_image(tmp_path / "nonexistent.jpg")

    def test_corrupted_file_raises(self, corrupted_file_path):
        """Corrupted/invalid image file should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot open or decode"):
            load_and_preprocess_image(corrupted_file_path)

    def test_custom_target_size(self, rgb_image_path):
        """Should resize to custom target size."""
        result = load_and_preprocess_image(rgb_image_path, target_size=(128, 128))
        assert result.shape == (128, 128, 3)


class TestPreprocessForPrediction:
    """Tests for preprocess_for_prediction()."""

    def test_batch_dimension_added(self, rgb_image_path):
        """Output should have batch dimension: (1, 224, 224, 3)."""
        result = preprocess_for_prediction(rgb_image_path)
        assert result.shape == (1, 224, 224, 3)

    def test_values_normalized(self, rgb_image_path):
        """Values should be in [0, 1]."""
        result = preprocess_for_prediction(rgb_image_path)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


class TestPreprocessPilImage:
    """Tests for preprocess_pil_image()."""

    def test_pil_image_preprocessed(self):
        """PIL Image should be preprocessed correctly."""
        img = Image.new("RGB", (500, 400), color=(100, 200, 50))
        result = preprocess_pil_image(img)
        assert result.shape == (1, 224, 224, 3)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_grayscale_pil_image(self):
        """Grayscale PIL Image should be converted to RGB."""
        img = Image.new("L", (300, 300), color=128)
        result = preprocess_pil_image(img)
        assert result.shape == (1, 224, 224, 3)

    def test_rgba_pil_image(self):
        """RGBA PIL Image should be converted to RGB."""
        img = Image.new("RGBA", (300, 300), color=(100, 200, 50, 128))
        result = preprocess_pil_image(img)
        assert result.shape == (1, 224, 224, 3)
