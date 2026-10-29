"""Tests for dataset discovery and validation."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
import config
from preprocessing import discover_dataset, get_split_dir, get_class_dir, validate_dataset


@pytest.fixture
def valid_dataset(tmp_path):
    """Create a valid dataset directory structure with test images."""
    ds_dir = tmp_path / "medicine_dataset"
    for split in ["Train", "Val", "Test"]:
        for cls in ["Real", "Fake"]:
            cls_dir = ds_dir / split / cls
            cls_dir.mkdir(parents=True)
            # Create 3 small test images per class
            for i in range(3):
                img = Image.new("RGB", (50, 50), color=(i * 50, i * 30, i * 20))
                img.save(cls_dir / f"img_{i}.jpg")
    return ds_dir


@pytest.fixture
def missing_class_dataset(tmp_path):
    """Create a dataset with missing class folders."""
    ds_dir = tmp_path / "medicine_dataset"
    for split in ["Train", "Val", "Test"]:
        # Only create Real, not Fake
        real_dir = ds_dir / split / "Real"
        real_dir.mkdir(parents=True)
        img = Image.new("RGB", (50, 50), color=(100, 100, 100))
        img.save(real_dir / "img.jpg")
    return ds_dir


@pytest.fixture
def missing_split_dataset(tmp_path):
    """Create a dataset with missing split folders."""
    ds_dir = tmp_path / "medicine_dataset"
    # Only create Train, not Val or Test
    for cls in ["Real", "Fake"]:
        cls_dir = ds_dir / "Train" / cls
        cls_dir.mkdir(parents=True)
        img = Image.new("RGB", (50, 50), color=(100, 100, 100))
        img.save(cls_dir / "img.jpg")
    return ds_dir


@pytest.fixture
def empty_class_dataset(tmp_path):
    """Create a dataset with empty class folders."""
    ds_dir = tmp_path / "medicine_dataset"
    for split in ["Train", "Val", "Test"]:
        for cls in ["Real", "Fake"]:
            (ds_dir / split / cls).mkdir(parents=True)
    return ds_dir


class TestDiscoverDataset:
    """Tests for discover_dataset()."""

    def test_valid_dataset_recognized(self, valid_dataset):
        """Valid dataset should pass validation."""
        result = discover_dataset(valid_dataset)
        assert result["valid"] is True
        assert result["total_images"] == 18  # 3 images * 2 classes * 3 splits
        assert len(result["errors"]) == 0

    def test_missing_directory_detected(self, tmp_path):
        """Non-existent directory should produce helpful error."""
        result = discover_dataset(tmp_path / "nonexistent")
        assert result["valid"] is False
        assert len(result["errors"]) > 0
        assert "not found" in result["errors"][0].lower()

    def test_missing_class_detected(self, missing_class_dataset):
        """Missing class folders should be detected."""
        result = discover_dataset(missing_class_dataset)
        assert result["valid"] is False
        assert any("Fake" in e for e in result["errors"])

    def test_missing_split_detected(self, missing_split_dataset):
        """Missing split folders should be detected."""
        result = discover_dataset(missing_split_dataset)
        assert result["valid"] is False
        assert any("Val" in e for e in result["errors"])

    def test_empty_folders_detected(self, empty_class_dataset):
        """Empty class folders should report errors."""
        result = discover_dataset(empty_class_dataset)
        assert result["valid"] is False
        assert any("No valid images" in e for e in result["errors"])

    def test_class_counts_accurate(self, valid_dataset):
        """Image counts per class and split should be correct."""
        result = discover_dataset(valid_dataset)
        for split_name, split_info in result["splits"].items():
            assert split_info["class_counts"]["Real"] == 3
            assert split_info["class_counts"]["Fake"] == 3
            assert split_info["total_images"] == 6

    def test_non_image_files_ignored(self, valid_dataset):
        """Non-image files should not be counted."""
        # Add a non-image file
        (valid_dataset / "Train" / "Real" / "readme.txt").write_text("not an image")
        result = discover_dataset(valid_dataset)
        assert result["valid"] is True
        assert result["splits"]["Train"]["class_counts"]["Real"] == 3  # Still only 3 images

    def test_invalid_image_reported(self, valid_dataset):
        """Corrupted image files should be reported."""
        corrupt = valid_dataset / "Train" / "Real" / "corrupt.jpg"
        corrupt.write_text("not a valid image")
        result = discover_dataset(valid_dataset)
        # Should still be valid (3 good images per class remain)
        assert result["valid"] is True
        train_info = result["splits"]["Train"]
        assert len(train_info["invalid_images"]) > 0

    def test_alternative_folder_names(self, tmp_path):
        """Should recognize alternative folder names like 'train', 'validation'."""
        ds_dir = tmp_path / "medicine_dataset"
        for split in ["train", "validation", "test"]:
            for cls in ["Real", "Fake"]:
                cls_dir = ds_dir / split / cls
                cls_dir.mkdir(parents=True)
                img = Image.new("RGB", (50, 50), color=(100, 100, 100))
                img.save(cls_dir / "img.jpg")
        result = discover_dataset(ds_dir)
        assert result["valid"] is True


class TestGetSplitDir:
    """Tests for get_split_dir()."""

    def test_finds_standard_split(self, valid_dataset):
        """Should find standard split directories."""
        train_dir = get_split_dir("Train", valid_dataset)
        assert train_dir is not None
        assert train_dir.name == "Train"

    def test_returns_none_for_missing(self, tmp_path):
        """Should return None for missing splits."""
        result = get_split_dir("Train", tmp_path / "nonexistent")
        assert result is None


class TestGetClassDir:
    """Tests for get_class_dir()."""

    def test_finds_class_folders(self, valid_dataset):
        """Should find Real and Fake class folders."""
        train_dir = get_split_dir("Train", valid_dataset)
        real_dir = get_class_dir(train_dir, "Real")
        fake_dir = get_class_dir(train_dir, "Fake")
        assert real_dir is not None
        assert fake_dir is not None

    def test_returns_none_for_missing_class(self, valid_dataset):
        """Should return None for non-existent class."""
        train_dir = get_split_dir("Train", valid_dataset)
        result = get_class_dir(train_dir, "Unknown")
        assert result is None


class TestClassEncoding:
    """Tests for class encoding consistency."""

    def test_class_labels_consistent(self):
        """Class labels should be Real=0, Fake=1."""
        assert config.CLASS_LABELS["Real"] == 0
        assert config.CLASS_LABELS["Fake"] == 1

    def test_class_names_order(self):
        """CLASS_NAMES order should match label encoding."""
        assert config.CLASS_NAMES[0] == "Real"
        assert config.CLASS_NAMES[1] == "Fake"

    def test_threshold_default(self):
        """Default threshold should be 0.5."""
        assert config.CLASSIFICATION_THRESHOLD == 0.5


class TestValidateDataset:
    """Tests for validate_dataset() and lowercase split support."""

    def test_validate_dataset_reports_counts(self, valid_dataset):
        """validate_dataset should report split and class counts correctly."""
        info = validate_dataset(valid_dataset, verbose=False)
        assert info["valid"] is True
        assert info["total_images"] == 18
        for split in ["Train", "Val", "Test"]:
            assert info["splits"][split]["class_counts"]["Real"] == 3
            assert info["splits"][split]["class_counts"]["Fake"] == 3

    def test_validate_dataset_warns_on_empty_folder(self, empty_class_dataset):
        """Empty class folders should trigger warnings."""
        info = validate_dataset(empty_class_dataset, verbose=False)
        assert info["valid"] is False
        assert any("empty" in w.lower() for w in info["warnings"])

    def test_validate_dataset_warns_on_missing_folder(self, missing_split_dataset):
        """Missing split folders should trigger warnings."""
        info = validate_dataset(missing_split_dataset, verbose=False)
        assert info["valid"] is False
        assert any("missing" in w.lower() for w in info["warnings"])

    def test_lowercase_split_folders_supported(self, tmp_path):
        """Lowercase train, val, test folder names must be discovered."""
        ds_dir = tmp_path / "dataset"
        for split in ["train", "val", "test"]:
            for cls in ["Real", "Fake"]:
                cls_dir = ds_dir / split / cls
                cls_dir.mkdir(parents=True)
                img = Image.new("RGB", (50, 50), color=(10, 20, 30))
                img.save(cls_dir / "test.jpg")

        info = validate_dataset(ds_dir, verbose=False)
        assert info["valid"] is True
        assert info["total_images"] == 6
        assert get_split_dir("train", ds_dir) is not None
        assert get_split_dir("val", ds_dir) is not None
        assert get_split_dir("test", ds_dir) is not None

    def test_configured_dataset_path_validity(self):
        """Configured DATASET_DIR should point to downloaded dataset and be valid."""
        if config.DOWNLOADED_DATASET_DIR.exists():
            info = validate_dataset(config.DATASET_DIR, verbose=False)
            assert info["valid"] is True
            assert info["total_images"] > 0
            assert "Train" in info["splits"]
            assert "Val" in info["splits"]
            assert "Test" in info["splits"]
            assert info["splits"]["Train"]["class_counts"]["Real"] == 421
            assert info["splits"]["Train"]["class_counts"]["Fake"] == 240
