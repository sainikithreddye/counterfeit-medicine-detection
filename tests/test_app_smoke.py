"""Smoke tests for application imports, configuration, and startup behavior."""

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
import preprocessing


class TestSmokeAndImports:
    """Basic smoke and integrity tests."""

    def test_config_paths_and_constants(self):
        """Configuration paths and model constants must match specification."""
        assert config.IMAGE_HEIGHT == 224
        assert config.IMAGE_WIDTH == 224
        assert config.IMAGE_CHANNELS == 3
        assert config.INPUT_SHAPE == (224, 224, 3)
        assert config.CLASS_LABELS == {"Real": 0, "Fake": 1}
        assert config.CLASSIFICATION_THRESHOLD == 0.5
        assert config.PROJECT_ROOT.exists()
        assert config.DATA_DIR.exists()
        assert config.MODELS_DIR.exists()
        assert config.RESULTS_DIR.exists()

    def test_preprocessing_module_exports(self):
        """Preprocessing module must export core functions."""
        assert callable(preprocessing.load_and_preprocess_image)
        assert callable(preprocessing.preprocess_for_prediction)
        assert callable(preprocessing.preprocess_pil_image)
        assert callable(preprocessing.discover_dataset)
        assert callable(preprocessing.validate_dataset)

    def test_model_availability_check(self):
        """Model check should correctly reflect whether best_model.keras exists."""
        from app import check_model_available
        # When no model is trained yet
        if not config.MODEL_PATH.exists():
            assert check_model_available() is False
        else:
            assert check_model_available() is True

    def test_evaluation_results_loader_without_file(self, tmp_path, monkeypatch):
        """Evaluation loader should gracefully return None when file is missing."""
        from app import load_evaluation_results
        monkeypatch.setattr(config, "EVALUATION_REPORT_PATH", tmp_path / "nonexistent.json")
        res = load_evaluation_results()
        assert res is None

    def test_dataset_discovery_smoke(self):
        """Dataset discovery should run without crashing on existing directory."""
        ds_info = preprocessing.discover_dataset(config.DATASET_DIR)
        assert isinstance(ds_info, dict)
        assert "valid" in ds_info
        assert "splits" in ds_info
        assert "errors" in ds_info
