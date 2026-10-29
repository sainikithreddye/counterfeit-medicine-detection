"""Tests for the CNN model architecture."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestBuildModel:
    """Tests for model building and architecture verification."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Import TensorFlow and build model."""
        try:
            import tensorflow as tf
            from model import build_cnn_model, compile_model
            self.build_cnn_model = build_cnn_model
            self.compile_model = compile_model
            self.tf = tf
        except (ImportError, Exception) as e:
            pytest.skip(f"TensorFlow runtime unavailable in current environment: {e}")

    def test_model_builds_successfully(self):
        """Model should build without errors."""
        model = self.build_cnn_model((224, 224, 3))
        assert model is not None

    def test_model_input_shape(self):
        """Model should accept (None, 224, 224, 3) input."""
        model = self.build_cnn_model((224, 224, 3))
        # input_shape includes batch dim as None
        expected = (None, 224, 224, 3)
        actual = model.input_shape
        assert actual == expected, f"Expected input shape {expected}, got {actual}"

    def test_model_output_shape(self):
        """Model output should be (None, 1) for binary classification."""
        model = self.build_cnn_model((224, 224, 3))
        assert model.output_shape == (None, 1)

    def test_forward_pass(self):
        """Model should accept a batch and produce output in [0, 1]."""
        model = self.build_cnn_model((224, 224, 3))
        self.compile_model(model)
        dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
        output = model.predict(dummy_input, verbose=0)
        assert output.shape == (1, 1), f"Expected (1,1), got {output.shape}"
        assert 0.0 <= float(output[0, 0]) <= 1.0, "Output outside [0, 1]"

    def test_batch_forward_pass(self):
        """Model should handle a batch of multiple images."""
        model = self.build_cnn_model((224, 224, 3))
        self.compile_model(model)
        dummy_input = np.random.rand(4, 224, 224, 3).astype(np.float32)
        output = model.predict(dummy_input, verbose=0)
        assert output.shape == (4, 1)
        assert np.all(output >= 0.0) and np.all(output <= 1.0)

    def test_three_conv_layers(self):
        """Model should have exactly 3 Conv2D layers."""
        model = self.build_cnn_model((224, 224, 3))
        conv_layers = [l for l in model.layers if "conv2d" in l.__class__.__name__.lower()]
        assert len(conv_layers) == 3, f"Expected 3 Conv2D layers, found {len(conv_layers)}"

    def test_conv_filter_counts(self):
        """Conv2D layers should have 32, 64, 128 filters respectively."""
        model = self.build_cnn_model((224, 224, 3))
        conv_layers = [l for l in model.layers if "conv2d" in l.__class__.__name__.lower()]
        expected_filters = [32, 64, 128]
        actual_filters = [l.filters for l in conv_layers]
        assert actual_filters == expected_filters, (
            f"Expected filters {expected_filters}, got {actual_filters}"
        )

    def test_three_maxpool_layers(self):
        """Model should have exactly 3 MaxPooling2D layers."""
        model = self.build_cnn_model((224, 224, 3))
        pool_layers = [l for l in model.layers if "maxpooling" in l.__class__.__name__.lower()]
        assert len(pool_layers) == 3, f"Expected 3 MaxPooling2D, found {len(pool_layers)}"

    def test_global_average_pooling(self):
        """Model should have a GlobalAveragePooling2D layer."""
        model = self.build_cnn_model((224, 224, 3))
        gap_layers = [l for l in model.layers if "globalaveragepooling" in l.__class__.__name__.lower()]
        assert len(gap_layers) == 1, "Expected 1 GlobalAveragePooling2D layer"

    def test_dense_128_layer(self):
        """Model should have a Dense(128) layer with ReLU."""
        model = self.build_cnn_model((224, 224, 3))
        dense_layers = [l for l in model.layers if l.__class__.__name__ == "Dense"]
        # Should have 2 Dense layers: Dense(128) and Dense(1)
        assert len(dense_layers) == 2, f"Expected 2 Dense layers, found {len(dense_layers)}"
        # First dense should be 128 units
        assert dense_layers[0].units == 128

    def test_dropout_layer(self):
        """Model should have a Dropout(0.5) layer."""
        model = self.build_cnn_model((224, 224, 3))
        dropout_layers = [l for l in model.layers if "dropout" in l.__class__.__name__.lower()]
        assert len(dropout_layers) == 1, "Expected 1 Dropout layer"
        assert dropout_layers[0].rate == 0.5, f"Expected dropout rate 0.5, got {dropout_layers[0].rate}"

    def test_sigmoid_output(self):
        """Output layer should use sigmoid activation."""
        model = self.build_cnn_model((224, 224, 3))
        output_layer = model.layers[-1]
        activation_name = output_layer.activation.__name__
        assert activation_name == "sigmoid", f"Expected sigmoid, got {activation_name}"

    def test_model_compiles(self):
        """Model should compile with Adam + binary_crossentropy."""
        model = self.build_cnn_model((224, 224, 3))
        self.compile_model(model, learning_rate=0.001)
        assert model.optimizer is not None
        assert model.loss is not None
