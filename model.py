"""
CNN model architecture for counterfeit medicine detection.

Implements the exact architecture specified in the assignment:
- 3 Conv2D + MaxPooling2D stages
- GlobalAveragePooling2D
- Dense(128) + Dropout(0.5)
- Sigmoid output for binary classification
"""

import io
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def build_cnn_model(input_shape: Tuple[int, int, int] = (224, 224, 3)) -> "tf.keras.Model":
    """
    Build the specified CNN architecture for binary classification.

    Architecture:
        1. Input: (224, 224, 3)
        2. Conv2D(32, 3x3, ReLU, same) → MaxPooling2D(2x2)
        3. Conv2D(64, 3x3, ReLU, same) → MaxPooling2D(2x2)
        4. Conv2D(128, 3x3, ReLU, same) → MaxPooling2D(2x2)
        5. GlobalAveragePooling2D
        6. Dense(128, ReLU)
        7. Dropout(0.5)
        8. Dense(1, sigmoid)

    Label encoding:
        REAL = 0, FAKE = 1
        Sigmoid output = P(FAKE)

    Args:
        input_shape: Input image dimensions (H, W, C).

    Returns:
        Compiled-ready Keras Sequential model (not yet compiled).
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential(name="CounterfeitMedicineCNN")

    # Input layer
    model.add(layers.Input(shape=input_shape))

    # Block 1: Conv2D(32) + MaxPool
    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv2d_block1"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_block1"))

    # Block 2: Conv2D(64) + MaxPool
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2d_block2"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_block2"))

    # Block 3: Conv2D(128) + MaxPool
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv2d_block3"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_block3"))

    # Global Average Pooling
    model.add(layers.GlobalAveragePooling2D(name="global_avg_pool"))

    # Dense classification head
    model.add(layers.Dense(128, activation="relu", name="dense_128"))
    model.add(layers.Dropout(0.5, name="dropout"))

    # Output: single sigmoid neuron (P(FAKE))
    model.add(layers.Dense(1, activation="sigmoid", name="output"))

    return model


def compile_model(model, learning_rate: float = 0.001) -> None:
    """
    Compile the model with binary cross-entropy loss and Adam optimizer.

    Args:
        model: Keras model to compile.
        learning_rate: Adam optimizer learning rate.
    """
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )


def get_model_summary_string(model) -> str:
    """Capture model.summary() output as a string."""
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + "\n"))
    return stream.getvalue()


def save_model_summary(model, output_path) -> str:
    """Save model summary to a text file and return the summary string."""
    from pathlib import Path

    summary_str = get_model_summary_string(model)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(summary_str)
    logger.info(f"Model summary saved to {output_path}")
    return summary_str


def load_trained_model(model_path=None):
    """
    Load a trained model from disk.

    Args:
        model_path: Path to the saved model. Defaults to config.MODEL_PATH.

    Returns:
        Loaded Keras model.

    Raises:
        FileNotFoundError: If the model file doesn't exist.
        ValueError: If the model file is corrupted or incompatible.
    """
    import tensorflow as tf
    from pathlib import Path
    import config

    if model_path is None:
        model_path = config.MODEL_PATH

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            f"Please train the model first by running: python train.py"
        )

    try:
        model = tf.keras.models.load_model(str(model_path))
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        raise ValueError(
            f"Failed to load model from '{model_path}'. "
            f"The file may be corrupted or incompatible: {e}"
        )
