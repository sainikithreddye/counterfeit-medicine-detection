"""
Training pipeline for the Counterfeit Medicine Detection CNN.

Handles:
- Dataset loading and validation
- Model building and compilation
- Training with callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
- Saving model, training history, metadata, and plots
"""

import json
import logging
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import config
from model import build_cnn_model, compile_model, save_model_summary, get_model_summary_string
from preprocessing import (
    discover_dataset,
    create_data_generators,
    save_dataset_info,
    validate_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seeds(seed: int = config.RANDOM_SEED) -> None:
    """Set random seeds for reproducibility."""
    import tensorflow as tf

    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Random seeds set to {seed}")


def get_callbacks() -> list:
    """Create training callbacks."""
    import tensorflow as tf

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    callbacks = [
        # Save best model based on validation loss
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(config.MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor=config.EARLY_STOPPING_MONITOR,
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=config.EARLY_STOPPING_RESTORE_BEST,
            verbose=1,
        ),
        # Reduce learning rate on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=config.REDUCE_LR_MONITOR,
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=config.REDUCE_LR_MIN_LR,
            verbose=1,
        ),
    ]

    return callbacks


def save_training_history(history) -> None:
    """Save training history to JSON."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    history_dict = {}
    for key, values in history.history.items():
        history_dict[key] = [float(v) for v in values]

    with open(config.TRAINING_HISTORY_PATH, "w") as f:
        json.dump(history_dict, f, indent=2)

    logger.info(f"Training history saved to {config.TRAINING_HISTORY_PATH}")


def save_training_plots(history) -> None:
    """Generate and save training/validation accuracy and loss plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    hist = history.history

    # Accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(hist["accuracy"], label="Training Accuracy", linewidth=2)
    plt.plot(hist["val_accuracy"], label="Validation Accuracy", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.TRAINING_ACCURACY_PLOT_PATH, dpi=150)
    plt.close()
    logger.info(f"Accuracy plot saved to {config.TRAINING_ACCURACY_PLOT_PATH}")

    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(hist["loss"], label="Training Loss", linewidth=2)
    plt.plot(hist["val_loss"], label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.TRAINING_LOSS_PLOT_PATH, dpi=150)
    plt.close()
    logger.info(f"Loss plot saved to {config.TRAINING_LOSS_PLOT_PATH}")


def save_training_metadata(
    history,
    model,
    train_gen,
    val_gen,
    elapsed_seconds: float,
) -> None:
    """Save machine-readable training metadata."""
    import tensorflow as tf

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    best_epoch = int(np.argmin(history.history["val_loss"])) + 1

    metadata = {
        "project": "Counterfeit Medicine Detection CNN",
        "timestamp": datetime.now().isoformat(),
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "class_mapping": {"Real": 0, "Fake": 1},
        "class_indices": train_gen.class_indices,
        "image_dimensions": list(config.INPUT_SHAPE),
        "classification_threshold": config.CLASSIFICATION_THRESHOLD,
        "batch_size": config.BATCH_SIZE,
        "max_epochs": config.MAX_EPOCHS,
        "actual_epochs": len(history.history["loss"]),
        "best_epoch": best_epoch,
        "learning_rate": config.LEARNING_RATE,
        "random_seed": config.RANDOM_SEED,
        "training_samples": train_gen.samples,
        "validation_samples": val_gen.samples,
        "best_val_loss": float(min(history.history["val_loss"])),
        "best_val_accuracy": float(max(history.history["val_accuracy"])),
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_train_loss": float(history.history["loss"][-1]),
        "training_time_seconds": round(elapsed_seconds, 2),
        "model_file": str(config.MODEL_PATH),
    }

    with open(config.TRAINING_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Training metadata saved to {config.TRAINING_METADATA_PATH}")


def train() -> None:
    """
    Main training function.

    Validates dataset, builds model, trains, and saves all outputs.
    """
    logger.info("=" * 60)
    logger.info("COUNTERFEIT MEDICINE DETECTION — MODEL TRAINING")
    logger.info("=" * 60)

    # Check TensorFlow
    import tensorflow as tf
    logger.info(f"TensorFlow version: {tf.__version__}")
    logger.info(f"Python version: {platform.python_version()}")

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        logger.info(f"GPU(s) available: {[g.name for g in gpus]}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        logger.info("No GPU detected. Training will use CPU.")

    # Set seeds
    set_seeds(config.RANDOM_SEED)

    # Validate and save dataset info
    logger.info("\n--- Dataset Validation ---")
    ds_info = save_dataset_info()

    if not ds_info["valid"]:
        logger.error("Dataset validation failed:")
        for err in ds_info["errors"]:
            logger.error(f"  • {err}")
        logger.error(
            "\nPlease ensure the dataset is properly organized in:\n"
            f"  {config.DATASET_DIR}\n"
            "Expected structure (case-insensitive splits):\n"
            "  train/Fake/, train/Real/\n"
            "  val/Fake/, val/Real/\n"
            "  test/Fake/, test/Real/"
        )
        sys.exit(1)

    # Report dataset stats
    for split_name, split_info in ds_info["splits"].items():
        logger.info(
            f"  {split_name}: {split_info['total_images']} images "
            f"(Real={split_info['class_counts'].get('Real', 0)}, "
            f"Fake={split_info['class_counts'].get('Fake', 0)})"
        )

    if ds_info["warnings"]:
        for warn in ds_info["warnings"]:
            logger.warning(f"  ⚠ {warn}")

    # Create data generators
    logger.info("\n--- Creating Data Generators ---")
    train_gen, val_gen, test_gen = create_data_generators(
        dataset_dir=config.DATASET_DIR,
        batch_size=config.BATCH_SIZE,
        image_size=config.IMAGE_SIZE,
        augment_training=True,
    )

    logger.info(f"Training samples: {train_gen.samples}")
    logger.info(f"Validation samples: {val_gen.samples}")
    logger.info(f"Test samples: {test_gen.samples}")
    logger.info(f"Class indices: {train_gen.class_indices}")

    # Build model
    logger.info("\n--- Building CNN Model ---")
    model = build_cnn_model(config.INPUT_SHAPE)
    compile_model(model, config.LEARNING_RATE)

    # Print and save summary
    summary_str = save_model_summary(model, config.MODEL_SUMMARY_PATH)
    print(summary_str)

    # Callbacks
    callbacks = get_callbacks()

    # Train
    logger.info("\n--- Training Started ---")
    logger.info(f"Batch size: {config.BATCH_SIZE}")
    logger.info(f"Max epochs: {config.MAX_EPOCHS}")
    logger.info(f"Learning rate: {config.LEARNING_RATE}")

    start_time = time.time()

    history = model.fit(
        train_gen,
        epochs=config.MAX_EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
    )

    elapsed = time.time() - start_time
    logger.info(f"\n--- Training Completed in {elapsed:.1f}s ---")
    logger.info(f"Total epochs trained: {len(history.history['loss'])}")
    logger.info(f"Best validation loss: {min(history.history['val_loss']):.4f}")
    logger.info(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")

    # Save outputs
    logger.info("\n--- Saving Results ---")
    save_training_history(history)
    save_training_plots(history)
    save_training_metadata(history, model, train_gen, val_gen, elapsed)

    # Verify model was saved
    if config.MODEL_PATH.exists():
        model_size = config.MODEL_PATH.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Trained model saved: {config.MODEL_PATH} ({model_size:.2f} MB)")
    else:
        logger.error("✗ Model file was not saved!")

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Model: {config.MODEL_PATH}")
    logger.info(f"Results: {config.RESULTS_DIR}")
    logger.info("Next step: python evaluate.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    train()
