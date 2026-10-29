"""
Central configuration for the Counterfeit Medicine Detection project.

All configurable paths, hyperparameters, and constants are defined here
to ensure consistency across training, evaluation, prediction, and testing.
"""

from pathlib import Path

# =============================================================================
# Project Directories
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
# Downloaded dataset path with train/val/test splits
DOWNLOADED_DATASET_DIR = DATA_DIR / "downloaded_dataset" / "dataset"
LEGACY_DATASET_DIR = DATA_DIR / "medicine_dataset"
# Point to actual downloaded dataset path if it exists, otherwise fall back to legacy path
DATASET_DIR = DOWNLOADED_DATASET_DIR if DOWNLOADED_DATASET_DIR.exists() else LEGACY_DATASET_DIR

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# =============================================================================
# Dataset Paths & Configuration
# =============================================================================
# Recognized folder name variations for each split (lowercase preferred first)
TRAIN_FOLDER_NAMES = ["train", "Train", "training", "Training"]
VAL_FOLDER_NAMES = ["val", "Val", "validation", "Validation", "valid", "Valid"]
TEST_FOLDER_NAMES = ["test", "Test", "testing", "Testing"]

# Class names and label encoding
# REAL = 0, FAKE = 1  (sigmoid output = P(FAKE))
CLASS_NAMES = ["Real", "Fake"]
CLASS_LABELS = {"Real": 0, "Fake": 1}
POSITIVE_CLASS = "Fake"

# Recognized class folder name variations
REAL_FOLDER_NAMES = ["Real", "real", "REAL", "Genuine", "genuine", "Authentic", "authentic"]
FAKE_FOLDER_NAMES = ["Fake", "fake", "FAKE", "Counterfeit", "counterfeit", "Forged", "forged"]

# Supported image extensions
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# =============================================================================
# Image Preprocessing
# =============================================================================
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNELS = 3
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

# Pixel normalization: divide by this value to scale to [0, 1]
PIXEL_SCALE_FACTOR = 255.0

# =============================================================================
# Model Configuration
# =============================================================================
MODEL_FILENAME = "best_model.keras"
MODEL_PATH = MODELS_DIR / MODEL_FILENAME

# Decision threshold for binary classification
# sigmoid output >= threshold → FAKE, < threshold → REAL
CLASSIFICATION_THRESHOLD = 0.5

# =============================================================================
# Training Hyperparameters
# =============================================================================
BATCH_SIZE = 32
MAX_EPOCHS = 50
LEARNING_RATE = 0.001
RANDOM_SEED = 42

# Data augmentation parameters (training only)
ROTATION_RANGE = 15          # degrees
WIDTH_SHIFT_RANGE = 0.1      # fraction of total width
HEIGHT_SHIFT_RANGE = 0.1     # fraction of total height
ZOOM_RANGE = 0.15            # zoom factor
BRIGHTNESS_RANGE = (0.85, 1.15)  # brightness multiplier range

# EarlyStopping
EARLY_STOPPING_PATIENCE = 8
EARLY_STOPPING_MONITOR = "val_loss"
EARLY_STOPPING_RESTORE_BEST = True

# ReduceLROnPlateau
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_PATIENCE = 4
REDUCE_LR_MIN_LR = 1e-7
REDUCE_LR_MONITOR = "val_loss"

# =============================================================================
# Data Splitting (if no pre-existing splits)
# =============================================================================
TRAIN_SPLIT_RATIO = 0.70
VAL_SPLIT_RATIO = 0.15
TEST_SPLIT_RATIO = 0.15

# =============================================================================
# Output Files
# =============================================================================
TRAINING_HISTORY_PATH = RESULTS_DIR / "training_history.json"
MODEL_SUMMARY_PATH = RESULTS_DIR / "model_summary.txt"
EVALUATION_REPORT_PATH = RESULTS_DIR / "evaluation_report.json"
EVALUATION_TEXT_PATH = RESULTS_DIR / "evaluation_report.txt"
CLASSIFICATION_REPORT_PATH = RESULTS_DIR / "classification_report.csv"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
TRAINING_ACCURACY_PLOT_PATH = RESULTS_DIR / "training_accuracy.png"
TRAINING_LOSS_PLOT_PATH = RESULTS_DIR / "training_loss.png"
TRAINING_METADATA_PATH = RESULTS_DIR / "training_metadata.json"
DATASET_INFO_PATH = RESULTS_DIR / "dataset_info.json"
