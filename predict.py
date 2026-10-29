"""
Single-image prediction module for counterfeit medicine detection.

Provides:
- Command-line inference on a single image
- Reusable prediction function for the Streamlit app
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def predict_image(
    image_path: str | Path,
    model=None,
    threshold: float = config.CLASSIFICATION_THRESHOLD,
) -> Dict[str, Any]:
    """
    Predict whether a medicine package image is Real or Fake.

    Args:
        image_path: Path to the image file.
        model: Pre-loaded Keras model. If None, loads from config.MODEL_PATH.
        threshold: Classification threshold. >= threshold → FAKE.

    Returns:
        Dictionary with:
            - 'label': 'Real' or 'Fake'
            - 'fake_probability': float (0-1)
            - 'real_probability': float (0-1)
            - 'raw_score': float (raw sigmoid output)
            - 'threshold': float
            - 'image_path': str

    Raises:
        FileNotFoundError: If image or model file not found.
        ValueError: If image is invalid or model output is unexpected.
    """
    from preprocessing import preprocess_for_prediction
    from model import load_trained_model

    image_path = Path(image_path)

    # Validate image file
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if not image_path.is_file():
        raise ValueError(f"Path is not a file: {image_path}")

    suffix = image_path.suffix.lower()
    if suffix not in config.SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format '{suffix}'. "
            f"Supported: {sorted(config.SUPPORTED_IMAGE_EXTENSIONS)}"
        )

    # Load model if not provided
    if model is None:
        model = load_trained_model(config.MODEL_PATH)

    # Preprocess
    img_array = preprocess_for_prediction(image_path)

    # Predict
    prediction = model.predict(img_array, verbose=0)

    # Validate output shape
    if prediction.ndim == 0 or prediction.size != 1:
        raise ValueError(
            f"Unexpected model output shape: {prediction.shape}. "
            f"Expected shape (1, 1) for binary classification."
        )

    raw_score = float(prediction.flatten()[0])

    # Validate output range
    if not (0.0 <= raw_score <= 1.0):
        raise ValueError(
            f"Model output {raw_score} is outside expected [0, 1] range."
        )

    # Apply threshold: sigmoid output = P(FAKE)
    fake_prob = raw_score
    real_prob = 1.0 - raw_score
    label = "Fake" if raw_score >= threshold else "Real"

    result = {
        "label": label,
        "fake_probability": round(fake_prob, 6),
        "real_probability": round(real_prob, 6),
        "raw_score": round(raw_score, 6),
        "threshold": threshold,
        "image_path": str(image_path),
    }

    return result


def predict_pil_image(
    pil_image,
    model=None,
    threshold: float = config.CLASSIFICATION_THRESHOLD,
) -> Dict[str, Any]:
    """
    Predict from a PIL Image object (e.g., from Streamlit upload).

    Args:
        pil_image: PIL Image object.
        model: Pre-loaded Keras model.
        threshold: Classification threshold.

    Returns:
        Same dictionary structure as predict_image().
    """
    from preprocessing import preprocess_pil_image
    from model import load_trained_model

    if model is None:
        model = load_trained_model(config.MODEL_PATH)

    img_array = preprocess_pil_image(pil_image)

    prediction = model.predict(img_array, verbose=0)

    if prediction.ndim == 0 or prediction.size != 1:
        raise ValueError(f"Unexpected model output shape: {prediction.shape}")

    raw_score = float(prediction.flatten()[0])

    if not (0.0 <= raw_score <= 1.0):
        raise ValueError(f"Model output {raw_score} outside expected [0, 1] range.")

    fake_prob = raw_score
    real_prob = 1.0 - raw_score
    label = "Fake" if raw_score >= threshold else "Real"

    return {
        "label": label,
        "fake_probability": round(fake_prob, 6),
        "real_probability": round(real_prob, 6),
        "raw_score": round(raw_score, 6),
        "threshold": threshold,
        "image_path": "uploaded_image",
    }


def main():
    """Command-line interface for single-image prediction."""
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        print("Example: python predict.py test_medicine.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    logger.info(f"Predicting: {image_path}")

    try:
        result = predict_image(image_path)

        print("\n" + "=" * 50)
        print("COUNTERFEIT MEDICINE DETECTION — PREDICTION")
        print("=" * 50)
        print(f"Image:              {result['image_path']}")
        print(f"Predicted Class:    {result['label']}")
        print(f"FAKE probability:   {result['fake_probability'] * 100:.2f}%")
        print(f"REAL probability:   {result['real_probability'] * 100:.2f}%")
        print(f"Raw sigmoid score:  {result['raw_score']:.6f}")
        print(f"Threshold:          {result['threshold']}")
        print("=" * 50)
        print(
            "\n⚠ AI-assisted visual screening only. This result is not proof "
            "of medicine authenticity. Consult a qualified pharmacist or "
            "regulatory authority if authenticity is uncertain."
        )

    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Prediction error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
