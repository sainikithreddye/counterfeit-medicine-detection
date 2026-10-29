"""
Model evaluation on the held-out test set.

Computes and saves:
- Accuracy, Precision, Recall, F1-score
- Confusion matrix with heatmap
- Per-class classification report
- Detailed evaluation report (JSON, TXT, CSV)
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

import config
from model import load_trained_model
from preprocessing import create_data_generators, discover_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def evaluate() -> None:
    """
    Evaluate the trained model on the test dataset.

    Computes metrics, generates confusion matrix heatmap,
    and saves all results to the results directory.
    """
    logger.info("=" * 60)
    logger.info("COUNTERFEIT MEDICINE DETECTION — MODEL EVALUATION")
    logger.info("=" * 60)

    # --- Check model exists ---
    if not config.MODEL_PATH.exists():
        logger.error(
            f"Trained model not found at {config.MODEL_PATH}.\n"
            "Please train the model first: python train.py"
        )
        sys.exit(1)

    # --- Check dataset ---
    ds_info = discover_dataset(config.DATASET_DIR)
    if not ds_info["valid"]:
        logger.error("Dataset validation failed:")
        for err in ds_info["errors"]:
            logger.error(f"  • {err}")
        sys.exit(1)

    # --- Load model ---
    logger.info("Loading trained model...")
    model = load_trained_model(config.MODEL_PATH)

    # --- Create test data generator ---
    logger.info("Preparing test data...")
    _, _, test_gen = create_data_generators(
        dataset_dir=config.DATASET_DIR,
        batch_size=config.BATCH_SIZE,
        image_size=config.IMAGE_SIZE,
        augment_training=False,
    )

    logger.info(f"Test samples: {test_gen.samples}")
    logger.info(f"Class indices: {test_gen.class_indices}")

    # --- Get predictions ---
    logger.info("Running predictions on test set...")
    test_gen.reset()
    y_prob = model.predict(test_gen, verbose=1)
    y_prob = y_prob.flatten()

    # Apply threshold
    y_pred = (y_prob >= config.CLASSIFICATION_THRESHOLD).astype(int)

    # True labels
    y_true = test_gen.classes

    # Verify lengths match
    assert len(y_true) == len(y_pred), (
        f"Label count mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
    )

    # --- Compute metrics ---
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report,
    )

    accuracy = accuracy_score(y_true, y_pred)
    # FAKE = 1 is the positive class
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    # Confusion matrix: rows = true, cols = predicted
    # Labels order: [0=Real, 1=Fake]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # Classification report
    cls_report_str = classification_report(
        y_true, y_pred,
        target_names=config.CLASS_NAMES,  # ["Real", "Fake"]
        labels=[0, 1],
        zero_division=0,
    )

    cls_report_dict = classification_report(
        y_true, y_pred,
        target_names=config.CLASS_NAMES,
        labels=[0, 1],
        output_dict=True,
        zero_division=0,
    )

    correct = int(np.sum(y_true == y_pred))
    misclassified = int(np.sum(y_true != y_pred))

    # --- Print results ---
    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"Test images:       {len(y_true)}")
    logger.info(f"Correct:           {correct}")
    logger.info(f"Misclassified:     {misclassified}")
    logger.info(f"Overall Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
    logger.info(f"Fake Precision:    {precision:.4f}")
    logger.info(f"Fake Recall:       {recall:.4f}")
    logger.info(f"Fake F1-Score:     {f1:.4f}")
    logger.info(f"\nConfusion Matrix (rows=true, cols=predicted):")
    logger.info(f"  Label order: Real=0, Fake=1")
    logger.info(f"  {cm}")
    logger.info(f"\nClassification Report:\n{cls_report_str}")

    # --- Save results ---
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # JSON report
    eval_report = {
        "total_test_images": int(len(y_true)),
        "correctly_classified": correct,
        "misclassified": misclassified,
        "accuracy": float(accuracy),
        "fake_precision": float(precision),
        "fake_recall": float(recall),
        "fake_f1_score": float(f1),
        "confusion_matrix": {
            "label_order": ["Real", "Fake"],
            "matrix": cm.tolist(),
            "description": "Rows = true labels, Columns = predicted labels",
        },
        "classification_report": cls_report_dict,
        "classification_threshold": config.CLASSIFICATION_THRESHOLD,
        "note": (
            "These metrics reflect model performance on the held-out test set. "
            "Real-world performance depends on dataset quality, representativeness, "
            "and the independence of the test data."
        ),
    }

    with open(config.EVALUATION_REPORT_PATH, "w") as f:
        json.dump(eval_report, f, indent=2)
    logger.info(f"Evaluation report saved to {config.EVALUATION_REPORT_PATH}")

    # Text report
    with open(config.EVALUATION_TEXT_PATH, "w") as f:
        f.write("COUNTERFEIT MEDICINE DETECTION — EVALUATION REPORT\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Test images:       {len(y_true)}\n")
        f.write(f"Correct:           {correct}\n")
        f.write(f"Misclassified:     {misclassified}\n")
        f.write(f"Overall Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)\n")
        f.write(f"Fake Precision:    {precision:.4f}\n")
        f.write(f"Fake Recall:       {recall:.4f}\n")
        f.write(f"Fake F1-Score:     {f1:.4f}\n")
        f.write(f"\nConfusion Matrix (Real=0, Fake=1):\n{cm}\n")
        f.write(f"\nClassification Report:\n{cls_report_str}\n")
        f.write(
            "\nNote: These metrics reflect model performance on the held-out test set. "
            "Real-world performance depends on dataset quality and representativeness.\n"
        )
    logger.info(f"Text report saved to {config.EVALUATION_TEXT_PATH}")

    # CSV classification report
    import csv
    with open(config.CLASSIFICATION_REPORT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Class", "Precision", "Recall", "F1-Score", "Support"])
        for cls_name in config.CLASS_NAMES:
            if cls_name in cls_report_dict:
                row = cls_report_dict[cls_name]
                writer.writerow([
                    cls_name,
                    f"{row['precision']:.4f}",
                    f"{row['recall']:.4f}",
                    f"{row['f1-score']:.4f}",
                    int(row['support']),
                ])
        if "accuracy" in cls_report_dict:
            writer.writerow(["accuracy", "", "", f"{accuracy:.4f}", int(len(y_true))])
    logger.info(f"Classification report CSV saved to {config.CLASSIFICATION_REPORT_PATH}")

    # Confusion matrix heatmap
    _save_confusion_matrix_plot(cm)

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info(f"Results saved to: {config.RESULTS_DIR}")
    logger.info("=" * 60)


def _save_confusion_matrix_plot(cm) -> None:
    """Generate and save a confusion matrix heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
        cbar=True,
        square=True,
        linewidths=0.5,
    )
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Confusion Matrix\n(Real=0, Fake=1)", fontsize=14)
    plt.tight_layout()
    plt.savefig(config.CONFUSION_MATRIX_PATH, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix plot saved to {config.CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    evaluate()
