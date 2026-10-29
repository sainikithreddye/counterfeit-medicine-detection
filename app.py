"""
Streamlit Web Application for AI-Powered Counterfeit Medicine Detection.

A professional, healthcare-inspired interface for:
- Uploading medicine package images for classification
- Viewing model performance metrics
- Exploring dataset and model information
"""

import json
import sys
import os
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import numpy as np
from PIL import Image

import config


# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="Counterfeit Medicine Detection",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Custom CSS
# =============================================================================
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .main-header h1 {
        color: #1B4F72;
        font-size: 2.2rem;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #5D6D7E;
        font-size: 1.1rem;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #dee2e6;
        margin-bottom: 0.8rem;
    }
    .metric-card h3 {
        color: #1B4F72;
        font-size: 0.9rem;
        margin-bottom: 0.3rem;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2C3E50;
    }

    /* Result styling */
    .result-real {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 2px solid #28a745;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .result-fake {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 2px solid #dc3545;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }

    /* Warning box */
    .disclaimer-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #856404;
    }

    /* Workflow steps */
    .workflow-step {
        background: #f0f8ff;
        border-radius: 8px;
        padding: 0.8rem;
        text-align: center;
        border: 1px solid #b8daff;
        margin: 0.2rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem 0;
        margin-top: 2rem;
        border-top: 1px solid #dee2e6;
        color: #6c757d;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Helper Functions
# =============================================================================

def check_model_available() -> bool:
    """Check if a trained model file exists."""
    return config.MODEL_PATH.exists()


def check_dataset_available() -> dict:
    """Check dataset availability and return info."""
    from preprocessing import discover_dataset
    return discover_dataset(config.DATASET_DIR)


@st.cache_resource
def load_model_cached():
    """Load the trained model (cached across reruns)."""
    from model import load_trained_model
    return load_trained_model(config.MODEL_PATH)


def load_evaluation_results() -> dict | None:
    """Load saved evaluation results if available."""
    if config.EVALUATION_REPORT_PATH.exists():
        with open(config.EVALUATION_REPORT_PATH) as f:
            return json.load(f)
    return None


def load_training_history() -> dict | None:
    """Load saved training history if available."""
    if config.TRAINING_HISTORY_PATH.exists():
        with open(config.TRAINING_HISTORY_PATH) as f:
            return json.load(f)
    return None


def load_training_metadata() -> dict | None:
    """Load saved training metadata if available."""
    if config.TRAINING_METADATA_PATH.exists():
        with open(config.TRAINING_METADATA_PATH) as f:
            return json.load(f)
    return None


# =============================================================================
# Sidebar Navigation
# =============================================================================
st.sidebar.markdown("## 💊 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "🔬 Predict Medicine",
        "📊 Model Performance",
        "📁 Dataset Info",
        "🧠 Model Architecture",
        "⚠️ Limitations",
    ],
    label_visibility="collapsed",
)

# Status indicators in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### System Status")

model_available = check_model_available()
if model_available:
    st.sidebar.success("✅ Trained model loaded")
else:
    st.sidebar.error("❌ No trained model found")

ds_info = check_dataset_available()
if ds_info["valid"]:
    st.sidebar.success(f"✅ Dataset: {ds_info['total_images']} images")
else:
    st.sidebar.warning("⚠️ Dataset not found/incomplete")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small>AI-Powered Counterfeit Medicine Detection<br>"
    "Deep Learning Assignment Project</small>",
    unsafe_allow_html=True,
)

# Reload model button
if model_available:
    if st.sidebar.button("🔄 Reload Model"):
        load_model_cached.clear()
        st.rerun()


# =============================================================================
# Page: Home
# =============================================================================
if page == "🏠 Home":
    st.markdown(
        '<div class="main-header">'
        '<h1>💊 AI-Powered Counterfeit Medicine Detection</h1>'
        '<p>Using Convolutional Neural Networks for Visual Screening</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown("### 🎯 Project Objective")
    st.markdown(
        "This application uses a custom-built **Convolutional Neural Network (CNN)** "
        "to classify images of medicine packages as visually resembling either "
        "**REAL** or **FAKE** medicines based on patterns learned from the training dataset."
    )

    st.markdown(
        '<div class="disclaimer-box">'
        "⚠️ <strong>Important:</strong> AI-assisted visual screening only. "
        "This result is not proof of medicine authenticity. A medicine package "
        "predicted as REAL may still be counterfeit. Consult a qualified pharmacist, "
        "manufacturer, or appropriate regulatory authority if authenticity is uncertain."
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### 🔄 System Workflow")
    cols = st.columns(5)
    steps = [
        ("📸", "Medicine\nPackage Image"),
        ("⚙️", "Image\nPreprocessing"),
        ("🧠", "CNN Feature\nExtraction"),
        ("📊", "Binary\nClassification"),
        ("✅", "Result:\nReal / Fake"),
    ]
    for col, (icon, label) in zip(cols, steps):
        with col:
            st.markdown(
                f'<div class="workflow-step">'
                f'<div style="font-size:2rem">{icon}</div>'
                f'<div style="font-size:0.85rem; margin-top:0.3rem">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("### 🔍 How It Works")
    st.markdown(
        "The CNN analyzes visual characteristics of medicine packaging images, "
        "including:\n"
        "- **Shapes and edges** in the packaging\n"
        "- **Colors and textures** of labels and materials\n"
        "- **Logo patterns** and branding elements\n"
        "- **Text and label arrangements**\n"
        "- **Overall packaging structure**\n\n"
        "The prediction is based entirely on visual patterns learned from "
        "labeled training images. It does not chemically analyze medicine "
        "or verify manufacturing records."
    )

    st.markdown("### 🚀 Quick Start")
    if model_available:
        st.info("✅ Model is ready! Navigate to **🔬 Predict Medicine** to classify an image.")
    else:
        st.warning(
            "**Model not yet trained.** To get started:\n\n"
            f"1. Dataset location: `{config.DATASET_DIR}` (detected and validated)\n"
            "2. Run `python train.py` to train the CNN\n"
            "3. Run `python evaluate.py` to evaluate performance\n"
            "4. Return here to use the prediction feature"
        )


# =============================================================================
# Page: Predict Medicine
# =============================================================================
elif page == "🔬 Predict Medicine":
    st.markdown("## 🔬 Medicine Image Prediction")
    st.markdown("Upload a medicine package image to classify it as **Real** or **Fake**.")

    if not model_available:
        st.error(
            "### ❌ Trained Model Not Available\n\n"
            "The prediction feature requires a trained CNN model.\n\n"
            "**Setup Instructions:**\n"
            f"1. Dataset is configured at `{config.DATASET_DIR}`\n"
            "2. Run `python train.py` to train the model\n"
            "3. Restart this application\n\n"
            f"Expected model path: `{config.MODEL_PATH}`"
        )
    else:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### 📤 Upload Image")
            uploaded_file = st.file_uploader(
                "Choose a medicine package image",
                type=["jpg", "jpeg", "png"],
                help="Upload a clear photo of a medicine package (JPG or PNG format)",
            )

            if uploaded_file is not None:
                try:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                except Exception as e:
                    st.error(f"Could not open uploaded file: {e}")
                    uploaded_file = None

        with col2:
            st.markdown("### 🔍 Prediction Result")

            if uploaded_file is not None:
                predict_btn = st.button("🔬 Analyze Image", type="primary", use_container_width=True)

                if predict_btn:
                    with st.spinner("Running CNN inference..."):
                        try:
                            image = Image.open(uploaded_file)
                            model = load_model_cached()
                            from predict import predict_pil_image
                            result = predict_pil_image(image, model=model)

                            # Store in session state
                            st.session_state["prediction_result"] = result
                        except Exception as e:
                            st.error(f"Prediction failed: {e}")

                # Display result
                if "prediction_result" in st.session_state:
                    result = st.session_state["prediction_result"]
                    label = result["label"]
                    fake_pct = result["fake_probability"] * 100
                    real_pct = result["real_probability"] * 100

                    if label == "Real":
                        st.markdown(
                            f'<div class="result-real">'
                            f'<h2 style="color:#155724;">✅ Predicted: REAL</h2>'
                            f'<p>Model-estimated Real probability: <strong>{real_pct:.2f}%</strong></p>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="result-fake">'
                            f'<h2 style="color:#721c24;">⚠️ Predicted: FAKE</h2>'
                            f'<p>Model-estimated Fake probability: <strong>{fake_pct:.2f}%</strong></p>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    st.markdown("")

                    # Probability bars
                    st.markdown("**Model-Estimated Probabilities:**")
                    rcol, fcol = st.columns(2)
                    with rcol:
                        st.metric("Real", f"{real_pct:.2f}%")
                        st.progress(result["real_probability"])
                    with fcol:
                        st.metric("Fake", f"{fake_pct:.2f}%")
                        st.progress(result["fake_probability"])

                    st.markdown(
                        f"*Raw sigmoid score: {result['raw_score']:.6f} | "
                        f"Threshold: {result['threshold']}*"
                    )

                    st.markdown(
                        '<div class="disclaimer-box">'
                        "⚠️ This is an AI-assisted visual screening result based on "
                        "patterns learned from training images. It is not proof of "
                        "medicine authenticity. Consult a qualified pharmacist or "
                        "regulatory authority if authenticity is uncertain."
                        '</div>',
                        unsafe_allow_html=True,
                    )

                    if st.button("🔄 Clear & Start Over"):
                        if "prediction_result" in st.session_state:
                            del st.session_state["prediction_result"]
                        st.rerun()

            else:
                st.info("👆 Upload an image to get started.")


# =============================================================================
# Page: Model Performance
# =============================================================================
elif page == "📊 Model Performance":
    st.markdown("## 📊 Model Performance Metrics")

    eval_results = load_evaluation_results()
    training_history = load_training_history()
    metadata = load_training_metadata()

    if eval_results is None and training_history is None:
        st.warning(
            "### Metrics Not Available\n\n"
            "No evaluation results or training history found.\n\n"
            "To generate metrics:\n"
            "1. Train the model: `python train.py`\n"
            "2. Evaluate: `python evaluate.py`\n"
            "3. Refresh this page"
        )
    else:
        # Evaluation metrics
        if eval_results:
            st.markdown("### 📈 Test Set Evaluation")

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(
                    f'<div class="metric-card"><h3>Accuracy</h3>'
                    f'<div class="value">{eval_results["accuracy"]*100:.2f}%</div></div>',
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f'<div class="metric-card"><h3>Precision (Fake)</h3>'
                    f'<div class="value">{eval_results["fake_precision"]*100:.2f}%</div></div>',
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f'<div class="metric-card"><h3>Recall (Fake)</h3>'
                    f'<div class="value">{eval_results["fake_recall"]*100:.2f}%</div></div>',
                    unsafe_allow_html=True,
                )
            with c4:
                st.markdown(
                    f'<div class="metric-card"><h3>F1-Score (Fake)</h3>'
                    f'<div class="value">{eval_results["fake_f1_score"]*100:.2f}%</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"**Test images:** {eval_results['total_test_images']} | "
                f"**Correct:** {eval_results['correctly_classified']} | "
                f"**Misclassified:** {eval_results['misclassified']}"
            )

            # Confusion matrix image
            if config.CONFUSION_MATRIX_PATH.exists():
                st.markdown("### Confusion Matrix")
                st.image(str(config.CONFUSION_MATRIX_PATH), use_container_width=False)

            # Classification report
            if "classification_report" in eval_results:
                st.markdown("### Classification Report")
                report = eval_results["classification_report"]
                report_data = []
                for cls_name in config.CLASS_NAMES:
                    if cls_name in report:
                        r = report[cls_name]
                        report_data.append({
                            "Class": cls_name,
                            "Precision": f"{r['precision']:.4f}",
                            "Recall": f"{r['recall']:.4f}",
                            "F1-Score": f"{r['f1-score']:.4f}",
                            "Support": int(r["support"]),
                        })
                st.table(report_data)

            st.info(
                "ℹ️ These metrics are computed from actual model predictions on the "
                "held-out test set. Real-world performance depends on dataset quality "
                "and representativeness."
            )

        # Training curves
        if training_history:
            st.markdown("### 📉 Training Curves")
            tc1, tc2 = st.columns(2)

            with tc1:
                if config.TRAINING_ACCURACY_PLOT_PATH.exists():
                    st.image(str(config.TRAINING_ACCURACY_PLOT_PATH), caption="Accuracy Curves")
                else:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.plot(training_history.get("accuracy", []), label="Train")
                    ax.plot(training_history.get("val_accuracy", []), label="Validation")
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Accuracy")
                    ax.set_title("Training & Validation Accuracy")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)

            with tc2:
                if config.TRAINING_LOSS_PLOT_PATH.exists():
                    st.image(str(config.TRAINING_LOSS_PLOT_PATH), caption="Loss Curves")
                else:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.plot(training_history.get("loss", []), label="Train")
                    ax.plot(training_history.get("val_loss", []), label="Validation")
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Loss")
                    ax.set_title("Training & Validation Loss")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)

        # Training metadata
        if metadata:
            st.markdown("### 🔧 Training Details")
            meta_cols = st.columns(3)
            with meta_cols[0]:
                st.metric("Total Epochs", metadata.get("actual_epochs", "N/A"))
                st.metric("Best Epoch", metadata.get("best_epoch", "N/A"))
            with meta_cols[1]:
                st.metric("Training Samples", metadata.get("training_samples", "N/A"))
                st.metric("Validation Samples", metadata.get("validation_samples", "N/A"))
            with meta_cols[2]:
                val_acc = metadata.get("best_val_accuracy")
                st.metric(
                    "Best Val Accuracy",
                    f"{val_acc*100:.2f}%" if val_acc else "N/A",
                )
                time_s = metadata.get("training_time_seconds")
                st.metric(
                    "Training Time",
                    f"{time_s:.1f}s" if time_s else "N/A",
                )


# =============================================================================
# Page: Dataset Info
# =============================================================================
elif page == "📁 Dataset Info":
    st.markdown("## 📁 Dataset Information")

    st.markdown(
        "**Dataset:** Fake vs Real Medicine Dataset  \n"
        "**Source:** Kaggle  \n"
        "**Classes:** Real, Fake  \n"
        "**Model Input:** 224 × 224 × 3 (RGB)  \n"
    )

    if ds_info["valid"]:
        st.success(f"✅ Dataset found with **{ds_info['total_images']}** total images")

        st.markdown("### 📊 Image Distribution")
        for split_name, split_info in ds_info["splits"].items():
            st.markdown(f"**{split_name}:**")
            cols = st.columns(3)
            with cols[0]:
                st.metric("Real", split_info["class_counts"].get("Real", 0))
            with cols[1]:
                st.metric("Fake", split_info["class_counts"].get("Fake", 0))
            with cols[2]:
                st.metric("Total", split_info["total_images"])

        if ds_info["warnings"]:
            st.markdown("### ⚠️ Warnings")
            for w in ds_info["warnings"]:
                st.warning(w)
    else:
        st.error("### ❌ Dataset Not Found or Incomplete")
        for err in ds_info["errors"]:
            st.error(err)

        st.markdown(
            "### 📥 Setup Instructions\n\n"
            f"1. Extract the dataset into `{config.DATASET_DIR}`\n"
            "2. Ensure the following structure (lowercase or capitalized):\n"
            "```\n"
            "dataset/\n"
            "├── train/ (or Train/)\n"
            "│   ├── Fake/\n"
            "│   └── Real/\n"
            "├── val/ (or Val/)\n"
            "│   ├── Fake/\n"
            "│   └── Real/\n"
            "└── test/ (or Test/)\n"
            "    ├── Fake/\n"
            "    └── Real/\n"
            "```\n"
            "3. Refresh the application"
        )

    st.markdown("### ⚙️ Preprocessing Pipeline")
    st.markdown(
        "| Step | Details |\n"
        "|------|--------|\n"
        "| Load | Read image file safely |\n"
        "| Color | Convert to RGB |\n"
        "| Resize | Scale to 224 × 224 pixels |\n"
        "| Normalize | Divide pixel values by 255 → [0, 1] |\n"
        "| Batch | Add batch dimension for model input |\n"
    )

    st.markdown("### 🔀 Training Augmentation")
    st.markdown(
        "Applied **only** to training images:\n"
        "- Random rotation (±15°)\n"
        "- Horizontal/vertical shift (±10%)\n"
        "- Zoom (±15%)\n"
        "- Brightness variation (85%–115%)\n\n"
        "No augmentation is applied to validation or test images."
    )


# =============================================================================
# Page: Model Architecture
# =============================================================================
elif page == "🧠 Model Architecture":
    st.markdown("## 🧠 CNN Architecture")
    st.markdown(
        "The model uses a custom Convolutional Neural Network designed for "
        "binary classification of medicine packaging images."
    )

    st.markdown("### Layer Configuration")
    arch_data = [
        {"#": 1, "Layer": "Input", "Configuration": "224 × 224 × 3", "Purpose": "Accept RGB image input"},
        {"#": 2, "Layer": "Conv2D", "Configuration": "32 filters, 3×3, ReLU, same padding", "Purpose": "Extract low-level features (edges, textures)"},
        {"#": 3, "Layer": "MaxPooling2D", "Configuration": "2×2 pool size", "Purpose": "Reduce spatial dimensions, retain features"},
        {"#": 4, "Layer": "Conv2D", "Configuration": "64 filters, 3×3, ReLU, same padding", "Purpose": "Extract mid-level features (shapes, patterns)"},
        {"#": 5, "Layer": "MaxPooling2D", "Configuration": "2×2 pool size", "Purpose": "Further spatial reduction"},
        {"#": 6, "Layer": "Conv2D", "Configuration": "128 filters, 3×3, ReLU, same padding", "Purpose": "Extract high-level features (logos, text patterns)"},
        {"#": 7, "Layer": "MaxPooling2D", "Configuration": "2×2 pool size", "Purpose": "Final spatial reduction"},
        {"#": 8, "Layer": "GlobalAvgPool2D", "Configuration": "—", "Purpose": "Aggregate spatial features into fixed-length vector"},
        {"#": 9, "Layer": "Dense", "Configuration": "128 neurons, ReLU", "Purpose": "Learn classification decision boundary"},
        {"#": 10, "Layer": "Dropout", "Configuration": "Rate = 0.5", "Purpose": "Prevent overfitting during training"},
        {"#": 11, "Layer": "Dense (Output)", "Configuration": "1 neuron, Sigmoid", "Purpose": "Output probability of FAKE class"},
    ]
    st.table(arch_data)

    st.markdown("### 🏗️ Architecture Concepts")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            "**Feature Extraction (Conv2D)**  \n"
            "Convolutional layers apply learnable filters to detect "
            "visual patterns. Deeper layers capture increasingly complex features.\n\n"
            "**Pooling (MaxPooling2D)**  \n"
            "Reduces spatial dimensions while preserving the most "
            "important features, making the model efficient and translation-invariant."
        )
    with c2:
        st.markdown(
            "**Global Average Pooling**  \n"
            "Converts feature maps into a compact vector by averaging "
            "each feature map, reducing overfitting compared to Flatten.\n\n"
            "**Dropout**  \n"
            "Randomly deactivates 50% of neurons during training to "
            "prevent the model from memorizing training examples."
        )

    st.markdown("### 🏷️ Label Encoding")
    st.markdown(
        "| Class | Label | Interpretation |\n"
        "|-------|-------|---------------|\n"
        "| Real | 0 | Sigmoid output < 0.5 |\n"
        "| Fake | 1 | Sigmoid output ≥ 0.5 |\n\n"
        "The sigmoid output represents the model's estimated probability "
        "that the input belongs to the **Fake** class."
    )

    # Show actual model summary if available
    if config.MODEL_SUMMARY_PATH.exists():
        st.markdown("### 📋 Model Summary")
        with open(config.MODEL_SUMMARY_PATH) as f:
            st.code(f.read(), language="text")


# =============================================================================
# Page: Limitations
# =============================================================================
elif page == "⚠️ Limitations":
    st.markdown("## ⚠️ Limitations & Disclaimer")

    st.markdown(
        '<div class="disclaimer-box" style="font-size:1rem;">'
        "⚠️ <strong>AI-assisted visual screening only.</strong> This result is not proof "
        "of medicine authenticity. A medicine package predicted as REAL may still be "
        "counterfeit. Consult a qualified pharmacist, manufacturer, or appropriate "
        "regulatory authority if authenticity is uncertain."
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### What This Application Does")
    st.markdown(
        "- ✅ Classifies **visual resemblance** to the training classes\n"
        "- ✅ Provides a model-estimated probability score\n"
        "- ✅ Uses a CNN trained on labeled medicine packaging images\n"
    )

    st.markdown("### What This Application Does **NOT** Do")
    st.markdown(
        "- ❌ Does **not** test chemical composition or active ingredients\n"
        "- ❌ Does **not** establish manufacturer authorization or legal authenticity\n"
        "- ❌ Does **not** verify barcodes, batch numbers, or supply chain records\n"
        "- ❌ Does **not** guarantee that a REAL prediction means the medicine is safe\n"
        "- ❌ Must **not** be used as the sole basis for medical or purchasing decisions\n"
    )

    st.markdown("### Known Limitations")
    st.markdown(
        "- May incorrectly classify **unfamiliar medicine brands** not in the training data\n"
        "- May produce errors on **poorly photographed**, blurry, or occluded images\n"
        "- May **generalize poorly** beyond the distribution of the training dataset\n"
        "- Performance depends on the **quality and representativeness** of the training data\n"
        "- The model learns visual correlations, not causal indicators of authenticity\n"
    )

    st.markdown("### Ethical Considerations")
    st.markdown(
        "This tool is intended as a supplementary screening aid for educational "
        "and research purposes. It should **always** be used in conjunction with "
        "professional pharmaceutical verification procedures. A predicted REAL "
        "result must **never** be interpreted as a guarantee of safety."
    )


# =============================================================================
# Footer
# =============================================================================
st.markdown("---")
st.markdown(
    '<div class="footer">'
    '💊 <strong>AI-Powered Counterfeit Medicine Detection</strong> — '
    'Deep Learning Assignment Project<br>'
    '<em>AI-assisted visual screening only. Not a substitute for professional '
    'pharmaceutical verification.</em>'
    '</div>',
    unsafe_allow_html=True,
)
