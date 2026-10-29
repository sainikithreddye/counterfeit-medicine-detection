# 💊 AI-Powered Counterfeit Medicine Detection Using CNN

## Project Overview

This project implements an AI-powered application that uses a **Convolutional Neural Network (CNN)** to classify images of medicine packages as either **REAL** or **FAKE** based on visual patterns learned from the *Fake vs Real Medicine Dataset*.

> ⚠️ **Disclaimer:** AI-assisted visual screening only. This result is not proof of medicine authenticity. A medicine package predicted as REAL may still be counterfeit. Consult a qualified pharmacist, manufacturer, or appropriate regulatory authority if authenticity is uncertain.

## Problem Statement

Counterfeit medicines are a significant global health threat. This project explores the use of deep learning to assist in visual screening of medicine packaging, leveraging a CNN trained on labeled images of genuine and counterfeit medicine packages.

The model analyzes visual characteristics including shapes, edges, colors, textures, logo patterns, text arrangements, and overall packaging structure — but does **not** perform chemical analysis, verify manufacturing records, or establish legal authenticity.

## Objectives

1. Train a custom CNN on the Fake vs Real Medicine Dataset
2. Achieve binary classification (Real vs Fake) of medicine packaging images
3. Provide a professional web interface for image upload and prediction
4. Generate comprehensive model evaluation metrics
5. Maintain clear limitations and medical disclaimers

## Main Features

- ✅ Custom CNN architecture (3 Conv blocks + Dense classifier)
- ✅ Image preprocessing with training augmentation
- ✅ Complete training pipeline with callbacks (EarlyStopping, ReduceLR, ModelCheckpoint)
- ✅ Model evaluation with accuracy, precision, recall, F1-score, and confusion matrix
- ✅ Single-image prediction with probability scores
- ✅ Professional Streamlit web application
- ✅ Comprehensive automated tests
- ✅ Robust dataset validation and error handling

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12 |
| Deep Learning | TensorFlow / Keras |
| Image Processing | Pillow, TensorFlow image utilities |
| Numerical | NumPy |
| Evaluation | scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |
| Testing | pytest |
| Model Format | Keras (.keras) |

## CNN Architecture

| # | Layer | Configuration | Purpose |
|---|-------|--------------|---------|
| 1 | Input | 224 × 224 × 3 | Accept RGB image |
| 2 | Conv2D | 32 filters, 3×3, ReLU, same | Low-level features |
| 3 | MaxPooling2D | 2×2 | Spatial reduction |
| 4 | Conv2D | 64 filters, 3×3, ReLU, same | Mid-level features |
| 5 | MaxPooling2D | 2×2 | Spatial reduction |
| 6 | Conv2D | 128 filters, 3×3, ReLU, same | High-level features |
| 7 | MaxPooling2D | 2×2 | Spatial reduction |
| 8 | GlobalAvgPool2D | — | Feature aggregation |
| 9 | Dense | 128, ReLU | Classification head |
| 10 | Dropout | 0.5 | Regularization |
| 11 | Dense (Output) | 1, Sigmoid | P(Fake) output |

**Label Encoding:** Real = 0, Fake = 1. Sigmoid output ≥ 0.5 → Fake.

## System Workflow

```
Medicine Package Image → Image Preprocessing → CNN Feature Extraction → Binary Classification → Result (Real/Fake)
```

## Dataset

**Source:** [Fake vs Real Medicine Dataset](https://www.kaggle.com/) (Kaggle)

### Expected Folder Structure

```
data/medicine_dataset/
├── Train/
│   ├── Fake/
│   └── Real/
├── Val/
│   ├── Fake/
│   └── Real/
└── Test/
    ├── Fake/
    └── Real/
```

The system also recognizes folder name variations: `train`/`Training`, `val`/`Validation`, `test`/`Testing`.

## Installation & Setup

### Prerequisites

- **Python 3.12** (required for TensorFlow compatibility)
- 8+ GB RAM recommended
- GPU optional (CUDA-compatible NVIDIA GPU for faster training)

### Step 1: Clone/Navigate to the Project

```powershell
cd counterfeit-medicine-detection
```

### Step 2: Create and Activate Virtual Environment

**Windows PowerShell:**
```powershell
python3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If `python3.12` is not on PATH, use the full path to the Python 3.12 executable, e.g.:
```powershell
& "$env:USERPROFILE\.local\bin\python3.12.exe" -m venv .venv
```

Or if using uv:
```powershell
python -m uv venv .venv --python 3.12
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 4: Download and Setup Dataset

1. Download the **Fake vs Real Medicine Dataset** from Kaggle
2. Extract the archive into `data/medicine_dataset/`
3. Verify the folder structure matches the expected layout above

### Step 5: Train the Model

```powershell
python train.py
```

This will:
- Validate the dataset
- Build the CNN
- Train with augmentation, early stopping, and learning rate scheduling
- Save the best model to `models/best_model.keras`
- Generate training plots in `results/`

### Step 6: Evaluate the Model

```powershell
python evaluate.py
```

Generates evaluation metrics (accuracy, precision, recall, F1, confusion matrix) on the test set.

### Step 7: Launch the Web Application

```powershell
streamlit run app.py
```

Opens at `http://localhost:8501`

### Single-Image Prediction (CLI)

```powershell
python predict.py path/to/medicine_image.jpg
```

## Expected Output Files

| File | Location | Description |
|------|----------|-------------|
| `best_model.keras` | `models/` | Trained CNN model |
| `training_history.json` | `results/` | Epoch-by-epoch metrics |
| `training_metadata.json` | `results/` | Training configuration & versions |
| `model_summary.txt` | `results/` | Model architecture summary |
| `training_accuracy.png` | `results/` | Accuracy curve plot |
| `training_loss.png` | `results/` | Loss curve plot |
| `evaluation_report.json` | `results/` | Test set metrics (JSON) |
| `evaluation_report.txt` | `results/` | Test set metrics (text) |
| `classification_report.csv` | `results/` | Per-class metrics (CSV) |
| `confusion_matrix.png` | `results/` | Confusion matrix heatmap |
| `dataset_info.json` | `results/` | Dataset validation info |

## Running Tests

```powershell
python -m pytest tests/ -v
```

Tests cover:
- Image preprocessing (resize, normalize, format conversion)
- CNN architecture (layers, shapes, activations)
- Dataset validation (structure, missing data, edge cases)
- Prediction pipeline (output format, probabilities, error handling)

## Troubleshooting

### TensorFlow not found / installation fails
- Ensure you're using **Python 3.12** (not 3.13+)
- Install `uv` and use `python -m uv python install 3.12` to get a compatible Python

### Windows Smart App Control (SAC) / Application Control Policy
On Windows 11 systems where **Smart App Control** is active in Enforce mode (`VerifiedAndReputablePolicyState: 1`), Windows Code Integrity blocks unsigned `.pyd` binaries with:
`ImportError: DLL load failed while importing _tf_stack: An Application Control policy has blocked this file.`
To resolve this:
- **Recommended:** Run under **WSL2 (Windows Subsystem for Linux)**: Ubuntu on WSL runs TensorFlow natively with full CPU/GPU support and is unaffected by Windows SAC policy.
- Or run inside a Linux Docker container (`docker run -it -p 8501:8501 tensorflow/tensorflow`).
- Or configure Smart App Control in Windows Security (App & browser control > Smart App Control).

### Out of Memory during training
- Reduce `BATCH_SIZE` in `config.py` (try 16 or 8)
- Close other applications
- Use CPU training if GPU memory is limited

### Dataset validation errors
- Check folder structure matches the expected layout
- Ensure images are valid JPG/PNG files
- Check for empty class folders

### Model file not found
- Run `python train.py` first to generate the model
- Check that `models/best_model.keras` exists

### Streamlit won't start
- Ensure all dependencies are installed
- Try `python -m streamlit run app.py`

## Limitations

- Classifies **visual resemblance** only — not chemical composition
- May fail on medicine brands not in the training dataset
- May produce errors on blurry, occluded, or poorly lit images
- Does **not** verify manufacturer authorization or legal authenticity
- A REAL prediction does **not** guarantee medicine safety
- Performance is bounded by training data quality and diversity

## Medical Authenticity Disclaimer

> ⚠️ **This application is an AI-assisted visual screening tool for educational and research purposes.** It does not test chemical composition, verify manufacturing records, or establish pharmaceutical authenticity. A medicine package predicted as REAL may still be counterfeit. Do not use this application as the sole basis for any medical, purchasing, or regulatory decision. Always consult a qualified pharmacist, manufacturer, or appropriate regulatory authority if medicine authenticity is uncertain.

## Project Structure

```
counterfeit-medicine-detection/
├── app.py                 # Streamlit web application
├── train.py               # Training pipeline
├── evaluate.py            # Model evaluation
├── predict.py             # Single-image inference
├── model.py               # CNN architecture
├── preprocessing.py       # Image loading & dataset utilities
├── config.py              # Central configuration
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── pytest.ini             # Test configuration
├── .gitignore             # Version control exclusions
├── data/
│   └── medicine_dataset/  # Dataset (user-supplied)
├── models/
│   └── best_model.keras   # Trained model (generated)
├── results/               # Training & evaluation outputs
├── tests/
│   ├── test_preprocessing.py
│   ├── test_model.py
│   ├── test_dataset.py
│   ├── test_prediction.py
│   └── test_app_smoke.py
└── .streamlit/
    └── config.toml        # Streamlit theme configuration
```

## Reproducing the Experiment

1. Set up the environment as described above
2. Download and place the exact dataset
3. Run `python train.py` — training uses seed 42 for reproducibility
4. Run `python evaluate.py` — metrics are computed on the held-out test set
5. Compare results with the assignment report

---

*Deep Learning Assignment Project — AI-Powered Counterfeit Medicine Detection Using CNN*
