# Time-Series Signal Processing & Biometric Authentication

A PPG-based biometric identification project using a one-dimensional Convolutional Neural Network (1D CNN). The academic report describes a workflow covering signal acquisition, preprocessing, segmentation, classification and evaluation.

## Project overview

Photoplethysmography (PPG) is a non-invasive optical measurement that captures blood-volume-related changes over time. This project explores using temporal patterns in PPG waveforms for multi-class user identification.

The report describes preprocessing with noise filtering, baseline correction and z-score normalization, followed by segmentation into fixed windows of 300 samples. **The original preprocessing implementation was not recovered**, so this repository documents the reported method but does not claim to reproduce that preprocessing stage.

## CNN model

The recovered training script, `train_ppg_cnn.py`, loads already-segmented arrays and trains a 1D CNN with:

- Input shape: `(300, 1)`
- Conv1D: 32 filters, kernel size 3, ReLU
- Max pooling: pool size 2
- Conv1D: 64 filters, kernel size 3, ReLU
- Max pooling: pool size 2
- Flatten
- Dense: 128 units, ReLU
- Softmax output layer

**Configuration discrepancy:** The supplied training script uses a two-class output layer and `sparse_categorical_crossentropy`. The academic report's architecture appendix instead describes a three-class output layer, dropout (0.5), and `categorical_crossentropy`. The supplied script is preserved as recovered; these artifacts should be reconciled against the original dataset and experiment files before claiming exact reproduction.

## Dataset format

The script expects two files in the working directory:

- `X.npy`: segmented PPG samples with expected shape `(2892, 300, 1)`
- `y.npy`: integer class labels with expected shape `(2892,)`

The data files are not included because PPG recordings are physiological and potentially identifying biometric data. Do not publish raw recordings without appropriate rights and permission. See [DATASET_README.md](DATASET_README.md) for details.

## Setup

Python 3.10–3.11 is recommended for broad TensorFlow compatibility.

```bash
python -m venv .venv
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Place `X.npy` and `y.npy` in the working directory, then run:

```bash
python train_ppg_cnn.py
```

The script performs an 80:20 stratified train/test split, reserves 20% of the training split for validation, trains for 20 epochs with batch size 16, prints accuracy, weighted precision/recall/F1, ROC-AUC, confusion matrix and classification report, then saves `PPG_CNN_Model.keras`.

## Reported result

The academic report records an experimental accuracy of approximately **92–93%** and describes stable training/validation accuracy curves with minor overfitting. The exact metric values are not embedded in the supplied training script, and the original data and preprocessing code are unavailable here. Treat 92–93% as the result reported in the project document, not as a metric independently reproduced from this repository.

A public text edition of the report, with the institutional email address removed, is available at [report/ppg_biometric_recognition_report.pdf](report/ppg_biometric_recognition_report.pdf). Additional methodology and reproducibility notes are in [REPORT_NOTES.md](REPORT_NOTES.md).

## Repository contents

```text
.
├── train_ppg_cnn.py
├── requirements.txt
├── DATASET_README.md
├── REPORT_NOTES.md
├── report/
│   └── ppg_biometric_recognition_report.pdf
└── .gitignore
```

## Scope and limitations

- This repository preserves the supplied CNN training script.
- The original signal filtering, baseline-correction and z-score preprocessing code was not recovered.
- The report and supplied script disagree on the number of output classes and training configuration.
- Dataset-level reproduction and independent verification of the reported accuracy require the original processed data and exact experiment configuration.
- The report discusses ESP32-based PPG acquisition and a Doppler motion-sensing observation; the runnable Python entry point focuses on CNN training from pre-segmented PPG arrays.
