# Time-Series Signal Processing & Biometric Authentication

A PPG-based biometric identification project using a one-dimensional Convolutional Neural Network (1D CNN). The project report describes a workflow covering signal acquisition, preprocessing, segmentation, classification and evaluation.

## Project overview
Photoplethysmography (PPG) is a non-invasive optical measurement that captures blood-volume-related changes over time. This project explores using temporal patterns in PPG waveforms for multi-class user identification.

The accompanying report describes preprocessing with noise filtering, baseline correction and z-score normalization, followed by segmentation into fixed windows of 300 samples. **The original preprocessing implementation was not recovered**, so this repository documents the reported method but does not claim to reproduce that preprocessing stage.

## CNN model
The training script in `train_ppg_cnn.py` loads already-segmented arrays and trains a 1D CNN with:

- Input shape: `(300, 1)`
- Conv1D: 32 filters, kernel size 3, ReLU
- Max pooling: pool size 2
- Conv1D: 64 filters, kernel size 3, ReLU
- Max pooling: pool size 2
- Flatten
- Dense: 128 units, ReLU
- Softmax output layer

The supplied training script uses a **two-class output layer** and `sparse_categorical_crossentropy`. The project report's architecture diagram and appendix instead describe a **three-class output layer**, dropout (0.5), and `categorical_crossentropy`. These are inconsistent project artifacts. The supplied script is preserved as received; do not assume it reproduces the reported three-class experiment without resolving the configuration and data labels.

## Dataset format
The script expects two files in the repository root (or run directory):

- `X.npy`: segmented PPG samples with expected shape `(2892, 300, 1)`
- `y.npy`: integer class labels with expected shape `(2892,)`

The actual dataset is not included because PPG is biometric data. Do not upload identifiable or sensitive raw physiological recordings to a public repository. See [`DATASET_README.md`](DATASET_README.md) for details.

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
The accompanying academic report records an experimental accuracy of approximately **92–93%** and describes stable training/validation accuracy curves with minor overfitting. The exact metric values are not embedded in the supplied training script, and the original data and preprocessing code are unavailable here. Treat 92–93% as the result reported in the project document, not as a metric independently reproduced from this repository.

## Repository contents

```text
.
├── README.md
├── train_ppg_cnn.py
├── DATASET_README.md
├── REPORT_NOTES.md
├── requirements.txt
└── .gitignore
```

## Scope and limitations
- This repository preserves the provided CNN training script.
- The original signal filtering, baseline-correction and z-score preprocessing code was not recovered.
- The report and supplied script disagree on the number of output classes and training configuration.
- Dataset-level reproduction and independent verification of the reported accuracy require the original processed data and exact experiment configuration.
- The report discusses ESP32-based PPG acquisition and a Doppler motion-sensing observation; this repository's runnable Python entry point focuses on CNN training from pre-segmented PPG arrays.

## Citation / attribution
The companion report is summarized in `REPORT_NOTES.md`. The original PDF is not included in this repository at present.
