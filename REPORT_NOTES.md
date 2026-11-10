# Academic report notes

The accompanying academic report is titled **Analog Biometric Recognition System**. It describes PPG-based user identification using a 1D CNN.

## Reported methodology

- PPG signals are acquired and processed as time series.
- Preprocessing is described as noise filtering, baseline correction and z-score normalization.
- Signals are segmented into windows of 300 samples.
- The described CNN uses Conv1D layers with 32 and 64 filters, max pooling, flattening, a 128-unit dense layer and a softmax output.
- The report records approximately 92–93% test accuracy and discusses training/validation curves with minor overfitting.
- The report also documents signal observations using an ESP32-connected PPG sensor and a Doppler motion-sensing module.

## Configuration discrepancy

The recovered Python script and the report do not fully agree:

| Detail | Recovered training script | Report appendix |
|---|---|---|
| Output classes | 2 | 3 |
| Dropout | Not present | Dropout 0.5 |
| Loss | Sparse categorical cross-entropy | Categorical cross-entropy |
| Training batch size | 16 | 32 |
| Validation | 20% validation split from training data | Validation data shown as test split in appendix |

The source script has been preserved as supplied. The configuration should be reconciled against the original dataset and experiment files before claiming exact reproduction.

## Reproducibility status

The original preprocessing code and original data files were not recovered. The reported accuracy is therefore attributed to the academic report and is not independently reproduced by this repository.
