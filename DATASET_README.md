# Dataset format and handling

The supplied training script expects two NumPy files in its working directory:

- `X.npy`: segmented PPG samples; expected shape `(2892, 300, 1)`
- `y.npy`: integer class labels; expected shape `(2892,)`

These files are not included in this public repository. PPG recordings are physiological and potentially identifying biometric data. Only publish data when you have the rights and explicit permission to do so.

## Preprocessing described in the report

The academic report describes the following preprocessing stages:

1. Noise filtering
2. Baseline correction
3. Z-score normalization
4. Segmentation into fixed windows of 300 samples

The original preprocessing implementation was not recovered. This file documents the reported methodology but does not claim to reproduce it.
