# Group 7 — AI-Driven Deepfake Detection System

**Module:** AI Systems Engineering (CMP-L044) — Part 2 Artefact  
**Group:** Group 7  
**Members:** Isha Luhar (A00085061) · Nishtha Solanki (A00087199) · Om Mistry (A00067376) · Rushitkumar Patel (A00085504)

---

## Overview

This repository contains the full implementation of an end-to-end deepfake detection pipeline, built as part of the CMP-L044 Part 2 assessment. The system detects AI-generated (deepfake) content in both video frames and audio clips using a multi-modal deep learning approach.

The pipeline combines:
- **EfficientNet-B4 + Temporal Transformer** for video frame classification
- **ResNet-18 on log-Mel spectrograms** for audio classification
- **Late fusion meta-learner** (calibrated logistic regression) combining both modality scores
- **Grad-CAM explainability** for human review of borderline predictions
- **MLflow** experiment tracking for full reproducibility

---

## Key Results

| Metric | Video Model | Audio Model | Target (Part 1 NFR) |
|--------|------------|-------------|----------------------|
| AUC | **0.9321** | **0.9999** | ≥ 0.92 |
| Accuracy | 0.8445 | 0.9924 | — |
| FPR | 0.2015 | 0.0000 | ≤ 0.03 |
| F1 Score | 0.8455 | 0.9953 | — |

**Latency (video branch, T4 GPU, 50 runs):**  
Mean: 94.7 ms · P50: 87.0 ms · P95: 143.1 ms · Max: 154.4 ms · NFR target: ≤ 500 ms 

**Robustness (video model AUC under perturbation):**  
Gaussian noise: 0.9089 · Low brightness: 0.9065 · JPEG compression (q=30): 0.8845

**Fairness (demographic parity difference, ITA proxy):** 0.0152 — meets NFR target of ≤ 0.05 

> **Note on fusion pipeline:** The late fusion AUC of 0.5000 is a known dataset alignment bug (video validation set: 2,469 samples; audio validation set: 527 samples). The fusion architecture itself is sound — see Section 5.3 of the report for full discussion.

---

## Repository Structure

```
Group_7_Deepfake_Detection/
├── Group_7_Deepfake_Detection.ipynb   # Main notebook (run top-to-bottom)
├── requirements.txt                    # Pinned Python dependencies
└── README.md                           # This file
```

### Notebook Sections

| Section | Lead | Description |
|---------|------|-------------|
| Section 0 | Rushitkumar Patel (A00085504) | Environment setup, dataset download, global config |
| Section 1 | Isha Luhar (A00085061) | Data ingestion, MTCNN preprocessing, Mel spectrograms |
| Section 2 | Nishtha Solanki (A00087199) | Video pipeline — EfficientNet-B4 + Transformer |
| Section 3 | Nishtha Solanki (A00087199) | Audio pipeline — ResNet-18 on Mel spectrograms |
| Section 4 | Rushitkumar Patel (A00085504) | Late fusion meta-learner + Grad-CAM explainability |
| Section 5 | Om Mistry (A00067376) | Evaluation, robustness, fairness, MLflow logging |

---

## Getting Started

### Requirements

- Google Colab with **T4 GPU** (Runtime → Change runtime type → T4 GPU)
- A Kaggle account with a **Legacy API key** (`kaggle.json`)

### Datasets

The notebook automatically downloads two Kaggle datasets:

| Modality | Dataset | Kaggle ID |
|----------|---------|-----------|
| Video | Deep Fake Detection (Cropped) | `ucimachinelearning/deep-fake-detection-cropped-dataset` |
| Audio | Audio Deepfake Detection | `adarshsingh0903/audio-deepfake-detection-dataset` |

These are proxy datasets used in place of the full DFDC and ASVspoof 2021 datasets, which are too large for free Colab sessions.

### Running the Notebook

1. Open the notebook in Google Colab using the badge at the top of this README.
2. Set the runtime to **T4 GPU**.
3. In **Section 0**, upload your `kaggle.json` file when prompted.
4. Run all cells **strictly top to bottom** — sections are interdependent.
5. If Colab disconnects at any point, re-run from Section 0 (packages reset on each session restart).

> **Demo mode:** If you do not have Kaggle access, set `DEMO_MODE = True` in Section 0. The pipeline will run on synthetic random data so you can verify the code structure without real datasets. Performance metrics in demo mode are not meaningful.

---

## Architecture

### Video Pipeline

```
Input Frame (224×224 RGB)
    └─► MTCNN Face Detector (graceful fallback to centre-crop)
        └─► EfficientNet-B4 (ImageNet pretrained, timm)
            └─► Temporal Transformer Encoder (2-layer, 8-head)
                └─► P(fake_video) ∈ [0, 1]
```

- **EfficientNet-B4** acts as a spatial feature extractor per frame
- **Temporal Transformer** captures inter-frame consistency patterns (blink artefacts, temporal flicker)
- Wang et al. (2022) showed this hybrid achieves within 2.1% AUC of full ViT at 61% lower compute

### Audio Pipeline

```
Input Audio (WAV, 16 kHz)
    └─► Log-Mel Spectrogram (128 bins, 25 ms window, 10 ms hop)
        └─► ResNet-18 (single-channel adapted, ImageNet pretrained)
            └─► P(fake_audio) ∈ [0, 1]
```

- First convolutional layer adapted from 3-channel RGB to 1-channel by averaging pretrained weights
- Targets spectral smoothing artefacts and phase discontinuities characteristic of TTS systems

### Fusion & Explainability

```
P(fake_video) + P(fake_audio)
    └─► Calibrated Logistic Regression (Platt scaling)
        └─► P(fake_fused) ∈ [0, 1]
            └─► Label + Confidence Score
                └─► [0.4–0.6] → flag_for_human_review (EU AI Act compliance)
```

- **Grad-CAM** heatmaps are generated for every prediction to support human oversight
- The confidence-based flagging mechanism aligns with EU AI Act high-risk AI system requirements

---

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimiser | AdamW |
| Learning rate | 1e-4 |
| LR schedule | Cosine annealing |
| Weight decay | 1e-4 |
| Label smoothing | 0.1 |
| Gradient clipping | max norm = 1.0 |
| Epochs (video) | 5 |
| Epochs (audio) | 5 |
| Batch size | 16 |
| Image size | 224 × 224 |
| Random seed | 42 |

---

## Reproducibility

Reproducibility was treated as a first-class requirement:

- **Random seed 42** set across Python, NumPy, PyTorch, and CUDA
- **Dataset splits are deterministic** — `create_splits()` moves files rather than sampling, so splits are identical on every run
- **MLflow** tracks all hyperparameters, dataset versions, and metrics under experiment `Group7_Deepfake_Detection`
- **Model checkpoints** saved to Google Drive whenever validation AUC improves

To reproduce any result in the report, run the notebook from top to bottom on a T4 GPU with the same Kaggle datasets.

---

## Limitations

- **Fusion AUC = 0.5000**: Dataset alignment bug — audio and video validation sets have different sizes. See Section 5.3 of the report. Fix requires a matched audio-visual dataset (e.g., FakeAVCeleb).
- **Video FPR = 0.2015**: Does not meet the NFR target of ≤ 0.03. Likely due to limited training data (≈11,500 frames) and short training (5 epochs). Hard negative mining and focal loss are proposed improvements.
- **Audio dataset imbalance**: 100 real vs 427 fake samples (4:1 ratio). FPR of 0.0000 may partially reflect the small real sample count.
- **Proxy datasets**: The Kaggle datasets used here are smaller alternatives to the benchmark DFDC and ASVspoof 2021 datasets. Generalisation to those benchmarks is not guaranteed.
- **Colab dependency**: Full execution requires a Kaggle account and Colab T4 GPU. Future work would package the pipeline as a Docker image.

