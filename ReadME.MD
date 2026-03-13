# Structural Damage Detection Using 1D Convolutional Autoencoder

A deep learning framework for **structural health monitoring (SHM)** and **damage detection** in structures (wind turbines, dams, PZT-instrumented panels) using 1D Convolutional Autoencoders (Conv1D-AE) with PyTorch. The pipeline covers signal preprocessing, unsupervised feature extraction, damage classification, and regression-based damage quantification.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Results](#results)
- [Dependencies](#dependencies)
- [License](#license)

---

## Project Overview

This project implements an end-to-end deep learning pipeline for **non-destructive structural damage detection** using guided ultrasonic wave signals captured by PZT sensors. The framework:

1. **Preprocesses** raw FEM/experimental vibration and wave propagation signals
2. **Extracts features** using wavelet transform (CWT) and signal processing
3. **Trains a 1D Convolutional Autoencoder** to learn compact latent representations of healthy/damaged signals
4. **Classifies** damage categories (undamaged vs. various damage levels) using the AE bottleneck features
5. **Quantifies** damage severity via regression on the latent space

The system is validated on both **FEM-simulated** and **experimental** datasets involving off-center and on-center debonding damage scenarios at multiple SNR levels (55, 60, 65 dB).

---

## Repository Structure

```
.
├── ConvAE_BNL_4.py                      # Conv1D-AE with 4-dim bottleneck layer
├── ConvAE_BNL_8.py                      # Conv1D-AE with 8-dim bottleneck layer
├── ConvAE_BNL_16.py                     # Conv1D-AE with 16-dim bottleneck layer
│
├── Dam_cat_1DCNN_AE_TS150ms.py          # AE training on 150ms time-series features
├── Dam_cat_1DCNN_AE_TS150ms_cls.py      # AE + downstream classifier (damage category)
├── Dam_cat_1DCNN_AE_TS150ms_reg.py      # AE + downstream regressor (damage size)
├── Dam_cat_1DCNN_AE_TS150ms_reg-2.py   # AE + regressor (alternate config)
│
├── Dam_cat_class.py                     # Damage classification (standalone CNN)
├── Dam_cat_class_TS150ms.py            # Damage classification on 150ms TS data
├── Dam_cat_class-2.py                  # Damage classification (variant)
├── Dam_cat_reg.py                      # Damage regression (standalone)
├── Dam_Cat_file_handle.py              # Data file I/O utilities
│
├── PyTorch_stiffned_damage_autoemcoder_1.py  # AE for stiffness-based damage detection
├── Pytorch_reg_trial.py                # Regression trial with PyTorch
│
├── Vib_data_1DCNN_autoemcoder.py       # AE trained on raw vibration data
├── Vib_data_1DCNN_autoemcoder_class.py # AE + classifier on vibration data
├── Vib_data_autoemcoder_1.py           # Vibration AE baseline
│
├── WT_dam_model_exp.py                 # Wind turbine damage — experimental data
├── WT_dam_model_fem.py                 # Wind turbine damage — FEM simulation data
├── WT_dam_model_fem-2.py              # WT FEM model (alternate preprocessing)
├── WT_off_center_dam_model.py          # Off-center damage model for WT
├── WT_off_dam_pzt_model.py             # PZT-based off-center damage model
│
├── Scatter_anl_WT.py                   # Scatter analysis and wave propagation plots
├── SVR_Trial.py                        # Support Vector Regression baseline
├── load_save_model.py                  # Model checkpoint save/load utilities
│
├── train_loss_fcl_4.txt                # Training loss log (bottleneck=4)
└── ReadME.MD                           # This file
```

---

## Architecture

### 1D Convolutional Autoencoder (Conv1D-AE)

The core model follows an **encoder–bottleneck–decoder** architecture operating on 1D signal windows of length 41.

```
Input (1 × 41)
    │
    ▼
┌─────────────────────────────────┐
│  ENCODER                        │
│  Conv1d(1→32, k=4, s=2) + BN   │
│  Conv1d(32→64, k=4, s=2) + BN  │
│  Conv1d(64→128, k=4, s=1) + BN │
│  Flatten → FC(640→320→80→N)     │
│  Bottleneck: N = {4, 8, 16}     │
└─────────────────────────────────┘
    │
    ▼  Latent Space (N-dim features)
    │
┌─────────────────────────────────┐
│  DECODER                        │
│  FC(N→16→80→320→640)            │
│  Reshape → ConvTranspose1d ×3   │
│  Output (1 × 41)                │
└─────────────────────────────────┘
```

Three bottleneck variants are provided — **4-dim**, **8-dim**, and **16-dim** — allowing comparison of reconstruction quality vs. compression ratio.

### Downstream Tasks

After training the autoencoder, the **bottleneck features** are used as inputs to:

| Task | Model | Output |
|---|---|---|
| Damage Classification | FCN / CNN classifier | Damage category label |
| Damage Quantification | FC Regression network | Damage size (mm) |
| Anomaly Detection | Reconstruction error | Damage / No-damage score |

---

## Dataset

The models are trained and evaluated on:

| Data Source | Description |
|---|---|
| `X_FEATURE_snr55_60_65.mat` | Extracted signal features at SNR 55, 60, 65 dB |
| FEM `.rpt` files | ABAQUS FEM simulation output (displacement time-series) |
| Experimental data | PZT sensor recordings on physical specimens |

**Damage scenarios include:**
- Undamaged (healthy baseline)
- On-center debonding at 10 mm
- Off-center debonding at 10 mm (at positions 100, 120, 140, 160, 170, 180 mm)

Signal features are extracted using **Continuous Wavelet Transform (CWT)** with 8 voices per octave over 10–600 kHz.

---

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.10+
- CUDA (optional, for GPU acceleration)

### Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install numpy scipy matplotlib scikit-learn
pip install obspy mlxtend pandas
```

Or install all at once:

```bash
pip install -r requirements.txt
```

### requirements.txt

```
torch>=1.10.0
numpy>=1.21
scipy>=1.7
matplotlib>=3.4
scikit-learn>=0.24
obspy>=1.3
mlxtend>=0.19
pandas>=1.3
```

---

## Usage

### 1. Train the Autoencoder

Select the desired bottleneck size (4, 8, or 16):

```bash
python Dam_cat_1DCNN_AE_TS150ms.py
```

This will:
- Load `X_FEATURE_snr55_60_65.mat`
- Train the Conv1D-AE for 15,000 epochs
- Plot training loss (log scale)
- Visualize reconstructed signals

### 2. Train Damage Classifier

After training and saving the AE model weights:

```bash
python Dam_cat_1DCNN_AE_TS150ms_cls.py
```

This loads the pretrained AE (`AE_CF_16_snr55_60_65-2.pth`), extracts bottleneck features, and trains a downstream fully-connected classifier. Outputs include confusion matrices and per-class accuracy.

### 3. Train Damage Regression Model

```bash
python Dam_cat_1DCNN_AE_TS150ms_reg.py
```

Trains a regression head on top of AE features to predict damage size in mm.

### 4. Save and Load Models

```python
# Save
torch.save(AE_model.state_dict(), 'AE_CF_16_snr55_60_65-2.pth')

# Load
from ConvAE_BNL_16 import Conv1D_AE_BNL_16
model = Conv1D_AE_BNL_16()
model.load_state_dict(torch.load('AE_CF_16_snr55_60_65-2.pth'))
model.eval()
```

See `load_save_model.py` for full utilities.

### 5. Signal Preprocessing (FEM/Experimental)

For wavelet transform processing of FEM `.rpt` output files:

```bash
python WT_dam_model_fem.py
```

---

## Models

| File | Bottleneck | Purpose |
|---|---|---|
| `ConvAE_BNL_4.py` | 4-dim | Maximum compression |
| `ConvAE_BNL_8.py` | 8-dim | Balanced compression |
| `ConvAE_BNL_16.py` | 16-dim | Best reconstruction quality |

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.005 |
| Weight Decay | 1e-5 |
| Loss Function | MSELoss |
| Batch Size | 120 |
| Max Epochs | 15,000 |

---

## Results

Training loss is logged in `train_loss_fcl_4.txt`. The autoencoder achieves low reconstruction error on healthy signals; damaged signals produce **elevated reconstruction error**, forming the basis for anomaly detection.

Downstream classifiers achieve high accuracy across damage categories when trained on the 16-dim bottleneck features. Regression models predict debonding sizes with low mean absolute error.

---

## Dependencies

| Library | Purpose |
|---|---|
| `torch` | Deep learning framework |
| `numpy` | Numerical computation |
| `scipy` | Signal processing, `.mat` file loading |
| `matplotlib` | Visualization |
| `obspy` | Continuous Wavelet Transform (CWT) |
| `scikit-learn` | PCA, SVR baseline models |
| `mlxtend` | Confusion matrix plotting |
| `pandas` | Data handling for FEM outputs |

---

## Citation

If you use this code in your research, please cite accordingly.

---

## License

This project is for research and academic use. Please contact the authors for commercial use inquiries.
