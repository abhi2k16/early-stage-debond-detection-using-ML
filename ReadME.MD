# Structural Damage Detection Using 1D Convolutional Autoencoder

A deep learning framework for **structural health monitoring (SHM)** and **damage detection** in PZT-instrumented panels using 1D Convolutional Autoencoders (Conv1D-AE) with PyTorch. The pipeline covers signal preprocessing via CWT, unsupervised feature extraction, damage classification, and regression-based damage quantification.

---

## Repository Structure

```
.
├── ConvAE_BNL_4.py                      # Conv1D-AE model definition — 4-dim bottleneck
├── ConvAE_BNL_8.py                      # Conv1D-AE model definition — 8-dim bottleneck
├── ConvAE_BNL_16.py                     # Conv1D-AE model definition — 16-dim bottleneck
│
├── Dam_cat_1DCNN_AE_TS150ms.py          # AE training script (inline model, 8-dim BNL)
├── Dam_cat_1DCNN_AE_TS150ms_cls.py      # AE (16-dim) + downstream classifiers (SVM, RF, AdaBoost, GBC, Stack, Vote)
├── Dam_cat_1DCNN_AE_TS150ms_reg.py      # AE (snr35_40) + downstream regressors (SVR, RF, AdaBoost, GBR, Stack, Vote)
├── Dam_cat_1DCNN_AE_TS150ms_reg-2.py   # AE (16-dim, snr55_60_65) + downstream regressors
│
├── Dam_cat_class.py                     # Standalone FC classifier on X_feature.mat (177 samples, 4 classes)
├── Dam_cat_class_TS150ms.py            # FC classifier on X_SubSam_TimeSegment data (1344 samples, 4 classes)
├── Dam_cat_class-2.py                  # FC classifier variant on X_feature.mat (same architecture as TS150ms)
├── Dam_cat_reg.py                      # FC regression on X_feature.mat (MSELoss, SGD)
├── Dam_Cat_file_handle.py              # Loads damage category Excel files and computes FFT
│
├── CAE_PCA_compare.py                  # Compares AE (4/8/16-dim) vs PCA reconstruction MSE
│
├── PyTorch_stiffned_damage_autoemcoder_1.py  # FC autoencoder (416→25→416) on StiffenDebondData2.csv
├── Pytorch_reg_trial.py                # Minimal PyTorch linear regression demo (sin curve)
│
├── Vib_data_autoemcoder_1.py           # FC AE (416→25) on vibration CSV data
├── Vib_data_1DCNN_autoemcoder.py       # Conv1D AE (416-len signals) on vibration CSV data
├── Vib_data_1DCNN_autoemcoder_class.py # Conv1D AE + downstream 17-class classifier head
│
├── WT_dam_model_fem.py                 # CWT scalogram of FEM .rpt signal (3-column, 140 kHz)
├── WT_dam_model_fem-2.py              # CWT scalogram of FEM .rpt signal (2-column variant)
├── WT_off_center_dam_model.py          # CWT + Hilbert transform + debonding location estimation
├── WT_off_dam_pzt_model.py             # CWT scalogram of off-center PZT FEM data
├── WT_dam_model_exp.py                 # CWT scalogram of experimental PZT scope data (.xlsx)
│
├── Scatter_anl_WT.py                   # Scatter signal analysis: CWT + WTC plot from .rpt files
├── SVR_Trial.py                        # SVR baseline demo (RBF / Linear / Poly kernels on sin data)
├── load_save_model.py                  # Load pretrained Conv1D_AE from AE_model.pth
│
├── AE_CF_4_snr55_60_65*.pth            # Saved weights — BNL-4 model
├── AE_CF_8_snr55_60_65*.pth            # Saved weights — BNL-8 model
├── AE_CF_16_snr55_60_65*.pth           # Saved weights — BNL-16 model
├── AE_model*.pth                       # Saved weights — other AE variants
│
├── X_FEATURE_snr55_60_65.mat           # CWT features at SNR 55/60/65 dB (3600 × 41)
├── X_FEATURE_snr35_40.mat              # CWT features at SNR 35/40 dB (2688 × 41)
├── X_FEATURE_snr100_50.mat             # CWT features at SNR 100/50 dB
├── X_FEATURE_snr30_35.mat              # CWT features at SNR 30/35 dB
├── X_feature.mat                       # Raw CWT features (177 samples × 41)
├── X_SubSam_TimeSegment_norm_feature*.mat  # Sub-sampled time-segment features
├── X_SubSam_TimeSegment_target.mat     # Targets for time-segment data
├── Targ_reg_norm*.mat                  # Normalized regression targets (debonding size)
├── T_reg.mat / Targ_reg.mat            # Raw regression targets
├── StiffenDebondData2.csv              # Stiffness-based debond dataset (5434 × 417)
│
└── train_loss_fcl_4.txt                # Training loss log (BNL-4 run)
```

---

## Architecture

### 1D Convolutional Autoencoder (Conv1D-AE)

Encoder–bottleneck–decoder operating on 1D signal windows of length 41.

```
Input (1 × 41)
    │
    ▼
┌─────────────────────────────────────┐
│  ENCODER                            │
│  Conv1d(1→32, k=4, s=2) + BN       │
│  Conv1d(32→64, k=4, s=2) + BN      │
│  Conv1d(64→128, k=4, s=1) + BN     │
│  Flatten → FC(640→320→80→16→N)     │
│  Bottleneck: N ∈ {4, 8, 16}        │
└─────────────────────────────────────┘
    │  Latent Space (N-dim)
    ▼
┌─────────────────────────────────────┐
│  DECODER                            │
│  FC(N→16→80→320→640) + Tanh        │
│  Reshape → ConvTranspose1d ×3       │
│  Output (1 × 41)                    │
└─────────────────────────────────────┘
```

Three bottleneck variants: **4-dim**, **8-dim**, **16-dim**.

### Downstream Tasks

Bottleneck features feed into:

| Task | Script | Classifiers / Models |
|---|---|---|
| Damage Classification | `Dam_cat_1DCNN_AE_TS150ms_cls.py` | SVM, RF, AdaBoost, GBC, Stacking, Voting |
| Damage Regression | `Dam_cat_1DCNN_AE_TS150ms_reg.py` / `reg-2.py` | SVR, RFR, AdaBoost, GBR, Stacking, Voting |
| Anomaly Detection | Reconstruction MSE | Elevated error → damage |
| AE vs PCA | `CAE_PCA_compare.py` | MSE comparison across component counts |

---

## Dataset

| File | Description |
|---|---|
| `X_FEATURE_snr55_60_65.mat` | 3600 × 41 CWT features, SNR 55/60/65 dB |
| `X_FEATURE_snr35_40.mat` | 2688 × 41 CWT features, SNR 35/40 dB |
| `X_feature.mat` | 177 × 41 features (initial small dataset) |
| `StiffenDebondData2.csv` | 5434 samples × 416 features, stiffness-based debond |
| FEM `.rpt` files | ABAQUS displacement time-series output |
| Experimental `.xlsx` | PZT scope recordings on physical specimens |

**Damage scenarios:**
- Undamaged (healthy baseline)
- On-center debonding at 10 mm
- Off-center debonding at 10 mm (positions: 100, 120, 140, 160, 170, 180, 215 mm)

Signal features extracted via **CWT** (obspy), 8 voices/octave, 10–600 kHz.

---

## Installation

```bash
pip install torch torchvision torchaudio
pip install numpy scipy matplotlib scikit-learn obspy mlxtend pandas openpyxl
```

---

## Usage

### 1. Train the Autoencoder

```bash
python Dam_cat_1DCNN_AE_TS150ms.py
```

Loads `X_FEATURE_snr55_60_65.mat`, trains Conv1D-AE for 15,000 epochs (Adam, lr=0.005, batch=120), plots log-scale training loss and reconstructed signals.

### 2. Train Damage Classifier (AE features → SVM / RF / ensemble)

```bash
python Dam_cat_1DCNN_AE_TS150ms_cls.py
```

Loads pretrained `AE_CF_16_snr55_60_65-2.pth`, extracts 16-dim bottleneck features via forward hook, trains and evaluates SVM, RF, AdaBoost, GBC, Stacking, and Voting classifiers. Outputs confusion matrices, ROC curves, and precision-recall curves.

### 3. Train Damage Regression Model

```bash
python Dam_cat_1DCNN_AE_TS150ms_reg-2.py   # uses snr55_60_65 + AE-16 features
python Dam_cat_1DCNN_AE_TS150ms_reg.py     # uses snr35_40 features
```

Trains SVR, RFR, AdaBoost, GBR, Stacking, and Voting regressors. Reports R² scores and plots predicted vs. actual debonding size.

### 4. Compare AE vs PCA Reconstruction

```bash
python CAE_PCA_compare.py
```

Computes per-sample MSE for AE (4/8/16-dim) and PCA (4/15 components) and plots average reconstruction error vs. number of components.

### 5. Signal Preprocessing (CWT Scalogram)

```bash
python WT_dam_model_fem.py        # FEM .rpt → CWT scalogram
python WT_dam_model_exp.py        # Experimental .xlsx → CWT scalogram
python WT_off_center_dam_model.py # CWT + Hilbert + debonding location estimate
python Scatter_anl_WT.py          # Scatter signal CWT analysis
```

### 6. Save / Load Model

```python
# Save
torch.save(AE_model.state_dict(), 'AE_CF_16_snr55_60_65-2.pth')

# Load
from ConvAE_BNL_16 import Conv1D_AE_BNL_16
model = Conv1D_AE_BNL_16()
model.load_state_dict(torch.load('AE_CF_16_snr55_60_65-2.pth'))
model.eval()
```

See `load_save_model.py` for a self-contained example.

---

## Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.005 |
| Weight Decay | 1e-5 |
| Loss | MSELoss |
| Batch Size | 120 |
| Epochs | 15,000 |

---

## Dependencies

| Library | Purpose |
|---|---|
| `torch` | Deep learning (AE, classifiers, regressors) |
| `numpy` | Numerical computation |
| `scipy` | `.mat` file loading, signal processing |
| `matplotlib` | Visualization |
| `obspy` | Continuous Wavelet Transform (CWT) |
| `scikit-learn` | PCA, SVM, RF, ensemble models, metrics |
| `mlxtend` | Confusion matrix plotting |
| `pandas` | FEM `.rpt` / `.xlsx` data loading |

---
- Original paper: [Early-Stage Debonding Detection and Assessment]..(https://doi.org/10.1115/1.4064612)
## License
This project is for research and academic use. Please contact the authors for commercial use inquiries.
