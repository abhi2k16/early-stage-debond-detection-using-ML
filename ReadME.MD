# Structural Damage Detection Using 1D Convolutional Autoencoder

A deep learning framework for **structural health monitoring (SHM)** and **damage detection** in PZT-instrumented panels using 1D Convolutional Autoencoders (Conv1D-AE) with PyTorch. The pipeline covers signal preprocessing via CWT, unsupervised feature extraction, damage classification, and regression-based damage quantification.

---

## Repository Structure

```
.
├── main.py                              # Pipeline entry point — runs all steps or individual steps via --step flag
│
├── roc_auc_utils.py                     # Shared ROC/AUC and Precision-Recall utility functions
│
├── ConvAE_BNL_4.py                      # Conv1D-AE model definition — 4-dim bottleneck
├── ConvAE_BNL_8.py                      # Conv1D-AE model definition — 8-dim bottleneck
├── ConvAE_BNL_16.py                     # Conv1D-AE model definition — 16-dim bottleneck
│
├── Dam_cat_1DCNN_AE_TS150ms.py          # Conv1D-AE training script (8-dim BNL, snr55_60_65)
├── Dam_cat_1DCNN_AE_TS150ms_cls.py      # AE (16-dim) + downstream classifiers (SVM, RF, AdaBoost, GBC, Stack, Vote)
├── Dam_cat_1DCNN_AE_TS150ms_reg.py      # AE (snr35_40) + downstream regressors (SVR, RF, AdaBoost, GBR, Stack, Vote)
│
├── Dam_cat_class.py                     # Standalone FC classifier on X_feature.mat (177 samples, 4 classes)
│
├── CAE_PCA_compare.py                   # Compares AE (4/8/16-dim) vs PCA reconstruction MSE
│
├── AE_CF_4_snr55_60_65*.pth            # Saved weights — BNL-4 model
├── AE_CF_8_snr55_60_65*.pth            # Saved weights — BNL-8 model
├── AE_CF_16_snr55_60_65*.pth           # Saved weights — BNL-16 model
├── AE_model*.pth                        # Saved weights — other AE variants
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

## Module Responsibilities

### Entry Point

| File | Role |
|---|---|
| `main.py` | Orchestrates the full pipeline. Accepts `--step` argument to run individual stages or all at once. |

### Model Definitions (importable, no side effects)

| File | Class | Bottleneck |
|---|---|---|
| `ConvAE_BNL_4.py` | `Conv1D_AE_BNL_4` | 4-dim |
| `ConvAE_BNL_8.py` | `Conv1D_AE_BNL_8` | 8-dim |
| `ConvAE_BNL_16.py` | `Conv1D_AE_BNL_16` | 16-dim |

### Shared Utilities

| File | Exported Functions |
|---|---|
| `roc_auc_utils.py` | `ROC_Curve_avg`, `ROC_Curve_all`, `precision_recall_all`, `precision_recall_avg` |

### Pipeline Scripts (executable via `if __name__ == '__main__'`)

| File | Purpose | Key Classes / Functions |
|---|---|---|
| `Dam_cat_1DCNN_AE_TS150ms.py` | Trains Conv1D-AE (8-dim BNL) on `X_FEATURE_snr55_60_65.mat` | `Conv1D_AE`, `train`, `test_model` |
| `Dam_cat_1DCNN_AE_TS150ms_cls.py` | Loads AE-16 weights, extracts bottleneck features, trains and evaluates 6 classifiers | `nn_model`, `train_model`, `get_activation`, `evaluate` |
| `Dam_cat_1DCNN_AE_TS150ms_reg.py` | Loads AE weights (snr35_40), extracts features, trains and evaluates 6 regressors | `Conv1D_AE`, `nn_model_reg`, `train_model`, `get_activation` |
| `Dam_cat_class.py` | Simple FC classifier (41→20→10→4) on `X_feature.mat` | `nn_model`, `train`, `saveModel`, `testAccuracy` |
| `CAE_PCA_compare.py` | Loads all 3 AE variants, computes per-sample and average MSE vs PCA | `cal_mse_pca`, `cal_mse_pca_2`, `cal_mse`, `MSE` |

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
| Damage Regression | `Dam_cat_1DCNN_AE_TS150ms_reg.py` | SVR, RFR, AdaBoost, GBR, Stacking, Voting |
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

### Run the full pipeline

```bash
python main.py --step all
```

### Run individual steps

```bash
python main.py --step ae_train    # Step 1: Train Conv1D Autoencoder
python main.py --step classify    # Step 2: AE feature extraction + classification
python main.py --step regress     # Step 3: AE feature extraction + regression
python main.py --step compare     # Step 4: AE vs PCA reconstruction comparison
python main.py --step nn_class    # Step 5: Simple FC classifier on X_feature.mat
```

### Run scripts directly

#### 1. Train the Autoencoder

```bash
python Dam_cat_1DCNN_AE_TS150ms.py
```

Loads `X_FEATURE_snr55_60_65.mat`, trains Conv1D-AE (8-dim bottleneck) for 15,000 epochs (Adam, lr=0.005, batch=120), plots log-scale training loss, initial/trained/updated encoder weights, and reconstructed signals.

#### 2. Train Damage Classifiers (AE features → SVM / RF / ensemble)

```bash
python Dam_cat_1DCNN_AE_TS150ms_cls.py
```

Loads pretrained `AE_CF_16_snr55_60_65-2.pth`, extracts 16-dim bottleneck features via forward hook, trains and evaluates SVM, RF, AdaBoost, GBC, Stacking, and Voting classifiers. Outputs confusion matrices, ROC curves, and precision-recall curves via `roc_auc_utils`.

#### 3. Train Damage Regression Models

```bash
python Dam_cat_1DCNN_AE_TS150ms_reg.py
```

Loads `AE_model_snr35_40.pth`, extracts 16-dim features, trains SVR, RFR, AdaBoost, GBR, Stacking, and Voting regressors. Reports R² scores and plots predicted vs. actual debonding size.

#### 4. Compare AE vs PCA Reconstruction

```bash
python CAE_PCA_compare.py
```

Loads all three pretrained AE variants (BNL-4/8/16), computes per-sample MSE against PCA (4 and 15 components), and plots average reconstruction error vs. number of components.

#### 5. Simple FC Classifier

```bash
python Dam_cat_class.py
```

Trains a 3-layer FC network (41→20→10→4) on `X_feature.mat` (177 samples, 4 damage classes). Saves model to `dam_cat_classification.pth`.

### Save / Load Model

```python
# Save
torch.save(AE_model.state_dict(), 'AE_CF_16_snr55_60_65-2.pth')

# Load
from ConvAE_BNL_16 import Conv1D_AE_BNL_16
model = Conv1D_AE_BNL_16()
model.load_state_dict(torch.load('AE_CF_16_snr55_60_65-2.pth'))
model.eval()
```

### Using ROC/AUC utilities

```python
from roc_auc_utils import ROC_Curve_avg, ROC_Curve_all, precision_recall_all, precision_recall_avg
import matplotlib.pyplot as plt

# Per-classifier full ROC plot
ROC_Curve_all(y_true_ohe, y_prob, clf_name='SVM')

# Overlay multiple classifiers on one ROC plot
for name, prob, color in zip(clf_names, y_probs, colors):
    ROC_Curve_avg(y_true_ohe, prob, name=name, color=color)
plt.show()
```

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

- Original paper: [Early-Stage Debonding Detection and Assessment](https://doi.org/10.1115/1.4064612)

## License
This project is for research and academic use. Please contact the authors for commercial use inquiries.
