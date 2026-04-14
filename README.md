# Group2 Project (SPH6004 AY2526 Sem2)

## Overview
This repository contains ICU prediction pipelines built on MIMIC-IV derived data.  
The project focuses on:
- Task 1: Predict whether a patient can be discharged from ICU within the next 24 hours.
- Task 2: Predict whether the patient will **not** return to ICU within 72 hours after discharge.

The main recommended baseline is the **XGBoost multi-task pipeline** (`model/xgboostmulti.py`).

## Project Structure
```text
.
├── dataset/                     # Input CSV files
│   ├── MIMIC-IV-static(Group Assignment).csv
│   ├── MIMIC-IV-text(Group Assignment).csv
│   └── MIMIC-IV-time_series(Group Assignment).csv
├── model/                       # Train/eval entry scripts
│   ├── xgboostmulti.py          # Multi-task XGBoost (recommended)
│   ├── logisticmulti.py         # Multi-task logistic baseline
│   └── 1.py                     # Quick xgboost version check
├── utils/
│   └── multitask_common.py      # Shared data processing and evaluation
└── result/                      # Output artifacts
    ├── xgboostmultiresult/
    └── logisticmultiresult/
```

## Requirements
- Python 3.11 (recommended)
- `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `joblib`, `xgboost`
- Optional: `shap` (only needed for SHAP plots in `xgboost111.py`)
- `git-lfs` (required to pull large files in `dataset/`)

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Swiaple/Group2_Project_SPH6004_AY2526Sem2.git
   cd Group2_Project_SPH6004_AY2526Sem2
   ```
2. Pull large dataset files (LFS):
   ```bash
   git lfs pull
   ```
3. Install Python dependencies (example):
   ```bash
   pip install numpy pandas scikit-learn matplotlib joblib xgboost
   ```

## Quick Start (XGBoost Multi-Task)
Run from project root:
```bash
python model/xgboostmulti.py
```

Outputs will be written to:
```text
result/xgboostmultiresult/
```

Key files include:
- `metrics_summary.csv`
- `test_predictions.csv`
- `train_val_loss_curve.csv`
- `run_config.json`

## Common Run Commands
XGBoost multi-task:
```bash
python model/xgboostmulti.py
```

Logistic multi-task:
```bash
python model/logisticmulti.py
```


## Optional Runtime Parameters
You can control training via environment variables.

Example (XGBoost multi-task):
```bash
DEBUG_MAX_STAYS=0 XGB_N_ESTIMATORS=200 XGB_LR=0.05 XGB_MAX_DEPTH=5 MODEL_N_JOBS=1 python model/xgboostmulti.py
```

Important notes:
- `DEBUG_MAX_STAYS=0` means full dataset.
- `DEBUG_MAX_STAYS>0` enables fast debug subsampling.

## Evaluation Notes
- Data split is group-aware by `subject_id` into train/val/test.
- Task 2 metrics are computed on masked samples only (`task2_mask == 1`), i.e., windows eligible for discharge interpretation.
- The scripts save confusion matrices, ROC curves, PR summaries, and probability outputs for reproducibility.

## Troubleshooting
- If `xgboost` import fails on macOS, install OpenMP runtime (`libomp`) and reinstall xgboost.
