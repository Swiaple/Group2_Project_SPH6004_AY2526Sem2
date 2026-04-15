# Model Execution Playbook

This document converts the three modeling ideas into executable implementation plans for this repository.

## Project Baseline and Constraints

- Current tasks:
  - Task 1: predict `label_can_discharge_24h`
  - Task 2: predict `label_no_return_72h` on `task2_mask == 1`
- Current baseline code:
  - `utils/multitask_common.py`
  - `model/xgboostmulti.py`
  - `model/logisticmulti.py`
- Current data:
  - `dataset/MIMIC-IV-static(Group Assignment).csv`
  - `dataset/MIMIC-IV-text(Group Assignment).csv`
  - `dataset/MIMIC-IV-time_series(Group Assignment).csv`

## Shared Preparation (Run Once)

1. Create or activate Python environment.
2. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn matplotlib joblib xgboost shap torch torchvision torchaudio transformers
   ```
3. Run baseline once for reference metrics:
   ```bash
   python model/xgboostmulti.py
   python model/logisticmulti.py
   ```
4. Save baseline metrics snapshot:
   - Copy `result/xgboostmultiresult/metrics_summary.csv`
   - Copy `result/logisticmultiresult/metrics_summary.csv`

---

## Plan A: XMI-ICU Pseudo-Dynamic Explainable XGBoost (Fastest to Land)

### Objective

- Upgrade current XGBoost pipeline to multi-horizon pseudo-dynamic prediction.
- Add time-resolved SHAP explanations.
- Keep full compatibility with existing split and evaluation style.

### Files to Add

- `utils/xmi_features.py`
- `model/xmi_icu_multihorizon.py`
- `result/xmi_icu_result/` (runtime outputs)

### Step-by-Step Execution

1. Implement XMI-style feature builder in `utils/xmi_features.py`.
   - Input: raw time-series table with `stay_id`, `hour_ts`.
   - For each stay and each landmark `t_end`, generate:
     - mean and std per variable in window `[0, t_end]`
     - optional trend slope per variable
     - missingness ratio per variable
2. Define prediction horizons in code:
   - `HORIZONS = [6, 12, 18, 24]`
3. Build labels per horizon in `model/xmi_icu_multihorizon.py`:
   - `y_discharge_h`: discharge within next `h` hours and alive
   - `y_death_h`: ICU death within next `h` hours
4. Train one XGBoost head per horizon and per target (or a compact looped head structure).
5. Reuse group-aware split by `subject_id` to avoid leakage.
6. Save evaluation outputs:
   - `metrics_by_horizon.csv`
   - `predictions_by_horizon.csv`
   - `roc_task1_h{h}.png`, `pr_task1_h{h}.png`
7. Add SHAP analysis:
   - generate top-20 feature importance per horizon
   - export `shap_top_features_h{h}.csv`
   - save summary plot `shap_summary_h{h}.png`
8. Run:
   ```bash
   python model/xmi_icu_multihorizon.py
   ```

### Acceptance Criteria

- Multi-horizon metrics are generated for all `6/12/18/24h`.
- SHAP outputs exist for every horizon.
- At least one horizon improves PR-AUC or AUC against existing 24h XGBoost baseline.

### Estimated Effort

- 1 to 2 days.

---

## Plan B: Pooled Landmark Cause-Specific Hazard Supermodel (Main Statistical Model)

### Objective

- Implement dynamic competing-risk modeling that explicitly separates:
  - safe ICU discharge event
  - ICU death event
- Produce clinically interpretable cumulative incidence outputs.

### Files to Add

- `utils/landmark_builder.py`
- `model/pooled_landmark_supermodel.py`
- `result/pooled_landmark_result/` (runtime outputs)

### Step-by-Step Execution

1. Build landmark dataset in `utils/landmark_builder.py`.
   - Landmarks: `24, 48, 72, ...` hours (or `24 + 12k` to reuse current step size).
   - Keep only rows still at risk at each landmark.
2. For each landmark row, create residual-time targets:
   - event type in prediction window: `discharge`, `death`, or `none`
   - optional horizon index (24h/48h/72h)
3. Build pooled supermodel design matrix:
   - predictors at landmark
   - landmark time term
   - key predictor-by-landmark interactions
4. Train cause-specific hazard heads:
   - Head 1: hazard of safe discharge
   - Head 2: hazard of ICU death
5. Convert hazards to cumulative incidence functions (CIF):
   - `CIF_discharge(t | landmark)`
   - `CIF_death(t | landmark)`
6. Export dynamic risk outputs:
   - `cif_curves_by_landmark.csv`
   - `prob_next_24_48_72h.csv`
7. Compare to non-pooled alternative:
   - fit separate per-landmark models
   - report variance and performance gap vs pooled supermodel
8. Run:
   ```bash
   python model/pooled_landmark_supermodel.py
   ```

### Acceptance Criteria

- Valid CIF outputs are generated for discharge and death.
- Competing-risk probabilities are calibrated and non-negative.
- Pooled model shows lower variance or better stability than separate-landmark models.

### Estimated Effort

- 2 to 4 days.

---

## Plan C: CMA-Surv Cross-Modal Attention Survival (Advanced Deep Model)

### Objective

- Build a multimodal deep survival model using:
  - time-series encoder
  - text encoder
  - static encoder
  - gated cross-modal attention
- Provide competing-risk survival probabilities and attention-based interpretability.

### Files to Add

- `utils/cma_dataset.py`
- `model/cma_surv.py`
- `model/cma_train.py`
- `result/cma_surv_result/` (runtime outputs)

### Step-by-Step Execution

1. Create multimodal sample builder in `utils/cma_dataset.py`.
   - For each `(stay_id, t_end)` create:
     - time tensor from last 24h (or fixed history length)
     - aggregated text up to `t_end`
     - static tabular vector
2. Text pipeline:
   - Start with lightweight tokenizer + embedding.
   - Optional upgrade path: ClinicalBERT encoder if GPU memory permits.
3. Model implementation in `model/cma_surv.py`:
   - Bi-LSTM (or GRU) for time-series branch
   - Text encoder branch
   - MLP for static branch
   - gated cross-modal attention fusion
   - discrete-time competing-risk output heads
4. Training loop in `model/cma_train.py`:
   - group-aware split by `subject_id`
   - early stopping by validation total loss
   - save best checkpoint
5. Evaluation outputs:
   - `metrics_summary.csv`
   - `cif_predictions.csv`
   - `attention_heatmap_examples.png`
   - `run_config.json`
6. Inference compatibility:
   - produce `P(discharge within 24h)` aligned with Task 1 reporting
7. Run:
   ```bash
   python model/cma_train.py
   ```

### Acceptance Criteria

- End-to-end training converges and saves valid checkpoints.
- Competing-risk outputs are produced for both discharge and death.
- Attention visualization is generated for at least 10 test examples.
- Performance is benchmarked against XGBoost and pooled landmark baselines.

### Estimated Effort

- 4 to 7 days.

---

## Recommended Rollout Order

1. Plan A (XMI-ICU): fastest and closest to current code.
2. Plan B (Pooled Landmark): strongest statistical main-model candidate for dynamic competing risk.
3. Plan C (CMA-Surv): highest ceiling, highest implementation and compute cost.

## Minimal Milestone Schedule

- Milestone 1: XMI-ICU multi-horizon + SHAP complete.
- Milestone 2: pooled landmark CIF pipeline complete.
- Milestone 3: CMA-Surv prototype complete.
- Milestone 4: unified comparison table and final report-ready figures.

---

## Appendix A: DOI 10.1145/3777577.3777712 Adaptation Add-on (2026-04-15)

### A.1 Confirmed paper metadata

- DOI: `10.1145/3777577.3777712`
- Title: `A Multimodal Deep Learning Framework for Predicting Cardiovascular Deterioration Based on MIMIC-IV Dataset`
- Venue: `ISAIMS 2025`
- Pages: `837-842`
- Data source mentioned: `MIMIC-IV`

### A.2 Access caveat

- ACM full text and PDF are blocked by Cloudflare challenge in the current runtime.
- This appendix therefore follows a "metadata-grounded + transparent inference" approach for method landing.

### A.3 How to apply it to this repository

1. Keep `model/xgboostmulti.py` as baseline.
2. Build a multimodal deep main model with three branches:
   - static (MLP),
   - time-series (GRU/TCN),
   - text (ClinicalBERT or TF-IDF fallback).
3. Use multi-head outputs:
   - Task 1 `label_can_discharge_24h`,
   - Task 2 `label_no_return_72h` (masked),
   - optional auxiliary `deterioration_24h`.
4. Preserve current anti-leakage policies:
   - split by `subject_id`,
   - strict text time gating (`note_time <= t_end`).

### A.4 Concrete file-level landing

- Add `utils/multimodal_dataset.py`:
  - unify `(stay_id, t_end)` samples across modalities.
- Add `model/multimodal_fusion.py`:
  - encoders + fusion + heads.
- Add `model/train_multimodal_main.py`:
  - train/eval/export pipeline.
- Output to `result/multimodal_main_<timestamp>/` (no overwrite baseline).

### A.5 Minimal training/evaluation contract

- Primary metrics: AUROC/AUPRC for Task 1 and Task 2(masked).
- Add calibration (Brier + reliability curve) and `first_careunit` subgroup.
- Persist reproducibility fields in `run_config.json`:
  - split seed,
  - label rule version,
  - modality toggles,
  - loss weights.

### A.6 Suggested launch command for upcoming main model

```bash
MM_EPOCHS=30 MM_BATCH_SIZE=128 MM_LR=1e-3 MM_USE_TEXT=1 MM_USE_GPU=1 python model/train_multimodal_main.py
```

### A.7 Metadata references used for this appendix

- DOI resolver: <https://doi.org/10.1145/3777577.3777712>
- Crossref metadata: <https://api.crossref.org/works/10.1145/3777577.3777712>
- OpenAlex metadata: <https://api.openalex.org/works/doi:10.1145/3777577.3777712>
