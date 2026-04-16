import os
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.xgboostmulti import fit_xgboost_multitask, predict_prob_xgb
from utils.multitask_common import (
    TASK1_COL,
    TASK2_COL,
    TASK2_MASK_COL,
    save_json,
    save_loss_curve,
    save_metrics_bundle,
    split_group_train_val_test,
)
from utils.xmi_common import build_xmi_feature_matrices, prepare_master_table_xmi


def _to_abs_shap(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 3:
        arr = arr[0]
    return np.abs(arr)


def save_shap_outputs(
    model,
    x_ref: pd.DataFrame,
    out_dir: Path,
    prefix: str,
    max_samples: int = 2000,
) -> Optional[Path]:
    if not hasattr(model, "get_booster"):
        print(f"[Info] {prefix}: constant model, SHAP skipped.")
        return None

    try:
        import shap
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[Warning] {prefix}: SHAP dependencies unavailable, skipped. {e}")
        return None

    if x_ref.empty:
        print(f"[Info] {prefix}: empty matrix, SHAP skipped.")
        return None

    rng = np.random.default_rng(42)
    n_samples = min(max_samples, len(x_ref))
    sample_idx = rng.choice(np.arange(len(x_ref)), size=n_samples, replace=False)
    x_sample = x_ref.iloc[sample_idx].copy()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_sample)
    shap_abs = _to_abs_shap(shap_values)
    mean_abs = shap_abs.mean(axis=0)

    importance_df = pd.DataFrame(
        {
            "feature": x_sample.columns,
            "mean_abs_shap": mean_abs,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    imp_path = out_dir / f"{prefix}_shap_importance.csv"
    importance_df.to_csv(imp_path, index=False)

    # Save bar summary for quick comparison in report.
    shap.summary_plot(shap_values, x_sample, max_display=20, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_shap_summary_bar.png", dpi=180)
    plt.close()

    print(f"[SHAP] Saved: {imp_path}")
    return imp_path


def main():
    result_subdir = (os.getenv("RESULT_SUBDIR", "xmi_icu_result") or "xmi_icu_result").strip()
    result_dir = PROJECT_ROOT / "result" / result_subdir
    result_dir.mkdir(parents=True, exist_ok=True)

    debug_max_stays = int(os.getenv("DEBUG_MAX_STAYS", "0") or "0")
    n_estimators = int(os.getenv("XGB_N_ESTIMATORS", "300") or "300")
    learning_rate = float(os.getenv("XGB_LR", "0.05") or "0.05")
    max_depth = int(os.getenv("XGB_MAX_DEPTH", "5") or "5")
    model_n_jobs = int(os.getenv("MODEL_N_JOBS", "1") or "1")
    use_gpu = (os.getenv("XGB_USE_GPU", "1") or "1").strip().lower() in {"1", "true", "yes", "y", "on"}
    min_landmark_hour = int(os.getenv("XMI_MIN_LANDMARK", "24") or "24")
    step_size = int(os.getenv("XMI_STEP_SIZE", "12") or "12")
    run_shap = (os.getenv("XMI_RUN_SHAP", "1") or "1").strip().lower() in {"1", "true", "yes", "y", "on"}
    shap_max_samples = int(os.getenv("XMI_SHAP_MAX_SAMPLES", "2000") or "2000")

    print("Preparing XMI pseudo-dynamic master table...")
    df_master = prepare_master_table_xmi(
        debug_max_stays=debug_max_stays,
        min_landmark_hour=min_landmark_hour,
        step_size=step_size,
    )

    print("Splitting train/val/test (group-aware)...")
    df_train, df_val, df_test = split_group_train_val_test(df_master)

    print("Building XMI feature matrices...")
    bundle = build_xmi_feature_matrices(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        corr_threshold=0.95,
    )

    x_train_df = bundle["X_train"]
    x_val_df = bundle["X_val"]
    x_test_df = bundle["X_test"]

    x_train = x_train_df.to_numpy(dtype=np.float32, copy=False)
    x_val = x_val_df.to_numpy(dtype=np.float32, copy=False)
    x_test = x_test_df.to_numpy(dtype=np.float32, copy=False)

    y1_train = bundle["y1_train"]
    y1_val = bundle["y1_val"]
    y1_test = bundle["y1_test"]

    y2_train = bundle["y2_train"]
    y2_val = bundle["y2_val"]
    y2_test = bundle["y2_test"]

    mask2_train = bundle["mask2_train"]
    mask2_val = bundle["mask2_val"]
    mask2_test = bundle["mask2_test"]

    print("Training XMI-ICU (XGBoost multi-head, task2 masked)...")
    model1, model2, loss_df, train_info = fit_xgboost_multitask(
        x_train=x_train,
        x_val=x_val,
        y1_train=y1_train,
        y1_val=y1_val,
        y2_train=y2_train,
        y2_val=y2_val,
        mask2_train=mask2_train,
        mask2_val=mask2_val,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        model_n_jobs=model_n_jobs,
        use_gpu=use_gpu,
    )

    save_loss_curve(
        loss_df=loss_df,
        out_path_png=result_dir / "train_val_loss_curve.png",
        out_path_csv=result_dir / "train_val_loss_curve.csv",
        title="XMI-ICU Train/Val Loss",
    )

    best_round = int(train_info["best_round"])

    if hasattr(model1, "save_model"):
        model1.save_model(result_dir / "xmi_task1_model.json")
    else:
        save_json({"constant_positive_prob": float(model1.positive_prob)}, result_dir / "xmi_task1_constant.json")

    if hasattr(model2, "save_model"):
        model2.save_model(result_dir / "xmi_task2_model.json")
    else:
        save_json({"constant_positive_prob": float(model2.positive_prob)}, result_dir / "xmi_task2_constant.json")

    joblib.dump(
        {
            "preprocess_artifacts": bundle["artifacts"],
            "train_info": train_info,
            "task_columns": {
                "task1": TASK1_COL,
                "task2": TASK2_COL,
                "task2_mask": TASK2_MASK_COL,
            },
        },
        result_dir / "xmi_icu_artifacts.joblib",
    )

    y1_prob_test = predict_prob_xgb(model1, x_test, best_round=best_round)
    y2_prob_test = predict_prob_xgb(model2, x_test, best_round=best_round)

    metrics_df = save_metrics_bundle(
        result_dir=result_dir,
        y1_true=y1_test,
        y1_prob=y1_prob_test,
        y2_true=y2_test,
        y2_prob=y2_prob_test,
        y2_mask=mask2_test,
        threshold=0.5,
    )

    if run_shap:
        save_shap_outputs(
            model=model1,
            x_ref=x_test_df,
            out_dir=result_dir,
            prefix="task1",
            max_samples=shap_max_samples,
        )
        task2_ref_df = x_test_df[mask2_test].copy() if mask2_test.any() else x_test_df.iloc[:0].copy()
        save_shap_outputs(
            model=model2,
            x_ref=task2_ref_df,
            out_dir=result_dir,
            prefix="task2_masked",
            max_samples=shap_max_samples,
        )

    save_json(
        {
            "result_dir": str(result_dir),
            "result_subdir": result_subdir,
            "debug_max_stays": debug_max_stays,
            "feature_mode": "xmi_pseudo_dynamic",
            "xmi_min_landmark": min_landmark_hour,
            "xmi_step_size": step_size,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "model_n_jobs": model_n_jobs,
            "xgb_use_gpu": use_gpu,
            "xmi_run_shap": run_shap,
            "xmi_shap_max_samples": shap_max_samples,
            **train_info,
        },
        result_dir / "run_config.json",
    )

    print("\nSaved XMI-ICU outputs to:", result_dir)
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
