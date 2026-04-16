import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.multitask_common import (
    TASK1_COL,
    TASK2_COL,
    TASK2_MASK_COL,
    build_feature_matrices,
    prepare_master_table,
    save_json,
    save_loss_curve,
    save_metrics_bundle,
    safe_binary_log_loss,
    split_group_train_val_test,
)


class ConstantProbModel:
    """Fallback model when a task has only one class in train subset."""

    def __init__(self, positive_prob: float):
        self.positive_prob = float(np.clip(positive_prob, 1e-6, 1 - 1e-6))

    def predict_proba(self, x):
        n = len(x)
        p = np.full(n, self.positive_prob, dtype=np.float64)
        return np.column_stack([1.0 - p, p])


def fit_xgboost_multitask(
    x_train: np.ndarray,
    x_val: np.ndarray,
    y1_train: np.ndarray,
    y1_val: np.ndarray,
    y2_train: np.ndarray,
    y2_val: np.ndarray,
    mask2_train: np.ndarray,
    mask2_val: np.ndarray,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    model_n_jobs: int,
) -> Tuple[object, object, pd.DataFrame, Dict[str, float]]:
    try:
        from xgboost import XGBClassifier
    except Exception as e:
        raise RuntimeError(
            "xgboost import failed. Please install/link libomp first.\n"
            f"Original error:\n{e}"
        )

    common_params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": model_n_jobs,
        "eval_metric": "logloss",
        "objective": "binary:logistic",
        "tree_method": "hist",
    }

    # Task 1
    if len(np.unique(y1_train)) < 2:
        model1 = ConstantProbModel(float(np.mean(y1_train)))
        trainable_head1 = False
    else:
        n_pos = int(y1_train.sum())
        n_neg = int(len(y1_train) - n_pos)
        spw1 = (n_neg / n_pos) if n_pos > 0 else 1.0

        model1 = XGBClassifier(**common_params, scale_pos_weight=spw1)
        model1.fit(x_train, y1_train, eval_set=[(x_train, y1_train), (x_val, y1_val)], verbose=False)
        trainable_head1 = True

    # Task 2 (masked)
    y2_train_masked = y2_train[mask2_train]
    y2_val_masked = y2_val[mask2_val]

    if len(y2_train_masked) == 0 or len(np.unique(y2_train_masked)) < 2:
        base_prob = float(np.mean(y2_train_masked)) if len(y2_train_masked) > 0 else 0.5
        model2 = ConstantProbModel(base_prob)
        trainable_head2 = False
    else:
        n_pos2 = int(y2_train_masked.sum())
        n_neg2 = int(len(y2_train_masked) - n_pos2)
        spw2 = (n_neg2 / n_pos2) if n_pos2 > 0 else 1.0

        model2 = XGBClassifier(**common_params, scale_pos_weight=spw2)

        if len(y2_val_masked) > 0:
            eval_set2 = [(x_train[mask2_train], y2_train_masked), (x_val[mask2_val], y2_val_masked)]
        else:
            # Very rare fallback if val has no masked rows
            eval_set2 = [(x_train[mask2_train], y2_train_masked), (x_train[mask2_train], y2_train_masked)]

        model2.fit(x_train[mask2_train], y2_train_masked, eval_set=eval_set2, verbose=False)
        trainable_head2 = True

    # Build per-round losses
    epochs = np.arange(1, n_estimators + 1)

    if trainable_head1:
        res1 = model1.evals_result()
        l1_train_curve = np.array(res1["validation_0"]["logloss"], dtype=float)
        l1_val_curve = np.array(res1["validation_1"]["logloss"], dtype=float)
    else:
        p_const = np.full(len(y1_train), float(np.mean(y1_train)), dtype=float)
        l1_train_const = safe_binary_log_loss(y1_train, p_const)
        p_const_val = np.full(len(y1_val), float(np.mean(y1_train)), dtype=float)
        l1_val_const = safe_binary_log_loss(y1_val, p_const_val)
        l1_train_curve = np.repeat(l1_train_const, n_estimators)
        l1_val_curve = np.repeat(l1_val_const, n_estimators)

    if trainable_head2:
        res2 = model2.evals_result()
        l2_train_curve = np.array(res2["validation_0"]["logloss"], dtype=float)
        l2_val_curve = np.array(res2["validation_1"]["logloss"], dtype=float)
    else:
        if len(y2_train_masked) > 0:
            p2c_train = np.full(len(y2_train_masked), float(np.mean(y2_train_masked)), dtype=float)
            l2_train_const = safe_binary_log_loss(y2_train_masked, p2c_train)
        else:
            l2_train_const = np.nan

        if len(y2_val_masked) > 0:
            p2c_val = np.full(len(y2_val_masked), float(np.mean(y2_train_masked)) if len(y2_train_masked) > 0 else 0.5, dtype=float)
            l2_val_const = safe_binary_log_loss(y2_val_masked, p2c_val)
        else:
            l2_val_const = np.nan

        l2_train_curve = np.repeat(l2_train_const, n_estimators)
        l2_val_curve = np.repeat(l2_val_const, n_estimators)

    # Adaptive lambda and total loss curve
    lambda_weight = 1.0
    rows = []

    for i in range(n_estimators):
        l1_train = l1_train_curve[i]
        l1_val = l1_val_curve[i]
        l2_train = l2_train_curve[i]
        l2_val = l2_val_curve[i]

        if np.isfinite(l2_train) and l2_train > 0:
            ratio = l1_train / (l2_train + 1e-8)
            lambda_weight = 0.9 * lambda_weight + 0.1 * ratio
            lambda_weight = float(np.clip(lambda_weight, 0.1, 10.0))

        total_train = l1_train + (0.0 if not np.isfinite(l2_train) else lambda_weight * l2_train)
        total_val = l1_val + (0.0 if not np.isfinite(l2_val) else lambda_weight * l2_val)

        rows.append(
            {
                "epoch": i + 1,
                "train_loss_task1": l1_train,
                "train_loss_task2_masked": l2_train,
                "train_loss_total": total_train,
                "val_loss_task1": l1_val,
                "val_loss_task2_masked": l2_val,
                "val_loss_total": total_val,
                "lambda": lambda_weight,
            }
        )

    loss_df = pd.DataFrame(rows)

    if loss_df["val_loss_total"].notna().any():
        best_idx = int(loss_df["val_loss_total"].idxmin())
    else:
        best_idx = len(loss_df) - 1

    best_round = int(loss_df.loc[best_idx, "epoch"])
    best_lambda = float(loss_df.loc[best_idx, "lambda"])
    best_val_total_loss = float(loss_df.loc[best_idx, "val_loss_total"])

    train_info = {
        "best_round": best_round,
        "best_lambda": best_lambda,
        "best_val_total_loss": best_val_total_loss,
        "trainable_head1": bool(trainable_head1),
        "trainable_head2": bool(trainable_head2),
    }

    print(f"Best boosting round by val total loss: {best_round}")

    return model1, model2, loss_df, train_info


def predict_prob_xgb(model, x: np.ndarray, best_round: int) -> np.ndarray:
    if isinstance(model, ConstantProbModel):
        return model.predict_proba(x)[:, 1]

    # iteration_range end is exclusive
    return model.predict_proba(x, iteration_range=(0, best_round))[:, 1]


def main():
    result_dir = PROJECT_ROOT / "result" / "xgboostmultiresult"
    result_dir.mkdir(parents=True, exist_ok=True)

    debug_max_stays = int(os.getenv("DEBUG_MAX_STAYS", "0") or "0")
    n_estimators = int(os.getenv("XGB_N_ESTIMATORS", "300") or "300")
    learning_rate = float(os.getenv("XGB_LR", "0.05") or "0.05")
    max_depth = int(os.getenv("XGB_MAX_DEPTH", "5") or "5")
    model_n_jobs = int(os.getenv("MODEL_N_JOBS", "1") or "1")

    print("Preparing shared master table...")
    df_master = prepare_master_table(debug_max_stays=debug_max_stays)

    print("Splitting train/val/test (group-aware)...")
    df_train, df_val, df_test = split_group_train_val_test(df_master)

    print("Building features (shared pipeline)...")
    bundle = build_feature_matrices(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        scale_for_linear=False,
        tfidf_max_features=300,
    )

    x_train = bundle["X_train"].to_numpy(dtype=np.float32, copy=False)
    x_val = bundle["X_val"].to_numpy(dtype=np.float32, copy=False)
    x_test = bundle["X_test"].to_numpy(dtype=np.float32, copy=False)

    y1_train = bundle["y1_train"]
    y1_val = bundle["y1_val"]
    y1_test = bundle["y1_test"]

    y2_train = bundle["y2_train"]
    y2_val = bundle["y2_val"]
    y2_test = bundle["y2_test"]

    mask2_train = bundle["mask2_train"]
    mask2_val = bundle["mask2_val"]
    mask2_test = bundle["mask2_test"]

    print("Training xgboost multi-head (task2 masked)...")
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
    )

    save_loss_curve(
        loss_df=loss_df,
        out_path_png=result_dir / "train_val_loss_curve.png",
        out_path_csv=result_dir / "train_val_loss_curve.csv",
        title="XGBoost Multi-Task Train/Val Loss",
    )

    best_round = int(train_info["best_round"])

    # Save models after val selection
    if hasattr(model1, "save_model"):
        model1.save_model(result_dir / "xgb_task1_model.json")
    else:
        save_json({"constant_positive_prob": float(model1.positive_prob)}, result_dir / "xgb_task1_constant.json")

    if hasattr(model2, "save_model"):
        model2.save_model(result_dir / "xgb_task2_model.json")
    else:
        save_json({"constant_positive_prob": float(model2.positive_prob)}, result_dir / "xgb_task2_constant.json")

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
        result_dir / "xgboostmulti_artifacts.joblib",
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

    save_json(
        {
            "result_dir": str(result_dir),
            "debug_max_stays": debug_max_stays,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "model_n_jobs": model_n_jobs,
            **train_info,
        },
        result_dir / "run_config.json",
    )

    print("\nSaved xgboost multi-task outputs to:", result_dir)
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
