import copy
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.multitask_common import (
    RANDOM_STATE,
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
    """Fallback model when training labels contain only one class."""

    def __init__(self, positive_prob: float):
        self.positive_prob = float(np.clip(positive_prob, 1e-6, 1 - 1e-6))

    def predict_proba(self, x):
        n = len(x)
        p = np.full(n, self.positive_prob, dtype=np.float64)
        return np.column_stack([1.0 - p, p])


def make_sgd(alpha: float) -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        learning_rate="optimal",
        random_state=RANDOM_STATE,
    )


def predict_positive_prob(model, x: np.ndarray) -> np.ndarray:
    return model.predict_proba(x)[:, 1]


def fit_logistic_multitask(
    x_train: np.ndarray,
    x_val: np.ndarray,
    y1_train: np.ndarray,
    y1_val: np.ndarray,
    y2_train: np.ndarray,
    y2_val: np.ndarray,
    mask2_train: np.ndarray,
    mask2_val: np.ndarray,
    n_epochs: int,
    alpha: float,
    patience: int,
) -> Tuple[object, object, pd.DataFrame, Dict[str, float]]:
    classes = np.array([0, 1], dtype=int)

    # Head 1 model
    if len(np.unique(y1_train)) < 2:
        model1 = ConstantProbModel(float(np.mean(y1_train)))
        trainable_head1 = False
    else:
        model1 = make_sgd(alpha=alpha)
        trainable_head1 = True

    # Head 2 model (masked)
    y2_train_masked = y2_train[mask2_train]
    if len(y2_train_masked) == 0 or len(np.unique(y2_train_masked)) < 2:
        base_prob = float(np.mean(y2_train_masked)) if len(y2_train_masked) > 0 else 0.5
        model2 = ConstantProbModel(base_prob)
        trainable_head2 = False
    else:
        model2 = make_sgd(alpha=alpha)
        trainable_head2 = True

    best_model1 = copy.deepcopy(model1)
    best_model2 = copy.deepcopy(model2)
    best_epoch = 1
    best_lambda = 1.0
    best_val_total = np.inf

    lambda_weight = 1.0
    no_improve_rounds = 0

    loss_rows = []

    for epoch in range(1, n_epochs + 1):
        # Train one epoch
        if trainable_head1:
            if epoch == 1:
                model1.partial_fit(x_train, y1_train, classes=classes)
            else:
                model1.partial_fit(x_train, y1_train)

        if trainable_head2:
            if epoch == 1:
                model2.partial_fit(x_train[mask2_train], y2_train[mask2_train], classes=classes)
            else:
                model2.partial_fit(x_train[mask2_train], y2_train[mask2_train])

        # Predict
        p1_train = predict_positive_prob(model1, x_train)
        p1_val = predict_positive_prob(model1, x_val)

        p2_train = predict_positive_prob(model2, x_train)
        p2_val = predict_positive_prob(model2, x_val)

        # Losses (task2 only on masked samples)
        l1_train = safe_binary_log_loss(y1_train, p1_train)
        l1_val = safe_binary_log_loss(y1_val, p1_val)

        if mask2_train.sum() > 0:
            l2_train = safe_binary_log_loss(y2_train[mask2_train], p2_train[mask2_train])
        else:
            l2_train = np.nan

        if mask2_val.sum() > 0:
            l2_val = safe_binary_log_loss(y2_val[mask2_val], p2_val[mask2_val])
        else:
            l2_val = np.nan

        if np.isfinite(l2_train) and l2_train > 0:
            ratio = l1_train / (l2_train + 1e-8)
            lambda_weight = 0.9 * lambda_weight + 0.1 * ratio
            lambda_weight = float(np.clip(lambda_weight, 0.1, 10.0))

        total_train = l1_train + (0.0 if not np.isfinite(l2_train) else lambda_weight * l2_train)
        total_val = l1_val + (0.0 if not np.isfinite(l2_val) else lambda_weight * l2_val)

        loss_rows.append(
            {
                "epoch": epoch,
                "train_loss_task1": l1_train,
                "train_loss_task2_masked": l2_train,
                "train_loss_total": total_train,
                "val_loss_task1": l1_val,
                "val_loss_task2_masked": l2_val,
                "val_loss_total": total_val,
                "lambda": lambda_weight,
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"L1(train/val)={l1_train:.4f}/{l1_val:.4f} | "
            f"L2(train/val)={l2_train:.4f}/{l2_val:.4f} | "
            f"lambda={lambda_weight:.4f} | "
            f"L_total(val)={total_val:.4f}"
        )

        if total_val + 1e-8 < best_val_total:
            best_val_total = total_val
            best_model1 = copy.deepcopy(model1)
            best_model2 = copy.deepcopy(model2)
            best_epoch = epoch
            best_lambda = lambda_weight
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1

        if no_improve_rounds >= patience:
            print(f"Early stopping at epoch {epoch} (patience={patience}).")
            break

    loss_df = pd.DataFrame(loss_rows)
    train_info = {
        "best_epoch": int(best_epoch),
        "best_lambda": float(best_lambda),
        "best_val_total_loss": float(best_val_total),
        "trainable_head1": bool(trainable_head1),
        "trainable_head2": bool(trainable_head2),
    }

    return best_model1, best_model2, loss_df, train_info


def main():
    result_dir = PROJECT_ROOT / "result" / "logisticmultiresult"
    result_dir.mkdir(parents=True, exist_ok=True)

    debug_max_stays = int(os.getenv("DEBUG_MAX_STAYS", "0") or "0")
    n_epochs = int(os.getenv("LOGI_EPOCHS", "40") or "40")
    alpha = float(os.getenv("LOGI_ALPHA", "0.0001") or "0.0001")
    patience = int(os.getenv("LOGI_PATIENCE", "8") or "8")

    print("Preparing shared master table...")
    df_master = prepare_master_table(debug_max_stays=debug_max_stays)

    print("Splitting train/val/test (group-aware)...")
    df_train, df_val, df_test = split_group_train_val_test(df_master)

    print("Building features (shared pipeline)...")
    bundle = build_feature_matrices(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        scale_for_linear=True,
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

    print("Training logistic multi-head (task2 masked)...")
    model1, model2, loss_df, train_info = fit_logistic_multitask(
        x_train=x_train,
        x_val=x_val,
        y1_train=y1_train,
        y1_val=y1_val,
        y2_train=y2_train,
        y2_val=y2_val,
        mask2_train=mask2_train,
        mask2_val=mask2_val,
        n_epochs=n_epochs,
        alpha=alpha,
        patience=patience,
    )

    save_loss_curve(
        loss_df=loss_df,
        out_path_png=result_dir / "train_val_loss_curve.png",
        out_path_csv=result_dir / "train_val_loss_curve.csv",
        title="Logistic Multi-Task Train/Val Loss",
    )

    # Save model at best validation epoch
    model_bundle = {
        "model_task1": model1,
        "model_task2": model2,
        "preprocess_artifacts": bundle["artifacts"],
        "train_info": train_info,
        "task_columns": {
            "task1": TASK1_COL,
            "task2": TASK2_COL,
            "task2_mask": TASK2_MASK_COL,
        },
    }
    joblib.dump(model_bundle, result_dir / "logisticmulti_model.joblib")

    # Test predictions and metrics
    y1_prob_test = predict_positive_prob(model1, x_test)
    y2_prob_test = predict_positive_prob(model2, x_test)

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
            "n_epochs": n_epochs,
            "alpha": alpha,
            "patience": patience,
            **train_info,
        },
        result_dir / "run_config.json",
    )

    print("\nSaved logistic multi-task outputs to:", result_dir)
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
