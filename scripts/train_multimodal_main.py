from __future__ import annotations

import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.calibration import calibration_curve
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.multimodal_fusion import MultimodalFusionModel
from utils.multimodal_dataset import MultimodalArtifacts, build_multimodal_data_bundle
from utils.multitask_common import (
    TASK1_COL,
    TASK2_COL,
    TASK2_MASK_COL,
    evaluate_binary_task,
    save_json,
    save_loss_curve,
    save_metrics_bundle,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_task2_loss(task2_logits: torch.Tensor, task2_label: torch.Tensor, task2_mask: torch.Tensor) -> torch.Tensor:
    mask = task2_mask > 0.5
    if mask.any():
        return F.binary_cross_entropy_with_logits(task2_logits[mask], task2_label[mask])
    return torch.zeros((), device=task2_logits.device, dtype=task2_logits.dtype)


def survival_loss(survival_logits: torch.Tensor, survival_target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(survival_logits.reshape(-1, 3), survival_target.reshape(-1))


def compute_cif(discharge_hazard: np.ndarray, death_hazard: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = discharge_hazard.shape[0]
    cif_discharge = np.zeros((n, 3), dtype=np.float64)
    cif_death = np.zeros((n, 3), dtype=np.float64)
    survival = np.ones(n, dtype=np.float64)

    for i in range(3):
        hazard_sum = discharge_hazard[:, i] + death_hazard[:, i]
        over_mask = hazard_sum > 0.999
        if np.any(over_mask):
            scale = 0.999 / np.maximum(hazard_sum[over_mask], 1e-8)
            discharge_hazard[over_mask, i] *= scale
            death_hazard[over_mask, i] *= scale

        cif_discharge[:, i] = (cif_discharge[:, i - 1] if i > 0 else 0.0) + survival * discharge_hazard[:, i]
        cif_death[:, i] = (cif_death[:, i - 1] if i > 0 else 0.0) + survival * death_hazard[:, i]
        survival = survival * np.clip(1.0 - discharge_hazard[:, i] - death_hazard[:, i], 0.0, 1.0)
    return cif_discharge, cif_death


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: AdamW | None,
    scaler: torch.cuda.amp.GradScaler,
    grad_accum: int,
    use_amp: bool,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_surv = 0.0
    total_task2 = 0.0
    total_all = 0.0
    n_batches = 0

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, start=1):
        time_x = batch["time_series"].to(device, non_blocking=True)
        static_x = batch["static"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        task2_label = batch["task2_label"].to(device, non_blocking=True)
        task2_mask = batch["task2_mask"].to(device, non_blocking=True)
        surv_target = batch["survival_target"].to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(
                    time_x=time_x,
                    static_x=static_x,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_attention=False,
                )
                l_surv = survival_loss(out["survival_logits"], surv_target)
                l_task2 = masked_task2_loss(out["task2_logits"], task2_label, task2_mask)
                l_total = l_surv + l_task2

            if is_train:
                l_total_scaled = l_total / max(grad_accum, 1)
                scaler.scale(l_total_scaled).backward()
                if step % max(grad_accum, 1) == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

        total_surv += float(l_surv.detach().cpu().item())
        total_task2 += float(l_task2.detach().cpu().item())
        total_all += float(l_total.detach().cpu().item())
        n_batches += 1

    if is_train and (len(loader) % max(grad_accum, 1) != 0):
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    denom = max(n_batches, 1)
    return {
        "loss_survival": total_surv / denom,
        "loss_task2": total_task2 / denom,
        "loss_total": total_all / denom,
    }


def save_attention_examples(
    attention_rows: List[np.ndarray],
    metadata_rows: List[Tuple[int, int, float]],
    output_png: Path,
) -> None:
    if len(attention_rows) == 0:
        plt.figure(figsize=(6, 3))
        plt.text(0.5, 0.5, "No attention rows collected.", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_png, dpi=180)
        plt.close()
        return

    n = len(attention_rows)
    n_cols = 2
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.7 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    x_axis = np.arange(-len(attention_rows[0]) + 1, 1)
    for i in range(len(axes)):
        ax = axes[i]
        if i >= n:
            ax.axis("off")
            continue
        arr = attention_rows[i]
        stay_id, subject_id, t_end = metadata_rows[i]
        ax.plot(x_axis, arr, marker="o", linewidth=1.2)
        ax.set_title(f"stay={stay_id}, subject={subject_id}, t_end={t_end:.0f}h")
        ax.set_xlabel("Hour offset to t_end")
        ax.set_ylabel("Attention")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close()


def save_reliability_plot(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, title: str) -> None:
    if len(np.unique(y_true)) < 2:
        plt.figure(figsize=(6, 6))
        plt.text(0.5, 0.5, "Only one class in y_true; calibration unavailable", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=180)
        plt.close()
        return

    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "--", alpha=0.5, label="Perfect calibration")
    plt.plot(mean_pred, frac_pos, marker="o", linewidth=1.5, label="Model")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def build_first_careunit_subgroup_metrics(
    test_df: pd.DataFrame,
    y1_true: np.ndarray,
    y1_prob: np.ndarray,
    y2_true: np.ndarray,
    y2_prob: np.ndarray,
    y2_mask: np.ndarray,
) -> pd.DataFrame:
    if "first_careunit" not in test_df.columns:
        return pd.DataFrame()

    rows: List[Dict[str, float]] = []
    grouped = test_df["first_careunit"].fillna("UNKNOWN").astype(str)
    for unit, idx in grouped.groupby(grouped).groups.items():
        idx_arr = np.array(list(idx), dtype=int)
        m1 = evaluate_binary_task(y1_true[idx_arr], y1_prob[idx_arr], threshold=0.5)
        rows.append({"group": unit, "task": "task1_discharge", "n_eval": int(len(idx_arr)), **m1})

        mask_local = y2_mask[idx_arr].astype(bool)
        if mask_local.sum() > 0:
            y2_local_true = y2_true[idx_arr][mask_local]
            y2_local_prob = y2_prob[idx_arr][mask_local]
            m2 = evaluate_binary_task(y2_local_true, y2_local_prob, threshold=0.5)
            rows.append({"group": unit, "task": "task2_no_return72_masked", "n_eval": int(mask_local.sum()), **m2})

    return pd.DataFrame(rows)


def main() -> None:
    default_subdir = f"multimodal_main_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    result_subdir = (os.getenv("RESULT_SUBDIR", default_subdir) or default_subdir).strip()
    result_dir = PROJECT_ROOT / "result" / result_subdir
    result_dir.mkdir(parents=True, exist_ok=True)

    debug_max_stays = int(os.getenv("DEBUG_MAX_STAYS", "0") or "0")
    mm_epochs = int(os.getenv("MM_EPOCHS", "30") or "30")
    batch_size = int(os.getenv("MM_BATCH_SIZE", "32") or "32")
    lr = float(os.getenv("MM_LR", "1e-4") or "1e-4")
    weight_decay = float(os.getenv("MM_WEIGHT_DECAY", "1e-2") or "1e-2")
    max_text_len = int(os.getenv("MM_MAX_TEXT_LEN", "128") or "128")
    grad_accum = int(os.getenv("MM_GRAD_ACCUM", "1") or "1")
    use_gpu = (os.getenv("MM_USE_GPU", "1") or "1").strip().lower() in {"1", "true", "yes", "y", "on"}
    resume = (os.getenv("MM_RESUME", "0") or "0").strip().lower() in {"1", "true", "yes", "y", "on"}
    patience = int(os.getenv("MM_PATIENCE", "5") or "5")
    seed = int(os.getenv("MM_SEED", "42") or "42")
    num_workers = int(os.getenv("MM_NUM_WORKERS", "0") or "0")
    freeze_bert = (os.getenv("MM_FREEZE_BERT", "0") or "0").strip().lower() in {"1", "true", "yes", "y", "on"}
    bert_model_name = (
        os.getenv("MM_MODEL_NAME", "emilyalsentzer/Bio_ClinicalBERT") or "emilyalsentzer/Bio_ClinicalBERT"
    ).strip()
    use_text = (os.getenv("MM_USE_TEXT", "1") or "1").strip().lower() in {"1", "true", "yes", "y", "on"}
    use_time = (os.getenv("MM_USE_TIME", "1") or "1").strip().lower() in {"1", "true", "yes", "y", "on"}
    use_static = (os.getenv("MM_USE_STATIC", "1") or "1").strip().lower() in {"1", "true", "yes", "y", "on"}

    set_seed(seed)
    use_cuda = use_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    use_amp = bool(use_cuda)

    print("Preparing multimodal data bundle...")
    bundle = build_multimodal_data_bundle(
        debug_max_stays=debug_max_stays,
        tokenizer_name=bert_model_name,
        max_text_len=max_text_len,
        use_text=use_text,
        use_time=use_time,
        use_static=use_static,
    )
    train_dataset = bundle["train_dataset"]
    val_dataset = bundle["val_dataset"]
    test_dataset = bundle["test_dataset"]
    artifacts: MultimodalArtifacts = bundle["artifacts"]

    print(f"Train rows={len(train_dataset)}, Val rows={len(val_dataset)}, Test rows={len(test_dataset)}")
    print(f"Static dim={bundle['static_feature_dim']}, Time dim={bundle['time_feature_dim']}")

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": bool(use_cuda),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    model = MultimodalFusionModel(
        time_input_dim=int(bundle["time_feature_dim"]),
        static_input_dim=int(bundle["static_feature_dim"]),
        bert_model_name=bert_model_name,
        freeze_bert=freeze_bert,
        use_text=use_text,
        use_time=use_time,
        use_static=use_static,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    checkpoint_dir = result_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = checkpoint_dir / "latest.pt"
    best_ckpt = checkpoint_dir / "best.pt"

    start_epoch = 1
    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    loss_rows: List[Dict[str, float]] = []

    if resume and latest_ckpt.exists():
        print(f"Resuming from {latest_ckpt} ...")
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
        best_epoch = int(ckpt.get("best_epoch", best_epoch))
        no_improve = int(ckpt.get("no_improve", no_improve))
        loss_rows = ckpt.get("loss_rows", [])
        print(f"Resume start epoch={start_epoch}, best_val_loss={best_val_loss:.6f}")

    t0 = time.time()
    for epoch in range(start_epoch, mm_epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            grad_accum=grad_accum,
            use_amp=use_amp,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            optimizer=None,
            scaler=scaler,
            grad_accum=1,
            use_amp=use_amp,
        )

        row = {
            "epoch": epoch,
            "train_loss_survival": train_metrics["loss_survival"],
            "train_loss_task2_masked": train_metrics["loss_task2"],
            "train_loss_total": train_metrics["loss_total"],
            "val_loss_survival": val_metrics["loss_survival"],
            "val_loss_task2_masked": val_metrics["loss_task2"],
            "val_loss_total": val_metrics["loss_total"],
        }
        loss_rows.append(row)

        print(
            f"Epoch {epoch:03d} | train_total={row['train_loss_total']:.5f} | "
            f"val_total={row['val_loss_total']:.5f} | val_surv={row['val_loss_survival']:.5f} | "
            f"val_task2={row['val_loss_task2_masked']:.5f}"
        )

        improved = row["val_loss_total"] + 1e-8 < best_val_loss
        if improved:
            best_val_loss = float(row["val_loss_total"])
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                    "no_improve": no_improve,
                    "loss_rows": loss_rows,
                },
                best_ckpt,
            )
        else:
            no_improve += 1

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "no_improve": no_improve,
                "loss_rows": loss_rows,
            },
            latest_ckpt,
        )

        if no_improve >= patience:
            print(f"Early stopping at epoch={epoch}, patience={patience}.")
            break

    elapsed = time.time() - t0
    print(f"Training finished in {elapsed:.1f}s. Best epoch={best_epoch}, best_val_loss={best_val_loss:.6f}")

    if best_ckpt.exists():
        best_state = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(best_state["model"])

    loss_df = pd.DataFrame(loss_rows)
    save_loss_curve(
        loss_df=loss_df,
        out_path_png=result_dir / "train_val_loss_curve.png",
        out_path_csv=result_dir / "train_val_loss_curve.csv",
        title="Multimodal Main Train/Val Loss",
    )

    model.eval()
    all_y1_true: List[np.ndarray] = []
    all_y2_true: List[np.ndarray] = []
    all_y2_mask: List[np.ndarray] = []
    all_y1_prob: List[np.ndarray] = []
    all_y2_prob: List[np.ndarray] = []
    all_surv_prob: List[np.ndarray] = []
    all_stay: List[np.ndarray] = []
    all_subject: List[np.ndarray] = []
    all_t_end: List[np.ndarray] = []

    attention_examples: List[np.ndarray] = []
    attention_meta: List[Tuple[int, int, float]] = []

    with torch.no_grad():
        for batch in test_loader:
            time_x = batch["time_series"].to(device, non_blocking=True)
            static_x = batch["static"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            out = model(
                time_x=time_x,
                static_x=static_x,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_attention=True,
            )

            surv_prob = torch.softmax(out["survival_logits"], dim=-1).detach().cpu().numpy()
            task2_prob = torch.sigmoid(out["task2_logits"]).detach().cpu().numpy()
            task1_prob = surv_prob[:, 0, 1]

            all_surv_prob.append(surv_prob)
            all_y1_prob.append(task1_prob)
            all_y2_prob.append(task2_prob)
            all_y1_true.append(batch["task1_label"].cpu().numpy())
            all_y2_true.append(batch["task2_label"].cpu().numpy())
            all_y2_mask.append(batch["task2_mask"].cpu().numpy().astype(bool))
            all_stay.append(batch["stay_id"].cpu().numpy())
            all_subject.append(batch["subject_id"].cpu().numpy())
            all_t_end.append(batch["t_end"].cpu().numpy())

            if len(attention_examples) < 10:
                attn = out["attention_weights"].detach().cpu().numpy()
                stay_np = batch["stay_id"].cpu().numpy()
                subject_np = batch["subject_id"].cpu().numpy()
                t_end_np = batch["t_end"].cpu().numpy()
                for i in range(attn.shape[0]):
                    if len(attention_examples) >= 10:
                        break
                    attention_examples.append(attn[i].astype(np.float64))
                    attention_meta.append((int(stay_np[i]), int(subject_np[i]), float(t_end_np[i])))

    y1_true = np.concatenate(all_y1_true, axis=0).astype(int)
    y2_true = np.concatenate(all_y2_true, axis=0).astype(int)
    y2_mask = np.concatenate(all_y2_mask, axis=0).astype(bool)
    y1_prob = np.concatenate(all_y1_prob, axis=0).astype(np.float64)
    y2_prob = np.concatenate(all_y2_prob, axis=0).astype(np.float64)
    surv_prob_all = np.concatenate(all_surv_prob, axis=0).astype(np.float64)

    metrics_df = save_metrics_bundle(
        result_dir=result_dir,
        y1_true=y1_true,
        y1_prob=y1_prob,
        y2_true=y2_true,
        y2_prob=y2_prob,
        y2_mask=y2_mask,
        threshold=0.5,
    )

    discharge_h = surv_prob_all[:, :, 1]
    death_h = surv_prob_all[:, :, 2]
    cif_discharge, cif_death = compute_cif(discharge_h.copy(), death_h.copy())

    stay_ids = np.concatenate(all_stay, axis=0).astype(np.int64)
    subject_ids = np.concatenate(all_subject, axis=0).astype(np.int64)
    t_end = np.concatenate(all_t_end, axis=0).astype(np.float64)

    cif_df = pd.DataFrame(
        {
            "stay_id": stay_ids,
            "subject_id": subject_ids,
            "t_end": t_end,
            "task1_true": y1_true,
            "task1_prob": y1_prob,
            "task2_true": y2_true,
            "task2_mask": y2_mask.astype(int),
            "task2_prob": y2_prob,
            "hazard_discharge_0_24h": discharge_h[:, 0],
            "hazard_discharge_24_48h": discharge_h[:, 1],
            "hazard_discharge_48_72h": discharge_h[:, 2],
            "hazard_death_0_24h": death_h[:, 0],
            "hazard_death_24_48h": death_h[:, 1],
            "hazard_death_48_72h": death_h[:, 2],
            "cif_discharge_24h": cif_discharge[:, 0],
            "cif_discharge_48h": cif_discharge[:, 1],
            "cif_discharge_72h": cif_discharge[:, 2],
            "cif_death_24h": cif_death[:, 0],
            "cif_death_48h": cif_death[:, 1],
            "cif_death_72h": cif_death[:, 2],
        }
    )
    cif_df.to_csv(result_dir / "cif_predictions.csv", index=False)

    save_attention_examples(attention_examples, attention_meta, result_dir / "attention_heatmap_examples.png")

    brier_task1 = float(np.mean((y1_true - y1_prob) ** 2))
    if y2_mask.any():
        brier_task2_masked = float(np.mean((y2_true[y2_mask] - y2_prob[y2_mask]) ** 2))
    else:
        brier_task2_masked = float("nan")

    calib_rows = [
        {"task": "task1_discharge", "brier": brier_task1, "n_eval": int(len(y1_true))},
        {"task": "task2_no_return72_masked", "brier": brier_task2_masked, "n_eval": int(y2_mask.sum())},
    ]
    pd.DataFrame(calib_rows).to_csv(result_dir / "calibration_summary.csv", index=False)
    save_reliability_plot(y1_true, y1_prob, result_dir / "reliability_task1.png", "Task1 Reliability")
    if y2_mask.any():
        save_reliability_plot(
            y2_true[y2_mask],
            y2_prob[y2_mask],
            result_dir / "reliability_task2_masked.png",
            "Task2(masked) Reliability",
        )

    subgroup_df = build_first_careunit_subgroup_metrics(
        test_df=bundle["test_df"],
        y1_true=y1_true,
        y1_prob=y1_prob,
        y2_true=y2_true,
        y2_prob=y2_prob,
        y2_mask=y2_mask,
    )
    if not subgroup_df.empty:
        subgroup_df.to_csv(result_dir / "first_careunit_subgroup_metrics.csv", index=False)

    joblib.dump(
        {
            "artifacts": artifacts,
            "model_name": bert_model_name,
            "time_feature_dim": int(bundle["time_feature_dim"]),
            "static_feature_dim": int(bundle["static_feature_dim"]),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
        },
        result_dir / "multimodal_main_artifacts.joblib",
    )

    torch.save(model.state_dict(), result_dir / "multimodal_main_model.pt")

    run_config = {
        "result_dir": str(result_dir),
        "result_subdir": result_subdir,
        "debug_max_stays": debug_max_stays,
        "mm_epochs": mm_epochs,
        "mm_batch_size": batch_size,
        "mm_lr": lr,
        "mm_weight_decay": weight_decay,
        "mm_max_text_len": max_text_len,
        "mm_grad_accum": grad_accum,
        "mm_patience": patience,
        "mm_seed": seed,
        "mm_num_workers": num_workers,
        "mm_freeze_bert": freeze_bert,
        "mm_model_name": bert_model_name,
        "mm_resume": resume,
        "mm_use_text": use_text,
        "mm_use_time": use_time,
        "mm_use_static": use_static,
        "device": str(device),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "time_feature_dim": int(bundle["time_feature_dim"]),
        "static_feature_dim": int(bundle["static_feature_dim"]),
        "tokenizer_name": artifacts.cma_artifacts.tokenizer_name,
        "survival_intervals": artifacts.cma_artifacts.survival_intervals,
        "task_columns": {
            "task1": TASK1_COL,
            "task2": TASK2_COL,
            "task2_mask": TASK2_MASK_COL,
        },
    }
    save_json(run_config, result_dir / "run_config.json")

    print("\nSaved multimodal-main outputs to:", result_dir)
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
