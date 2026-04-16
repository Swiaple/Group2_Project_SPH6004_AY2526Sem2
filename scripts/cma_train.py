from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.cma_surv import CmaSurvModel
from utils.cma_dataset import CmaArtifacts, build_cma_data_bundle
from utils.multitask_common import (
    TASK1_COL,
    TASK2_COL,
    TASK2_MASK_COL,
    save_json,
    save_loss_curve,
    save_metrics_bundle,
)


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)) or str(default))


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _is_dist() else 0


def _world_size() -> int:
    return dist.get_world_size() if _is_dist() else 1


def _is_main() -> bool:
    return _rank() == 0


def _log(msg: str, main_only: bool = True) -> None:
    if (not main_only) or _is_main():
        print(msg, flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_distributed(use_cuda: bool) -> Tuple[bool, int, int, int]:
    rank = _env_int("RANK", 0)
    world_size = _env_int("WORLD_SIZE", 1)
    local_rank = _env_int("LOCAL_RANK", 0)
    use_ddp = world_size > 1

    if use_ddp and not _is_dist():
        backend = "nccl" if use_cuda else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    return use_ddp, rank, world_size, local_rank


def cleanup_distributed() -> None:
    if _is_dist():
        dist.destroy_process_group()


def state_dict_model(model: nn.Module) -> nn.Module:
    # Keep checkpoint format stable across single GPU / DDP.
    if isinstance(model, (DDP, nn.DataParallel)):
        return model.module
    return model


def masked_task2_loss(task2_logits: torch.Tensor, task2_label: torch.Tensor, task2_mask: torch.Tensor) -> torch.Tensor:
    mask = task2_mask > 0.5
    if mask.any():
        return F.binary_cross_entropy_with_logits(task2_logits[mask], task2_label[mask])
    return torch.zeros((), device=task2_logits.device, dtype=task2_logits.dtype)


def survival_loss(survival_logits: torch.Tensor, survival_target: torch.Tensor) -> torch.Tensor:
    # [B, 3 bins, 3 classes] -> average CE across all bins.
    return F.cross_entropy(survival_logits.reshape(-1, 3), survival_target.reshape(-1))


def compute_cif(discharge_hazard: np.ndarray, death_hazard: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Inputs: [N, 3], outputs cumulative incidence [N, 3].
    n = discharge_hazard.shape[0]
    cif_discharge = np.zeros((n, 3), dtype=np.float64)
    cif_death = np.zeros((n, 3), dtype=np.float64)
    survival = np.ones(n, dtype=np.float64)

    for i in range(3):
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
    epoch: int,
    split_name: str,
    heartbeat_steps: int,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_surv = 0.0
    total_task2 = 0.0
    total_all = 0.0
    n_batches = 0

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    t0 = time.time()
    loader_len = max(len(loader), 1)

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

            if is_train and step == 1 and device.type == "cuda":
                mem_alloc_mb = torch.cuda.memory_allocated(device) / (1024.0 ** 2)
                mem_reserved_mb = torch.cuda.memory_reserved(device) / (1024.0 ** 2)
                _log(
                    f"[GPU-TRAIN-CHECK] rank={_rank()} local_rank={os.getenv('LOCAL_RANK', '0')} "
                    f"epoch={epoch:03d} split={split_name} step=1 "
                    f"loss_total={l_total.item():.5f} "
                    f"mem_alloc_mb={mem_alloc_mb:.1f} mem_reserved_mb={mem_reserved_mb:.1f}",
                    main_only=False,
                )

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

        if is_train and heartbeat_steps > 0 and step % heartbeat_steps == 0 and _is_main():
            elapsed = max(time.time() - t0, 1e-6)
            it_s = step / elapsed
            _log(
                f"[HB] epoch={epoch:03d} split={split_name} "
                f"step={step}/{loader_len} "
                f"loss_total={l_total.item():.5f} loss_surv={l_surv.item():.5f} loss_task2={l_task2.item():.5f} "
                f"it/s={it_s:.2f}"
            )

    if is_train and (len(loader) % max(grad_accum, 1) != 0):
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    metric_tensor = torch.tensor(
        [total_surv, total_task2, total_all, float(max(n_batches, 1))],
        device=device,
        dtype=torch.float64,
    )
    if _is_dist():
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)

    denom = float(metric_tensor[3].item())
    return {
        "loss_survival": float(metric_tensor[0].item() / denom),
        "loss_task2": float(metric_tensor[1].item() / denom),
        "loss_total": float(metric_tensor[2].item() / denom),
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


def main() -> None:
    result_subdir = (os.getenv("RESULT_SUBDIR", "cma_surv_result") or "cma_surv_result").strip()
    result_dir = PROJECT_ROOT / "result" / result_subdir
    result_dir.mkdir(parents=True, exist_ok=True)

    debug_max_stays = int(os.getenv("DEBUG_MAX_STAYS", "0") or "0")
    cma_epochs = int(os.getenv("CMA_EPOCHS", "30") or "30")
    batch_size = int(os.getenv("CMA_BATCH_SIZE", "32") or "32")
    lr = float(os.getenv("CMA_LR", "1e-4") or "1e-4")
    weight_decay = float(os.getenv("CMA_WEIGHT_DECAY", "1e-2") or "1e-2")
    max_text_len = int(os.getenv("CMA_MAX_TEXT_LEN", "128") or "128")
    grad_accum = int(os.getenv("CMA_GRAD_ACCUM", "1") or "1")
    heartbeat_steps = int(os.getenv("CMA_HEARTBEAT_STEPS", "200") or "200")
    use_gpu = (os.getenv("CMA_USE_GPU", "1") or "1").strip().lower() in {"1", "true", "yes", "y", "on"}
    resume = (os.getenv("CMA_RESUME", "0") or "0").strip().lower() in {"1", "true", "yes", "y", "on"}
    patience = int(os.getenv("CMA_PATIENCE", "5") or "5")
    seed = int(os.getenv("CMA_SEED", "42") or "42")
    num_workers = int(os.getenv("CMA_NUM_WORKERS", "0") or "0")
    freeze_bert = (os.getenv("CMA_FREEZE_BERT", "0") or "0").strip().lower() in {"1", "true", "yes", "y", "on"}
    enable_multi_gpu = (os.getenv("CMA_MULTI_GPU", "1") or "1").strip().lower() in {"1", "true", "yes", "y", "on"}
    bert_model_name = (os.getenv("CMA_MODEL_NAME", "emilyalsentzer/Bio_ClinicalBERT") or "emilyalsentzer/Bio_ClinicalBERT").strip()
    ddp_find_unused_env = (os.getenv("CMA_DDP_FIND_UNUSED", "auto") or "auto").strip().lower()

    use_cuda = use_gpu and torch.cuda.is_available()
    use_ddp, rank, world_size, local_rank = setup_distributed(use_cuda=use_cuda)
    if world_size > 1 and not enable_multi_gpu:
        _log("WORLD_SIZE>1 detected; forcing DDP despite CMA_MULTI_GPU=0.", main_only=False)
    use_ddp = bool(world_size > 1)

    if use_cuda:
        if use_ddp:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    use_amp = bool(use_cuda)
    detected_gpu_count = torch.cuda.device_count() if use_cuda else 0

    set_seed(seed + rank)
    _log(
        f"Torch runtime: version={torch.__version__}, torch.cuda={torch.version.cuda}, "
        f"cuda_available={torch.cuda.is_available()}, cuda_device_count={torch.cuda.device_count()}",
        main_only=False,
    )
    _log(
        f"Runtime setup: rank={rank}, world_size={world_size}, local_rank={local_rank}, "
        f"use_ddp={use_ddp}, device={device}, freeze_bert={freeze_bert}, heartbeat_steps={heartbeat_steps}",
        main_only=False,
    )

    _log("Preparing CMA data bundle...")
    bundle = build_cma_data_bundle(
        debug_max_stays=debug_max_stays,
        tokenizer_name=bert_model_name,
        max_text_len=max_text_len,
    )
    train_dataset = bundle["train_dataset"]
    val_dataset = bundle["val_dataset"]
    test_dataset = bundle["test_dataset"]
    artifacts: CmaArtifacts = bundle["artifacts"]

    _log(f"Train rows={len(train_dataset)}, Val rows={len(val_dataset)}, Test rows={len(test_dataset)}")
    _log(f"Static dim={bundle['static_feature_dim']}, Time dim={bundle['time_feature_dim']}")

    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if use_ddp
        else None
    )
    val_sampler = (
        DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if use_ddp
        else None
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": bool(use_cuda),
    }
    train_loader = DataLoader(train_dataset, shuffle=(train_sampler is None), sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, sampler=val_sampler, **loader_kwargs)
    test_loader = None
    if _is_main():
        test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    model = CmaSurvModel(
        time_input_dim=int(bundle["time_feature_dim"]),
        static_input_dim=int(bundle["static_feature_dim"]),
        bert_model_name=bert_model_name,
        freeze_bert=freeze_bert,
    ).to(device)
    if use_ddp:
        if ddp_find_unused_env in {"1", "true", "yes", "y", "on"}:
            ddp_find_unused = True
        elif ddp_find_unused_env in {"0", "false", "no", "n", "off"}:
            ddp_find_unused = False
        else:
            # Default to safer behavior when BERT is unfrozen.
            ddp_find_unused = not freeze_bert
        _log("Wrapping model with DistributedDataParallel.")
        _log(f"DDP config: find_unused_parameters={ddp_find_unused}", main_only=False)
        model = DDP(
            model,
            device_ids=[local_rank] if use_cuda else None,
            output_device=local_rank if use_cuda else None,
            find_unused_parameters=ddp_find_unused,
        )

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
        _log(f"Resuming from {latest_ckpt} ...")
        ckpt = torch.load(latest_ckpt, map_location=device)
        state_dict_model(model).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
        best_epoch = int(ckpt.get("best_epoch", best_epoch))
        no_improve = int(ckpt.get("no_improve", no_improve))
        loss_rows = ckpt.get("loss_rows", [])
        _log(f"Resume start epoch={start_epoch}, best_val_loss={best_val_loss:.6f}")

    t0 = time.time()
    for epoch in range(start_epoch, cma_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            grad_accum=grad_accum,
            use_amp=use_amp,
            epoch=epoch,
            split_name="train",
            heartbeat_steps=heartbeat_steps,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            optimizer=None,
            scaler=scaler,
            grad_accum=1,
            use_amp=use_amp,
            epoch=epoch,
            split_name="val",
            heartbeat_steps=0,
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

        stop_now = 0
        if _is_main():
            loss_rows.append(row)
            _log(
                f"Epoch {epoch:03d} | "
                f"train_total={row['train_loss_total']:.5f} | "
                f"val_total={row['val_loss_total']:.5f} | "
                f"val_surv={row['val_loss_survival']:.5f} | "
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
                        "model": state_dict_model(model).state_dict(),
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
                    "model": state_dict_model(model).state_dict(),
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
                _log(f"Early stopping at epoch={epoch}, patience={patience}.")
                stop_now = 1

        if _is_dist():
            stop_tensor = torch.tensor([stop_now], device=device, dtype=torch.int32)
            dist.broadcast(stop_tensor, src=0)
            stop_now = int(stop_tensor.item())
        if stop_now == 1:
            break

    elapsed = time.time() - t0
    _log(f"Training finished in {elapsed:.1f}s.")

    if _is_dist():
        dist.barrier()

    if not _is_main():
        cleanup_distributed()
        return

    if best_ckpt.exists():
        best_state = torch.load(best_ckpt, map_location=device)
        state_dict_model(model).load_state_dict(best_state["model"])

    loss_df = pd.DataFrame(loss_rows)
    save_loss_curve(
        loss_df=loss_df,
        out_path_png=result_dir / "train_val_loss_curve.png",
        out_path_csv=result_dir / "train_val_loss_curve.csv",
        title="CMA-Surv Train/Val Loss",
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

    assert test_loader is not None
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
    cif_discharge, cif_death = compute_cif(discharge_h, death_h)

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

    joblib.dump(
        {
            "artifacts": artifacts,
            "model_name": bert_model_name,
            "time_feature_dim": int(bundle["time_feature_dim"]),
            "static_feature_dim": int(bundle["static_feature_dim"]),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
        },
        result_dir / "cma_surv_artifacts.joblib",
    )

    torch.save(state_dict_model(model).state_dict(), result_dir / "cma_surv_model.pt")

    run_config = {
        "result_dir": str(result_dir),
        "result_subdir": result_subdir,
        "debug_max_stays": debug_max_stays,
        "cma_epochs": cma_epochs,
        "cma_batch_size": batch_size,
        "cma_lr": lr,
        "cma_weight_decay": weight_decay,
        "cma_max_text_len": max_text_len,
        "cma_grad_accum": grad_accum,
        "cma_patience": patience,
        "cma_seed": seed,
        "cma_num_workers": num_workers,
        "cma_freeze_bert": freeze_bert,
        "cma_multi_gpu": enable_multi_gpu,
        "cma_ddp": bool(use_ddp),
        "cma_heartbeat_steps": heartbeat_steps,
        "cma_model_name": bert_model_name,
        "cma_resume": resume,
        "device": str(device),
        "detected_gpu_count": int(detected_gpu_count),
        "world_size": int(world_size),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "time_feature_dim": int(bundle["time_feature_dim"]),
        "static_feature_dim": int(bundle["static_feature_dim"]),
        "tokenizer_name": artifacts.tokenizer_name,
        "survival_intervals": artifacts.survival_intervals,
        "task_columns": {
            "task1": TASK1_COL,
            "task2": TASK2_COL,
            "task2_mask": TASK2_MASK_COL,
        },
    }
    save_json(run_config, result_dir / "run_config.json")

    _log(f"\nSaved CMA-Surv outputs to: {result_dir}")
    _log(metrics_df.to_string(index=False))

    cleanup_distributed()


if __name__ == "__main__":
    main()
