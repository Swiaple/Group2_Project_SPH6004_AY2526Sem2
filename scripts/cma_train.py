from __future__ import annotations

import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
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


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)) or str(default))


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


def masked_task2_loss(
    task2_logits: torch.Tensor,
    task2_label: torch.Tensor,
    task2_mask: torch.Tensor,
) -> Tuple[torch.Tensor, int]:
    mask = task2_mask > 0.5
    valid_count = int(mask.sum().item())
    if valid_count > 0:
        return F.binary_cross_entropy_with_logits(task2_logits[mask], task2_label[mask]), valid_count
    return torch.zeros((), device=task2_logits.device, dtype=task2_logits.dtype), valid_count


def survival_loss(survival_logits: torch.Tensor, survival_target: torch.Tensor) -> torch.Tensor:
    # [B, 3 bins, 3 classes] -> average CE across all bins.
    return F.cross_entropy(survival_logits.reshape(-1, 3), survival_target.reshape(-1))


def task1_logit_from_survival_logits(survival_logits: torch.Tensor) -> torch.Tensor:
    # task1 uses discharge probability in first survival bin.
    bin0 = survival_logits[:, 0, :]
    neg = torch.stack([bin0[:, 0], bin0[:, 2]], dim=-1)
    return bin0[:, 1] - torch.logsumexp(neg, dim=-1)


def task1_probability_from_survival_logits(survival_logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(task1_logit_from_survival_logits(survival_logits))


def joint_loss(
    task1_prob: torch.Tensor,
    task2_logits: torch.Tensor,
    task1_label: torch.Tensor,
    task2_label: torch.Tensor,
    use_mask_for_joint: bool,
    task2_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    task2_prob = torch.sigmoid(task2_logits)
    joint_prob = torch.clamp(task1_prob * task2_prob, min=1e-6, max=1.0 - 1e-6)
    joint_target = ((task1_label > 0.5) & (task2_label > 0.5)).to(joint_prob.dtype)
    if use_mask_for_joint and task2_mask is not None:
        mask = task2_mask > 0.5
        if mask.any():
            return F.binary_cross_entropy(joint_prob[mask], joint_target[mask])
        return torch.zeros((), device=joint_prob.device, dtype=joint_prob.dtype)
    return F.binary_cross_entropy(joint_prob, joint_target)


def build_optimizer(
    model: nn.Module,
    lr_head: float,
    lr_bert: float,
    weight_decay: float,
) -> AdamW:
    no_decay_tokens = ("bias", "LayerNorm.weight", "LayerNorm.bias")

    head_decay: List[torch.nn.Parameter] = []
    head_no_decay: List[torch.nn.Parameter] = []
    bert_decay: List[torch.nn.Parameter] = []
    bert_no_decay: List[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_bert = name.startswith("text_encoder.")
        use_decay = not any(token in name for token in no_decay_tokens)
        if is_bert:
            if use_decay:
                bert_decay.append(param)
            else:
                bert_no_decay.append(param)
        else:
            if use_decay:
                head_decay.append(param)
            else:
                head_no_decay.append(param)

    groups: List[Dict[str, object]] = []
    if head_decay:
        groups.append({"params": head_decay, "lr": lr_head, "weight_decay": weight_decay, "group_name": "head_decay"})
    if head_no_decay:
        groups.append({"params": head_no_decay, "lr": lr_head, "weight_decay": 0.0, "group_name": "head_no_decay"})
    if bert_decay:
        groups.append({"params": bert_decay, "lr": lr_bert, "weight_decay": weight_decay, "group_name": "bert_decay"})
    if bert_no_decay:
        groups.append({"params": bert_no_decay, "lr": lr_bert, "weight_decay": 0.0, "group_name": "bert_no_decay"})
    if not groups:
        raise RuntimeError("No trainable parameters found for optimizer.")
    return AdamW(groups)


def build_scheduler(
    optimizer: Optimizer,
    total_update_steps: int,
    warmup_ratio: float,
    scheduler_type: str,
    min_lr_ratio: float,
) -> Tuple[LambdaLR, int]:
    total_update_steps = max(int(total_update_steps), 1)
    warmup_ratio = min(max(float(warmup_ratio), 0.0), 0.95)
    warmup_steps = int(total_update_steps * warmup_ratio)
    min_lr_ratio = min(max(float(min_lr_ratio), 0.0), 1.0)
    scheduler_type = (scheduler_type or "cosine").strip().lower()
    if scheduler_type not in {"cosine", "linear"}:
        scheduler_type = "cosine"

    def lr_lambda(step_idx: int) -> float:
        if warmup_steps > 0 and step_idx < warmup_steps:
            return max(1e-8, float(step_idx + 1) / float(warmup_steps))
        if total_update_steps <= warmup_steps:
            return 1.0
        progress = float(step_idx - warmup_steps) / float(max(total_update_steps - warmup_steps, 1))
        progress = min(max(progress, 0.0), 1.0)
        if scheduler_type == "linear":
            return max(min_lr_ratio, 1.0 - (1.0 - min_lr_ratio) * progress)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=[lr_lambda] * len(optimizer.param_groups)), warmup_steps


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
    optimizer: Optional[Optimizer],
    scheduler: Optional[LambdaLR],
    scaler: torch.cuda.amp.GradScaler,
    grad_accum: int,
    use_amp: bool,
    epoch: int,
    split_name: str,
    heartbeat_steps: int,
    lambda_task2: float,
    loss_weight_surv: float,
    loss_weight_task1: float,
    loss_weight_joint: float,
    max_grad_norm: float,
    use_mask_for_joint: bool,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_surv = 0.0
    total_task1 = 0.0
    total_joint = 0.0
    total_l1 = 0.0
    total_task2 = 0.0
    total_task2_weighted = 0.0
    total_all = 0.0
    total_task2_valid = 0.0
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
        task1_label = batch["task1_label"].to(device, non_blocking=True)
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
                l_task2, task2_valid_count = masked_task2_loss(out["task2_logits"], task2_label, task2_mask)
                task1_logit = task1_logit_from_survival_logits(out["survival_logits"])
                task1_prob = torch.sigmoid(task1_logit)
                l_task1 = F.binary_cross_entropy_with_logits(task1_logit, task1_label)
                l_joint = joint_loss(
                    task1_prob=task1_prob,
                    task2_logits=out["task2_logits"],
                    task1_label=task1_label,
                    task2_label=task2_label,
                    use_mask_for_joint=use_mask_for_joint,
                    task2_mask=task2_mask,
                )
                l1 = loss_weight_surv * l_surv + loss_weight_task1 * l_task1 + loss_weight_joint * l_joint
                l_total = l1 + float(lambda_task2) * l_task2

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
                    scaler.unscale_(optimizer)
                    if max_grad_norm > 0:
                        clip_grad_norm_(state_dict_model(model).parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()

        total_surv += float(l_surv.detach().cpu().item())
        total_task1 += float(l_task1.detach().cpu().item())
        total_joint += float(l_joint.detach().cpu().item())
        total_l1 += float(l1.detach().cpu().item())
        total_task2 += float(l_task2.detach().cpu().item())
        total_all += float(l_total.detach().cpu().item())
        total_task2_valid += float(task2_valid_count)
        if task2_valid_count > 0:
            total_task2_weighted += float(l_task2.detach().cpu().item()) * float(task2_valid_count)
        n_batches += 1

        if is_train and heartbeat_steps > 0 and step % heartbeat_steps == 0 and _is_main():
            elapsed = max(time.time() - t0, 1e-6)
            it_s = step / elapsed
            lrs = [float(g["lr"]) for g in optimizer.param_groups] if optimizer is not None else [0.0]
            _log(
                f"[HB] epoch={epoch:03d} split={split_name} "
                f"step={step}/{loader_len} "
                f"loss_total={l_total.item():.5f} l1={l1.item():.5f} l2={l_task2.item():.5f} "
                f"surv={l_surv.item():.5f} task1={l_task1.item():.5f} joint={l_joint.item():.5f} "
                f"lambda={lambda_task2:.4f} lr_min={min(lrs):.2e} lr_max={max(lrs):.2e} it/s={it_s:.2f}"
            )

    if is_train and (len(loader) % max(grad_accum, 1) != 0):
        scaler.unscale_(optimizer)
        if max_grad_norm > 0:
            clip_grad_norm_(state_dict_model(model).parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()

    metric_tensor = torch.tensor(
        [
            total_surv,
            total_task1,
            total_joint,
            total_l1,
            total_task2,
            total_task2_weighted,
            total_all,
            total_task2_valid,
            float(max(n_batches, 1)),
        ],
        device=device,
        dtype=torch.float64,
    )
    if _is_dist():
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)

    denom = float(metric_tensor[8].item())
    valid_cnt = float(metric_tensor[7].item())
    task2_valid_loss = float(metric_tensor[5].item() / valid_cnt) if valid_cnt > 0 else 0.0
    return {
        "loss_survival": float(metric_tensor[0].item() / denom),
        "loss_task1": float(metric_tensor[1].item() / denom),
        "loss_joint": float(metric_tensor[2].item() / denom),
        "loss_l1": float(metric_tensor[3].item() / denom),
        "loss_task2": float(metric_tensor[4].item() / denom),
        "loss_task2_valid": task2_valid_loss,
        "loss_total": float(metric_tensor[6].item() / denom),
        "task2_valid_count": valid_cnt,
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
    lr_head = _env_float("CMA_LR_HEAD", _env_float("CMA_LR", 1e-4))
    lr_bert = _env_float("CMA_LR_BERT", max(lr_head * 0.1, 1e-6))
    weight_decay = float(os.getenv("CMA_WEIGHT_DECAY", "1e-2") or "1e-2")
    scheduler_type = (os.getenv("CMA_SCHEDULER", "cosine") or "cosine").strip().lower()
    warmup_ratio = _env_float("CMA_WARMUP_RATIO", 0.1)
    min_lr_ratio = _env_float("CMA_MIN_LR_RATIO", 0.05)
    max_grad_norm = _env_float("CMA_MAX_GRAD_NORM", 1.0)

    loss_weight_surv = _env_float("CMA_LOSS_WEIGHT_SURV", 1.0)
    loss_weight_task1 = _env_float("CMA_LOSS_WEIGHT_TASK1", 1.0)
    loss_weight_joint = _env_float("CMA_LOSS_WEIGHT_JOINT", 1.0)
    use_mask_for_joint = (os.getenv("CMA_USE_MASK_FOR_JOINT", "0") or "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }

    lambda_task2 = _env_float("CMA_LAMBDA_INIT", 1.0)
    lambda_smooth = _env_float("CMA_LAMBDA_SMOOTH", 0.9)
    lambda_min = _env_float("CMA_LAMBDA_MIN", 0.1)
    lambda_max = _env_float("CMA_LAMBDA_MAX", 10.0)
    lambda_min_valid = _env_int("CMA_LAMBDA_MIN_VALID", 64)

    max_text_len = int(os.getenv("CMA_MAX_TEXT_LEN", "128") or "128")
    grad_accum = int(os.getenv("CMA_GRAD_ACCUM", "1") or "1")
    heartbeat_steps = int(os.getenv("CMA_HEARTBEAT_STEPS", "200") or "200")
    use_gpu = (os.getenv("CMA_USE_GPU", "1") or "1").strip().lower() in {"1", "true", "yes", "y", "on"}
    resume = (os.getenv("CMA_RESUME", "0") or "0").strip().lower() in {"1", "true", "yes", "y", "on"}
    patience = int(os.getenv("CMA_PATIENCE", "5") or "5")
    seed = int(os.getenv("CMA_SEED", "42") or "42")
    num_workers = int(os.getenv("CMA_NUM_WORKERS", "0") or "0")
    freeze_bert_requested = (os.getenv("CMA_FREEZE_BERT", "0") or "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    freeze_bert = False
    enable_multi_gpu = (os.getenv("CMA_MULTI_GPU", "1") or "1").strip().lower() in {"1", "true", "yes", "y", "on"}
    bert_model_name = (
        os.getenv("CMA_MODEL_NAME", "emilyalsentzer/Bio_ClinicalBERT") or "emilyalsentzer/Bio_ClinicalBERT"
    ).strip()
    ddp_find_unused_env = (os.getenv("CMA_DDP_FIND_UNUSED", "0") or "0").strip().lower()
    ddp_static_graph = (os.getenv("CMA_DDP_STATIC_GRAPH", "1") or "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    ddp_bucket_cap_mb = _env_int("CMA_DDP_BUCKET_CAP_MB", 50)

    earlystop_mode = (os.getenv("CMA_EARLYSTOP_MODE", "aligned") or "aligned").strip().lower()
    if earlystop_mode not in {"legacy", "aligned"}:
        earlystop_mode = "aligned"

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
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
    use_amp = bool(use_cuda)
    detected_gpu_count = torch.cuda.device_count() if use_cuda else 0

    if freeze_bert_requested and _is_main():
        _log("CMA_FREEZE_BERT=1 was requested, but this pipeline enforces unfreezed BERT training.")

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

    loader_kwargs: Dict[str, object] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": bool(use_cuda),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

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
        else:
            ddp_find_unused = False
        _log("Wrapping model with DistributedDataParallel.")
        _log(
            f"DDP config: find_unused_parameters={ddp_find_unused}, static_graph={ddp_static_graph}, "
            f"bucket_cap_mb={ddp_bucket_cap_mb}",
            main_only=False,
        )
        ddp_kwargs: Dict[str, object] = {
            "device_ids": [local_rank] if use_cuda else None,
            "output_device": local_rank if use_cuda else None,
            "find_unused_parameters": ddp_find_unused,
            "broadcast_buffers": False,
            "gradient_as_bucket_view": True,
        }
        try:
            model = DDP(
                model,
                static_graph=ddp_static_graph,
                bucket_cap_mb=ddp_bucket_cap_mb,
                **ddp_kwargs,
            )
        except TypeError:
            model = DDP(model, **ddp_kwargs)

    optimizer = build_optimizer(
        model=state_dict_model(model),
        lr_head=lr_head,
        lr_bert=lr_bert,
        weight_decay=weight_decay,
    )
    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(grad_accum, 1)))
    total_update_steps = max(1, steps_per_epoch * max(cma_epochs, 1))
    scheduler, warmup_steps = build_scheduler(
        optimizer=optimizer,
        total_update_steps=total_update_steps,
        warmup_ratio=warmup_ratio,
        scheduler_type=scheduler_type,
        min_lr_ratio=min_lr_ratio,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if _is_main():
        group_summaries: List[str] = []
        for g in optimizer.param_groups:
            n_params = int(sum(int(p.numel()) for p in g["params"]))
            group_name = str(g.get("group_name", "group"))
            group_summaries.append(
                f"{group_name}(lr={float(g['lr']):.2e},wd={float(g['weight_decay']):.2e},n={n_params})"
            )
        _log("Optimizer groups: " + ", ".join(group_summaries))

    checkpoint_dir = result_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = checkpoint_dir / "latest.pt"
    best_ckpt = checkpoint_dir / "best.pt"

    start_epoch = 1
    best_val_loss_legacy = float("inf")
    best_val_loss_aligned = float("inf")
    best_epoch_legacy = 0
    best_epoch_aligned = 0
    no_improve_legacy = 0
    no_improve_aligned = 0
    loss_rows: List[Dict[str, float]] = []

    if resume and latest_ckpt.exists():
        _log(f"Resuming from {latest_ckpt} ...")
        ckpt = torch.load(latest_ckpt, map_location=device)
        state_dict_model(model).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_loss_legacy = float(ckpt.get("best_val_loss_legacy", best_val_loss_legacy))
        best_val_loss_aligned = float(ckpt.get("best_val_loss_aligned", best_val_loss_aligned))
        best_epoch_legacy = int(ckpt.get("best_epoch_legacy", best_epoch_legacy))
        best_epoch_aligned = int(ckpt.get("best_epoch_aligned", best_epoch_aligned))
        no_improve_legacy = int(ckpt.get("no_improve_legacy", no_improve_legacy))
        no_improve_aligned = int(ckpt.get("no_improve_aligned", no_improve_aligned))
        lambda_task2 = float(ckpt.get("lambda_task2", lambda_task2))
        loss_rows = ckpt.get("loss_rows", [])
        _log(
            f"Resume start epoch={start_epoch}, best_legacy={best_val_loss_legacy:.6f}, "
            f"best_aligned={best_val_loss_aligned:.6f}, lambda={lambda_task2:.4f}"
        )

    t0 = time.time()
    for epoch in range(start_epoch, cma_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        lambda_before = float(lambda_task2)
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            grad_accum=grad_accum,
            use_amp=use_amp,
            epoch=epoch,
            split_name="train",
            heartbeat_steps=heartbeat_steps,
            lambda_task2=lambda_before,
            loss_weight_surv=loss_weight_surv,
            loss_weight_task1=loss_weight_task1,
            loss_weight_joint=loss_weight_joint,
            max_grad_norm=max_grad_norm,
            use_mask_for_joint=use_mask_for_joint,
        )
        lambda_ratio = float("nan")
        if _is_main():
            if train_metrics["task2_valid_count"] >= float(lambda_min_valid) and train_metrics["loss_task2_valid"] > 0:
                ratio = train_metrics["loss_l1"] / max(train_metrics["loss_task2_valid"], 1e-8)
                lambda_ratio = float(ratio)
                lambda_task2 = float(
                    np.clip(lambda_smooth * lambda_task2 + (1.0 - lambda_smooth) * ratio, lambda_min, lambda_max)
                )
        if _is_dist():
            lambda_tensor = torch.tensor([lambda_task2], device=device, dtype=torch.float64)
            dist.broadcast(lambda_tensor, src=0)
            lambda_task2 = float(lambda_tensor.item())

        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            optimizer=None,
            scheduler=None,
            scaler=scaler,
            grad_accum=1,
            use_amp=use_amp,
            epoch=epoch,
            split_name="val",
            heartbeat_steps=0,
            lambda_task2=lambda_task2,
            loss_weight_surv=loss_weight_surv,
            loss_weight_task1=loss_weight_task1,
            loss_weight_joint=loss_weight_joint,
            max_grad_norm=0.0,
            use_mask_for_joint=use_mask_for_joint,
        )

        stop_now = 0
        if _is_main():
            val_total_legacy = float(val_metrics["loss_survival"] + val_metrics["loss_task2"])
            val_total_aligned = float(val_metrics["loss_total"])
            row = {
                "epoch": epoch,
                "train_loss_survival": train_metrics["loss_survival"],
                "train_loss_task1": train_metrics["loss_task1"],
                "train_loss_joint": train_metrics["loss_joint"],
                "train_loss_l1": train_metrics["loss_l1"],
                "train_loss_task2_masked": train_metrics["loss_task2"],
                "train_loss_task2_masked_valid_only": train_metrics["loss_task2_valid"],
                "train_task2_valid_count": train_metrics["task2_valid_count"],
                "train_loss_total": train_metrics["loss_total"],
                "val_loss_survival": val_metrics["loss_survival"],
                "val_loss_task1": val_metrics["loss_task1"],
                "val_loss_joint": val_metrics["loss_joint"],
                "val_loss_l1": val_metrics["loss_l1"],
                "val_loss_task2_masked": val_metrics["loss_task2"],
                "val_loss_task2_masked_valid_only": val_metrics["loss_task2_valid"],
                "val_task2_valid_count": val_metrics["task2_valid_count"],
                "val_loss_total_legacy": val_total_legacy,
                "val_loss_total_aligned": val_total_aligned,
                "lambda_task2_before": lambda_before,
                "lambda_task2_after": lambda_task2,
                "lambda_ratio_l1_over_l2valid": lambda_ratio,
            }
            loss_rows.append(row)
            _log(
                f"Epoch {epoch:03d} | monitor={earlystop_mode} "
                f"val_legacy={row['val_loss_total_legacy']:.5f} val_aligned={row['val_loss_total_aligned']:.5f} "
                f"surv={row['val_loss_survival']:.5f} task1={row['val_loss_task1']:.5f} "
                f"joint={row['val_loss_joint']:.5f} task2={row['val_loss_task2_masked']:.5f} "
                f"lambda={lambda_before:.4f}->{lambda_task2:.4f}"
            )

            improved_legacy = row["val_loss_total_legacy"] + 1e-8 < best_val_loss_legacy
            improved_aligned = row["val_loss_total_aligned"] + 1e-8 < best_val_loss_aligned

            if improved_legacy:
                best_val_loss_legacy = float(row["val_loss_total_legacy"])
                best_epoch_legacy = int(epoch)
                no_improve_legacy = 0
            else:
                no_improve_legacy += 1

            if improved_aligned:
                best_val_loss_aligned = float(row["val_loss_total_aligned"])
                best_epoch_aligned = int(epoch)
                no_improve_aligned = 0
            else:
                no_improve_aligned += 1

            improved_selected = improved_aligned if earlystop_mode == "aligned" else improved_legacy
            if improved_selected:
                torch.save(
                    {
                        "epoch": epoch,
                        "model": state_dict_model(model).state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "lambda_task2": lambda_task2,
                        "best_val_loss_legacy": best_val_loss_legacy,
                        "best_val_loss_aligned": best_val_loss_aligned,
                        "best_epoch_legacy": best_epoch_legacy,
                        "best_epoch_aligned": best_epoch_aligned,
                        "no_improve_legacy": no_improve_legacy,
                        "no_improve_aligned": no_improve_aligned,
                        "loss_rows": loss_rows,
                    },
                    best_ckpt,
                )

            torch.save(
                {
                    "epoch": epoch,
                    "model": state_dict_model(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "lambda_task2": lambda_task2,
                    "best_val_loss_legacy": best_val_loss_legacy,
                    "best_val_loss_aligned": best_val_loss_aligned,
                    "best_epoch_legacy": best_epoch_legacy,
                    "best_epoch_aligned": best_epoch_aligned,
                    "no_improve_legacy": no_improve_legacy,
                    "no_improve_aligned": no_improve_aligned,
                    "loss_rows": loss_rows,
                },
                latest_ckpt,
            )

            patience_counter = no_improve_aligned if earlystop_mode == "aligned" else no_improve_legacy
            if patience_counter >= patience:
                _log(
                    f"Early stopping at epoch={epoch}, patience={patience}, "
                    f"mode={earlystop_mode}, no_improve={patience_counter}."
                )
                stop_now = 1

        if _is_dist():
            stop_tensor = torch.tensor([stop_now], device=device, dtype=torch.int32)
            dist.broadcast(stop_tensor, src=0)
            stop_now = int(stop_tensor.item())
        if stop_now == 1:
            break

    elapsed = time.time() - t0
    _log(f"Training finished in {elapsed:.1f}s. monitor={earlystop_mode}")

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
            task1_prob = task1_probability_from_survival_logits(out["survival_logits"]).detach().cpu().numpy()

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

    best_epoch_selected = best_epoch_aligned if earlystop_mode == "aligned" else best_epoch_legacy
    best_val_selected = best_val_loss_aligned if earlystop_mode == "aligned" else best_val_loss_legacy

    joblib.dump(
        {
            "artifacts": artifacts,
            "model_name": bert_model_name,
            "time_feature_dim": int(bundle["time_feature_dim"]),
            "static_feature_dim": int(bundle["static_feature_dim"]),
            "best_epoch_selected": int(best_epoch_selected),
            "best_val_loss_selected": float(best_val_selected),
            "best_epoch_legacy": int(best_epoch_legacy),
            "best_val_loss_legacy": float(best_val_loss_legacy),
            "best_epoch_aligned": int(best_epoch_aligned),
            "best_val_loss_aligned": float(best_val_loss_aligned),
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
        "cma_lr": lr_head,
        "cma_lr_head": lr_head,
        "cma_lr_bert": lr_bert,
        "cma_weight_decay": weight_decay,
        "cma_scheduler": scheduler_type,
        "cma_warmup_ratio": warmup_ratio,
        "cma_warmup_steps": warmup_steps,
        "cma_min_lr_ratio": min_lr_ratio,
        "cma_max_grad_norm": max_grad_norm,
        "cma_loss_weight_surv": loss_weight_surv,
        "cma_loss_weight_task1": loss_weight_task1,
        "cma_loss_weight_joint": loss_weight_joint,
        "cma_use_mask_for_joint": use_mask_for_joint,
        "cma_lambda_init": _env_float("CMA_LAMBDA_INIT", 1.0),
        "cma_lambda_smooth": lambda_smooth,
        "cma_lambda_min": lambda_min,
        "cma_lambda_max": lambda_max,
        "cma_lambda_min_valid": lambda_min_valid,
        "cma_lambda_final": float(lambda_task2),
        "cma_max_text_len": max_text_len,
        "cma_grad_accum": grad_accum,
        "cma_patience": patience,
        "cma_earlystop_mode": earlystop_mode,
        "cma_seed": seed,
        "cma_num_workers": num_workers,
        "cma_freeze_bert_requested": bool(freeze_bert_requested),
        "cma_freeze_bert_effective": bool(freeze_bert),
        "cma_multi_gpu": enable_multi_gpu,
        "cma_ddp": bool(use_ddp),
        "cma_ddp_find_unused": ddp_find_unused_env,
        "cma_ddp_static_graph": bool(ddp_static_graph),
        "cma_ddp_bucket_cap_mb": ddp_bucket_cap_mb,
        "cma_heartbeat_steps": heartbeat_steps,
        "cma_model_name": bert_model_name,
        "cma_resume": resume,
        "device": str(device),
        "detected_gpu_count": int(detected_gpu_count),
        "world_size": int(world_size),
        "best_epoch_selected": int(best_epoch_selected),
        "best_val_loss_selected": float(best_val_selected),
        "best_epoch_legacy": int(best_epoch_legacy),
        "best_val_loss_legacy": float(best_val_loss_legacy),
        "best_epoch_aligned": int(best_epoch_aligned),
        "best_val_loss_aligned": float(best_val_loss_aligned),
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
