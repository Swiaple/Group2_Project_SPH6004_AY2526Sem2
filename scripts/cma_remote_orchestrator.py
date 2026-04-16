from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKLOG_PATH = PROJECT_ROOT / ".codex" / "WORKLOG.md"

REMOTE_HOST = os.getenv("REMOTE_HOST", "aspire2a.nus")
PBS_PROJECT_ID = (os.getenv("PBS_PROJECT_ID", "") or "").strip()
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "120") or "120")
DEFAULT_BATCH_SIZE = int(os.getenv("CMA_BATCH_SIZE", "32") or "32")
DEFAULT_GRAD_ACCUM = int(os.getenv("CMA_GRAD_ACCUM", "1") or "1")
DEFAULT_MODEL_NAME = os.getenv("CMA_MODEL_NAME", "emilyalsentzer/Bio_ClinicalBERT")
DEFAULT_CMA_EPOCHS = int(os.getenv("CMA_EPOCHS", "30") or "30")
DEFAULT_USE_GPU = os.getenv("CMA_USE_GPU", "1")
RUN_SMOKE = os.getenv("RUN_SMOKE", "1").strip().lower() not in {"0", "false", "no"}
DEFAULT_CONTAINER_IMAGE = os.getenv(
    "CMA_CONTAINER_IMAGE",
    "/scratch/GPFS/app/apps/containers/pytorch/pytorch-nvidia-22.12-py3.sif",
)

DATASET_FILES = [
    "MIMIC-IV-static(Group Assignment).csv",
    "MIMIC-IV-text(Group Assignment).csv",
    "MIMIC-IV-time_series(Group Assignment).csv",
]


@dataclass
class PhaseConfig:
    phase: str
    debug_max_stays: int
    epochs: int
    batch_size: int
    grad_accum: int
    resume: int
    attempt: int = 1

    @property
    def result_subdir(self) -> str:
        return f"cma_surv_{self.phase}_attempt{self.attempt}"


def append_worklog(title: str, lines: List[str]) -> None:
    ts = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    block = [f"\n## {ts} (Asia/Shanghai) - {title}"]
    block.extend([f"- {line}" for line in lines])
    WORKLOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(WORKLOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(block))


def run_local(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    print(f"[local] {' '.join(cmd)}")
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return proc


def run_ssh(host: str, command: str, check: bool = True) -> str:
    proc = run_local(["ssh", "-o", "BatchMode=yes", host, command], check=check)
    return (proc.stdout or "").strip()


def scp_put(local_path: Path, remote_spec: str) -> None:
    run_local(["scp", str(local_path), remote_spec], check=True)


def scp_get(remote_spec: str, local_path: Path) -> None:
    run_local(["scp", "-r", remote_spec, str(local_path)], check=True)


def connectivity_gate(host: str) -> None:
    out = run_ssh(host, "hostname && whoami", check=True)
    print("[remote connectivity]\n" + out)
    append_worklog(
        "CMA remote connectivity gate",
        [
            f"Host: {host}",
            "Connectivity check command passed: `ssh -o BatchMode=yes <host> \"hostname && whoami\"`.",
            f"Output: {out.replace(chr(10), ' | ')}",
        ],
    )


def detect_scratch_root(host: str) -> str:
    cmd = """candidates=()
if [ -n "$SCRATCH" ]; then candidates+=("$SCRATCH"); fi
candidates+=("$HOME/scratch" "/scratch/$USER" "/scratch/users/$USER" "/scratch/users/nus/$USER" "$HOME")
for p in "${candidates[@]}"; do
  if [ -d "$p" ] && [ -w "$p" ]; then
    readlink -f "$p"
    exit 0
  fi
done
readlink -f "$HOME"
"""
    out = run_ssh(host, cmd, check=True)
    root = out.splitlines()[0].strip()
    if not root:
        raise RuntimeError("Unable to detect remote scratch root.")
    return root


def create_remote_layout(host: str, remote_root: str) -> None:
    run_ssh(
        host,
        f'mkdir -p "{remote_root}/code/model" "{remote_root}/code/utils" "{remote_root}/dataset" "{remote_root}/result" "{remote_root}/logs"',
        check=True,
    )


def sync_code_and_data(host: str, remote_root: str) -> None:
    code_files = [
        PROJECT_ROOT / "model" / "cma_surv.py",
        PROJECT_ROOT / "model" / "cma_train.py",
        PROJECT_ROOT / "utils" / "cma_dataset.py",
        PROJECT_ROOT / "utils" / "multitask_common.py",
    ]
    for file_path in code_files:
        if not file_path.exists():
            raise FileNotFoundError(f"Local code file not found: {file_path}")
        subdir = "model" if file_path.parent.name == "model" else "utils"
        scp_put(file_path, f"{host}:{remote_root}/code/{subdir}/")

    for name in DATASET_FILES:
        local_data = PROJECT_ROOT / "dataset" / name
        if not local_data.exists():
            raise FileNotFoundError(f"Dataset file not found: {local_data}")
        scp_put(local_data, f"{host}:{remote_root}/dataset/")

    append_worklog(
        "CMA remote staging completed",
        [
            f"Remote root: `{remote_root}`",
            "Uploaded code files: model/cma_surv.py, scripts/cma_train.py, utils/cma_dataset.py, utils/multitask_common.py",
            "Uploaded dataset files: static/text/time_series CSV.",
        ],
    )


def _render_pbs_script(remote_root: str, cfg: PhaseConfig) -> str:
    pbs_project_line = f"#PBS -P {PBS_PROJECT_ID}\n" if PBS_PROJECT_ID else ""
    return f"""#!/bin/bash
#PBS -N cma_surv_{cfg.phase}
#PBS -q normal
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
#PBS -l walltime=24:00:00
#PBS -j oe
{pbs_project_line}
set -euo pipefail

BASE={remote_root}
CODE_DIR=${{BASE}}/code
DATA_DIR=${{BASE}}/dataset
RESULT_ROOT=${{BASE}}/result
RESULT_SUBDIR={cfg.result_subdir}
RUN_DIR=${{RESULT_ROOT}}/${{RESULT_SUBDIR}}
LOG_DIR=${{BASE}}/logs
TRAIN_LOG=${{RUN_DIR}}/train_${{PBS_JOBID}}.log
IMG={DEFAULT_CONTAINER_IMAGE}
VENV_DIR=${{BASE}}/venv/cma_surv

mkdir -p "${{RUN_DIR}}" "${{LOG_DIR}}" "${{BASE}}/tmp"

module load singularity
set +e
singularity exec -e --nv \\
  -B "${{BASE}}:${{BASE}}" \\
  -B /home/users/nus/e1553307:/home/users/nus/e1553307 \\
  "${{IMG}}" \\
  bash -lc "
set -euo pipefail

if [[ -x '${{VENV_DIR}}/bin/python3' ]]; then
  if ! '${{VENV_DIR}}/bin/python3' -V >/dev/null 2>&1; then
    rm -rf '${{VENV_DIR}}'
  fi
else
  rm -rf '${{VENV_DIR}}' || true
fi
if [[ ! -x '${{VENV_DIR}}/bin/python3' ]]; then
  python3 -m venv '${{VENV_DIR}}'
fi

source '${{VENV_DIR}}/bin/activate'
python -m pip install -U pip
python -m pip install -U numpy pandas scikit-learn matplotlib joblib torch torchvision torchaudio transformers

ln -sfn '${{DATA_DIR}}' '${{CODE_DIR}}/dataset'
ln -sfn '${{RESULT_ROOT}}' '${{CODE_DIR}}/result'
export PYTHONPATH='${{CODE_DIR}}:'\\\"\\${{PYTHONPATH:-}}\\\"
export RESULT_SUBDIR='{cfg.result_subdir}'
export DEBUG_MAX_STAYS='{cfg.debug_max_stays}'
export CMA_EPOCHS='{cfg.epochs}'
export CMA_BATCH_SIZE='{cfg.batch_size}'
export CMA_GRAD_ACCUM='{cfg.grad_accum}'
export CMA_RESUME='{cfg.resume}'
export CMA_USE_GPU='{DEFAULT_USE_GPU}'
export CMA_MODEL_NAME='{DEFAULT_MODEL_NAME}'

cd '${{CODE_DIR}}'
python scripts/cma_train.py > '${{TRAIN_LOG}}' 2>&1
"
RC=$?
set -e
echo "$RC" > "${{RUN_DIR}}/job_exit_code.txt"
echo "run_dir=${{RUN_DIR}}"
exit "$RC"
"""


def submit_job(host: str, remote_root: str, cfg: PhaseConfig) -> str:
    local_pbs = PROJECT_ROOT / ".codex" / f"cma_surv_{cfg.phase}_attempt{cfg.attempt}.pbs"
    local_pbs.parent.mkdir(parents=True, exist_ok=True)
    # Force Unix newlines so PBS shebang is valid on Linux compute nodes.
    with open(local_pbs, "w", encoding="utf-8", newline="\n") as f:
        f.write(_render_pbs_script(remote_root=remote_root, cfg=cfg))

    remote_pbs = f"{remote_root}/code/cma_surv_{cfg.phase}_attempt{cfg.attempt}.pbs"
    scp_put(local_pbs, f"{host}:{remote_pbs}")
    run_ssh(host, f"sed -i 's/\\r$//' \"{remote_pbs}\"", check=True)
    run_ssh(host, f'chmod +x "{remote_pbs}"', check=True)

    out = run_ssh(host, f'qsub "{remote_pbs}"', check=True)
    job_id = out.splitlines()[-1].strip()
    if not job_id:
        raise RuntimeError(f"Failed to parse qsub output: {out}")
    append_worklog(
        f"CMA {cfg.phase} job submitted",
        [
            f"Attempt: {cfg.attempt}",
            f"Job ID: {job_id}",
            f"Result subdir: {cfg.result_subdir}",
            f"PBS script: {remote_pbs}",
            f"Config: debug_max_stays={cfg.debug_max_stays}, epochs={cfg.epochs}, batch_size={cfg.batch_size}, grad_accum={cfg.grad_accum}, resume={cfg.resume}",
        ],
    )
    return job_id


def _find_latest_job_log(host: str, remote_root: str, job_id: str) -> str:
    job_num = job_id.split(".")[0]
    cmd = (
        f'ls -1t '
        f'"{remote_root}"/logs/*"{job_id}"*.log '
        f'"{remote_root}"/result/*/*"{job_id}"*.log '
        f'"$HOME"/cma_surv*.o{job_num} '
        f'"$HOME"/cma_surv*.e{job_num} '
        f'2>/dev/null | head -n 1'
    )
    return run_ssh(host, cmd, check=False).strip()


def _classify_failure(log_text: str) -> str:
    low = log_text.lower()
    if "modulenotfounderror" in low or "importerror" in low:
        return "missing_deps"
    if "cuda out of memory" in low or "out of memory" in low:
        return "oom"
    if "walltime" in low or "killed" in low or "signal 9" in low:
        return "walltime_or_killed"
    if "filenotfounderror" in low or "no such file or directory" in low:
        return "path_missing"
    return "unknown"


def monitor_job(host: str, remote_root: str, job_id: str, result_subdir: str, poll_seconds: int) -> Tuple[bool, str]:
    print(f"[monitor] job={job_id}, result_subdir={result_subdir}")
    job_token = job_id.split(".")[0]
    while True:
        qstat_out = run_ssh(host, f"qstat -ans 2>/dev/null | grep -w '{job_token}' || true", check=False)
        log_path = _find_latest_job_log(host, remote_root, job_id)
        if log_path:
            tail = run_ssh(host, f'tail -n 20 "{log_path}"', check=False)
            if tail.strip():
                print(f"[log tail] {log_path}\n{tail}\n")

        if qstat_out.strip():
            time.sleep(poll_seconds)
            continue
        break

    artifact_check = run_ssh(
        host,
        (
            f'for f in metrics_summary.csv test_predictions.csv run_config.json attention_heatmap_examples.png; do '
            f'if [ ! -s "{remote_root}/result/{result_subdir}/$f" ]; then echo "$f"; fi; done'
        ),
        check=False,
    )
    missing = [x.strip() for x in artifact_check.splitlines() if x.strip()]
    if not missing:
        return True, "success"

    log_path = _find_latest_job_log(host, remote_root, job_id)
    log_tail = run_ssh(host, f'tail -n 300 "{log_path}"', check=False) if log_path else ""
    reason = _classify_failure(log_tail)
    return False, reason


def apply_failure_fix(host: str, remote_root: str, reason: str, cfg: PhaseConfig) -> PhaseConfig:
    action_lines = [f"Failure reason: {reason}", f"Attempt before fix: {cfg.attempt}"]
    if reason == "missing_deps":
        action_lines.append("Applied fix: dependencies are installed in each container run; immediate resubmit.")
    elif reason == "oom":
        if cfg.batch_size > 8:
            cfg.batch_size = max(8, cfg.batch_size // 2)
            action_lines.append(f"Applied fix: reduced CMA_BATCH_SIZE to {cfg.batch_size}.")
        else:
            cfg.grad_accum = max(1, cfg.grad_accum * 2)
            action_lines.append(f"Applied fix: increased CMA_GRAD_ACCUM to {cfg.grad_accum}.")
    elif reason == "walltime_or_killed":
        cfg.resume = 1
        action_lines.append("Applied fix: set CMA_RESUME=1 for next attempt.")
    elif reason == "path_missing":
        sync_code_and_data(host, remote_root)
        action_lines.append("Applied fix: resynced code and dataset.")
    else:
        action_lines.append("Applied fix: no automatic parameter change; resubmitting with current settings.")

    cfg.attempt += 1
    action_lines.append(f"Next attempt: {cfg.attempt}")
    append_worklog(f"CMA {cfg.phase} failure auto-fix", action_lines)
    return cfg


def run_phase_with_retry(host: str, remote_root: str, cfg: PhaseConfig) -> str:
    while True:
        job_id = submit_job(host, remote_root, cfg)
        ok, reason = monitor_job(
            host=host,
            remote_root=remote_root,
            job_id=job_id,
            result_subdir=cfg.result_subdir,
            poll_seconds=POLL_SECONDS,
        )
        if ok:
            append_worklog(
                f"CMA {cfg.phase} phase success",
                [
                    f"Job ID: {job_id}",
                    f"Attempt: {cfg.attempt}",
                    f"Result subdir: {cfg.result_subdir}",
                ],
            )
            return cfg.result_subdir
        cfg = apply_failure_fix(host, remote_root, reason, cfg)


def pull_remote_result(host: str, remote_root: str, result_subdir: str, local_stamp: str) -> Path:
    local_dir = PROJECT_ROOT / "result" / f"cma_surv_remote_{local_stamp}"
    local_dir.mkdir(parents=True, exist_ok=True)
    scp_get(f"{host}:{remote_root}/result/{result_subdir}", local_dir)
    pulled = local_dir / result_subdir
    if not pulled.exists():
        raise RuntimeError(f"Result pull failed: {pulled} not found.")
    append_worklog(
        "CMA remote result pulled back",
        [
            f"Remote result: `{remote_root}/result/{result_subdir}`",
            f"Local destination: `{pulled}`",
        ],
    )
    return pulled


def run_local_comparison(cma_metrics_path: Path) -> None:
    env = os.environ.copy()
    env["CMA_METRICS_PATH"] = str(cma_metrics_path)
    cmd = ["python", str(PROJECT_ROOT / "model" / "cma_compare_results.py")]
    print(f"[local] {' '.join(cmd)} with CMA_METRICS_PATH={cma_metrics_path}")
    proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Comparison script failed.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    print(proc.stdout)
    append_worklog(
        "CMA baseline comparison generated",
        [
            f"Input CMA metrics: `{cma_metrics_path}`",
            "Ran `scripts/cma_compare_results.py` successfully.",
        ],
    )


def main() -> None:
    append_worklog(
        "CMA remote orchestration start",
        [
            f"Requested host: {REMOTE_HOST}",
            "Plan: connectivity gate -> staging -> smoke phase -> full phase with infinite retry -> pullback -> baseline comparison.",
        ],
    )

    connectivity_gate(REMOTE_HOST)
    scratch_root = detect_scratch_root(REMOTE_HOST)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    remote_root = f"{scratch_root}/sph6004/cma_surv_{ts}"
    create_remote_layout(REMOTE_HOST, remote_root)
    sync_code_and_data(REMOTE_HOST, remote_root)

    if RUN_SMOKE:
        smoke_cfg = PhaseConfig(
            phase="smoke",
            debug_max_stays=200,
            epochs=2,
            batch_size=DEFAULT_BATCH_SIZE,
            grad_accum=DEFAULT_GRAD_ACCUM,
            resume=0,
            attempt=1,
        )
        run_phase_with_retry(REMOTE_HOST, remote_root, smoke_cfg)
    else:
        append_worklog(
            "CMA smoke phase skipped",
            [
                "RUN_SMOKE=0 detected; skipping smoke and submitting full phase directly.",
            ],
        )

    full_cfg = PhaseConfig(
        phase="full",
        debug_max_stays=0,
        epochs=DEFAULT_CMA_EPOCHS,
        batch_size=DEFAULT_BATCH_SIZE,
        grad_accum=DEFAULT_GRAD_ACCUM,
        resume=0,
        attempt=1,
    )
    final_result_subdir = run_phase_with_retry(REMOTE_HOST, remote_root, full_cfg)

    local_result_dir = pull_remote_result(REMOTE_HOST, remote_root, final_result_subdir, ts)
    cma_metrics_path = local_result_dir / "metrics_summary.csv"
    run_local_comparison(cma_metrics_path)

    append_worklog(
        "CMA remote orchestration completed",
        [
            f"Remote root: `{remote_root}`",
            f"Final result subdir: `{final_result_subdir}`",
            f"Pulled local result: `{local_result_dir}`",
        ],
    )


if __name__ == "__main__":
    main()
