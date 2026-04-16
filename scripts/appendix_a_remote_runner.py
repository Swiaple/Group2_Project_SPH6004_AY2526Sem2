from __future__ import annotations

import os
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKLOG_PATH = PROJECT_ROOT / ".codex" / "WORKLOG.md"

HOST = os.getenv("ASPIRE_HOST", "aspire2a.nus")
REMOTE_BASE = os.getenv("ASPIRE_SCRATCH_ROOT", "/scratch/users/nus/e1553307/sph6004")
POLL_SECONDS = int(os.getenv("ASPIRE_POLL_SECONDS", "120"))
PBS_PROJECT = (os.getenv("ASPIRE_PBS_PROJECT", "") or "").strip()

DATASET_FILES = [
    "MIMIC-IV-static(Group Assignment).csv",
    "MIMIC-IV-text(Group Assignment).csv",
    "MIMIC-IV-time_series(Group Assignment).csv",
]

BASELINE_XGB = "result/xgboostmultiresult_gpu_full_20260415_1115/metrics_summary.csv"
BASELINE_LOGI = "result/logisticmultiresult/metrics_summary.csv"


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S %z")


def append_worklog(message: str) -> None:
    line = f"- [{now_str()}] 远端流程：{message}"
    with WORKLOG_PATH.open("a", encoding="utf-8") as f:
        f.write("\n" + line)


def run_cmd(cmd: List[str], *, check: bool = True, capture_output: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        capture_output=capture_output,
        cwd=PROJECT_ROOT,
    )


def ssh(remote_cmd: str, *, check: bool = True) -> subprocess.CompletedProcess:
    return run_cmd(["ssh", "-o", "BatchMode=yes", HOST, remote_cmd], check=check)


def scp_upload(local_path: Path, remote_target: str, recursive: bool = False) -> None:
    cmd = ["scp", "-o", "BatchMode=yes"]
    if recursive:
        cmd.append("-r")
    cmd.extend([str(local_path), f"{HOST}:{remote_target}"])
    run_cmd(cmd, check=True)


def scp_upload_dir_contents(local_dir: Path, remote_dir: str) -> None:
    if not local_dir.exists():
        raise FileNotFoundError(f"Missing local directory: {local_dir}")
    for item in sorted(local_dir.iterdir()):
        if item.is_dir():
            # Avoid OpenSSH/scp directory canonicalization issues on nested folders (for example __pycache__).
            continue
        scp_upload(item, f"{remote_dir}/", recursive=False)


def ensure_connectivity_gate() -> None:
    res = ssh("hostname && whoami", check=False)
    if res.returncode != 0:
        append_worklog(
            f"连通性门禁失败：host={HOST}，stderr={res.stderr.strip()[:300]}"
        )
        raise RuntimeError(f"SSH gate failed: {res.stderr.strip()}")
    append_worklog(f"连通性门禁通过：{res.stdout.strip().replace(chr(10), ' | ')}")


def detect_failure_signature(log_text: str) -> str:
    lower = log_text.lower()
    if "modulenotfounderror" in lower or "importerror" in lower:
        return "import_error"
    if "out of memory" in lower or "cuda oom" in lower or "cuda error: out of memory" in lower:
        return "cuda_oom"
    if "walltime" in lower or "killed" in lower:
        return "walltime_or_killed"
    if "no such file or directory" in lower or "filenotfounderror" in lower:
        return "path_or_dataset_missing"
    if "traceback" in lower or "error" in lower:
        return "generic_error"
    return "unknown"


def make_pbs_script(
    remote_run_dir: str,
    result_subdir: str,
    phase: str,
    attempt: int,
    envs: Dict[str, str],
) -> str:
    env_lines = "\n".join([f"export {k}={shlex.quote(str(v))}" for k, v in envs.items()])
    log_path = f"{remote_run_dir}/logs/{phase}_attempt{attempt}.log"
    project_line = f"#PBS -P {PBS_PROJECT}" if PBS_PROJECT else ""
    pbs = f"""#!/bin/bash
#PBS -q normal
#PBS -N appendixA_mm_{phase}
#PBS -l select=1:ncpus=8:mem=64gb:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe
{project_line}
#PBS -o {log_path}
set -euo pipefail

REMOTE_RUN_DIR={shlex.quote(remote_run_dir)}
cd "$REMOTE_RUN_DIR/code"

python3 -m venv .venv || true
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn matplotlib joblib torch torchvision torchaudio transformers

{env_lines}
export RESULT_SUBDIR={shlex.quote(result_subdir)}

python scripts/train_multimodal_main.py
"""
    return pbs


def write_remote_file(remote_path: str, content: str) -> None:
    quoted = shlex.quote(content)
    cmd = f"cat > {shlex.quote(remote_path)} <<'EOF'\n{content}\nEOF"
    ssh(cmd)


def upload_project_payload(remote_run_dir: str) -> None:
    ssh(
        f"mkdir -p {shlex.quote(remote_run_dir)}/code "
        f"{shlex.quote(remote_run_dir)}/code/model "
        f"{shlex.quote(remote_run_dir)}/code/utils "
        f"{shlex.quote(remote_run_dir)}/code/dataset "
        f"{shlex.quote(remote_run_dir)}/dataset "
        f"{shlex.quote(remote_run_dir)}/result "
        f"{shlex.quote(remote_run_dir)}/logs"
    )

    # Upload code directories used by multimodal experiment.
    scp_upload_dir_contents(PROJECT_ROOT / "model", f"{remote_run_dir}/code/model")
    scp_upload_dir_contents(PROJECT_ROOT / "utils", f"{remote_run_dir}/code/utils")

    # Upload dataset files (explicitly fixed strategy).
    for name in DATASET_FILES:
        src = PROJECT_ROOT / "dataset" / name
        if not src.exists():
            raise FileNotFoundError(f"Missing dataset file: {src}")
        scp_upload(src, f"{remote_run_dir}/dataset/{name}")
        scp_upload(src, f"{remote_run_dir}/code/dataset/{name}")

    append_worklog(f"同步完成：remote_run_dir={remote_run_dir}")


def submit_job(remote_run_dir: str, phase: str, attempt: int, envs: Dict[str, str], result_subdir: str) -> str:
    pbs_content = make_pbs_script(
        remote_run_dir=remote_run_dir,
        result_subdir=result_subdir,
        phase=phase,
        attempt=attempt,
        envs=envs,
    )
    remote_pbs = f"{remote_run_dir}/code/{phase}_attempt{attempt}.pbs"
    write_remote_file(remote_pbs, pbs_content)

    res = ssh(
        f"cd {shlex.quote(remote_run_dir)}/code && qsub {shlex.quote(remote_pbs)}",
        check=False,
    )
    if res.returncode != 0:
        msg = (res.stderr or res.stdout or "").strip()
        append_worklog(f"提交失败：phase={phase} attempt={attempt} stderr={msg[:400]}")
        raise RuntimeError(f"qsub failed: {msg}")
    job_id = res.stdout.strip().splitlines()[-1].strip()
    append_worklog(f"提交作业：phase={phase} attempt={attempt} job_id={job_id}")
    return job_id


def wait_job_finish(job_id: str) -> None:
    while True:
        stat = ssh(f"qstat -ans {shlex.quote(job_id)}", check=False)
        if stat.returncode != 0:
            break
        time.sleep(POLL_SECONDS)


def remote_file_nonempty(path: str) -> bool:
    res = ssh(f"test -s {shlex.quote(path)}", check=False)
    return res.returncode == 0


def read_remote_log(remote_run_dir: str, phase: str, attempt: int) -> str:
    log_path = f"{remote_run_dir}/logs/{phase}_attempt{attempt}.log"
    res = ssh(f"test -f {shlex.quote(log_path)} && tail -n 200 {shlex.quote(log_path)} || true")
    return res.stdout


def ensure_phase_success(remote_run_dir: str, result_subdir: str) -> bool:
    required = [
        f"{remote_run_dir}/result/{result_subdir}/metrics_summary.csv",
        f"{remote_run_dir}/result/{result_subdir}/test_predictions.csv",
        f"{remote_run_dir}/result/{result_subdir}/run_config.json",
        f"{remote_run_dir}/result/{result_subdir}/attention_heatmap_examples.png",
    ]
    return all(remote_file_nonempty(p) for p in required)


def repair_on_failure(signature: str, envs: Dict[str, str], remote_run_dir: str) -> Dict[str, str]:
    new_envs = dict(envs)
    if signature == "cuda_oom":
        batch = int(new_envs.get("MM_BATCH_SIZE", "32"))
        grad_accum = int(new_envs.get("MM_GRAD_ACCUM", "1"))
        if batch > 8:
            batch = batch // 2
        else:
            grad_accum = grad_accum * 2
        new_envs["MM_BATCH_SIZE"] = str(batch)
        new_envs["MM_GRAD_ACCUM"] = str(grad_accum)
        append_worklog(f"故障修复：CUDA OOM -> MM_BATCH_SIZE={batch}, MM_GRAD_ACCUM={grad_accum}")
        return new_envs

    if signature == "walltime_or_killed":
        new_envs["MM_RESUME"] = "1"
        append_worklog("故障修复：walltime/killed -> 启用 MM_RESUME=1")
        return new_envs

    if signature == "path_or_dataset_missing":
        for name in DATASET_FILES:
            src = PROJECT_ROOT / "dataset" / name
            scp_upload(src, f"{remote_run_dir}/dataset/{name}")
        append_worklog("故障修复：检测到路径/数据缺失 -> 重新上传三份 CSV")
        return new_envs

    if signature == "import_error":
        # PBS 脚本每次都会 pip install 常用依赖，这里仅记录并重提。
        append_worklog("故障修复：import 错误 -> 保持依赖安装步骤并重提")
        return new_envs

    append_worklog(f"故障修复：未识别签名({signature}) -> 原参数重提")
    return new_envs


def run_phase(remote_run_dir: str, phase: str, base_envs: Dict[str, str], result_subdir: str) -> Dict[str, str]:
    envs = dict(base_envs)
    attempt = 0

    while True:
        attempt += 1
        job_id = submit_job(
            remote_run_dir=remote_run_dir,
            phase=phase,
            attempt=attempt,
            envs=envs,
            result_subdir=result_subdir,
        )
        wait_job_finish(job_id)

        if ensure_phase_success(remote_run_dir, result_subdir):
            append_worklog(f"阶段成功：phase={phase} attempt={attempt} job_id={job_id}")
            return envs

        log_text = read_remote_log(remote_run_dir, phase, attempt)
        signature = detect_failure_signature(log_text)
        append_worklog(
            f"阶段失败：phase={phase} attempt={attempt} job_id={job_id} signature={signature}"
        )
        envs = repair_on_failure(signature, envs, remote_run_dir)


def pull_back_results_scp(remote_run_dir: str, remote_result_subdir: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_target = PROJECT_ROOT / "result" / f"multimodal_main_remote_{ts}"
    local_target.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            "scp",
            "-o",
            "BatchMode=yes",
            "-r",
            f"{HOST}:{remote_run_dir}/result/{remote_result_subdir}",
            str(local_target),
        ],
        check=True,
    )
    append_worklog(f"结果回传：{local_target}")
    return local_target


def main() -> None:
    ensure_connectivity_gate()

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    remote_run_dir = f"{REMOTE_BASE}/appendix_a_multimodal_{run_stamp}"
    upload_project_payload(remote_run_dir)

    smoke_result_subdir = "multimodal_main_remote_smoke"
    smoke_env = {
        "PYTHONPATH": f"{remote_run_dir}/code",
        "DATASET_DIR": f"{remote_run_dir}/dataset",
        "DEBUG_MAX_STAYS": "200",
        "MM_EPOCHS": "2",
        "MM_BATCH_SIZE": "32",
        "MM_LR": "1e-4",
        "MM_USE_GPU": "1",
        "MM_USE_TEXT": "1",
        "MM_USE_TIME": "1",
        "MM_USE_STATIC": "1",
        "MM_GRAD_ACCUM": "1",
        "MM_RESUME": "0",
        "MM_MODEL_NAME": "emilyalsentzer/Bio_ClinicalBERT",
    }
    run_phase(remote_run_dir, phase="smoke", base_envs=smoke_env, result_subdir=smoke_result_subdir)

    full_result_subdir = "multimodal_main_remote_full"
    full_env = {
        "PYTHONPATH": f"{remote_run_dir}/code",
        "DATASET_DIR": f"{remote_run_dir}/dataset",
        "DEBUG_MAX_STAYS": "0",
        "MM_EPOCHS": "30",
        "MM_BATCH_SIZE": "32",
        "MM_LR": "1e-4",
        "MM_USE_GPU": "1",
        "MM_USE_TEXT": "1",
        "MM_USE_TIME": "1",
        "MM_USE_STATIC": "1",
        "MM_GRAD_ACCUM": "1",
        "MM_RESUME": "0",
        "MM_MODEL_NAME": "emilyalsentzer/Bio_ClinicalBERT",
    }
    final_env = run_phase(remote_run_dir, phase="full", base_envs=full_env, result_subdir=full_result_subdir)
    append_worklog(f"full 阶段收敛参数：{final_env}")

    local_root = pull_back_results_scp(remote_run_dir, full_result_subdir)
    local_result_dir = local_root / full_result_subdir
    if not local_result_dir.exists():
        raise FileNotFoundError(f"Pulled result dir missing: {local_result_dir}")

    # Compare against fixed baselines.
    env = os.environ.copy()
    env["MM_METRICS_PATH"] = str(local_result_dir / "metrics_summary.csv")
    env["XGB_METRICS_PATH"] = str(PROJECT_ROOT / BASELINE_XGB)
    env["LOGI_METRICS_PATH"] = str(PROJECT_ROOT / BASELINE_LOGI)
    subprocess.run(
        ["python", "scripts/multimodal_compare_results.py"],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
    )
    append_worklog("远端 Appendix-A 全流程完成（含基线对比产物）。")


if __name__ == "__main__":
    main()
