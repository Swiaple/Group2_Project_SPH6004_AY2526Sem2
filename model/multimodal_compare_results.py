from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_metrics(path: Path, model_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{model_name} metrics not found: {path}")
    df = pd.read_csv(path).copy()
    df["model"] = model_name
    return df


def main() -> None:
    mm_metrics = Path(
        os.getenv(
            "MM_METRICS_PATH",
            str(PROJECT_ROOT / "result" / "multimodal_main_result" / "metrics_summary.csv"),
        )
    )
    xgb_metrics = Path(
        os.getenv(
            "XGB_METRICS_PATH",
            str(PROJECT_ROOT / "result" / "xgboostmultiresult_gpu_full_20260415_1115" / "metrics_summary.csv"),
        )
    )
    logistic_metrics = Path(
        os.getenv(
            "LOGI_METRICS_PATH",
            str(PROJECT_ROOT / "result" / "logisticmultiresult" / "metrics_summary.csv"),
        )
    )

    mm_df = _load_metrics(mm_metrics, "multimodal_main")
    xgb_df = _load_metrics(xgb_metrics, "xgboost")
    logi_df = _load_metrics(logistic_metrics, "logistic")
    all_df = pd.concat([mm_df, xgb_df, logi_df], ignore_index=True)

    metrics = ["auc", "pr_auc", "recall", "precision"]
    rows = []
    for task in sorted(all_df["task"].unique()):
        mm_row = mm_df[mm_df["task"] == task]
        xgb_row = xgb_df[xgb_df["task"] == task]
        logi_row = logi_df[logi_df["task"] == task]
        if mm_row.empty or xgb_row.empty or logi_row.empty:
            continue
        for metric in metrics:
            mm_val = float(mm_row.iloc[0][metric])
            xgb_val = float(xgb_row.iloc[0][metric])
            logi_val = float(logi_row.iloc[0][metric])
            rows.append(
                {
                    "task": task,
                    "metric": metric,
                    "multimodal_main": mm_val,
                    "xgboost": xgb_val,
                    "logistic": logi_val,
                    "delta_vs_xgboost": mm_val - xgb_val,
                    "delta_vs_logistic": mm_val - logi_val,
                }
            )

    cmp_df = pd.DataFrame(rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = PROJECT_ROOT / "result" / f"multimodal_vs_baselines_{ts}.csv"
    out_md = PROJECT_ROOT / "docs" / f"multimodal_vs_baselines_comparison_{ts}.md"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    cmp_df.to_csv(out_csv, index=False)

    md_lines = [
        "# Multimodal Main vs Baselines",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Multimodal metrics: `{mm_metrics}`",
        f"- XGBoost metrics: `{xgb_metrics}`",
        f"- Logistic metrics: `{logistic_metrics}`",
        "",
        "## Raw Metrics",
        "",
        all_df.to_markdown(index=False),
        "",
        "## Delta (Multimodal - Baseline)",
        "",
        cmp_df.to_markdown(index=False),
        "",
    ]
    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Saved comparison CSV: {out_csv}")
    print(f"Saved comparison Markdown: {out_md}")


if __name__ == "__main__":
    main()
