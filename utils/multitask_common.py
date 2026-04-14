import os
import re
import time
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Keep plotting robust in headless/sandbox environments.
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix,
    log_loss,
    recall_score,
    precision_score,
    average_precision_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
ID_COL = "stay_id"
GROUP_COL = "subject_id"
HADM_COL = "hadm_id"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"

TASK1_COL = "label_can_discharge_24h"
TASK2_COL = "label_no_return_72h"
TASK2_MASK_COL = "task2_mask"

WINDOW_SIZE = 24
STEP_SIZE = 12


def clean_medical_report(text: str) -> Optional[str]:
    if pd.isna(text):
        return np.nan

    text = str(text)
    text = re.sub(r"(Findings|Results)\s+were\s+communicated.*", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"electronically signed.*", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) == 0:
        return np.nan

    return text


def build_time_features_for_one_stay(
    stay_id: int,
    stay_df: pd.DataFrame,
    feature_cols: List[str],
    window_size: int = WINDOW_SIZE,
    step_size: int = STEP_SIZE,
) -> pd.DataFrame:
    """
    Build rolling features at t_end = 24, 36, 48, ... for one stay.
    Uses hourly-aligned vectorized rolling for speed.
    """
    if stay_df.empty:
        return pd.DataFrame()

    stay_work = stay_df[["relative_hour"] + feature_cols].copy()
    stay_work["relative_hour_int"] = np.rint(stay_work["relative_hour"]).astype(int)
    stay_work = stay_work.drop(columns=["relative_hour"])

    stay_hourly = stay_work.groupby("relative_hour_int", as_index=True)[feature_cols].mean()
    if stay_hourly.empty:
        return pd.DataFrame()

    max_hour = int(stay_hourly.index.max())
    if max_hour < window_size:
        return pd.DataFrame()

    full_hour_index = np.arange(0, max_hour + 1)
    observed = pd.Series(1.0, index=stay_hourly.index).reindex(full_hour_index, fill_value=0.0)
    stay_hourly = stay_hourly.reindex(full_hour_index)

    roll_mean = stay_hourly.rolling(window_size, min_periods=1).mean().add_suffix("_mean")
    roll_min = stay_hourly.rolling(window_size, min_periods=1).min().add_suffix("_min")
    roll_max = stay_hourly.rolling(window_size, min_periods=1).max().add_suffix("_max")
    roll_measured = stay_hourly.notna().astype(float).rolling(window_size, min_periods=1).max().add_suffix("_is_measured")

    observed_in_window = observed.rolling(window_size, min_periods=1).sum()

    stay_features = pd.concat([roll_mean, roll_min, roll_max, roll_measured], axis=1)

    t_end_values = np.arange(window_size, max_hour + 1, step_size)
    if len(t_end_values) == 0:
        return pd.DataFrame()

    valid_t_end = t_end_values[observed_in_window.loc[t_end_values].to_numpy() > 0]
    if len(valid_t_end) == 0:
        return pd.DataFrame()

    selected = stay_features.loc[valid_t_end].copy()
    selected.insert(0, ID_COL, stay_id)
    selected.insert(1, "t_start", valid_t_end - window_size)
    selected.insert(2, "t_end", valid_t_end.astype(float))
    return selected.reset_index(drop=True)


def build_window_text(master_df: pd.DataFrame, text_df: pd.DataFrame) -> pd.Series:
    """Build cumulative available text per (stay, t_end) window."""
    out = pd.Series("", index=master_df.index, dtype=object)

    if text_df.empty:
        return out

    grouped_text = {
        sid: g.sort_values("report_time_estimated")[["report_time_estimated", "clean_text"]]
        for sid, g in text_df.groupby(ID_COL)
    }

    for stay_id, idx in master_df.groupby(ID_COL).groups.items():
        reports = grouped_text.get(stay_id)
        if reports is None or reports.empty:
            continue

        idx_arr = np.array(list(idx), dtype=int)
        t_vals = master_df.loc[idx_arr, "t_end"].to_numpy()
        order = np.argsort(t_vals)
        sorted_idx = idx_arr[order]
        sorted_t = t_vals[order]

        rep_t = reports["report_time_estimated"].to_numpy()
        rep_txt = reports["clean_text"].astype(str).tolist()

        cursor = 0
        collected = []
        result_text = []

        for t in sorted_t:
            while cursor < len(rep_t) and rep_t[cursor] <= t:
                collected.append(rep_txt[cursor])
                cursor += 1
            result_text.append(" ".join(collected) if collected else "")

        out.loc[sorted_idx] = result_text

    return out


def derive_readmit72h_labels(df_static_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Build stay-level 72h ICU return labels within the same hospital admission.
    readmit72h = 1 means returns to ICU within 72h after ICU outtime.
    no_return_72h = 1 - readmit72h.
    """
    work = df_static_clean[[ID_COL, HADM_COL, "intime", "outtime"]].copy()
    work["intime"] = pd.to_datetime(work["intime"], errors="coerce")
    work["outtime"] = pd.to_datetime(work["outtime"], errors="coerce")

    work = work.sort_values([HADM_COL, "intime"]).copy()
    work["next_intime_same_hadm"] = work.groupby(HADM_COL)["intime"].shift(-1)

    delta_hours = (work["next_intime_same_hadm"] - work["outtime"]).dt.total_seconds() / 3600.0
    work["readmit_72h"] = ((delta_hours >= 0) & (delta_hours <= 72)).astype(int)
    work[TASK2_COL] = 1 - work["readmit_72h"]

    return work[[ID_COL, "readmit_72h", TASK2_COL]].copy()


def prepare_master_table(debug_max_stays: int = 0) -> pd.DataFrame:
    """Shared data-processing pipeline used by logisticmulti/xgboostmulti."""
    df_static = pd.read_csv(DATASET_DIR / "MIMIC-IV-static(Group Assignment).csv")
    df_text = pd.read_csv(DATASET_DIR / "MIMIC-IV-text(Group Assignment).csv")
    df_time = pd.read_csv(DATASET_DIR / "MIMIC-IV-time_series(Group Assignment).csv")

    print("Loaded:")
    print("static:", df_static.shape)
    print("text  :", df_text.shape)
    print("time  :", df_time.shape)

    # 1) Filter stays >=24h
    df_static["intime"] = pd.to_datetime(df_static["intime"], errors="coerce")
    df_static["outtime"] = pd.to_datetime(df_static["outtime"], errors="coerce")
    if "deathtime" in df_static.columns:
        df_static["deathtime"] = pd.to_datetime(df_static["deathtime"], errors="coerce")

    total_stays = len(df_static)
    short_stays = (df_static["icu_los_hours"] < 24).sum()
    print(f"Total stays: {total_stays}")
    print(f"Stays with ICU LOS < 24h: {short_stays} ({short_stays / max(total_stays, 1) * 100:.2f}%)")

    df_static_clean = df_static[df_static["icu_los_hours"] >= 24].copy()
    if df_static_clean.empty:
        raise ValueError("No stays left after filtering ICU LOS >= 24h.")

    if debug_max_stays > 0:
        all_stays = df_static_clean[ID_COL].dropna().unique()
        rng = np.random.default_rng(RANDOM_STATE)
        sampled_stays = rng.choice(all_stays, size=min(debug_max_stays, len(all_stays)), replace=False)
        df_static_clean = df_static_clean[df_static_clean[ID_COL].isin(sampled_stays)].copy()
        valid_stay_ids = set(sampled_stays)
        print(f"[Debug] DEBUG_MAX_STAYS={debug_max_stays}, using {len(valid_stay_ids)} stays.")
    else:
        valid_stay_ids = set(df_static_clean[ID_COL].unique())

    df_time_clean = df_time[df_time[ID_COL].isin(valid_stay_ids)].copy()
    df_text_clean = df_text[df_text[ID_COL].isin(valid_stay_ids)].copy()

    print("After removing short stays:")
    print("static:", df_static_clean.shape)
    print("text  :", df_text_clean.shape)
    print("time  :", df_time_clean.shape)

    # 2) Remove leakage/surrogate-outcome columns from static
    leakage_cols_in_static = [
        "spo2_min",
        "spo2_max",
        "spo2_mean",
        "gcs_min",
        "gcs_max",
        "gcs_mean",
        "platelets_min",
        "platelets_max",
        "platelets_mean",
        "wbc_min",
        "wbc_max",
        "wbc_mean",
        "bun_min",
        "bun_max",
        "bun_mean",
        "creatinine_min",
        "creatinine_max",
        "creatinine_mean",
        "sodium_min",
        "sodium_max",
        "sodium_mean",
        "inr_min",
        "inr_max",
        "inr_mean",
        "los",
        "icu_los_hours",
        "outtime",
        "deathtime",
        "hospital_expire_flag",
        "icu_death_flag",
        "radiology_note_count",
    ]

    leakage_keywords = ("discharge", "expire", "mortality", "outcome", "target", "label", "readmit")
    for c in df_static_clean.columns:
        cl = c.lower()
        if c in (ID_COL, GROUP_COL, HADM_COL, "intime"):
            continue
        if any(k in cl for k in leakage_keywords):
            leakage_cols_in_static.append(c)

    existing_leakage_cols = sorted(set(c for c in leakage_cols_in_static if c in df_static_clean.columns))
    df_static_subset = df_static_clean.drop(columns=existing_leakage_cols, errors="ignore").copy()

    print(f"Removed {len(existing_leakage_cols)} leakage/static-outcome columns.")

    # 3) Text preprocessing
    if "radiology_note_text" not in df_text_clean.columns:
        raise ValueError("Expected column 'radiology_note_text' not found in text data.")

    if "intime" not in df_text_clean.columns:
        df_text_clean = df_text_clean.merge(df_static_clean[[ID_COL, "intime"]], on=ID_COL, how="left")

    df_text_clean["report_list"] = df_text_clean["radiology_note_text"].fillna("").str.split(r"-{3,}")
    df_exploded = df_text_clean.explode("report_list").copy()
    df_exploded["single_report_text"] = df_exploded["report_list"].astype(str).str.strip()
    df_exploded["single_report_text"] = df_exploded["single_report_text"].replace("", np.nan)
    df_exploded = df_exploded.dropna(subset=["single_report_text"]).copy()
    df_exploded["clean_text"] = df_exploded["single_report_text"].apply(clean_medical_report)
    df_exploded = df_exploded.dropna(subset=["clean_text"]).copy()

    df_exploded["report_time_estimated"] = np.nan
    if "radiology_note_time_max" in df_exploded.columns:
        note_time = pd.to_datetime(df_exploded["radiology_note_time_max"], errors="coerce")
        intime = pd.to_datetime(df_exploded["intime"], errors="coerce")
        df_exploded["report_time_estimated"] = (note_time - intime).dt.total_seconds() / 3600.0
    elif "radiology_note_time_min" in df_exploded.columns:
        note_time = pd.to_datetime(df_exploded["radiology_note_time_min"], errors="coerce")
        intime = pd.to_datetime(df_exploded["intime"], errors="coerce")
        df_exploded["report_time_estimated"] = (note_time - intime).dt.total_seconds() / 3600.0
    elif "t_max_rel" in df_exploded.columns:
        df_exploded["report_time_estimated"] = pd.to_numeric(df_exploded["t_max_rel"], errors="coerce")
    elif "t_min_rel" in df_exploded.columns:
        df_exploded["report_time_estimated"] = pd.to_numeric(df_exploded["t_min_rel"], errors="coerce")

    df_exploded["report_time_estimated"] = df_exploded["report_time_estimated"].clip(lower=0)
    missing_report_time = int(df_exploded["report_time_estimated"].isna().sum())
    if missing_report_time > 0:
        print(f"[Warning] Dropping {missing_report_time} text rows with unknown note time.")
        df_exploded = df_exploded.dropna(subset=["report_time_estimated"]).copy()

    print("Text records after cleaning:", df_exploded.shape)

    # 4) Time preprocessing
    if "intime" not in df_time_clean.columns:
        df_time_clean = df_time_clean.merge(df_static_subset[[ID_COL, "intime"]], on=ID_COL, how="left")

    df_time_clean["hour_ts"] = pd.to_datetime(df_time_clean["hour_ts"], errors="coerce")
    df_time_clean["intime"] = pd.to_datetime(df_time_clean["intime"], errors="coerce")

    df_time_clean["relative_hour"] = (df_time_clean["hour_ts"] - df_time_clean["intime"]).dt.total_seconds() / 3600.0
    df_time_clean["relative_hour"] = df_time_clean["relative_hour"].clip(lower=0)

    print(
        "Unified relative time range:",
        f"{df_time_clean['relative_hour'].min():.1f}h to {df_time_clean['relative_hour'].max():.1f}h",
    )

    # 5) Sliding-window time-series features
    exclude_time_cols = {ID_COL, GROUP_COL, "hour_ts", "intime", "relative_hour"}

    candidate_numeric_cols = []
    for col in df_time_clean.columns:
        if col in exclude_time_cols:
            continue
        if pd.api.types.is_numeric_dtype(df_time_clean[col]):
            candidate_numeric_cols.append(col)

    if len(candidate_numeric_cols) == 0:
        raise ValueError("No numeric time-series variables found after preprocessing.")

    df_time_clean[candidate_numeric_cols] = df_time_clean[candidate_numeric_cols].apply(pd.to_numeric, errors="coerce")

    time_feature_parts = []
    n_total_stays = int(df_time_clean[ID_COL].nunique())
    start_t = time.time()

    for i, (stay_id, g) in enumerate(df_time_clean.groupby(ID_COL, sort=False), start=1):
        part = build_time_features_for_one_stay(
            stay_id=stay_id,
            stay_df=g.sort_values("relative_hour"),
            feature_cols=candidate_numeric_cols,
            window_size=WINDOW_SIZE,
            step_size=STEP_SIZE,
        )
        if not part.empty:
            time_feature_parts.append(part)

        if i % 5000 == 0:
            print(f"Processed stays: {i}/{n_total_stays} | elapsed={time.time() - start_t:.1f}s")

    df_time_features = pd.concat(time_feature_parts, ignore_index=True) if len(time_feature_parts) > 0 else pd.DataFrame()
    if df_time_features.empty:
        raise ValueError("No sliding-window time features were generated.")

    print("Sliding-window time feature table:", df_time_features.shape)

    # 6) Merge text into each prediction window
    master_base = df_time_features[[ID_COL, "t_end"]].drop_duplicates().copy()
    master_base["final_text"] = build_window_text(master_base, df_exploded)
    master_base["has_text"] = (master_base["final_text"].str.len() > 0).astype(int)

    df_master = df_time_features.merge(master_base, on=[ID_COL, "t_end"], how="left")
    df_master = df_master.merge(df_static_subset, on=ID_COL, how="left")

    print("Merged master table:", df_master.shape)

    # 7) Build labels
    stay_info_cols = [ID_COL, "intime", "outtime"]
    if "deathtime" in df_static_clean.columns:
        stay_info_cols.append("deathtime")

    stay_info = df_static_clean[stay_info_cols].copy()
    stay_info["intime"] = pd.to_datetime(stay_info["intime"], errors="coerce")
    stay_info["outtime"] = pd.to_datetime(stay_info["outtime"], errors="coerce")
    if "deathtime" in stay_info.columns:
        stay_info["deathtime"] = pd.to_datetime(stay_info["deathtime"], errors="coerce")

    stay_info["icu_out_hour"] = (stay_info["outtime"] - stay_info["intime"]).dt.total_seconds() / 3600.0
    stay_info["icu_death_flag"] = 0
    if "deathtime" in stay_info.columns:
        stay_info["icu_death_flag"] = (
            stay_info["deathtime"].notna() & (stay_info["deathtime"] <= stay_info["outtime"])
        ).astype(int)

    df_master = df_master.merge(stay_info[[ID_COL, "icu_out_hour", "icu_death_flag"]], on=ID_COL, how="left")

    before_count = len(df_master)
    df_master = df_master[df_master["t_end"] <= df_master["icu_out_hour"]].copy()
    print(f"Removed {before_count - len(df_master)} rows with t_end after ICU discharge.")

    # Task 1: can discharge in next 24h and not ICU death
    df_master["discharge_in_24h"] = ((df_master["icu_out_hour"] - df_master["t_end"]) <= 24).astype(int)
    df_master[TASK1_COL] = ((df_master["discharge_in_24h"] == 1) & (df_master["icu_death_flag"] == 0)).astype(int)

    # Task 2: no ICU return within 72h (stay-level label)
    readmit_df = derive_readmit72h_labels(df_static_clean)
    df_master = df_master.merge(readmit_df[[ID_COL, TASK2_COL]], on=ID_COL, how="left")
    if df_master[TASK2_COL].isna().any():
        # Conservative fill: if unavailable, treat as no-return unknown -> 0.
        df_master[TASK2_COL] = df_master[TASK2_COL].fillna(0)

    df_master[TASK2_COL] = df_master[TASK2_COL].astype(int)
    df_master[TASK2_MASK_COL] = (df_master[TASK1_COL] == 1).astype(int)

    required_cols = [ID_COL, GROUP_COL, TASK1_COL, TASK2_COL, TASK2_MASK_COL]
    missing_required = [c for c in required_cols if c not in df_master.columns]
    if missing_required:
        raise ValueError(f"Missing required columns before split: {missing_required}")

    before_dropna = len(df_master)
    df_master = df_master.dropna(subset=[ID_COL, GROUP_COL, TASK1_COL, TASK2_COL, TASK2_MASK_COL]).copy()
    print(f"Dropped {before_dropna - len(df_master)} rows with missing key cols.")

    print(df_master[[TASK1_COL, TASK2_COL, TASK2_MASK_COL]].mean())

    return df_master


def split_group_train_val_test(df_master: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df_master[GROUP_COL].nunique() < 3:
        raise ValueError("Need at least 3 unique subject_id groups for train/val/test split.")

    gss_outer = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=RANDOM_STATE)
    trainval_idx, test_idx = next(gss_outer.split(df_master, groups=df_master[GROUP_COL]))

    df_trainval = df_master.iloc[trainval_idx].copy()
    df_test = df_master.iloc[test_idx].copy()

    gss_inner = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=RANDOM_STATE + 1)
    train_idx, val_idx = next(gss_inner.split(df_trainval, groups=df_trainval[GROUP_COL]))

    df_train = df_trainval.iloc[train_idx].copy()
    df_val = df_trainval.iloc[val_idx].copy()

    print("-" * 50)
    print(f"[Train] rows={len(df_train)}, patients={df_train[GROUP_COL].nunique()}")
    print(f"[Val  ] rows={len(df_val)}, patients={df_val[GROUP_COL].nunique()}")
    print(f"[Test ] rows={len(df_test)}, patients={df_test[GROUP_COL].nunique()}")
    print(f"[Train] task1 pos rate={df_train[TASK1_COL].mean():.4f}")
    print(f"[Val  ] task1 pos rate={df_val[TASK1_COL].mean():.4f}")
    print(f"[Test ] task1 pos rate={df_test[TASK1_COL].mean():.4f}")

    return df_train, df_val, df_test


def drop_high_corr_features(df_num: pd.DataFrame, threshold: float = 0.95) -> Tuple[pd.DataFrame, List[str]]:
    if df_num.shape[1] <= 1:
        return df_num.copy(), []

    corr_matrix = df_num.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df_num.drop(columns=to_drop), to_drop


def build_feature_matrices(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    scale_for_linear: bool,
    tfidf_max_features: int = 300,
) -> Dict[str, object]:
    """Build train/val/test matrices using TRAIN-only fitted transformations."""
    safe_cols = [
        ID_COL,
        GROUP_COL,
        "t_start",
        "t_end",
        "icu_out_hour",
        "discharge_in_24h",
        "icu_death_flag",
        TASK1_COL,
        TASK2_COL,
        TASK2_MASK_COL,
        "final_text",
        "has_text",
        "intime",
        "outtime",
    ]

    missing_rates = df_train.isnull().mean()
    cols_to_drop = missing_rates[missing_rates > 0.90].index.tolist()
    cols_to_drop = [c for c in cols_to_drop if c not in safe_cols and not c.endswith("_is_measured")]

    df_train_work = df_train.drop(columns=cols_to_drop, errors="ignore").copy()
    df_val_work = df_val.drop(columns=cols_to_drop, errors="ignore").copy()
    df_test_work = df_test.drop(columns=cols_to_drop, errors="ignore").copy()

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=tfidf_max_features, ngram_range=(1, 2), min_df=5)

    train_text = df_train_work["final_text"].fillna("")
    val_text = df_val_work["final_text"].fillna("")
    test_text = df_test_work["final_text"].fillna("")

    tfidf_feature_names: List[str] = []
    if (train_text.str.len() > 0).any():
        try:
            X_train_text = tfidf.fit_transform(train_text)
            X_val_text = tfidf.transform(val_text)
            X_test_text = tfidf.transform(test_text)

            tfidf_feature_names = [f"tfidf_{w}" for w in tfidf.get_feature_names_out()]

            if len(tfidf_feature_names) > 0:
                train_tfidf_df = pd.DataFrame(X_train_text.toarray().astype(np.float32), index=df_train_work.index, columns=tfidf_feature_names)
                val_tfidf_df = pd.DataFrame(X_val_text.toarray().astype(np.float32), index=df_val_work.index, columns=tfidf_feature_names)
                test_tfidf_df = pd.DataFrame(X_test_text.toarray().astype(np.float32), index=df_test_work.index, columns=tfidf_feature_names)

                df_train_work = pd.concat([df_train_work, train_tfidf_df], axis=1)
                df_val_work = pd.concat([df_val_work, val_tfidf_df], axis=1)
                df_test_work = pd.concat([df_test_work, test_tfidf_df], axis=1)
        except (ValueError, MemoryError) as e:
            print(f"[Warning] TF-IDF skipped due to: {e}")
    else:
        print("[Info] No non-empty training text; TF-IDF skipped.")

    exclude_cols = [
        ID_COL,
        GROUP_COL,
        HADM_COL,
        "t_start",
        "t_end",
        "icu_out_hour",
        "intime",
        "outtime",
        "final_text",
        "discharge_in_24h",
        "icu_death_flag",
        TASK1_COL,
        TASK2_COL,
        TASK2_MASK_COL,
    ]

    all_features = [c for c in df_train_work.columns if c not in exclude_cols]
    # Defensive rule: never allow identifier-like features into model input.
    all_features = [c for c in all_features if not c.lower().endswith("_id")]
    numeric_features = [c for c in all_features if pd.api.types.is_numeric_dtype(df_train_work[c]) or pd.api.types.is_bool_dtype(df_train_work[c])]

    if len(numeric_features) == 0:
        raise ValueError("No numeric features left for modeling.")

    X_train = df_train_work[numeric_features].copy()
    X_val = df_val_work[numeric_features].copy()
    X_test = df_test_work[numeric_features].copy()

    y1_train = df_train_work[TASK1_COL].astype(int).to_numpy()
    y1_val = df_val_work[TASK1_COL].astype(int).to_numpy()
    y1_test = df_test_work[TASK1_COL].astype(int).to_numpy()

    y2_train = df_train_work[TASK2_COL].astype(int).to_numpy()
    y2_val = df_val_work[TASK2_COL].astype(int).to_numpy()
    y2_test = df_test_work[TASK2_COL].astype(int).to_numpy()

    mask2_train = df_train_work[TASK2_MASK_COL].astype(int).to_numpy().astype(bool)
    mask2_val = df_val_work[TASK2_MASK_COL].astype(int).to_numpy().astype(bool)
    mask2_test = df_test_work[TASK2_MASK_COL].astype(int).to_numpy().astype(bool)

    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    X_val_imp = pd.DataFrame(imputer.transform(X_val), index=X_val.index, columns=X_val.columns)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), index=X_test.index, columns=X_test.columns)

    scaler = None
    if scale_for_linear:
        scaler = StandardScaler()
        X_train_imp = pd.DataFrame(scaler.fit_transform(X_train_imp), index=X_train_imp.index, columns=X_train_imp.columns)
        X_val_imp = pd.DataFrame(scaler.transform(X_val_imp), index=X_val_imp.index, columns=X_val_imp.columns)
        X_test_imp = pd.DataFrame(scaler.transform(X_test_imp), index=X_test_imp.index, columns=X_test_imp.columns)

    X_train_imp = X_train_imp.astype(np.float32)
    X_val_imp = X_val_imp.astype(np.float32)
    X_test_imp = X_test_imp.astype(np.float32)

    X_train_uncorr, corr_drop_cols = drop_high_corr_features(X_train_imp, threshold=0.95)
    X_val_uncorr = X_val_imp.drop(columns=corr_drop_cols, errors="ignore")
    X_test_uncorr = X_test_imp.drop(columns=corr_drop_cols, errors="ignore")

    print(f"Columns dropped (>90% missing): {len(cols_to_drop)}")
    print(f"TF-IDF features added: {len(tfidf_feature_names)}")
    print(f"Correlated columns dropped: {len(corr_drop_cols)}")
    print("Final feature count:", X_train_uncorr.shape[1])

    return {
        "X_train": X_train_uncorr,
        "X_val": X_val_uncorr,
        "X_test": X_test_uncorr,
        "y1_train": y1_train,
        "y1_val": y1_val,
        "y1_test": y1_test,
        "y2_train": y2_train,
        "y2_val": y2_val,
        "y2_test": y2_test,
        "mask2_train": mask2_train,
        "mask2_val": mask2_val,
        "mask2_test": mask2_test,
        "artifacts": {
            "cols_to_drop_missing": cols_to_drop,
            "corr_drop_cols": corr_drop_cols,
            "numeric_features": numeric_features,
            "tfidf": tfidf,
            "tfidf_feature_names": tfidf_feature_names,
            "imputer": imputer,
            "scaler": scaler,
            "final_feature_names": X_train_uncorr.columns.tolist(),
        },
    }


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")


def safe_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2 and np.all(y_true == 0):
            return float("nan")
        return float(recall_score(y_true, y_pred, zero_division=0))
    except Exception:
        return float("nan")


def safe_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(precision_score(y_true, y_pred, zero_division=0))
    except Exception:
        return float("nan")


def safe_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(average_precision_score(y_true, y_prob))
    except Exception:
        return float("nan")


def safe_binary_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(log_loss(y_true, y_prob, labels=[0, 1]))
    except Exception:
        return float("nan")


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def evaluate_binary_task(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    conf = confusion_counts(y_true, y_pred)

    metrics = {
        "auc": safe_auc(y_true, y_prob),
        "pr_auc": safe_pr_auc(y_true, y_prob),
        "recall": safe_recall(y_true, y_pred),
        "precision": safe_precision(y_true, y_pred),
        "threshold": float(threshold),
        "n_eval": int(len(y_true)),
        **conf,
    }
    return metrics


def save_roc_plot(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure(figsize=(6, 6))

    if len(np.unique(y_true)) < 2:
        plt.text(0.5, 0.5, "Only one class in y_true; ROC unavailable", ha="center", va="center")
        plt.title(title)
        plt.axis("off")
    else:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"AUC = {auc_val:.4f}")
        plt.plot([0, 1], [0, 1], "--", alpha=0.5)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_loss_curve(loss_df: pd.DataFrame, out_path_png: Path, out_path_csv: Path, title: str) -> None:
    loss_df.to_csv(out_path_csv, index=False)

    plt.figure(figsize=(9, 5))
    for col in loss_df.columns:
        if col == "epoch":
            continue
        plt.plot(loss_df["epoch"], loss_df[col], label=col)
    plt.xlabel("Epoch / Boosting Round")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path_png, dpi=180)
    plt.close()


def save_metrics_bundle(
    result_dir: Path,
    y1_true: np.ndarray,
    y1_prob: np.ndarray,
    y2_true: np.ndarray,
    y2_prob: np.ndarray,
    y2_mask: np.ndarray,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Save per-task and joint metrics + ROC plots.

    Task2 metrics are computed only on y2_mask == True.
    Joint event is defined as (task1==1 and task2==1).
    Joint score uses p(task1)*p(task2).
    """
    result_dir.mkdir(parents=True, exist_ok=True)

    # Task 1
    m1 = evaluate_binary_task(y1_true, y1_prob, threshold=threshold)

    # Task 2 (masked)
    valid_mask = y2_mask.astype(bool)
    if valid_mask.sum() > 0:
        m2 = evaluate_binary_task(y2_true[valid_mask], y2_prob[valid_mask], threshold=threshold)
    else:
        m2 = {
            "auc": float("nan"),
            "pr_auc": float("nan"),
            "recall": float("nan"),
            "precision": float("nan"),
            "threshold": float(threshold),
            "n_eval": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }

    # Joint
    y_joint_true = ((y1_true == 1) & (y2_true == 1)).astype(int)
    y_joint_prob = y1_prob * y2_prob
    y_joint_pred = ((y1_prob >= threshold) & (y2_prob >= threshold)).astype(int)
    conf_joint = confusion_counts(y_joint_true, y_joint_pred)
    m_joint = {
        "auc": safe_auc(y_joint_true, y_joint_prob),
        "pr_auc": safe_pr_auc(y_joint_true, y_joint_prob),
        "recall": safe_recall(y_joint_true, y_joint_pred),
        "precision": safe_precision(y_joint_true, y_joint_pred),
        "threshold": float(threshold),
        "n_eval": int(len(y_joint_true)),
        **conf_joint,
    }

    metrics_df = pd.DataFrame(
        [
            {"task": "task1_discharge", **m1},
            {"task": "task2_no_return72_masked", **m2},
            {"task": "joint_both_positive", **m_joint},
        ]
    )

    metrics_df.to_csv(result_dir / "metrics_summary.csv", index=False)
    metrics_df[["task", "recall"]].to_csv(result_dir / "recall_summary.csv", index=False)
    metrics_df[["task", "precision", "recall", "pr_auc", "auc"]].to_csv(
        result_dir / "precision_recall_pr_auc_summary.csv",
        index=False,
    )

    # Confusion tables (separate files)
    pd.DataFrame([m1]).to_csv(result_dir / "confusion_task1.csv", index=False)
    pd.DataFrame([m2]).to_csv(result_dir / "confusion_task2_masked.csv", index=False)
    pd.DataFrame([m_joint]).to_csv(result_dir / "confusion_joint.csv", index=False)

    # ROC plots
    save_roc_plot(y1_true, y1_prob, result_dir / "roc_task1.png", "Task1 ROC (Can Discharge)")

    if valid_mask.sum() > 0:
        save_roc_plot(
            y2_true[valid_mask],
            y2_prob[valid_mask],
            result_dir / "roc_task2_masked.png",
            "Task2 ROC (No Return 72h | Masked)",
        )
    else:
        save_roc_plot(np.array([0]), np.array([0.0]), result_dir / "roc_task2_masked.png", "Task2 ROC Unavailable")

    save_roc_plot(y_joint_true, y_joint_prob, result_dir / "roc_joint.png", "Joint ROC (Task1 & Task2)")

    # Also save raw probabilities for reproducibility
    pred_df = pd.DataFrame(
        {
            "y1_true": y1_true,
            "y1_prob": y1_prob,
            "y2_true": y2_true,
            "y2_prob": y2_prob,
            "y2_mask": y2_mask.astype(int),
            "y_joint_true": y_joint_true,
            "y_joint_prob": y_joint_prob,
        }
    )
    pred_df.to_csv(result_dir / "test_predictions.csv", index=False)

    return metrics_df


def save_json(data: Dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
