from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from utils.multitask_common import (
    DATASET_DIR,
    GROUP_COL,
    HADM_COL,
    ID_COL,
    TASK1_COL,
    TASK2_COL,
    TASK2_MASK_COL,
    derive_readmit72h_labels,
    drop_high_corr_features,
)


def build_xmi_features_for_one_stay(
    stay_id: int,
    stay_df: pd.DataFrame,
    feature_cols: List[str],
    min_landmark_hour: int = 24,
    step_size: int = 12,
) -> pd.DataFrame:
    """
    XMI-style pseudo-dynamic features:
    for each t_end, compute cumulative statistics over [0, t_end].
    """
    if stay_df.empty:
        return pd.DataFrame()

    work = stay_df[["relative_hour"] + feature_cols].copy()
    work["relative_hour_int"] = np.floor(work["relative_hour"]).astype(int)
    work = work.drop(columns=["relative_hour"])

    hourly = work.groupby("relative_hour_int", as_index=True)[feature_cols].mean()
    if hourly.empty:
        return pd.DataFrame()

    max_hour = int(hourly.index.max())
    if max_hour < min_landmark_hour:
        return pd.DataFrame()

    full_hour_index = np.arange(0, max_hour + 1, dtype=int)
    observed = pd.Series(1.0, index=hourly.index).reindex(full_hour_index, fill_value=0.0)
    hourly = hourly.reindex(full_hour_index)

    cum_mean = hourly.expanding(min_periods=1).mean().add_suffix("_cum_mean")
    cum_std = hourly.expanding(min_periods=2).std().fillna(0.0).add_suffix("_cum_std")
    cum_min = hourly.expanding(min_periods=1).min().add_suffix("_cum_min")
    cum_max = hourly.expanding(min_periods=1).max().add_suffix("_cum_max")
    last_val = hourly.ffill().add_suffix("_last")
    measured_frac = hourly.notna().astype(float).expanding(min_periods=1).mean().add_suffix("_measured_frac")

    feature_table = pd.concat([cum_mean, cum_std, cum_min, cum_max, last_val, measured_frac], axis=1)

    t_end_values = np.arange(min_landmark_hour, max_hour + 1, step_size, dtype=int)
    if len(t_end_values) == 0:
        return pd.DataFrame()

    observed_cum = observed.cumsum()
    valid_t_end = t_end_values[observed_cum.loc[t_end_values].to_numpy() > 0]
    if len(valid_t_end) == 0:
        return pd.DataFrame()

    selected = feature_table.loc[valid_t_end].copy()
    selected.insert(0, ID_COL, stay_id)
    selected.insert(1, "t_start", 0.0)
    selected.insert(2, "t_end", valid_t_end.astype(float))
    return selected.reset_index(drop=True)


def _drop_static_leakage(df_static_clean: pd.DataFrame) -> pd.DataFrame:
    leakage_cols = [
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

    for col in df_static_clean.columns:
        cl = col.lower()
        if col in (ID_COL, GROUP_COL, HADM_COL, "intime"):
            continue
        if any(k in cl for k in leakage_keywords):
            leakage_cols.append(col)

    leakage_cols = sorted(set(c for c in leakage_cols if c in df_static_clean.columns))
    print(f"Removed {len(leakage_cols)} leakage/static-outcome columns for XMI pipeline.")
    return df_static_clean.drop(columns=leakage_cols, errors="ignore").copy()


def prepare_master_table_xmi(
    debug_max_stays: int = 0,
    min_landmark_hour: int = 24,
    step_size: int = 12,
) -> pd.DataFrame:
    df_static = pd.read_csv(DATASET_DIR / "MIMIC-IV-static(Group Assignment).csv")
    df_time = pd.read_csv(DATASET_DIR / "MIMIC-IV-time_series(Group Assignment).csv")

    print("Loaded:")
    print("static:", df_static.shape)
    print("time  :", df_time.shape)

    df_static["intime"] = pd.to_datetime(df_static["intime"], errors="coerce")
    df_static["outtime"] = pd.to_datetime(df_static["outtime"], errors="coerce")
    if "deathtime" in df_static.columns:
        df_static["deathtime"] = pd.to_datetime(df_static["deathtime"], errors="coerce")

    total_stays = len(df_static)
    short_stays = int((df_static["icu_los_hours"] < 24).sum())
    print(f"Total stays: {total_stays}")
    print(f"Stays with ICU LOS < 24h: {short_stays} ({short_stays / max(total_stays, 1) * 100:.2f}%)")

    df_static_clean = df_static[df_static["icu_los_hours"] >= 24].copy()
    if df_static_clean.empty:
        raise ValueError("No stays left after filtering ICU LOS >= 24h.")

    if debug_max_stays > 0:
        all_stays = df_static_clean[ID_COL].dropna().unique()
        rng = np.random.default_rng(42)
        sampled_stays = rng.choice(all_stays, size=min(debug_max_stays, len(all_stays)), replace=False)
        df_static_clean = df_static_clean[df_static_clean[ID_COL].isin(sampled_stays)].copy()
        valid_stay_ids = set(sampled_stays)
        print(f"[Debug] DEBUG_MAX_STAYS={debug_max_stays}, using {len(valid_stay_ids)} stays.")
    else:
        valid_stay_ids = set(df_static_clean[ID_COL].unique())

    df_time_clean = df_time[df_time[ID_COL].isin(valid_stay_ids)].copy()
    df_static_subset = _drop_static_leakage(df_static_clean)

    if "intime" not in df_time_clean.columns:
        df_time_clean = df_time_clean.merge(df_static_subset[[ID_COL, "intime"]], on=ID_COL, how="left")

    df_time_clean["hour_ts"] = pd.to_datetime(df_time_clean["hour_ts"], errors="coerce")
    df_time_clean["intime"] = pd.to_datetime(df_time_clean["intime"], errors="coerce")
    df_time_clean["relative_hour"] = (df_time_clean["hour_ts"] - df_time_clean["intime"]).dt.total_seconds() / 3600.0
    df_time_clean["relative_hour"] = df_time_clean["relative_hour"].clip(lower=0)

    exclude_cols = {ID_COL, GROUP_COL, "hour_ts", "intime", "relative_hour"}
    candidate_numeric_cols = []
    for col in df_time_clean.columns:
        if col in exclude_cols:
            continue
        if pd.api.types.is_numeric_dtype(df_time_clean[col]):
            candidate_numeric_cols.append(col)

    if len(candidate_numeric_cols) == 0:
        raise ValueError("No numeric time-series variables found for XMI pipeline.")

    df_time_clean[candidate_numeric_cols] = df_time_clean[candidate_numeric_cols].apply(pd.to_numeric, errors="coerce")

    feature_parts: List[pd.DataFrame] = []
    n_total_stays = int(df_time_clean[ID_COL].nunique())
    start_t = time.time()

    for i, (stay_id, g) in enumerate(df_time_clean.groupby(ID_COL, sort=False), start=1):
        part = build_xmi_features_for_one_stay(
            stay_id=stay_id,
            stay_df=g.sort_values("relative_hour"),
            feature_cols=candidate_numeric_cols,
            min_landmark_hour=min_landmark_hour,
            step_size=step_size,
        )
        if not part.empty:
            feature_parts.append(part)
        if i % 5000 == 0:
            print(f"Processed stays: {i}/{n_total_stays} | elapsed={time.time() - start_t:.1f}s")

    df_time_features = pd.concat(feature_parts, ignore_index=True) if len(feature_parts) > 0 else pd.DataFrame()
    if df_time_features.empty:
        raise ValueError("No XMI pseudo-dynamic features generated.")

    print("XMI time feature table:", df_time_features.shape)

    df_master = df_time_features.merge(df_static_subset, on=ID_COL, how="left")
    print("Merged master table:", df_master.shape)

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

    df_master["discharge_in_24h"] = ((df_master["icu_out_hour"] - df_master["t_end"]) <= 24).astype(int)
    df_master[TASK1_COL] = ((df_master["discharge_in_24h"] == 1) & (df_master["icu_death_flag"] == 0)).astype(int)

    readmit_df = derive_readmit72h_labels(df_static_clean)
    df_master = df_master.merge(readmit_df[[ID_COL, TASK2_COL]], on=ID_COL, how="left")
    if df_master[TASK2_COL].isna().any():
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


def build_xmi_feature_matrices(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    corr_threshold: float = 0.95,
) -> Dict[str, object]:
    safe_cols = [
        ID_COL,
        GROUP_COL,
        HADM_COL,
        "t_start",
        "t_end",
        "icu_out_hour",
        "discharge_in_24h",
        "icu_death_flag",
        "intime",
        "outtime",
        TASK1_COL,
        TASK2_COL,
        TASK2_MASK_COL,
    ]

    missing_rates = df_train.isnull().mean()
    cols_to_drop = missing_rates[missing_rates > 0.90].index.tolist()
    cols_to_drop = [c for c in cols_to_drop if c not in safe_cols and not c.endswith("_measured_frac")]

    train_work = df_train.drop(columns=cols_to_drop, errors="ignore").copy()
    val_work = df_val.drop(columns=cols_to_drop, errors="ignore").copy()
    test_work = df_test.drop(columns=cols_to_drop, errors="ignore").copy()

    exclude_cols = [
        ID_COL,
        GROUP_COL,
        HADM_COL,
        "t_start",
        "t_end",
        "icu_out_hour",
        "intime",
        "outtime",
        "discharge_in_24h",
        "icu_death_flag",
        TASK1_COL,
        TASK2_COL,
        TASK2_MASK_COL,
    ]

    all_features = [c for c in train_work.columns if c not in exclude_cols]
    all_features = [c for c in all_features if not c.lower().endswith("_id")]

    numeric_features = [
        c
        for c in all_features
        if pd.api.types.is_numeric_dtype(train_work[c]) or pd.api.types.is_bool_dtype(train_work[c])
    ]
    if len(numeric_features) == 0:
        raise ValueError("No numeric features left for XMI modeling.")

    x_train = train_work[numeric_features].copy()
    x_val = val_work[numeric_features].copy()
    x_test = test_work[numeric_features].copy()

    y1_train = train_work[TASK1_COL].astype(int).to_numpy()
    y1_val = val_work[TASK1_COL].astype(int).to_numpy()
    y1_test = test_work[TASK1_COL].astype(int).to_numpy()

    y2_train = train_work[TASK2_COL].astype(int).to_numpy()
    y2_val = val_work[TASK2_COL].astype(int).to_numpy()
    y2_test = test_work[TASK2_COL].astype(int).to_numpy()

    mask2_train = train_work[TASK2_MASK_COL].astype(int).to_numpy().astype(bool)
    mask2_val = val_work[TASK2_MASK_COL].astype(int).to_numpy().astype(bool)
    mask2_test = test_work[TASK2_MASK_COL].astype(int).to_numpy().astype(bool)

    imputer = SimpleImputer(strategy="median")
    x_train_imp = pd.DataFrame(imputer.fit_transform(x_train), index=x_train.index, columns=x_train.columns)
    x_val_imp = pd.DataFrame(imputer.transform(x_val), index=x_val.index, columns=x_val.columns)
    x_test_imp = pd.DataFrame(imputer.transform(x_test), index=x_test.index, columns=x_test.columns)

    x_train_imp = x_train_imp.astype(np.float32)
    x_val_imp = x_val_imp.astype(np.float32)
    x_test_imp = x_test_imp.astype(np.float32)

    x_train_uncorr, corr_drop_cols = drop_high_corr_features(x_train_imp, threshold=corr_threshold)
    x_val_uncorr = x_val_imp.drop(columns=corr_drop_cols, errors="ignore")
    x_test_uncorr = x_test_imp.drop(columns=corr_drop_cols, errors="ignore")

    print(f"Columns dropped (>90% missing): {len(cols_to_drop)}")
    print(f"Correlated columns dropped: {len(corr_drop_cols)}")
    print("Final feature count:", x_train_uncorr.shape[1])

    return {
        "X_train": x_train_uncorr,
        "X_val": x_val_uncorr,
        "X_test": x_test_uncorr,
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
            "imputer": imputer,
            "final_feature_names": x_train_uncorr.columns.tolist(),
            "feature_mode": "xmi_pseudo_dynamic",
            "corr_threshold": corr_threshold,
        },
    }
