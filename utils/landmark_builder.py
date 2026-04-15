from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from utils.multitask_common import DATASET_DIR, GROUP_COL, HADM_COL, ID_COL, drop_high_corr_features
from utils.xmi_common import prepare_master_table_xmi

RANDOM_STATE = 42

INTERVAL_BOUNDS: Tuple[int, int, int] = (24, 48, 72)


def _build_event_table() -> pd.DataFrame:
    static = pd.read_csv(DATASET_DIR / "MIMIC-IV-static(Group Assignment).csv")
    static["intime"] = pd.to_datetime(static["intime"], errors="coerce")
    static["outtime"] = pd.to_datetime(static["outtime"], errors="coerce")
    if "deathtime" in static.columns:
        static["deathtime"] = pd.to_datetime(static["deathtime"], errors="coerce")
    else:
        static["deathtime"] = pd.NaT

    event_hour = (static["outtime"] - static["intime"]).dt.total_seconds() / 3600.0
    death_hour = (static["deathtime"] - static["intime"]).dt.total_seconds() / 3600.0

    death_in_icu = static["deathtime"].notna() & (static["deathtime"] <= static["outtime"])
    event_type = np.where(death_in_icu, "death", "discharge")
    event_hour = np.where(death_in_icu, death_hour, event_hour)

    out = static[[ID_COL, GROUP_COL, HADM_COL]].copy()
    out["event_type"] = event_type
    out["event_hour"] = pd.to_numeric(event_hour, errors="coerce")
    out["death_hour"] = pd.to_numeric(death_hour, errors="coerce")
    out = out.dropna(subset=[ID_COL, GROUP_COL, "event_hour"]).copy()
    out = out[out["event_hour"] > 0].copy()
    return out


def prepare_landmark_table(
    debug_max_stays: int = 0,
    min_landmark_hour: int = 24,
    step_size: int = 12,
    landmark_bin_size: int = 24,
) -> pd.DataFrame:
    """
    Build pooled landmark dataset from pseudo-dynamic master features.
    Adds competing-risk interval labels for (0,24], (24,48], (48,72].
    """
    master = prepare_master_table_xmi(
        debug_max_stays=debug_max_stays,
        min_landmark_hour=min_landmark_hour,
        step_size=step_size,
    )
    events = _build_event_table()

    df = master.merge(
        events[[ID_COL, "event_type", "event_hour", "death_hour"]],
        on=ID_COL,
        how="left",
    )
    df = df.dropna(subset=["event_type", "event_hour"]).copy()
    df["event_hour"] = pd.to_numeric(df["event_hour"], errors="coerce")
    df["remaining_hours"] = df["event_hour"] - df["t_end"]
    df = df[df["remaining_hours"] > 0].copy()

    lower = 0
    for upper in INTERVAL_BOUNDS:
        d_col = f"event_discharge_{lower}_{upper}h"
        m_col = f"event_death_{lower}_{upper}h"
        in_interval = (df["remaining_hours"] > lower) & (df["remaining_hours"] <= upper)
        df[d_col] = ((df["event_type"] == "discharge") & in_interval).astype(int)
        df[m_col] = ((df["event_type"] == "death") & in_interval).astype(int)
        lower = upper

    for upper in INTERVAL_BOUNDS:
        df[f"event_discharge_within_{upper}h"] = (
            (df["event_type"] == "discharge") & (df["remaining_hours"] <= upper)
        ).astype(int)
        df[f"event_death_within_{upper}h"] = (
            (df["event_type"] == "death") & (df["remaining_hours"] <= upper)
        ).astype(int)

    # 24h bins for non-pooled comparison.
    df["landmark_bin"] = (
        np.floor(df["t_end"] / float(landmark_bin_size)).astype(int) * int(landmark_bin_size)
    )
    df["landmark_bin"] = np.maximum(df["landmark_bin"], int(landmark_bin_size))
    df["landmark_hour"] = pd.to_numeric(df["t_end"], errors="coerce")
    df["landmark_hour_sq"] = df["landmark_hour"] ** 2

    # Keep only rows with valid group key for split.
    df = df.dropna(subset=[GROUP_COL]).copy()
    return df


def split_landmark_train_val_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df[GROUP_COL].nunique() < 3:
        raise ValueError("Need at least 3 unique subject_id groups for train/val/test split.")

    gss_outer = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=RANDOM_STATE)
    trainval_idx, test_idx = next(gss_outer.split(df, groups=df[GROUP_COL]))
    df_trainval = df.iloc[trainval_idx].copy()
    df_test = df.iloc[test_idx].copy()

    gss_inner = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=RANDOM_STATE + 1)
    train_idx, val_idx = next(gss_inner.split(df_trainval, groups=df_trainval[GROUP_COL]))
    df_train = df_trainval.iloc[train_idx].copy()
    df_val = df_trainval.iloc[val_idx].copy()

    print("-" * 50)
    print(f"[Train] rows={len(df_train)}, patients={df_train[GROUP_COL].nunique()}")
    print(f"[Val  ] rows={len(df_val)}, patients={df_val[GROUP_COL].nunique()}")
    print(f"[Test ] rows={len(df_test)}, patients={df_test[GROUP_COL].nunique()}")
    return df_train, df_val, df_test


def _add_landmark_interactions(
    train_x: pd.DataFrame,
    val_x: pd.DataFrame,
    test_x: pd.DataFrame,
    max_interactions: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    base_exclude = {"landmark_hour", "landmark_hour_sq"}
    candidates = [c for c in train_x.columns if c not in base_exclude]
    if len(candidates) == 0 or max_interactions <= 0:
        return train_x, val_x, test_x, []

    var_rank = train_x[candidates].var(numeric_only=True).sort_values(ascending=False)
    top_cols = [c for c in var_rank.index.tolist()[:max_interactions] if c in train_x.columns]
    created = []

    for col in top_cols:
        new_col = f"{col}_x_landmark"
        train_x[new_col] = train_x[col] * train_x["landmark_hour"]
        val_x[new_col] = val_x[col] * val_x["landmark_hour"]
        test_x[new_col] = test_x[col] * test_x["landmark_hour"]
        created.append(new_col)

    return train_x, val_x, test_x, created


def build_landmark_feature_matrices(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    corr_threshold: float = 0.95,
    max_interactions: int = 12,
) -> Dict[str, object]:
    safe_cols = {
        ID_COL,
        GROUP_COL,
        HADM_COL,
        "t_start",
        "t_end",
        "icu_out_hour",
        "event_type",
        "event_hour",
        "death_hour",
        "remaining_hours",
        "landmark_bin",
    }
    for upper in INTERVAL_BOUNDS:
        safe_cols.add(f"event_discharge_within_{upper}h")
        safe_cols.add(f"event_death_within_{upper}h")
    lower = 0
    for upper in INTERVAL_BOUNDS:
        safe_cols.add(f"event_discharge_{lower}_{upper}h")
        safe_cols.add(f"event_death_{lower}_{upper}h")
        lower = upper

    missing_rates = df_train.isnull().mean()
    cols_to_drop = missing_rates[missing_rates > 0.90].index.tolist()
    cols_to_drop = [c for c in cols_to_drop if c not in safe_cols and not c.endswith("_measured_frac")]

    train_work = df_train.drop(columns=cols_to_drop, errors="ignore").copy()
    val_work = df_val.drop(columns=cols_to_drop, errors="ignore").copy()
    test_work = df_test.drop(columns=cols_to_drop, errors="ignore").copy()

    exclude_cols = set(safe_cols)
    exclude_cols.update(
        {
            "final_text",
            "intime",
            "outtime",
            "discharge_in_24h",
            "icu_death_flag",
            "label_can_discharge_24h",
            "label_no_return_72h",
            "task2_mask",
        }
    )
    feature_cols = [c for c in train_work.columns if c not in exclude_cols and not c.lower().endswith("_id")]
    numeric_features = [
        c
        for c in feature_cols
        if pd.api.types.is_numeric_dtype(train_work[c]) or pd.api.types.is_bool_dtype(train_work[c])
    ]
    if len(numeric_features) == 0:
        raise ValueError("No numeric features left for landmark modeling.")

    x_train = train_work[numeric_features].copy()
    x_val = val_work[numeric_features].copy()
    x_test = test_work[numeric_features].copy()

    if "landmark_hour" not in x_train.columns:
        x_train["landmark_hour"] = pd.to_numeric(train_work["t_end"], errors="coerce")
        x_val["landmark_hour"] = pd.to_numeric(val_work["t_end"], errors="coerce")
        x_test["landmark_hour"] = pd.to_numeric(test_work["t_end"], errors="coerce")
    if "landmark_hour_sq" not in x_train.columns:
        x_train["landmark_hour_sq"] = x_train["landmark_hour"] ** 2
        x_val["landmark_hour_sq"] = x_val["landmark_hour"] ** 2
        x_test["landmark_hour_sq"] = x_test["landmark_hour"] ** 2

    imputer = SimpleImputer(strategy="median")
    x_train = pd.DataFrame(imputer.fit_transform(x_train), index=x_train.index, columns=x_train.columns)
    x_val = pd.DataFrame(imputer.transform(x_val), index=x_val.index, columns=x_val.columns)
    x_test = pd.DataFrame(imputer.transform(x_test), index=x_test.index, columns=x_test.columns)

    x_train, x_val, x_test, interaction_cols = _add_landmark_interactions(
        train_x=x_train,
        val_x=x_val,
        test_x=x_test,
        max_interactions=max_interactions,
    )

    scaler = StandardScaler()
    x_train = pd.DataFrame(scaler.fit_transform(x_train), index=x_train.index, columns=x_train.columns)
    x_val = pd.DataFrame(scaler.transform(x_val), index=x_val.index, columns=x_val.columns)
    x_test = pd.DataFrame(scaler.transform(x_test), index=x_test.index, columns=x_test.columns)

    x_train_uncorr, corr_drop_cols = drop_high_corr_features(x_train.astype(np.float32), threshold=corr_threshold)
    x_val_uncorr = x_val.astype(np.float32).drop(columns=corr_drop_cols, errors="ignore")
    x_test_uncorr = x_test.astype(np.float32).drop(columns=corr_drop_cols, errors="ignore")

    print(f"Columns dropped (>90% missing): {len(cols_to_drop)}")
    print(f"Interaction cols added: {len(interaction_cols)}")
    print(f"Correlated columns dropped: {len(corr_drop_cols)}")
    print("Final feature count:", x_train_uncorr.shape[1])

    return {
        "X_train": x_train_uncorr,
        "X_val": x_val_uncorr,
        "X_test": x_test_uncorr,
        "artifacts": {
            "cols_to_drop_missing": cols_to_drop,
            "interaction_cols": interaction_cols,
            "corr_drop_cols": corr_drop_cols,
            "numeric_features": numeric_features,
            "imputer": imputer,
            "scaler": scaler,
            "final_feature_names": x_train_uncorr.columns.tolist(),
            "corr_threshold": corr_threshold,
            "interval_bounds": list(INTERVAL_BOUNDS),
        },
    }
