from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from utils.multitask_common import (
    DATASET_DIR,
    GROUP_COL,
    HADM_COL,
    ID_COL,
    TASK1_COL,
    TASK2_COL,
    TASK2_MASK_COL,
    prepare_master_table,
    split_group_train_val_test,
)

TIME_WINDOW_HOURS = 24
SURVIVAL_INTERVALS: Tuple[Tuple[int, int], ...] = ((0, 24), (24, 48), (48, 72))
SURVIVAL_CLASS_NONE = 0
SURVIVAL_CLASS_DISCHARGE = 1
SURVIVAL_CLASS_DEATH = 2


@dataclass
class CmaArtifacts:
    static_feature_cols: List[str]
    time_feature_cols: List[str]
    static_imputer: SimpleImputer
    static_scaler: StandardScaler
    time_mean: np.ndarray
    time_std: np.ndarray
    tokenizer_name: str
    max_text_len: int
    time_window_hours: int
    survival_intervals: List[Tuple[int, int]]


class CmaTensorDataset(Dataset):
    def __init__(
        self,
        time_tensor: np.ndarray,
        static_tensor: np.ndarray,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        task1: np.ndarray,
        task2: np.ndarray,
        task2_mask: np.ndarray,
        survival_targets: np.ndarray,
        stay_ids: np.ndarray,
        subject_ids: np.ndarray,
        t_end: np.ndarray,
    ) -> None:
        self.time_tensor = torch.from_numpy(time_tensor.astype(np.float32))
        self.static_tensor = torch.from_numpy(static_tensor.astype(np.float32))
        self.input_ids = torch.from_numpy(input_ids.astype(np.int64))
        self.attention_mask = torch.from_numpy(attention_mask.astype(np.int64))
        self.task1 = torch.from_numpy(task1.astype(np.float32))
        self.task2 = torch.from_numpy(task2.astype(np.float32))
        self.task2_mask = torch.from_numpy(task2_mask.astype(np.float32))
        self.survival_targets = torch.from_numpy(survival_targets.astype(np.int64))
        self.stay_ids = torch.from_numpy(stay_ids.astype(np.int64))
        self.subject_ids = torch.from_numpy(subject_ids.astype(np.int64))
        self.t_end = torch.from_numpy(t_end.astype(np.float32))

    def __len__(self) -> int:
        return self.time_tensor.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "time_series": self.time_tensor[idx],
            "static": self.static_tensor[idx],
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "task1_label": self.task1[idx],
            "task2_label": self.task2[idx],
            "task2_mask": self.task2_mask[idx],
            "survival_target": self.survival_targets[idx],
            "stay_id": self.stay_ids[idx],
            "subject_id": self.subject_ids[idx],
            "t_end": self.t_end[idx],
        }


def _drop_static_leakage(df_static: pd.DataFrame) -> pd.DataFrame:
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
    for col in df_static.columns:
        col_l = col.lower()
        if col in (ID_COL, GROUP_COL, HADM_COL, "intime"):
            continue
        if any(k in col_l for k in leakage_keywords):
            leakage_cols.append(col)

    leakage_cols = sorted(set(c for c in leakage_cols if c in df_static.columns))
    return df_static.drop(columns=leakage_cols, errors="ignore").copy()


def _build_static_table(stay_ids: Iterable[int]) -> Tuple[pd.DataFrame, List[str]]:
    static_path = DATASET_DIR / "MIMIC-IV-static(Group Assignment).csv"
    df_static = pd.read_csv(static_path)

    df_static["intime"] = pd.to_datetime(df_static["intime"], errors="coerce")
    df_static["outtime"] = pd.to_datetime(df_static["outtime"], errors="coerce")
    if "deathtime" in df_static.columns:
        df_static["deathtime"] = pd.to_datetime(df_static["deathtime"], errors="coerce")
    else:
        df_static["deathtime"] = pd.NaT

    df_static = df_static[df_static["icu_los_hours"] >= 24].copy()
    df_static = df_static[df_static[ID_COL].isin(set(stay_ids))].copy()

    df_static["event_type"] = np.where(
        df_static["deathtime"].notna() & (df_static["deathtime"] <= df_static["outtime"]),
        "death",
        "discharge",
    )
    discharge_hour = (df_static["outtime"] - df_static["intime"]).dt.total_seconds() / 3600.0
    death_hour = (df_static["deathtime"] - df_static["intime"]).dt.total_seconds() / 3600.0
    df_static["event_hour"] = np.where(df_static["event_type"] == "death", death_hour, discharge_hour)
    df_static["event_hour"] = pd.to_numeric(df_static["event_hour"], errors="coerce")

    static_no_leak = _drop_static_leakage(df_static)

    base_exclude = {
        ID_COL,
        GROUP_COL,
        HADM_COL,
        "intime",
        "outtime",
        "deathtime",
        "event_type",
        "event_hour",
    }
    static_cols: List[str] = []
    for col in static_no_leak.columns:
        if col in base_exclude:
            continue
        if col.lower().endswith("_id"):
            continue
        if pd.api.types.is_numeric_dtype(static_no_leak[col]) or pd.api.types.is_bool_dtype(static_no_leak[col]):
            static_cols.append(col)

    prefixed_map = {col: f"static__{col}" for col in static_cols}
    keep_cols = [ID_COL, GROUP_COL, "event_type", "event_hour", *static_cols]
    out = static_no_leak[keep_cols].rename(columns=prefixed_map).copy()
    return out, list(prefixed_map.values())


def _build_time_hourly_map(stay_ids: Sequence[int]) -> Tuple[Dict[int, np.ndarray], List[str]]:
    time_path = DATASET_DIR / "MIMIC-IV-time_series(Group Assignment).csv"
    df_time = pd.read_csv(time_path)
    df_time = df_time[df_time[ID_COL].isin(set(stay_ids))].copy()

    # Borrow per-stay intime from static table if not present in time table.
    if "intime" not in df_time.columns:
        intime_table = pd.read_csv(DATASET_DIR / "MIMIC-IV-static(Group Assignment).csv", usecols=[ID_COL, "intime"])
        intime_table["intime"] = pd.to_datetime(intime_table["intime"], errors="coerce")
        df_time = df_time.merge(intime_table, on=ID_COL, how="left")
    else:
        df_time["intime"] = pd.to_datetime(df_time["intime"], errors="coerce")

    df_time["hour_ts"] = pd.to_datetime(df_time["hour_ts"], errors="coerce")
    df_time["relative_hour"] = (df_time["hour_ts"] - df_time["intime"]).dt.total_seconds() / 3600.0
    df_time["relative_hour"] = df_time["relative_hour"].clip(lower=0)
    df_time["relative_hour_int"] = np.floor(df_time["relative_hour"]).astype("Int64")
    df_time = df_time.dropna(subset=["relative_hour_int"]).copy()
    df_time["relative_hour_int"] = df_time["relative_hour_int"].astype(int)

    exclude_cols = {ID_COL, GROUP_COL, HADM_COL, "hour_ts", "intime", "relative_hour", "relative_hour_int"}
    time_feature_cols: List[str] = []
    for col in df_time.columns:
        if col in exclude_cols:
            continue
        if col.lower().endswith("_id"):
            continue
        if pd.api.types.is_numeric_dtype(df_time[col]) or pd.api.types.is_bool_dtype(df_time[col]):
            time_feature_cols.append(col)

    df_time[time_feature_cols] = df_time[time_feature_cols].apply(pd.to_numeric, errors="coerce")

    stay_map: Dict[int, np.ndarray] = {}
    grouped = df_time.groupby(ID_COL, sort=False)
    for stay_id, g in grouped:
        g_hourly = g.groupby("relative_hour_int", as_index=True)[time_feature_cols].mean()
        if g_hourly.empty:
            stay_map[int(stay_id)] = np.empty((0, len(time_feature_cols)), dtype=np.float32)
            continue

        max_hour = int(g_hourly.index.max())
        hourly = g_hourly.reindex(np.arange(0, max_hour + 1, dtype=int))
        stay_map[int(stay_id)] = hourly.to_numpy(dtype=np.float32, copy=True)

    # Keep table and map aligned in presence of stays without time rows.
    for stay_id in stay_ids:
        if int(stay_id) not in stay_map:
            stay_map[int(stay_id)] = np.empty((0, len(time_feature_cols)), dtype=np.float32)

    return stay_map, time_feature_cols


def _build_survival_targets(df: pd.DataFrame) -> np.ndarray:
    remaining = pd.to_numeric(df["event_hour"], errors="coerce") - pd.to_numeric(df["t_end"], errors="coerce")
    event_is_discharge = df["event_type"].astype(str).to_numpy() == "discharge"
    event_is_death = df["event_type"].astype(str).to_numpy() == "death"

    targets = np.full((len(df), len(SURVIVAL_INTERVALS)), SURVIVAL_CLASS_NONE, dtype=np.int64)
    rem = remaining.to_numpy(dtype=np.float64)
    for i, (lower, upper) in enumerate(SURVIVAL_INTERVALS):
        in_bin = (rem > float(lower)) & (rem <= float(upper))
        targets[in_bin & event_is_discharge, i] = SURVIVAL_CLASS_DISCHARGE
        targets[in_bin & event_is_death, i] = SURVIVAL_CLASS_DEATH

    return targets


def _extract_time_windows(
    df: pd.DataFrame,
    stay_map: Dict[int, np.ndarray],
    n_features: int,
    window_hours: int,
) -> np.ndarray:
    out = np.full((len(df), window_hours, n_features), np.nan, dtype=np.float32)
    if len(df) == 0:
        return out

    key_df = df[[ID_COL, "t_end"]].copy()
    for row_idx, (stay_id, t_end) in enumerate(key_df.itertuples(index=False, name=None)):
        stay_arr = stay_map.get(int(stay_id))
        if stay_arr is None or stay_arr.size == 0:
            continue
        t_end_int = int(math.floor(float(t_end)))
        hours = np.arange(t_end_int - window_hours + 1, t_end_int + 1, dtype=int)
        valid = (hours >= 0) & (hours < stay_arr.shape[0])
        if np.any(valid):
            out[row_idx, valid, :] = stay_arr[hours[valid], :]
    return out


def _tokenize_text(texts: Sequence[str], tokenizer: AutoTokenizer, max_text_len: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(texts) == 0:
        return (
            np.zeros((0, max_text_len), dtype=np.int64),
            np.zeros((0, max_text_len), dtype=np.int64),
        )

    encoded = tokenizer(
        list(texts),
        max_length=max_text_len,
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
    )
    input_ids = np.asarray(encoded["input_ids"], dtype=np.int64)
    attention_mask = np.asarray(encoded["attention_mask"], dtype=np.int64)
    return input_ids, attention_mask


def _normalize_time_by_train(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time_mean = np.nanmean(x_train, axis=(0, 1))
    time_std = np.nanstd(x_train, axis=(0, 1))

    time_mean = np.where(np.isfinite(time_mean), time_mean, 0.0).astype(np.float32)
    time_std = np.where(np.isfinite(time_std) & (time_std > 1e-6), time_std, 1.0).astype(np.float32)

    def _norm(arr: np.ndarray) -> np.ndarray:
        norm = (arr - time_mean.reshape(1, 1, -1)) / time_std.reshape(1, 1, -1)
        return np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return _norm(x_train), _norm(x_val), _norm(x_test), time_mean, time_std


def _build_split_dataset(
    df_split: pd.DataFrame,
    time_tensor: np.ndarray,
    static_tensor: np.ndarray,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
) -> CmaTensorDataset:
    surv_targets = df_split[[f"surv_label_bin{i + 1}" for i in range(len(SURVIVAL_INTERVALS))]].to_numpy(dtype=np.int64)
    task1 = df_split[TASK1_COL].astype(int).to_numpy()
    task2 = df_split[TASK2_COL].astype(int).to_numpy()
    task2_mask = df_split[TASK2_MASK_COL].astype(int).to_numpy()
    stay_ids = df_split[ID_COL].astype(int).to_numpy()
    subject_ids = df_split[GROUP_COL].astype(int).to_numpy()
    t_end = pd.to_numeric(df_split["t_end"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    return CmaTensorDataset(
        time_tensor=time_tensor,
        static_tensor=static_tensor,
        input_ids=input_ids,
        attention_mask=attention_mask,
        task1=task1,
        task2=task2,
        task2_mask=task2_mask,
        survival_targets=surv_targets,
        stay_ids=stay_ids,
        subject_ids=subject_ids,
        t_end=t_end,
    )


def build_cma_data_bundle(
    debug_max_stays: int = 0,
    tokenizer_name: str = "emilyalsentzer/Bio_ClinicalBERT",
    max_text_len: int = 128,
    time_window_hours: int = TIME_WINDOW_HOURS,
) -> Dict[str, object]:
    df_master = prepare_master_table(debug_max_stays=debug_max_stays)
    static_table, static_feature_cols = _build_static_table(df_master[ID_COL].unique().tolist())

    df_all = df_master.merge(static_table, on=[ID_COL, GROUP_COL], how="left")
    df_all = df_all.dropna(subset=["event_type", "event_hour"]).copy()

    surv_targets = _build_survival_targets(df_all)
    for i in range(len(SURVIVAL_INTERVALS)):
        df_all[f"surv_label_bin{i + 1}"] = surv_targets[:, i]

    df_train, df_val, df_test = split_group_train_val_test(df_all)

    stay_map, time_feature_cols = _build_time_hourly_map(stay_ids=df_all[ID_COL].astype(int).unique().tolist())

    x_time_train = _extract_time_windows(df_train, stay_map, n_features=len(time_feature_cols), window_hours=time_window_hours)
    x_time_val = _extract_time_windows(df_val, stay_map, n_features=len(time_feature_cols), window_hours=time_window_hours)
    x_time_test = _extract_time_windows(df_test, stay_map, n_features=len(time_feature_cols), window_hours=time_window_hours)
    x_time_train, x_time_val, x_time_test, time_mean, time_std = _normalize_time_by_train(
        x_time_train,
        x_time_val,
        x_time_test,
    )

    # Keep all static columns to preserve model input dimensionality even when
    # some columns are fully missing in sampled debug splits.
    static_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    static_scaler = StandardScaler()

    x_static_train = static_imputer.fit_transform(df_train[static_feature_cols].astype(np.float32))
    x_static_val = static_imputer.transform(df_val[static_feature_cols].astype(np.float32))
    x_static_test = static_imputer.transform(df_test[static_feature_cols].astype(np.float32))

    x_static_train = static_scaler.fit_transform(x_static_train).astype(np.float32)
    x_static_val = static_scaler.transform(x_static_val).astype(np.float32)
    x_static_test = static_scaler.transform(x_static_test).astype(np.float32)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    train_text = df_train["final_text"].fillna("").astype(str).tolist()
    val_text = df_val["final_text"].fillna("").astype(str).tolist()
    test_text = df_test["final_text"].fillna("").astype(str).tolist()

    train_ids, train_mask = _tokenize_text(train_text, tokenizer, max_text_len=max_text_len)
    val_ids, val_mask = _tokenize_text(val_text, tokenizer, max_text_len=max_text_len)
    test_ids, test_mask = _tokenize_text(test_text, tokenizer, max_text_len=max_text_len)

    train_dataset = _build_split_dataset(df_train, x_time_train, x_static_train, train_ids, train_mask)
    val_dataset = _build_split_dataset(df_val, x_time_val, x_static_val, val_ids, val_mask)
    test_dataset = _build_split_dataset(df_test, x_time_test, x_static_test, test_ids, test_mask)

    artifacts = CmaArtifacts(
        static_feature_cols=static_feature_cols,
        time_feature_cols=time_feature_cols,
        static_imputer=static_imputer,
        static_scaler=static_scaler,
        time_mean=time_mean,
        time_std=time_std,
        tokenizer_name=tokenizer_name,
        max_text_len=max_text_len,
        time_window_hours=time_window_hours,
        survival_intervals=list(SURVIVAL_INTERVALS),
    )

    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "train_df": df_train.reset_index(drop=True),
        "val_df": df_val.reset_index(drop=True),
        "test_df": df_test.reset_index(drop=True),
        "artifacts": artifacts,
        "time_feature_dim": int(len(time_feature_cols)),
        "static_feature_dim": int(len(static_feature_cols)),
    }
