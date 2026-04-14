# ============================================
# 0. Imports
# ============================================
import re
import warnings
from typing import List, Tuple, Optional, Dict
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Avoid matplotlib/fontconfig cache permission issues in sandboxed envs.
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

# Use non-interactive backend so the script can run in terminal/headless envs.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Optional xgboost backend. If unavailable (e.g., missing libomp on macOS),
# we automatically fall back to LogisticRegression so the pipeline still runs.
try:
    from xgboost import XGBClassifier  # type: ignore

    XGBOOST_AVAILABLE = True
    XGBOOST_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover
    XGBClassifier = None
    XGBOOST_AVAILABLE = False
    XGBOOST_IMPORT_ERROR = e

RANDOM_STATE = 42
TARGET_COL = "label_stable_discharge"
GROUP_COL = "subject_id"
ID_COL = "stay_id"
MODEL_N_JOBS = int(os.getenv("MODEL_N_JOBS", "1") or "1")
CV_N_JOBS = int(os.getenv("CV_N_JOBS", "1") or "1")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
RESULT_DIR = PROJECT_ROOT / "result" / "xgboost111result"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================
# Helper utilities
# ============================================
def build_model(scale_pos_weight: float, random_state: int):
    """Build XGBoost if available; otherwise use LogisticRegression fallback."""
    if XGBOOST_AVAILABLE:
        return XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=MODEL_N_JOBS,
            importance_type="gain",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
        )

    class_weight = {0: 1.0, 1: max(1.0, float(scale_pos_weight))}
    return LogisticRegression(
        solver="saga",
        penalty="l2",
        C=1.0,
        max_iter=500,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=MODEL_N_JOBS,
    )


def extract_feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    """Unified feature-importance extraction for tree and linear models."""
    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        if coef.ndim == 1:
            importance = np.abs(coef)
        else:
            importance = np.mean(np.abs(coef), axis=0)
    else:
        raise AttributeError("Current model backend has neither feature_importances_ nor coef_.")

    if importance.shape[0] != len(feature_names):
        raise ValueError("Feature-importance length does not match feature_names length.")

    return pd.DataFrame({"Feature": feature_names, "Importance": importance})


def safe_metric(func, *args, default=np.nan, **kwargs):
    try:
        return float(func(*args, **kwargs))
    except Exception:
        return default


def plot_save(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


# ============================================
# 1. Load data
# ============================================
df_static = pd.read_csv(DATASET_DIR / "MIMIC-IV-static(Group Assignment).csv")
df_text = pd.read_csv(DATASET_DIR / "MIMIC-IV-text(Group Assignment).csv")
df_time = pd.read_csv(DATASET_DIR / "MIMIC-IV-time_series(Group Assignment).csv")

print("Loaded:")
print("static:", df_static.shape)
print("text  :", df_text.shape)
print("time  :", df_time.shape)

if not XGBOOST_AVAILABLE:
    print("\n[Warning] xgboost import failed; fallback model will be used.")
    print("Reason:", str(XGBOOST_IMPORT_ERROR))
    print("Tip: on macOS, install OpenMP runtime (e.g., `brew install libomp`) to enable xgboost.")
else:
    print("\nUsing model backend: xgboost")


# ============================================
# 2. Basic cleaning: remove stays < 24h
# ============================================
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

debug_max_stays = int(os.getenv("DEBUG_MAX_STAYS", "0") or "0")
if debug_max_stays > 0:
    sampled_stays = df_static_clean[ID_COL].dropna().unique()[:debug_max_stays]
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


# ============================================
# 3. Remove leakage/surrogate-outcome columns from static
# ============================================
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

# Extra keyword-based safety net for accidental outcome proxies.
leakage_keywords = (
    "discharge",
    "expire",
    "mortality",
    "outcome",
    "target",
    "label",
    "readmit",
)
for c in df_static_clean.columns:
    cl = c.lower()
    if c in (ID_COL, GROUP_COL, "hadm_id", "intime"):
        continue
    if any(k in cl for k in leakage_keywords):
        leakage_cols_in_static.append(c)

existing_leakage_cols = sorted(set(c for c in leakage_cols_in_static if c in df_static_clean.columns))
df_static_subset = df_static_clean.drop(columns=existing_leakage_cols, errors="ignore").copy()

print(f"Removed {len(existing_leakage_cols)} leakage/static-outcome columns.")


# ============================================
# 4. Text preprocessing
# ============================================
def clean_medical_report(text: str) -> Optional[str]:
    if pd.isna(text):
        return np.nan
    text = str(text)

    # remove common communication/footer patterns
    text = re.sub(r"(Findings|Results)\s+were\s+communicated.*", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"electronically signed.*", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) == 0:
        return np.nan
    return text


if "radiology_note_text" not in df_text_clean.columns:
    raise ValueError("Expected column 'radiology_note_text' not found in text data.")

# Need intime for converting absolute text timestamps into relative ICU hours.
if "intime" not in df_text_clean.columns:
    df_text_clean = df_text_clean.merge(df_static_clean[[ID_COL, "intime"]], on=ID_COL, how="left")

df_text_clean["report_list"] = df_text_clean["radiology_note_text"].fillna("").str.split(r"-{3,}")
df_exploded = df_text_clean.explode("report_list").copy()
df_exploded["single_report_text"] = df_exploded["report_list"].astype(str).str.strip()
df_exploded["single_report_text"] = df_exploded["single_report_text"].replace("", np.nan)
df_exploded = df_exploded.dropna(subset=["single_report_text"]).copy()
df_exploded["clean_text"] = df_exploded["single_report_text"].apply(clean_medical_report)
df_exploded = df_exploded.dropna(subset=["clean_text"]).copy()
df_exploded["report_seq"] = df_exploded.groupby(ID_COL).cumcount() + 1

# Robust and leakage-safe time assignment for text.
# Priority:
# 1) absolute timestamps (radiology_note_time_max/min) converted to relative ICU hour.
# 2) precomputed relative timestamps (t_max_rel/t_min_rel).
# 3) if unavailable, drop those text rows from time-aware modeling (avoid future leakage).
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
    print(
        f"[Warning] Dropping {missing_report_time} text rows with unknown note time "
        f"to avoid temporal leakage."
    )
    df_exploded = df_exploded.dropna(subset=["report_time_estimated"]).copy()

print("Text records after cleaning:", df_exploded.shape)


# ============================================
# 5. Time preprocessing
# ============================================
if "intime" not in df_time_clean.columns:
    df_time_clean = df_time_clean.merge(df_static_subset[[ID_COL, "intime"]], on=ID_COL, how="left")

df_time_clean["hour_ts"] = pd.to_datetime(df_time_clean["hour_ts"], errors="coerce")
df_time_clean["intime"] = pd.to_datetime(df_time_clean["intime"], errors="coerce")

df_time_clean["relative_hour"] = (df_time_clean["hour_ts"] - df_time_clean["intime"]).dt.total_seconds() / 3600.0

# fix negative relative time
df_time_clean["relative_hour"] = df_time_clean["relative_hour"].clip(lower=0)

print(
    "Unified relative time range:",
    f"{df_time_clean['relative_hour'].min():.1f}h to {df_time_clean['relative_hour'].max():.1f}h",
)


# ============================================
# 6. Build sliding-window time-series features
# ============================================
WINDOW_SIZE = 24
STEP_SIZE = 12

exclude_time_cols = {
    ID_COL,
    GROUP_COL,
    "hour_ts",
    "intime",
    "relative_hour",
}

candidate_numeric_cols = []
for col in df_time_clean.columns:
    if col in exclude_time_cols:
        continue
    if pd.api.types.is_numeric_dtype(df_time_clean[col]):
        candidate_numeric_cols.append(col)

if len(candidate_numeric_cols) == 0:
    raise ValueError("No numeric time-series variables found after preprocessing.")

print(f"Number of numeric time-series variables: {len(candidate_numeric_cols)}")

# Ensure numeric columns are consistently numeric before rolling operations.
df_time_clean[candidate_numeric_cols] = df_time_clean[candidate_numeric_cols].apply(pd.to_numeric, errors="coerce")


def build_time_features_for_one_stay(
    stay_id: int,
    stay_df: pd.DataFrame,
    feature_cols: List[str],
    window_size: int = 24,
    step_size: int = 12,
) -> pd.DataFrame:
    """
    Build rolling 24h features at t_end = 24, 36, 48, ... for one stay.
    Uses vectorized rolling over hourly-aligned rows for speed.
    """
    if stay_df.empty:
        return pd.DataFrame()

    stay_work = stay_df[["relative_hour"] + feature_cols].copy()
    # Relative time is hourly in this dataset; round to the nearest hour for stable indexing.
    stay_work["relative_hour_int"] = np.rint(stay_work["relative_hour"]).astype(int)
    stay_work = stay_work.drop(columns=["relative_hour"])

    # Aggregate duplicates within the same hour (if any).
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
    roll_measured = (
        stay_hourly.notna().astype(float).rolling(window_size, min_periods=1).max().add_suffix("_is_measured")
    )
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


time_feature_parts = []
n_total_stays = int(df_time_clean[ID_COL].nunique())
start_time = time.time()

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
        elapsed = time.time() - start_time
        print(f"Processed stays: {i}/{n_total_stays} | elapsed={elapsed:.1f}s")

df_time_features = (
    pd.concat(time_feature_parts, ignore_index=True) if len(time_feature_parts) > 0 else pd.DataFrame()
)
if df_time_features.empty:
    raise ValueError("No sliding-window time features were generated.")

print("Sliding-window time feature table:", df_time_features.shape)


# ============================================
# 7. Merge text into each prediction window
# ============================================
master_base = df_time_features[[ID_COL, "t_end"]].drop_duplicates().copy()


def build_window_text(master_df: pd.DataFrame, text_df: pd.DataFrame) -> pd.Series:
    """Efficiently build cumulative available text per (stay, t_end) window."""
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


master_base["final_text"] = build_window_text(master_base, df_exploded)
master_base["has_text"] = (master_base["final_text"].str.len() > 0).astype(int)

df_master = df_time_features.merge(master_base, on=[ID_COL, "t_end"], how="left")
df_master = df_master.merge(df_static_subset, on=ID_COL, how="left")

print("Merged master table:", df_master.shape)


# ============================================
# 8. Create labels
# ============================================
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

# Only keep windows that happen before ICU discharge
before_count = len(df_master)
df_master = df_master[df_master["t_end"] <= df_master["icu_out_hour"]].copy()
print(f"Removed {before_count - len(df_master)} rows with t_end after ICU discharge.")

# Label: discharge within next 24h
df_master["discharge_in_24h"] = ((df_master["icu_out_hour"] - df_master["t_end"]) <= 24).astype(int)

# Stable discharge: discharge within 24h and not ICU death
df_master[TARGET_COL] = ((df_master["discharge_in_24h"] == 1) & (df_master["icu_death_flag"] == 0)).astype(int)

# Mandatory column sanity checks before split
required_cols = [ID_COL, GROUP_COL, TARGET_COL]
missing_required = [c for c in required_cols if c not in df_master.columns]
if missing_required:
    raise ValueError(f"Missing required columns before split: {missing_required}")

before_dropna = len(df_master)
df_master = df_master.dropna(subset=[ID_COL, GROUP_COL, TARGET_COL]).copy()
print(f"Dropped {before_dropna - len(df_master)} rows with missing ID/group/target.")

print(df_master[[TARGET_COL, "discharge_in_24h", "icu_death_flag"]].mean())


# ============================================
# 9. Group-aware train/test split
# ============================================
print("Start group-aware 8:2 split based on subject_id.")
if df_master[GROUP_COL].nunique() < 2:
    raise ValueError("Need at least 2 unique subject_id groups for train/test split.")

gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(df_master, groups=df_master[GROUP_COL]))

df_train = df_master.iloc[train_idx].copy()
df_test = df_master.iloc[test_idx].copy()

print("-" * 40)
print(f"[Train] rows={len(df_train)}, patients={df_train[GROUP_COL].nunique()}")
print(f"[Test ] rows={len(df_test)}, patients={df_test[GROUP_COL].nunique()}")
print(f"[Train] positive rate={df_train[TARGET_COL].mean():.4f}")
print(f"[Test ] positive rate={df_test[TARGET_COL].mean():.4f}")


# ============================================
# 10. Drop extremely sparse columns based on TRAIN only
# ============================================
safe_cols = [
    ID_COL,
    GROUP_COL,
    "t_start",
    "t_end",
    "icu_out_hour",
    "discharge_in_24h",
    "icu_death_flag",
    TARGET_COL,
    "final_text",
    "has_text",
    "intime",
    "outtime",
]

missing_rates = df_train.isnull().mean()
cols_to_drop = missing_rates[missing_rates > 0.90].index.tolist()
cols_to_drop = [c for c in cols_to_drop if c not in safe_cols and not c.endswith("_is_measured")]

print(f"Columns to drop due to >90% missingness: {len(cols_to_drop)}")

df_train = df_train.drop(columns=cols_to_drop, errors="ignore")
df_test = df_test.drop(columns=cols_to_drop, errors="ignore")


# ============================================
# 11. TF-IDF text features
# ============================================
TFIDF_MAX_FEATURES = 300
tfidf = TfidfVectorizer(
    max_features=TFIDF_MAX_FEATURES,
    ngram_range=(1, 2),
    min_df=5,
)

train_text = df_train["final_text"].fillna("")
test_text = df_test["final_text"].fillna("")

tfidf_feature_names = []
if (train_text.str.len() > 0).any():
    try:
        X_train_text = tfidf.fit_transform(train_text)
        X_test_text = tfidf.transform(test_text)

        tfidf_feature_names = [f"tfidf_{w}" for w in tfidf.get_feature_names_out()]

        if len(tfidf_feature_names) > 0:
            df_train_tfidf = pd.DataFrame(
                X_train_text.toarray().astype(np.float32),
                index=df_train.index,
                columns=tfidf_feature_names,
            )
            df_test_tfidf = pd.DataFrame(
                X_test_text.toarray().astype(np.float32),
                index=df_test.index,
                columns=tfidf_feature_names,
            )

            df_train = pd.concat([df_train, df_train_tfidf], axis=1)
            df_test = pd.concat([df_test, df_test_tfidf], axis=1)
    except (ValueError, MemoryError) as e:
        print(f"[Warning] TF-IDF skipped due to: {e}")
else:
    print("[Info] No non-empty text in training set; TF-IDF skipped.")

print("Added TF-IDF features:", len(tfidf_feature_names))


# ============================================
# 12. Build modeling matrix
# ============================================
exclude_cols = [
    ID_COL,
    GROUP_COL,
    "t_start",
    "t_end",
    "icu_out_hour",
    "intime",
    "outtime",
    "final_text",
    "discharge_in_24h",
    "icu_death_flag",
    TARGET_COL,
]

all_features = [c for c in df_train.columns if c not in exclude_cols]

# Keep only numeric/bool
numeric_features = []
for c in all_features:
    if pd.api.types.is_numeric_dtype(df_train[c]) or pd.api.types.is_bool_dtype(df_train[c]):
        numeric_features.append(c)

X_train = df_train[numeric_features].copy()
X_test = df_test[numeric_features].copy()
y_train = df_train[TARGET_COL].astype(int).copy()
y_test = df_test[TARGET_COL].astype(int).copy()

if X_train.shape[1] == 0:
    raise ValueError("No modeling features are left after preprocessing.")

print("Initial modeling feature count:", X_train.shape[1])


# ============================================
# 13. Imputation (fit on train only)
# ============================================
imputer = SimpleImputer(strategy="median")
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
X_test_imp = pd.DataFrame(imputer.transform(X_test), index=X_test.index, columns=X_test.columns)

# For linear fallback models, standardize features so coefficient-based
# importance and optimization are better behaved.
if not XGBOOST_AVAILABLE:
    scaler = StandardScaler()
    X_train_imp = pd.DataFrame(scaler.fit_transform(X_train_imp), index=X_train_imp.index, columns=X_train_imp.columns)
    X_test_imp = pd.DataFrame(scaler.transform(X_test_imp), index=X_test_imp.index, columns=X_test_imp.columns)

X_train_imp = X_train_imp.astype(np.float32)
X_test_imp = X_test_imp.astype(np.float32)


# ============================================
# 14. Remove highly correlated features on TRAIN only
# ============================================
def drop_high_corr_features(df_num: pd.DataFrame, threshold: float = 0.95) -> Tuple[pd.DataFrame, List[str]]:
    if df_num.shape[1] <= 1:
        return df_num.copy(), []

    corr_matrix = df_num.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df_num.drop(columns=to_drop), to_drop


X_train_uncorr, corr_drop_cols = drop_high_corr_features(X_train_imp, threshold=0.95)
X_test_uncorr = X_test_imp.drop(columns=corr_drop_cols, errors="ignore")

print(f"Removed {len(corr_drop_cols)} highly correlated features.")
print("Remaining feature count:", X_train_uncorr.shape[1])


# ============================================
# 15. Feature ranking on TRAIN only
# ============================================
n_pos = int(y_train.sum())
n_neg = int(len(y_train) - n_pos)
scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

ranker_model = build_model(scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE)
ranker_model.fit(X_train_uncorr, y_train)

importance_df = (
    extract_feature_importance(ranker_model, X_train_uncorr.columns.tolist())
    .sort_values(by="Importance", ascending=False)
    .reset_index(drop=True)
)

importance_sum = float(importance_df["Importance"].sum())
if importance_sum > 0:
    importance_df["Cumulative_Importance"] = importance_df["Importance"].cumsum() / importance_sum
else:
    importance_df["Cumulative_Importance"] = np.linspace(1 / max(len(importance_df), 1), 1.0, len(importance_df))

print(importance_df.head(20))


# ============================================
# 16. Group-aware CV for selecting top-K
# ============================================
sorted_features = importance_df["Feature"].tolist()
if len(sorted_features) == 0:
    raise ValueError("No ranked features available for CV selection.")

k_values = [10, 20, 30, 40, 50, 60, 80, 100, min(150, len(sorted_features)), len(sorted_features)]
k_values = sorted(set([k for k in k_values if k <= len(sorted_features)]))

group_level_target = df_train.groupby(GROUP_COL)[TARGET_COL].max()
min_groups_per_class = min((group_level_target == 0).sum(), (group_level_target == 1).sum())
n_splits = min(5, int(min_groups_per_class))

auc_scores = []
auc_stds = []
groups_train = df_train[GROUP_COL]

if n_splits >= 2:
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    for k in k_values:
        top_k_features = sorted_features[:k]
        X_subset = X_train_uncorr[top_k_features]

        model_cv = build_model(scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE)

        scores = cross_val_score(
            model_cv,
            X_subset,
            y_train,
            cv=cv,
            groups=groups_train,
            scoring="roc_auc",
            n_jobs=CV_N_JOBS,
            error_score=np.nan,
        )
        valid_scores = scores[~np.isnan(scores)]

        if len(valid_scores) == 0:
            mean_auc = np.nan
            std_auc = np.nan
        else:
            mean_auc = float(valid_scores.mean())
            std_auc = float(valid_scores.std())

        auc_scores.append(mean_auc)
        auc_stds.append(std_auc)
        print(f"Top-{k:<3d} | Group-CV ROC-AUC = {mean_auc:.4f} ± {std_auc:.4f}")
else:
    print("[Warning] Not enough positive/negative patient groups for StratifiedGroupKFold.")
    # Fallback: still keep structures consistent and choose a default K later.
    auc_scores = [np.nan for _ in k_values]
    auc_stds = [np.nan for _ in k_values]

cv_result_df = pd.DataFrame({"k": k_values, "cv_auc_mean": auc_scores, "cv_auc_std": auc_stds})

if cv_result_df["cv_auc_mean"].notna().any():
    cv_result_df = cv_result_df.sort_values("cv_auc_mean", ascending=False)
else:
    cv_result_df = cv_result_df.sort_values("k")

print("\nBest K by group-aware CV:")
print(cv_result_df.head())


# ============================================
# 17. Select final K
# ============================================
if cv_result_df["cv_auc_mean"].notna().any():
    best_k = int(cv_result_df.iloc[0]["k"])
else:
    best_k = min(40, len(sorted_features))
    print(f"[Info] CV AUC unavailable; fallback to K={best_k}")

print(f"Selected K = {best_k}")

final_selected_features = sorted_features[:best_k]

X_train_selected = X_train_uncorr[final_selected_features].copy()
X_test_selected = X_test_uncorr[final_selected_features].copy()

print("Final train shape:", X_train_selected.shape)
print("Final test shape :", X_test_selected.shape)


# ============================================
# 18. Train final model
# ============================================
final_model = build_model(scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE)
final_model.fit(X_train_selected, y_train)


# ============================================
# 19. Evaluate on independent TEST set
# ============================================
if hasattr(final_model, "predict_proba"):
    y_prob = final_model.predict_proba(X_test_selected)[:, 1]
else:
    # Generic fallback for estimators without predict_proba
    raw_score = final_model.decision_function(X_test_selected)
    y_prob = 1.0 / (1.0 + np.exp(-raw_score))

y_pred = (y_prob >= 0.5).astype(int)

test_metrics = {
    "ROC-AUC": safe_metric(roc_auc_score, y_test, y_prob),
    "PR-AUC": safe_metric(average_precision_score, y_test, y_prob),
    "Accuracy": safe_metric(accuracy_score, y_test, y_pred),
    "F1": safe_metric(f1_score, y_test, y_pred, zero_division=0),
    "Precision": safe_metric(precision_score, y_test, y_pred, zero_division=0),
    "Recall": safe_metric(recall_score, y_test, y_pred, zero_division=0),
}

print("\nIndependent TEST metrics")
for k, v in test_metrics.items():
    if np.isnan(v):
        print(f"{k:10s}: nan")
    else:
        print(f"{k:10s}: {v:.4f}")

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))


# ============================================
# 20. Plot CV curve
# ============================================
plt.figure(figsize=(8, 5))
plt.errorbar(k_values, auc_scores, yerr=auc_stds, marker="o")
plt.xlabel("Number of Top Features")
plt.ylabel("Group-CV ROC-AUC")
plt.title("Feature Count Selection with Group-Aware CV")
plt.grid(True, alpha=0.3)
plot_save(RESULT_DIR / "group_cv_feature_selection_curve.png")


# ============================================
# 21. Plot ROC / PR on test
# ============================================
fig, ax = plt.subplots(figsize=(6, 6))
RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax)
ax.set_title("Test ROC Curve")
plot_save(RESULT_DIR / "test_roc_curve.png")

fig, ax = plt.subplots(figsize=(6, 6))
PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=ax)
ax.set_title("Test Precision-Recall Curve")
plot_save(RESULT_DIR / "test_pr_curve.png")


# ============================================
# 22. Plot top feature importance
# ============================================
final_importance = extract_feature_importance(final_model, final_selected_features).sort_values("Importance", ascending=False)

plt.figure(figsize=(8, 10))
top_n = min(20, len(final_importance))
plt.barh(final_importance["Feature"].head(top_n)[::-1], final_importance["Importance"].head(top_n)[::-1])
plt.xlabel("Importance")
plt.title("Top Final Model Features")
plot_save(RESULT_DIR / "final_top_feature_importance.png")


# ============================================
# 23. SHAP analysis
# ============================================
try:
    import shap

    if not XGBOOST_AVAILABLE:
        raise RuntimeError("SHAP tree plots are skipped for non-tree fallback model.")

    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_test_selected)

    shap.summary_plot(shap_values, X_test_selected, plot_type="bar", show=False)
    plt.title("Overall SHAP Feature Importance - Test Set")
    plot_save(RESULT_DIR / "shap_summary_bar.png")

    shap.summary_plot(shap_values, X_test_selected, show=False)
    plt.title("Overall SHAP Feature Contribution - Test Set")
    plot_save(RESULT_DIR / "shap_summary_beeswarm.png")

    # early vs late window SHAP
    X_test_with_time = X_test_selected.copy()
    X_test_with_time["t_end"] = df_test.loc[X_test_selected.index, "t_end"].values

    median_t_end = X_test_with_time["t_end"].median()

    early_mask = X_test_with_time["t_end"] <= median_t_end
    late_mask = X_test_with_time["t_end"] > median_t_end

    X_test_early = X_test_with_time.loc[early_mask].drop(columns=["t_end"])
    X_test_late = X_test_with_time.loc[late_mask].drop(columns=["t_end"])

    if len(X_test_early) > 0:
        shap_values_early = explainer.shap_values(X_test_early)
        shap.summary_plot(shap_values_early, X_test_early, plot_type="bar", show=False)
        plt.title("SHAP Feature Importance - Early Prediction Windows")
        plot_save(RESULT_DIR / "shap_early_bar.png")

    if len(X_test_late) > 0:
        shap_values_late = explainer.shap_values(X_test_late)
        shap.summary_plot(shap_values_late, X_test_late, plot_type="bar", show=False)
        plt.title("SHAP Feature Importance - Late Prediction Windows")
        plot_save(RESULT_DIR / "shap_late_bar.png")

except Exception as e:
    print("SHAP analysis skipped due to error:")
    print(e)


# ============================================
# 24. Save outputs for report
# ============================================
cv_result_df.to_csv(RESULT_DIR / "group_cv_feature_selection_results.csv", index=False)
final_importance.to_csv(RESULT_DIR / "final_model_feature_importance.csv", index=False)

pd.DataFrame({"metric": list(test_metrics.keys()), "value": list(test_metrics.values())}).to_csv(
    RESULT_DIR / "test_metrics.csv",
    index=False,
)

print("\nSaved:")
print("- group_cv_feature_selection_results.csv")
print("- final_model_feature_importance.csv")
print("- test_metrics.csv")
print("- group_cv_feature_selection_curve.png")
print("- test_roc_curve.png")
print("- test_pr_curve.png")
print("- final_top_feature_importance.png")
print("Output dir:", RESULT_DIR)
