"""
data_cleaning.py
----------------
Loads city_day.csv, validates, cleans, and exports a processed dataset
ready for model training. Produces data/processed_city_day.csv and
prints a brief data-quality report.

Usage:
    python src/data_cleaning.py
"""

import os
import pandas as pd
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
RAW_PATH = "city_day.csv"
OUT_PATH  = "data/processed_city_day.csv"
REPORT_PATH = "data/data_quality_report.txt"

# Features used for modelling (Xylene dropped – >60 % missing)
FEATURE_COLS = ["PM2.5", "PM10", "NO", "NO2", "NOx",
                "NH3", "CO", "SO2", "O3", "Benzene", "Toluene"]
TARGET_COL   = "AQI_Bucket"
AUX_COLS     = ["City", "Date", "AQI"]

# Canonical order / label encoding kept consistent with training
BUCKET_ORDER = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    print(f"[INFO] Loaded {path}: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def drop_missing_target(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=[TARGET_COL, "AQI"])
    after  = len(df)
    print(f"[INFO] Dropped {before - after} rows with missing AQI / AQI_Bucket  "
          f"({after} rows remain)")
    return df


def drop_xylene(df: pd.DataFrame) -> pd.DataFrame:
    """Remove Xylene – over 62 % of values are missing."""
    if "Xylene" in df.columns:
        df = df.drop(columns=["Xylene"])
        print("[INFO] Dropped 'Xylene' column (>62 % missing)")
    return df


def cap_outliers(df: pd.DataFrame, cols: list, upper_quantile: float = 0.999
                 ) -> pd.DataFrame:
    """Cap extreme sensor outliers at the 99.9th percentile."""
    for col in cols:
        if col in df.columns:
            cap = df[col].quantile(upper_quantile)
            before = (df[col] > cap).sum()
            df[col] = df[col].clip(upper=cap)
            if before:
                print(f"[INFO] Capped {before} outliers in '{col}' at {cap:.2f}")
    return df


def impute_features(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Median imputation per city for pollutant features."""
    for col in cols:
        if col not in df.columns:
            continue
        missing_before = df[col].isna().sum()
        if missing_before == 0:
            continue
        # city-level median first, global median as fallback
        df[col] = df.groupby("City")[col].transform(
            lambda x: x.fillna(x.median())
        )
        df[col] = df[col].fillna(df[col].median())
        missing_after = df[col].isna().sum()
        print(f"[INFO] Imputed '{col}': {missing_before} -> {missing_after} missing")
    return df


def validate_bucket_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows whose AQI_Bucket is in the known set."""
    valid_buckets = set(BUCKET_ORDER)
    before = len(df)
    df = df[df[TARGET_COL].isin(valid_buckets)].copy()
    after  = len(df)
    if before != after:
        print(f"[WARN] Removed {before - after} rows with unrecognised AQI_Bucket labels")
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive month and day-of-year from Date for seasonal patterns."""
    df["Month"]      = df["Date"].dt.month
    df["DayOfYear"]  = df["Date"].dt.dayofyear
    return df


def write_report(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        "Air Quality Dataset - Data Quality Report",
        "=" * 45,
        f"Total rows after cleaning : {len(df)}",
        f"Date range                : {df['Date'].min().date()} to {df['Date'].max().date()}",
        f"Cities covered            : {df['City'].nunique()} "
          f"({', '.join(sorted(df['City'].unique())[:6])} ...)",
        "",
        "Target distribution (AQI_Bucket):",
    ]
    for bucket in BUCKET_ORDER:
        n   = (df[TARGET_COL] == bucket).sum()
        pct = 100 * n / len(df)
        lines.append(f"  {bucket:<15}: {n:>5}  ({pct:.1f} %)")
    lines += [
        "",
        "Remaining missing values per feature:",
    ]
    existing_feats = [c for c in FEATURE_COLS if c in df.columns]
    for col in existing_feats:
        lines.append(f"  {col:<12}: {df[col].isna().sum()}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[INFO] Data quality report saved -> {path}")


def run_cleaning_pipeline() -> pd.DataFrame:
    os.makedirs("data", exist_ok=True)
    df = load_raw(RAW_PATH)
    df = drop_missing_target(df)
    df = drop_xylene(df)
    df = validate_bucket_labels(df)
    existing_feats = [c for c in FEATURE_COLS if c in df.columns]
    df = cap_outliers(df, existing_feats)
    df = impute_features(df, existing_feats)
    df = add_temporal_features(df)

    keep_cols = AUX_COLS + existing_feats + ["Month", "DayOfYear", TARGET_COL]
    df = df[keep_cols].reset_index(drop=True)

    df.to_csv(OUT_PATH, index=False)
    print(f"[INFO] Processed dataset saved -> {OUT_PATH}  ({len(df)} rows)")
    write_report(df, REPORT_PATH)
    return df


if __name__ == "__main__":
    run_cleaning_pipeline()
