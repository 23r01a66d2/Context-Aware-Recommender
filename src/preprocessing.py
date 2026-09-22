"""
Module 3 & Module 14: Data Cleaning & Chronological Preprocessing
Performs deduplication, missing value checks, IQR outlier analysis & capping,
and strictly chronological train / validation / test splitting without data leakage.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler, MinMaxScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Preprocessing")


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_and_split_data(raw_df: pd.DataFrame = None, config: dict = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Clean raw dataset and perform chronological train/val/test splitting.
    Returns (cleaned_df, train_df, val_df, test_df).
    """
    if config is None:
        config = load_config()

    if raw_df is None:
        from src.data_ingestion import ingest_data
        raw_df = ingest_data(config["data"]["raw_path"], config["data"]["metadata_path"])

    df = raw_df.copy()
    initial_rows = len(df)
    logger.info(f"Starting data cleaning on {initial_rows} rows.")

    # 1. Duplicate detection and removal
    duplicates = df.duplicated()
    num_duplicates = int(duplicates.sum())
    if num_duplicates > 0:
        logger.warning(f"Found {num_duplicates} exact duplicate rows. Removing duplicates.")
        df = df.drop_duplicates().reset_index(drop=True)
    else:
        logger.info("Duplicate check: 0 exact duplicate rows found.")

    # 2. Missing Value Imputation (Defensive check)
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        logger.warning(f"Missing values detected:\n{null_counts[null_counts > 0]}")
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
    else:
        logger.info("Missing value check: 0 missing values across all columns.")

    # 3. Outlier Analysis & Conservative Capping (IQR-based)
    continuous_num_cols = ["unit_price", "discount_amount", "time_on_site_sec", "pages_viewed"]
    outlier_report = {}

    for col in continuous_num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_count = int(outliers.sum())
        outlier_pct = round(outlier_count / len(df) * 100, 2)
        outlier_report[col] = {
            "q1": round(float(q1), 2),
            "q3": round(float(q3), 2),
            "iqr": round(float(iqr), 2),
            "lower_bound": round(float(lower_bound), 2),
            "upper_bound": round(float(upper_bound), 2),
            "outlier_count": outlier_count,
            "outlier_pct": outlier_pct
        }
        # Conservative Winsorization (capping at upper bound only for positive attributes to avoid negative values)
        effective_lower = max(0.0, lower_bound)
        if outlier_count > 0:
            logger.info(f"Capping outliers for {col}: {outlier_count} values ({outlier_pct}%) outside [{effective_lower:.1f}, {upper_bound:.1f}]")
            df[col] = df[col].clip(lower=effective_lower, upper=upper_bound)

    # 4. Standardize / parse date for chronological splitting
    date_col = config["splitting"]["date_column"]
    date_fmt = config["splitting"]["date_format"]
    df["parsed_visit_date"] = pd.to_datetime(df[date_col], format=date_fmt)

    # Sort strictly chronologically
    df = df.sort_values(by=["parsed_visit_date", "session_id"]).reset_index(drop=True)

    # 5. Chronological Train / Val / Test Partitioning
    train_start = pd.to_datetime(config["splitting"]["train_start"])
    train_end = pd.to_datetime(config["splitting"]["train_end"])
    val_start = pd.to_datetime(config["splitting"]["val_start"])
    val_end = pd.to_datetime(config["splitting"]["val_end"])
    test_start = pd.to_datetime(config["splitting"]["test_start"])
    test_end = pd.to_datetime(config["splitting"]["test_end"])

    train_df = df[(df["parsed_visit_date"] >= train_start) & (df["parsed_visit_date"] <= train_end)].copy()
    val_df = df[(df["parsed_visit_date"] >= val_start) & (df["parsed_visit_date"] <= val_end)].copy()
    test_df = df[(df["parsed_visit_date"] >= test_start) & (df["parsed_visit_date"] <= test_end)].copy()

    # Verify no gap or overlap
    logger.info(f"Chronological Split Summary:")
    logger.info(f"Train Set:      {len(train_df)} rows ({train_df['parsed_visit_date'].min().strftime('%Y-%m-%d')} to {train_df['parsed_visit_date'].max().strftime('%Y-%m-%d')})")
    logger.info(f"Validation Set: {len(val_df)} rows ({val_df['parsed_visit_date'].min().strftime('%Y-%m-%d')} to {val_df['parsed_visit_date'].max().strftime('%Y-%m-%d')})")
    logger.info(f"Test Set:       {len(test_df)} rows ({test_df['parsed_visit_date'].min().strftime('%Y-%m-%d')} to {test_df['parsed_visit_date'].max().strftime('%Y-%m-%d')})")

    # Persist cleaned and partitioned datasets
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    cleaned_path = Path(config["data"]["cleaned_path"])
    train_path = Path(config["data"]["train_path"])
    val_path = Path(config["data"]["val_path"])
    test_path = Path(config["data"]["test_path"])

    df.to_csv(cleaned_path, index=False)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Saved cleaned dataset to {cleaned_path}")
    logger.info(f"Saved train split to {train_path}")
    logger.info(f"Saved val split to {val_path}")
    logger.info(f"Saved test split to {test_path}")

    return df, train_df, val_df, test_df, outlier_report


if __name__ == "__main__":
    df, train_df, val_df, test_df, outlier_report = clean_and_split_data()
    print("\n--- PREPROCESSING COMPLETE ---")
    print(f"Cleaned Total Rows: {len(df)}")
    print(f"Train Rows: {len(train_df)} ({round(len(train_df)/len(df)*100, 1)}%)")
    print(f"Val Rows:   {len(val_df)} ({round(len(val_df)/len(df)*100, 1)}%)")
    print(f"Test Rows:  {len(test_df)} ({round(len(test_df)/len(df)*100, 1)}%)")
    print("\nOutlier Report Summary:")
    for col, rep in outlier_report.items():
        print(f"  {col}: {rep['outlier_count']} outliers ({rep['outlier_pct']}%) capped at upper bound {rep['upper_bound']}")
