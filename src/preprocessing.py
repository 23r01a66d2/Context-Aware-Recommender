"""
Module 3 & Module 14: Data Cleaning & Chronological Preprocessing
Performs deduplication, missing value checks, IQR outlier analysis & capping,
and strictly chronological train / validation / test splitting without data leakage.
Supports both global configuration (config.yaml) and client-specific schema mappings (ClientSchemaConfig).
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union, List

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.schema_mapper import ClientSchemaConfig, SchemaRole, get_or_create_client_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Preprocessing")


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_and_split_data(
    raw_df: pd.DataFrame = None,
    config: dict = None,
    schema_config: Union[ClientSchemaConfig, dict, str] = None,
    output_dir: Union[str, Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Clean raw dataset and perform chronological train/val/test splitting.
    Supports either standard config.yaml (backward-compatible) or dynamic ClientSchemaConfig.
    Returns (cleaned_df, train_df, val_df, test_df, outlier_report).
    """
    if config is None:
        config = load_config()

    # Resolve schema_config if provided
    client_cfg: Optional[ClientSchemaConfig] = None
    if schema_config is not None:
        if isinstance(schema_config, ClientSchemaConfig):
            client_cfg = schema_config
        elif isinstance(schema_config, str):
            client_cfg = get_or_create_client_config(schema_config)
        elif isinstance(schema_config, dict):
            client_cfg = ClientSchemaConfig.from_dict(schema_config)

    if raw_df is None:
        from src.data_ingestion import ingest_data
        raw_df = ingest_data(config["data"]["raw_path"], config["data"]["metadata_path"])

    df = raw_df.copy()
    initial_rows = len(df)
    logger.info(f"Starting data cleaning on {initial_rows} rows (Client: {client_cfg.client_id if client_cfg else 'default'}).")

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
    if client_cfg and client_cfg.outlier_columns:
        continuous_num_cols = [c for c in client_cfg.outlier_columns if c in df.columns]
    elif client_cfg:
        # Auto-detect numeric feature columns with cardinality > 10
        feature_cols = client_cfg.get_behavior_cols() + client_cfg.get_content_cols()
        continuous_num_cols = [
            c for c in feature_cols
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 10
        ]
    else:
        continuous_num_cols = ["unit_price", "discount_amount", "time_on_site_sec", "pages_viewed"]

    outlier_report = {}
    for col in continuous_num_cols:
        if col not in df.columns:
            continue
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
        # Conservative Winsorization (capping at upper bound only for non-negative attributes)
        effective_lower = max(0.0, lower_bound) if (df[col] >= 0).all() else lower_bound
        if outlier_count > 0:
            logger.info(f"Capping outliers for {col}: {outlier_count} values ({outlier_pct}%) outside [{effective_lower:.1f}, {upper_bound:.1f}]")
            df[col] = df[col].clip(lower=effective_lower, upper=upper_bound)

    # 4. Standardize / parse date for chronological splitting
    if client_cfg:
        date_col = client_cfg.get_timestamp_col()
        date_fmt = client_cfg.timestamp_format
    else:
        date_col = config["splitting"]["date_column"]
        date_fmt = config["splitting"]["date_format"]

    try:
        if date_fmt:
            df["parsed_visit_date"] = pd.to_datetime(df[date_col], format=date_fmt)
        else:
            df["parsed_visit_date"] = pd.to_datetime(df[date_col])
    except Exception:
        df["parsed_visit_date"] = pd.to_datetime(df[date_col], format="mixed")

    # Sort strictly chronologically
    sort_keys = ["parsed_visit_date"]
    if "session_id" in df.columns:
        sort_keys.append("session_id")
    elif client_cfg and client_cfg.get_user_id_col() in df.columns:
        sort_keys.append(client_cfg.get_user_id_col())

    df = df.sort_values(by=sort_keys).reset_index(drop=True)

    # 5. Chronological Train / Val / Test Partitioning
    split_info = client_cfg.split_config if client_cfg else config["splitting"]
    has_date_intervals = (
        client_cfg is None or client_cfg.client_id == "demo_ecommerce"
    ) and "train_start" in split_info and "train_end" in split_info

    if has_date_intervals:
        train_start = pd.to_datetime(split_info["train_start"])
        train_end = pd.to_datetime(split_info["train_end"])
        val_start = pd.to_datetime(split_info["val_start"])
        val_end = pd.to_datetime(split_info["val_end"])
        test_start = pd.to_datetime(split_info["test_start"])
        test_end = pd.to_datetime(split_info["test_end"])

        train_df = df[(df["parsed_visit_date"] >= train_start) & (df["parsed_visit_date"] <= train_end)].copy()
        val_df = df[(df["parsed_visit_date"] >= val_start) & (df["parsed_visit_date"] <= val_end)].copy()
        test_df = df[(df["parsed_visit_date"] >= test_start) & (df["parsed_visit_date"] <= test_end)].copy()

        if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
            has_date_intervals = False

    if not has_date_intervals:
        # Ratio-based chronological split (preserving time order)
        ratios = split_info.get("split_ratios", [0.6, 0.2, 0.2])
        n = len(df)
        n_train = max(1, int(n * ratios[0]))
        n_val = max(1, int(n * ratios[1]))
        train_df = df.iloc[:n_train].copy()
        val_df = df.iloc[n_train:n_train + n_val].copy()
        test_df = df.iloc[n_train + n_val:].copy()
        if len(test_df) == 0 and len(val_df) > 1:
            test_df = val_df.iloc[-1:].copy()
            val_df = val_df.iloc[:-1].copy()

    # Verify no gap or overlap
    def _format_date_range(sub_df):
        if len(sub_df) == 0 or "parsed_visit_date" not in sub_df.columns:
            return "empty"
        d_min = sub_df["parsed_visit_date"].min()
        d_max = sub_df["parsed_visit_date"].max()
        s_min = d_min.strftime('%Y-%m-%d') if pd.notnull(d_min) else "N/A"
        s_max = d_max.strftime('%Y-%m-%d') if pd.notnull(d_max) else "N/A"
        return f"{s_min} to {s_max}"

    logger.info("Chronological Split Summary:")
    logger.info(f"Train Set:      {len(train_df)} rows ({_format_date_range(train_df)})")
    logger.info(f"Validation Set: {len(val_df)} rows ({_format_date_range(val_df)})")
    logger.info(f"Test Set:       {len(test_df)} rows ({_format_date_range(test_df)})")

    # Persist cleaned and partitioned datasets
    if output_dir is not None:
        save_dir = Path(output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        cleaned_path = save_dir / "cleaned_dataset.csv"
        train_path = save_dir / "train.csv"
        val_path = save_dir / "validation.csv"
        test_path = save_dir / "test.csv"
    elif client_cfg and client_cfg.client_id != "demo_ecommerce":
        save_dir = Path(f"clients/{client_cfg.client_id}/data/processed")
        save_dir.mkdir(parents=True, exist_ok=True)
        cleaned_path = save_dir / "cleaned_dataset.csv"
        train_path = save_dir / "train.csv"
        val_path = save_dir / "validation.csv"
        test_path = save_dir / "test.csv"
    else:
        cleaned_path = Path(config["data"]["cleaned_path"])
        train_path = Path(config["data"]["train_path"])
        val_path = Path(config["data"]["val_path"])
        test_path = Path(config["data"]["test_path"])
        cleaned_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(cleaned_path, index=False)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    # Also write val.csv as alias if validation.csv is used
    if val_path.name == "validation.csv":
        val_alias = val_path.parent / "val.csv"
        val_df.to_csv(val_alias, index=False)

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
