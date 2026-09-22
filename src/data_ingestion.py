"""
Module 1: Data Ingestion
Loads raw dataset, validates format and encodings, extracts metadata,
and saves dataset summary to outputs/dataset_metadata.json.
"""

import os
import sys
import json
import logging
from pathlib import Path
import pandas as pd
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DataIngestion")


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ingest_data(raw_path: str = None, metadata_path: str = None) -> pd.DataFrame:
    """
    Ingest raw dataset with fallback encoding detection,
    compute detailed schema metadata, and persist metadata.
    """
    if raw_path is None or metadata_path is None:
        cfg = load_config()
        raw_path = raw_path or cfg["data"]["raw_path"]
        metadata_path = metadata_path or cfg["data"]["metadata_path"]

    raw_file = Path(raw_path)
    if not raw_file.exists():
        err_msg = f"Raw dataset file not found at {raw_file.resolve()}"
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    # Detect file format
    file_ext = raw_file.suffix.lower()
    logger.info(f"Loading raw file: {raw_file} (Format: {file_ext})")

    # Handle encoding
    encodings_to_try = ["utf-8", "latin-1", "cp1252"]
    df = None
    last_error = None

    for enc in encodings_to_try:
        try:
            if file_ext == ".csv":
                df = pd.read_csv(raw_file, encoding=enc)
            elif file_ext in [".xlsx", ".xls"]:
                df = pd.read_excel(raw_file)
            else:
                raise ValueError(f"Unsupported file extension: {file_ext}")
            logger.info(f"Successfully loaded dataset using {enc} encoding.")
            break
        except Exception as e:
            last_error = e
            logger.warning(f"Failed to load with encoding {enc}: {e}")

    if df is None:
        logger.error(f"Fatal error loading {raw_file}: {last_error}")
        raise last_error

    # Compute schema & statistics
    n_rows, n_cols = df.shape
    memory_usage_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)

    column_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
    missing_values = df.isnull().sum().to_dict()
    unique_counts = {col: int(df[col].nunique()) for col in df.columns}

    metadata = {
        "file_name": raw_file.name,
        "file_format": file_ext,
        "rows": n_rows,
        "columns": n_cols,
        "memory_usage_mb": memory_usage_mb,
        "column_names": list(df.columns),
        "data_types": column_types,
        "missing_values": missing_values,
        "unique_counts": unique_counts,
        "first_5_records": df.head(5).to_dict(orient="records")
    }

    # Ensure output directory exists
    meta_path = Path(metadata_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Dataset ingested successfully. Shape: ({n_rows}, {n_cols}), Memory: {memory_usage_mb} MB")
    logger.info(f"Metadata saved to {meta_path.resolve()}")

    return df


if __name__ == "__main__":
    df = ingest_data()
    print("\n--- DATASET INGESTION SUMMARY ---")
    print(f"Shape: {df.shape}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    print(f"Memory: {round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)} MB")
    print("\nFirst 3 records:")
    print(df.head(3))
