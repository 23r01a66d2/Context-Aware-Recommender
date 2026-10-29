"""
MLVis Data Preprocessing and Feature Extraction Pipeline
---------------------------------------------------------
Extracts dataset- and column-level meta-features from MLVis raw datasets
and pairs them with visualization records from vis_all.csv.

Produces:
1. Feature vectors for each column pair (X, Y) or single column candidate
2. Target visualization types (bar, line, scatter, box, histogram, heatmap)
3. Leakage-aware Train / Val / Test splits grouped by dataset_id (70% / 15% / 15%)
4. Preprocessing artifacts & metadata schema for consistent inference
"""

import json
import logging
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mlvis_preprocessing")

# Supported chart types in Visual Analytics & MLVis
CANONICAL_CHART_TYPES = ["bar", "line", "scatter", "box", "histogram", "heatmap"]
CHART_TYPE_TO_ID = {t: i for i, t in enumerate(CANONICAL_CHART_TYPES)}
ID_TO_CHART_TYPE = {i: t for i, t in enumerate(CANONICAL_CHART_TYPES)}

# Feature schema keys
NUMERIC_FEATURE_NAMES = [
    # Dataset level
    "ds_log_rows",
    "ds_num_cols",
    "ds_numeric_ratio",
    "ds_categorical_ratio",
    # Column X features
    "x_is_numeric",
    "x_is_categorical",
    "x_is_datetime",
    "x_unique_ratio",
    "x_missing_ratio",
    "x_entropy",
    "x_skewness",
    "x_mean_magnitude",
    # Column Y features
    "y_is_numeric",
    "y_is_categorical",
    "y_is_datetime",
    "y_unique_ratio",
    "y_missing_ratio",
    "y_entropy",
    "y_skewness",
    "y_mean_magnitude",
    # Pair relational features
    "xy_is_both_numeric",
    "xy_is_cat_num",
    "xy_is_num_cat",
    "xy_is_both_cat",
    "xy_correlation",
    "xy_cardinality_ratio",
]


def resolve_chart_type(type_val: Any, mode_val: Any) -> Optional[str]:
    """Resolves raw MLVis/Plotly trace type and mode to canonical chart type."""
    t = str(type_val).strip().lower() if pd.notna(type_val) else ""
    m = str(mode_val).strip().lower() if pd.notna(mode_val) else ""

    if not t or t == "nan":
        if "line" in m:
            return "line"
        return "scatter"
    if t in ["scatter", "scattergl"]:
        if "line" in m:
            return "line"
        return "scatter"
    if t in ["scattergeo", "scattermapbox"]:
        return "scatter"
    if t in CANONICAL_CHART_TYPES:
        return t
    return None


def parse_src_tag(src: Any) -> Tuple[Optional[str], Optional[str]]:
    """
    Parses MLVis source reference like 'author:dataset_num:col_hash'
    Returns (dataset_id, col_hash)
    """
    if not isinstance(src, str) or not src.strip():
        return None, None
    parts = src.strip().split(":")
    if len(parts) >= 3:
        return f"{parts[0]}:{parts[1]}", parts[2]
    if len(parts) == 2:
        return f"{parts[0]}:{parts[1]}", None
    return None, None


def compute_column_meta(series: pd.Series, max_sample: int = 5000) -> Dict[str, float]:
    """Computes statistical and structural meta-features for a single column series."""
    # Subsample if large
    if len(series) > max_sample:
        series = series.sample(n=max_sample, random_state=42)

    total_len = max(len(series), 1)
    missing_count = series.isna().sum()
    valid_series = series.dropna()
    valid_len = len(valid_series)

    missing_ratio = float(missing_count / total_len)

    # Detect datetime
    is_datetime = 0.0
    if valid_len > 0 and (series.dtype == object or pd.api.types.is_string_dtype(series) or pd.api.types.is_datetime64_any_dtype(series)):
        # Check first few items
        try:
            pd.to_datetime(valid_series.iloc[:10])
            is_datetime = 1.0
        except Exception:
            is_datetime = 0.0

    # Try numeric conversion
    numeric_series = pd.to_numeric(valid_series, errors="coerce")
    valid_num_len = numeric_series.notna().sum()
    is_numeric = 1.0 if (valid_num_len / max(valid_len, 1)) > 0.8 else 0.0
    is_categorical = 1.0 if (is_numeric == 0.0 and is_datetime == 0.0) else 0.0

    unique_count = valid_series.nunique()
    unique_ratio = float(unique_count / max(valid_len, 1))

    # Shannon Entropy approximation on categorical / binned
    try:
        val_counts = valid_series.value_counts(normalize=True).values
        entropy = -float(np.sum(val_counts * np.log2(val_counts + 1e-12)))
        # Normalize entropy
        entropy = min(entropy / 10.0, 1.0)
    except Exception:
        entropy = 0.0

    # Numeric distribution stats
    skewness = 0.0
    mean_mag = 0.0
    if is_numeric == 1.0 and valid_num_len > 2:
        clean_num = numeric_series.dropna()
        try:
            skew = clean_num.skew()
            skewness = float(np.clip(skew if pd.notna(skew) else 0.0, -10.0, 10.0) / 10.0)
        except Exception:
            skewness = 0.0
        mean_val = clean_num.mean()
        if pd.notna(mean_val) and mean_val != 0:
            mean_mag = float(np.log1p(abs(mean_val)) / 20.0)

    return {
        "is_numeric": is_numeric,
        "is_categorical": is_categorical,
        "is_datetime": is_datetime,
        "unique_ratio": unique_ratio,
        "missing_ratio": missing_ratio,
        "entropy": entropy,
        "skewness": skewness,
        "mean_magnitude": mean_mag,
    }


def extract_pair_features(
    ds_meta: Dict[str, float],
    x_meta: Dict[str, float],
    y_meta: Dict[str, float],
    correlation: float = 0.0,
    cardinality_ratio: float = 1.0,
) -> List[float]:
    """Combines dataset, X-column, Y-column, and relational stats into a flat feature vector."""
    x_is_num = x_meta.get("is_numeric", 0.0)
    y_is_num = y_meta.get("is_numeric", 0.0)
    x_is_cat = x_meta.get("is_categorical", 0.0)
    y_is_cat = y_meta.get("is_categorical", 0.0)

    features = [
        ds_meta.get("ds_log_rows", 0.0),
        ds_meta.get("ds_num_cols", 0.0),
        ds_meta.get("ds_numeric_ratio", 0.0),
        ds_meta.get("ds_categorical_ratio", 0.0),
        # X features
        x_meta.get("is_numeric", 0.0),
        x_meta.get("is_categorical", 0.0),
        x_meta.get("is_datetime", 0.0),
        x_meta.get("unique_ratio", 0.0),
        x_meta.get("missing_ratio", 0.0),
        x_meta.get("entropy", 0.0),
        x_meta.get("skewness", 0.0),
        x_meta.get("mean_magnitude", 0.0),
        # Y features
        y_meta.get("is_numeric", 0.0),
        y_meta.get("is_categorical", 0.0),
        y_meta.get("is_datetime", 0.0),
        y_meta.get("unique_ratio", 0.0),
        y_meta.get("missing_ratio", 0.0),
        y_meta.get("entropy", 0.0),
        y_meta.get("skewness", 0.0),
        y_meta.get("mean_magnitude", 0.0),
        # Relational
        1.0 if (x_is_num > 0.5 and y_is_num > 0.5) else 0.0,
        1.0 if (x_is_cat > 0.5 and y_is_num > 0.5) else 0.0,
        1.0 if (x_is_num > 0.5 and y_is_cat > 0.5) else 0.0,
        1.0 if (x_is_cat > 0.5 and y_is_cat > 0.5) else 0.0,
        float(np.clip(correlation, -1.0, 1.0)),
        float(np.clip(cardinality_ratio, 0.0, 10.0) / 10.0),
    ]
    return features


def load_dataset_columns(file_path: str, nrows: int = 5000) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Loads dataset CSV, identifies whether row 0 has hash identifiers,
    and returns (cleaned_dataframe, hash_to_col_name).
    """
    df_raw = pd.read_csv(file_path, nrows=nrows + 2, low_memory=False)
    if len(df_raw) == 0:
        return df_raw, {}

    # Check if row 0 has 6-character hex-like hashes
    row0_vals = df_raw.iloc[0].astype(str).tolist()
    is_hash_row = sum(len(v) == 6 for v in row0_vals) >= max(1, len(row0_vals) // 2)

    if is_hash_row:
        hash_to_col = dict(zip(row0_vals, df_raw.columns.tolist()))
        df_clean = df_raw.iloc[1:].reset_index(drop=True)
    else:
        # Fallback hash mapping to column names directly
        hash_to_col = {col: col for col in df_raw.columns}
        df_clean = df_raw

    return df_clean, hash_to_col


def run_mlvis_preprocessing(
    raw_dir: str = "data/mlvis/raw",
    processed_dir: str = "data/mlvis/processed",
    max_datasets: Optional[int] = None,
) -> Dict[str, Any]:
    """Runs the full MLVis pairing and feature extraction pipeline."""
    os.makedirs(processed_dir, exist_ok=True)
    vis_path = os.path.join(raw_dir, "vis_all.csv")
    index_path = os.path.join(raw_dir, "dataset_index.json")

    logger.info("Loading vis_all.csv and dataset_index.json...")
    vis_df = pd.read_csv(vis_path, low_memory=False)
    with open(index_path, "r", encoding="utf-8") as f:
        dataset_index = json.load(f)

    # 1. Map visualization records
    records_by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for idx, row in vis_df.iterrows():
        chart_type = resolve_chart_type(row.get("type"), row.get("mode"))
        if not chart_type or chart_type not in CHART_TYPE_TO_ID:
            continue

        # Look for source in primary coordinate keys
        xsrc = row.get("xsrc")
        ysrc = row.get("ysrc")
        lonsrc = row.get("lonsrc")
        latsrc = row.get("latsrc")
        valsrc = row.get("valuessrc")

        # Pick lead source to find dataset_id
        lead_src = ysrc or xsrc or lonsrc or valsrc
        d_id, _ = parse_src_tag(lead_src)
        if not d_id or d_id not in dataset_index:
            continue

        _, x_hash = parse_src_tag(xsrc or lonsrc)
        _, y_hash = parse_src_tag(ysrc or latsrc or valsrc)

        records_by_dataset.setdefault(d_id, []).append(
            {
                "chart_type": chart_type,
                "type_id": CHART_TYPE_TO_ID[chart_type],
                "x_hash": x_hash,
                "y_hash": y_hash,
            }
        )

    logger.info(
        f"Grouped visualization records across {len(records_by_dataset)} paired datasets."
    )

    all_dataset_ids = sorted(list(records_by_dataset.keys()))
    if max_datasets:
        all_dataset_ids = all_dataset_ids[:max_datasets]

    # Pre-extract meta-features per dataset & build dataset-grouped feature records
    feature_rows = []
    dataset_group_ids = []
    labels = []
    metadata_records = []

    processed_count = 0
    for d_id in all_dataset_ids:
        ds_info = dataset_index[d_id]
        csv_path = os.path.join(raw_dir, "datasets", ds_info["safe_filename"])
        if not os.path.exists(csv_path):
            continue

        try:
            df, hash_to_col = load_dataset_columns(csv_path, nrows=3000)
        except Exception as e:
            logger.debug(f"Failed to read {csv_path}: {e}")
            continue

        if len(df) == 0 or len(df.columns) == 0:
            continue

        # Dataset-level meta-features
        n_rows = len(df)
        n_cols = len(df.columns)
        ds_meta = {
            "ds_log_rows": float(np.log1p(n_rows) / 15.0),
            "ds_num_cols": float(min(n_cols, 100) / 100.0),
        }

        # Profile columns
        col_meta_cache: Dict[str, Dict[str, float]] = {}
        for col_name in df.columns:
            col_meta_cache[col_name] = compute_column_meta(df[col_name])

        num_numeric = sum(m["is_numeric"] for m in col_meta_cache.values())
        ds_meta["ds_numeric_ratio"] = float(num_numeric / max(n_cols, 1))
        ds_meta["ds_categorical_ratio"] = float(1.0 - ds_meta["ds_numeric_ratio"])

        # Create inverse lookup from hash to column name
        col_to_hash = {v: k for k, v in hash_to_col.items()}

        # Match positive visualization records
        vis_list = records_by_dataset[d_id]
        for vis in vis_list:
            c_type = vis["chart_type"]
            c_type_id = vis["type_id"]
            x_h = vis["x_hash"]
            y_h = vis["y_hash"]

            # Map hashes to actual column names
            x_col = hash_to_col.get(x_h) if x_h else None
            y_col = hash_to_col.get(y_h) if y_h else None

            # Fallback for single-column visualizations like histogram/box if X is None
            if not x_col and y_col:
                # If only Y is given, use Y as target
                x_col = y_col
            elif not y_col and x_col:
                y_col = x_col

            if not x_col or not y_col or x_col not in col_meta_cache or y_col not in col_meta_cache:
                continue

            x_m = col_meta_cache[x_col]
            y_m = col_meta_cache[y_col]

            # Compute correlation if both numeric
            corr = 0.0
            if x_m["is_numeric"] > 0.5 and y_m["is_numeric"] > 0.5:
                try:
                    s_x = pd.to_numeric(df[x_col], errors="coerce")
                    s_y = pd.to_numeric(df[y_col], errors="coerce")
                    corr_val = s_x.corr(s_y)
                    corr = float(corr_val) if pd.notna(corr_val) else 0.0
                except Exception:
                    corr = 0.0

            card_ratio = float((x_m["unique_ratio"] + 1e-4) / (y_m["unique_ratio"] + 1e-4))
            feats = extract_pair_features(ds_meta, x_m, y_m, corr, card_ratio)

            feature_rows.append(feats)
            dataset_group_ids.append(d_id)
            labels.append(c_type_id)
            metadata_records.append(
                {
                    "dataset_id": d_id,
                    "x_col": str(x_col),
                    "y_col": str(y_col),
                    "chart_type": c_type,
                    "type_id": c_type_id,
                }
            )

        processed_count += 1
        if processed_count % 150 == 0:
            logger.info(
                f"Processed {processed_count}/{len(all_dataset_ids)} datasets, total samples: {len(feature_rows)}"
            )

    X = np.array(feature_rows, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    groups = np.array(dataset_group_ids)

    logger.info(
        f"Feature extraction complete! Total instances: {len(X)} across {len(set(groups))} distinct datasets."
    )

    # 2. Leakage-aware Train / Val / Test splitting grouped by dataset_id
    # Split 1: Train+Val (85%) vs Test (15%)
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_val_idx, test_idx = next(gss_test.split(X, y, groups))

    X_train_val, y_train_val, groups_train_val = (
        X[train_val_idx],
        y[train_val_idx],
        groups[train_val_idx],
    )
    X_test, y_test, groups_test = X[test_idx], y[test_idx], groups[test_idx]

    # Split 2: Train (70% total) vs Val (15% total -> ~17.6% of train_val)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=42)
    train_sub_idx, val_sub_idx = next(gss_val.split(X_train_val, y_train_val, groups_train_val))

    X_train, y_train, groups_train = (
        X_train_val[train_sub_idx],
        y_train_val[train_sub_idx],
        groups_train_val[train_sub_idx],
    )
    X_val, y_val, groups_val = (
        X_train_val[val_sub_idx],
        y_train_val[val_sub_idx],
        groups_train_val[val_sub_idx],
    )

    logger.info(
        f"Splits summary (Grouped by dataset_id): "
        f"Train={len(X_train)} samples ({len(set(groups_train))} datasets), "
        f"Val={len(X_val)} samples ({len(set(groups_val))} datasets), "
        f"Test={len(X_test)} samples ({len(set(groups_test))} datasets)"
    )

    # 3. Save preprocessed arrays and schemas
    np.savez_compressed(
        os.path.join(processed_dir, "mlvis_splits.npz"),
        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,
        X_val=X_val,
        y_val=y_val,
        groups_val=groups_val,
        X_test=X_test,
        y_test=y_test,
        groups_test=groups_test,
    )

    schema_metadata = {
        "version": "1.0.0",
        "num_features": len(NUMERIC_FEATURE_NAMES),
        "feature_names": NUMERIC_FEATURE_NAMES,
        "chart_types": CANONICAL_CHART_TYPES,
        "chart_type_to_id": CHART_TYPE_TO_ID,
        "id_to_chart_type": ID_TO_CHART_TYPE,
        "total_instances": len(X),
        "total_datasets": len(set(groups)),
        "splits": {
            "train_samples": int(len(X_train)),
            "train_datasets": int(len(set(groups_train))),
            "val_samples": int(len(X_val)),
            "val_datasets": int(len(set(groups_val))),
            "test_samples": int(len(X_test)),
            "test_datasets": int(len(set(groups_test))),
        },
    }

    with open(os.path.join(processed_dir, "feature_schema.json"), "w", encoding="utf-8") as f:
        json.dump(schema_metadata, f, indent=2)

    logger.info("Saved mlvis_splits.npz and feature_schema.json successfully.")
    return schema_metadata


if __name__ == "__main__":
    run_mlvis_preprocessing()
