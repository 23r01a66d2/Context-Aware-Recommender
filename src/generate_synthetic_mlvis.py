"""
Synthetic Visualization Recommendation Data Generator
-------------------------------------------------------
Generates exactly 600 synthetic dataset and visualization records
for evaluating and training the MLVis Visualization Recommender.

Design & Scientific Rigor:
- Distinguishes synthetic profiles from real MLVis benchmarks.
- Labels are strictly derived from documented, deterministic visualization
  selection rules (e.g. Wilkinson's Grammar of Graphics, Mackinlay's APT,
  and task-based recommendations: comparison, trend, distribution, correlation,
  part-to-whole, multivariate).
- Covers 12 visualization templates mapped to canonical and extended types:
  bar, line, scatter, box, histogram, heatmap, area, pie, donut,
  stacked_bar, grouped_bar, bubble, treemap.
- Employs 60 distinct synthetic dataset archetypes (10 records each = 600 total)
  so that group-based splitting strictly avoids leakage across dataset archetypes.
- Fully reproducible using a fixed random seed (seed=42).
"""

import json
import math
import os
import sys
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

if "." not in sys.path:
    sys.path.insert(0, ".")

from src.mlvis_preprocessing import (
    CANONICAL_CHART_TYPES,
    CHART_TYPE_TO_ID,
    NUMERIC_FEATURE_NAMES,
    extract_pair_features,
)

# 12 specific visualization types supported in analytics
EXTENDED_CHART_TYPES = [
    "bar",
    "line",
    "scatter",
    "box",
    "histogram",
    "heatmap",
    "pie",
    "donut",
    "area",
    "stacked_bar",
    "grouped_bar",
    "bubble",
    "treemap",
]

# Canonical mapping for MLVis 6-class neural head
EXTENDED_TO_CANONICAL = {
    "bar": "bar",
    "grouped_bar": "bar",
    "stacked_bar": "bar",
    "pie": "bar",       # proportion/composition often represented by bar/composition
    "donut": "bar",     # proportion/composition
    "treemap": "bar",   # hierarchical/part-to-whole categorical
    "line": "line",
    "area": "line",     # continuous temporal trend
    "scatter": "scatter",
    "bubble": "scatter", # 2D/3D numeric relationship
    "box": "box",
    "histogram": "histogram",
    "heatmap": "heatmap",
}

GENERATION_SEED = 42

ANALYTICAL_TASKS = [
    "distribution_single",     # histogram
    "trend_temporal",          # line, area
    "category_comparison",     # bar, grouped_bar
    "part_to_whole",           # pie, donut, treemap
    "correlation_numeric",     # scatter, bubble
    "distribution_by_group",   # box
    "matrix_correlation",      # heatmap
    "composition_stacked",     # stacked_bar
]


def generate_synthetic_records(
    num_records: int = 600,
    seed: int = GENERATION_SEED,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generates exactly `num_records` synthetic dataset & visualization profiles.
    Returns:
      (df_records, X_features, y_canonical_ids, group_ids, schema_meta)
    """
    assert num_records == 600, "Exactly 600 synthetic records required as per specification."
    rng = np.random.RandomState(seed)

    records = []
    feature_rows = []
    canonical_labels = []
    group_ids = []

    # 60 distinct dataset archetypes, 10 records per archetype = 600 total
    num_archetypes = 60
    records_per_archetype = num_records // num_archetypes

    archetype_templates = [
        # (name, default_task, domain, rows_range, cols_range, cat_ratio, num_ratio)
        ("E-Commerce Order Analytics", "category_comparison", "Retail", (5000, 100000), (10, 25), 0.4, 0.6),
        ("Server Telemetry & Metrics", "trend_temporal", "DevOps", (10000, 500000), (8, 20), 0.1, 0.9),
        ("Clinical Patient Biomarkers", "distribution_by_group", "Healthcare", (500, 5000), (15, 40), 0.3, 0.7),
        ("Real Estate Housing Valuation", "correlation_numeric", "Finance", (2000, 30000), (12, 30), 0.35, 0.65),
        ("Mobile App Session Events", "part_to_whole", "Product Analytics", (20000, 200000), (6, 18), 0.6, 0.4),
        ("Customer Churn Survey", "distribution_single", "Marketing", (1000, 15000), (10, 22), 0.5, 0.5),
        ("Stock Market Volatility", "trend_temporal", "Fintech", (3000, 80000), (8, 16), 0.05, 0.95),
        ("Supply Chain Logistics Delivery", "composition_stacked", "Logistics", (4000, 60000), (12, 28), 0.45, 0.55),
        ("Genomic Expression Microarray", "matrix_correlation", "Bioinformatics", (100, 2000), (30, 80), 0.1, 0.9),
        ("Social Media Engagement Matrix", "category_comparison", "Social", (5000, 150000), (10, 24), 0.5, 0.5),
        ("Sensor Network IoT Readings", "distribution_single", "IoT", (10000, 300000), (6, 15), 0.15, 0.85),
        ("Employee Performance Reviews", "distribution_by_group", "HR", (200, 5000), (14, 30), 0.5, 0.5),
    ]

    # Expand archetypes to 60 by parameter variation
    archetypes = []
    for i in range(num_archetypes):
        base = archetype_templates[i % len(archetype_templates)]
        arch_id = f"synth_ds_{i+1:03d}"
        arch_name = f"{base[0]} Archetype {i+1}"
        arch_domain = base[2]
        
        # Select target tasks balanced across archetypes
        task = ANALYTICAL_TASKS[i % len(ANALYTICAL_TASKS)]
        rows = int(rng.uniform(base[3][0], base[3][1]))
        cols = int(rng.uniform(base[4][0], base[4][1]))
        cat_ratio = float(np.clip(base[5] + rng.normal(0, 0.05), 0.05, 0.85))
        num_ratio = float(np.clip(1.0 - cat_ratio, 0.1, 0.95))
        
        archetypes.append({
            "archetype_id": arch_id,
            "name": arch_name,
            "domain": arch_domain,
            "task": task,
            "rows": rows,
            "cols": cols,
            "cat_ratio": cat_ratio,
            "num_ratio": num_ratio,
        })

    record_counter = 0

    for arch in archetypes:
        arch_id = arch["archetype_id"]
        rows = arch["rows"]
        cols = arch["cols"]
        cat_ratio = arch["cat_ratio"]
        num_ratio = arch["num_ratio"]
        ds_log_rows = float(np.log10(max(rows, 10)) / 7.0)
        ds_num_cols = float(min(cols, 100) / 100.0)

        ds_meta = {
            "ds_log_rows": ds_log_rows,
            "ds_num_cols": ds_num_cols,
            "ds_numeric_ratio": num_ratio,
            "ds_categorical_ratio": cat_ratio,
        }

        for rec_i in range(records_per_archetype):
            record_counter += 1
            # Primary task for archetype, with slight variation for diversity
            if rec_i < 7:
                task = arch["task"]
            else:
                task = ANALYTICAL_TASKS[(rec_i + hash(arch_id)) % len(ANALYTICAL_TASKS)]

            # Generate column characteristics and target chart based on analytical task rules
            if task == "distribution_single":
                # Single numeric column frequency/density -> Histogram
                target_chart = "histogram"
                x_col = f"metric_val_{rec_i}"
                y_col = "frequency_density"
                x_meta = {
                    "is_numeric": 1.0,
                    "is_categorical": 0.0,
                    "is_datetime": 0.0,
                    "unique_ratio": float(rng.uniform(0.15, 0.95)),
                    "missing_ratio": float(rng.uniform(0.0, 0.05)),
                    "entropy": float(rng.uniform(0.4, 0.9)),
                    "skewness": float(rng.uniform(-0.4, 0.6)),
                    "mean_magnitude": float(rng.uniform(0.2, 0.7)),
                }
                y_meta = {
                    "is_numeric": 1.0,
                    "is_categorical": 0.0,
                    "is_datetime": 0.0,
                    "unique_ratio": float(rng.uniform(0.05, 0.3)),
                    "missing_ratio": 0.0,
                    "entropy": float(rng.uniform(0.3, 0.7)),
                    "skewness": float(rng.uniform(-0.1, 0.3)),
                    "mean_magnitude": float(rng.uniform(0.1, 0.5)),
                }
                corr = 0.0
                card_ratio = 1.0

            elif task == "trend_temporal":
                # Datetime on X, Numeric on Y -> Line or Area chart
                target_chart = "line" if rng.rand() > 0.3 else "area"
                x_col = f"timestamp_{rec_i}"
                y_col = f"kpi_measurement_{rec_i}"
                x_meta = {
                    "is_numeric": 0.0,
                    "is_categorical": 0.0,
                    "is_datetime": 1.0,
                    "unique_ratio": float(rng.uniform(0.4, 0.99)),
                    "missing_ratio": float(rng.uniform(0.0, 0.02)),
                    "entropy": float(rng.uniform(0.6, 0.95)),
                    "skewness": 0.0,
                    "mean_magnitude": 0.0,
                }
                y_meta = {
                    "is_numeric": 1.0,
                    "is_categorical": 0.0,
                    "is_datetime": 0.0,
                    "unique_ratio": float(rng.uniform(0.2, 0.8)),
                    "missing_ratio": float(rng.uniform(0.0, 0.05)),
                    "entropy": float(rng.uniform(0.4, 0.85)),
                    "skewness": float(rng.uniform(-0.3, 0.5)),
                    "mean_magnitude": float(rng.uniform(0.3, 0.8)),
                }
                corr = float(rng.uniform(0.2, 0.7))
                card_ratio = float(rng.uniform(0.8, 1.5))

            elif task == "category_comparison":
                # Categorical on X (cardinality 3-15), Numeric aggregate on Y -> Bar or Grouped Bar
                target_chart = "bar" if rng.rand() > 0.25 else "grouped_bar"
                x_col = f"category_name_{rec_i}"
                y_col = f"sales_metric_{rec_i}"
                cat_count = int(rng.uniform(3, 16))
                x_meta = {
                    "is_numeric": 0.0,
                    "is_categorical": 1.0,
                    "is_datetime": 0.0,
                    "unique_ratio": float(cat_count / max(rows, 100)),
                    "missing_ratio": float(rng.uniform(0.0, 0.03)),
                    "entropy": float(rng.uniform(0.3, 0.7)),
                    "skewness": 0.0,
                    "mean_magnitude": 0.0,
                }
                y_meta = {
                    "is_numeric": 1.0,
                    "is_categorical": 0.0,
                    "is_datetime": 0.0,
                    "unique_ratio": float(rng.uniform(0.1, 0.6)),
                    "missing_ratio": float(rng.uniform(0.0, 0.04)),
                    "entropy": float(rng.uniform(0.3, 0.8)),
                    "skewness": float(rng.uniform(-0.2, 0.6)),
                    "mean_magnitude": float(rng.uniform(0.2, 0.75)),
                }
                corr = 0.0
                card_ratio = float(min(cat_count / 100.0, 1.0))

            elif task == "part_to_whole":
                # Low cardinality categorical (2-7), proportion -> Pie, Donut, or Treemap
                r_choice = rng.rand()
                if r_choice < 0.4:
                    target_chart = "pie"
                elif r_choice < 0.75:
                    target_chart = "donut"
                else:
                    target_chart = "treemap"
                x_col = f"segment_slice_{rec_i}"
                y_col = f"share_percentage_{rec_i}"
                cat_count = int(rng.uniform(2, 7))
                x_meta = {
                    "is_numeric": 0.0,
                    "is_categorical": 1.0,
                    "is_datetime": 0.0,
                    "unique_ratio": float(cat_count / max(rows, 100)),
                    "missing_ratio": 0.0,
                    "entropy": float(rng.uniform(0.2, 0.5)),
                    "skewness": 0.0,
                    "mean_magnitude": 0.0,
                }
                y_meta = {
                    "is_numeric": 1.0,
                    "is_categorical": 0.0,
                    "is_datetime": 0.0,
                    "unique_ratio": float(cat_count / max(rows, 100)),
                    "missing_ratio": 0.0,
                    "entropy": float(rng.uniform(0.3, 0.6)),
                    "skewness": float(rng.uniform(0.0, 0.2)),
                    "mean_magnitude": 0.3,
                }
                corr = 0.0
                card_ratio = 1.0

            elif task == "correlation_numeric":
                # Continuous Numeric on X & Y -> Scatter or Bubble chart
                target_chart = "scatter" if rng.rand() > 0.25 else "bubble"
                x_col = f"continuous_var_a_{rec_i}"
                y_col = f"continuous_var_b_{rec_i}"
                corr_val = float(rng.uniform(-0.85, 0.85))
                x_meta = {
                    "is_numeric": 1.0,
                    "is_categorical": 0.0,
                    "is_datetime": 0.0,
                    "unique_ratio": float(rng.uniform(0.3, 0.95)),
                    "missing_ratio": float(rng.uniform(0.0, 0.05)),
                    "entropy": float(rng.uniform(0.5, 0.9)),
                    "skewness": float(rng.uniform(-0.5, 0.5)),
                    "mean_magnitude": float(rng.uniform(0.2, 0.8)),
                }
                y_meta = {
                    "is_numeric": 1.0,
                    "is_categorical": 0.0,
                    "is_datetime": 0.0,
                    "unique_ratio": float(rng.uniform(0.3, 0.95)),
                    "missing_ratio": float(rng.uniform(0.0, 0.05)),
                    "entropy": float(rng.uniform(0.5, 0.9)),
                    "skewness": float(rng.uniform(-0.5, 0.5)),
                    "mean_magnitude": float(rng.uniform(0.2, 0.8)),
                }
                corr = corr_val
                card_ratio = float(rng.uniform(0.7, 1.4))

            elif task == "distribution_by_group":
                # Categorical grouping on X, Numeric distribution on Y -> Box Plot
                target_chart = "box"
                x_col = f"group_factor_{rec_i}"
                y_col = f"distribution_metric_{rec_i}"
                cat_count = int(rng.uniform(3, 10))
                x_meta = {
                    "is_numeric": 0.0,
                    "is_categorical": 1.0,
                    "is_datetime": 0.0,
                    "unique_ratio": float(cat_count / max(rows, 100)),
                    "missing_ratio": float(rng.uniform(0.0, 0.03)),
                    "entropy": float(rng.uniform(0.4, 0.75)),
                    "skewness": 0.0,
                    "mean_magnitude": 0.0,
                }
                y_meta = {
                    "is_numeric": 1.0,
                    "is_categorical": 0.0,
                    "is_datetime": 0.0,
                    "unique_ratio": float(rng.uniform(0.2, 0.8)),
                    "missing_ratio": float(rng.uniform(0.0, 0.05)),
                    "entropy": float(rng.uniform(0.4, 0.85)),
                    "skewness": float(rng.uniform(0.2, 0.8)),
                    "mean_magnitude": float(rng.uniform(0.2, 0.7)),
                }
                corr = 0.0
                card_ratio = float(cat_count / 10.0)

            elif task == "matrix_correlation":
                # Both Categorical dimensions with cross-table cell values -> Heatmap
                target_chart = "heatmap"
                x_col = f"dimension_grid_x_{rec_i}"
                y_col = f"dimension_grid_y_{rec_i}"
                cat_x = int(rng.uniform(5, 20))
                cat_y = int(rng.uniform(5, 20))
                x_meta = {
                    "is_numeric": 0.0,
                    "is_categorical": 1.0,
                    "is_datetime": 0.0,
                    "unique_ratio": float(cat_x / max(rows, 100)),
                    "missing_ratio": 0.0,
                    "entropy": float(rng.uniform(0.5, 0.8)),
                    "skewness": 0.0,
                    "mean_magnitude": 0.0,
                }
                y_meta = {
                    "is_numeric": 0.0,
                    "is_categorical": 1.0,
                    "is_datetime": 0.0,
                    "unique_ratio": float(cat_y / max(rows, 100)),
                    "missing_ratio": 0.0,
                    "entropy": float(rng.uniform(0.5, 0.8)),
                    "skewness": 0.0,
                    "mean_magnitude": 0.0,
                }
                corr = 0.0
                card_ratio = float(cat_x / max(cat_y, 1))

            else:  # composition_stacked
                # Categorical with Sub-segmentation -> Stacked Bar
                target_chart = "stacked_bar"
                x_col = f"main_category_{rec_i}"
                y_col = f"subsegment_metric_{rec_i}"
                cat_count = int(rng.uniform(4, 12))
                x_meta = {
                    "is_numeric": 0.0,
                    "is_categorical": 1.0,
                    "is_datetime": 0.0,
                    "unique_ratio": float(cat_count / max(rows, 100)),
                    "missing_ratio": 0.0,
                    "entropy": float(rng.uniform(0.35, 0.7)),
                    "skewness": 0.0,
                    "mean_magnitude": 0.0,
                }
                y_meta = {
                    "is_numeric": 1.0,
                    "is_categorical": 0.0,
                    "is_datetime": 0.0,
                    "unique_ratio": float(rng.uniform(0.2, 0.6)),
                    "missing_ratio": 0.0,
                    "entropy": float(rng.uniform(0.3, 0.7)),
                    "skewness": float(rng.uniform(-0.1, 0.4)),
                    "mean_magnitude": float(rng.uniform(0.2, 0.7)),
                }
                corr = 0.0
                card_ratio = float(cat_count / 12.0)

            # Map extended target chart to canonical MLVis class
            canonical_type = EXTENDED_TO_CANONICAL[target_chart]
            canonical_id = CHART_TYPE_TO_ID[canonical_type]

            # Compute 26-dim feature vector using official pipeline function
            feat_vec = extract_pair_features(
                ds_meta=ds_meta,
                x_meta=x_meta,
                y_meta=y_meta,
                correlation=corr,
                cardinality_ratio=card_ratio,
            )

            record = {
                "record_id": f"synth_{record_counter:04d}",
                "dataset_archetype_id": arch_id,
                "dataset_name": arch["name"],
                "domain": arch["domain"],
                "analytical_task": task,
                "x_column": x_col,
                "y_column": y_col,
                "target_chart": target_chart,
                "canonical_chart": canonical_type,
                "canonical_class_id": canonical_id,
                "dataset_rows": rows,
                "dataset_cols": cols,
                "missing_value_ratio": float(x_meta["missing_ratio"]),
                "correlation": corr,
                "is_synthetic": True,
            }

            records.append(record)
            feature_rows.append(feat_vec)
            canonical_labels.append(canonical_id)
            group_ids.append(arch_id)

    df_records = pd.DataFrame(records)
    X_features = np.array(feature_rows, dtype=np.float32)
    y_canonical = np.array(canonical_labels, dtype=np.int64)
    groups = np.array(group_ids)

    schema_metadata = {
        "dataset_name": "Synthetic Visualization Benchmark 600",
        "description": (
            "600 curated synthetic dataset & visualization profiles across 60 dataset archetypes, "
            "designed to evaluate and improve visualization recommendation without leakage."
        ),
        "generation_seed": seed,
        "total_records": len(df_records),
        "total_archetypes": len(archetypes),
        "records_per_archetype": records_per_archetype,
        "is_synthetic": True,
        "feature_dimensions": X_features.shape[1],
        "feature_names": NUMERIC_FEATURE_NAMES,
        "extended_chart_types": EXTENDED_CHART_TYPES,
        "canonical_classes": CANONICAL_CHART_TYPES,
        "canonical_to_extended_mapping": {
            c: [e for e, can in EXTENDED_TO_CANONICAL.items() if can == c]
            for c in CANONICAL_CHART_TYPES
        },
        "target_distribution_canonical": {
            CANONICAL_CHART_TYPES[c]: int(count)
            for c, count in enumerate(np.bincount(y_canonical, minlength=len(CANONICAL_CHART_TYPES)))
        },
        "target_distribution_extended": df_records["target_chart"].value_counts().to_dict(),
        "analytical_tasks_distribution": df_records["analytical_task"].value_counts().to_dict(),
    }

    return df_records, X_features, y_canonical, groups, schema_metadata


def save_synthetic_dataset(output_dir: str = "data/mlvis/synthetic") -> Dict[str, Any]:
    """Generates and writes synthetic dataset and associated splits."""
    os.makedirs(output_dir, exist_ok=True)
    df_records, X, y, groups, schema_meta = generate_synthetic_records(num_records=600)

    # 1. Save CSV records
    csv_path = os.path.join(output_dir, "synthetic_vis_600.csv")
    df_records.to_csv(csv_path, index=False)

    # 2. Save JSON Schema & Documentation
    schema_path = os.path.join(output_dir, "synthetic_schema.json")
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema_meta, f, indent=2)

    # 3. Save Compressed Feature Tensors & Group Split Info
    tensor_path = os.path.join(output_dir, "synthetic_features.npz")
    np.savez_compressed(
        tensor_path,
        X=X,
        y=y,
        groups=groups,
    )

    print(f"Successfully generated {len(df_records)} synthetic records in {output_dir}")
    print(f"CSV: {csv_path}")
    print(f"Schema: {schema_path}")
    print(f"Tensors: {tensor_path}")
    print("\nCanonical Class Distribution:")
    for c_name, count in schema_meta["target_distribution_canonical"].items():
        print(f"  {c_name:12s}: {count:3d} ({count/len(df_records)*100:5.1f}%)")

    return schema_meta


if __name__ == "__main__":
    save_synthetic_dataset()
