"""
Context-Aware Recommendation System for Visual Analytics Engine.
Provides dataset profiling, interactive chart generation, visualization candidate generation,
cold-start suitability ranking, and neural multimodal personalization integration.
"""

import os
import re
import json
import math
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.schema_mapper import infer_column_type
from src.mlvis_preprocessing import (
    compute_column_meta,
    extract_pair_features,
    CANONICAL_CHART_TYPES
)
from src.models.mlvis_recommender import MLVisInferenceEngine

logger = logging.getLogger("VisualAnalytics")

# Global MLVis Inference Engines cache
_mlvis_engines = {}

def get_mlvis_engine(version: str = "v1") -> MLVisInferenceEngine:
    global _mlvis_engines
    ver = "v2" if "v2" in str(version).lower() or str(version).startswith("2") else "v1"
    if ver not in _mlvis_engines:
        m_dir = "models/mlvis_v2" if ver == "v2" else "models/mlvis"
        _mlvis_engines[ver] = MLVisInferenceEngine(model_dir=m_dir)
    return _mlvis_engines[ver]


# =====================================================================
# 1. DATASET PROFILING
# =====================================================================
def profile_dataset(df: pd.DataFrame, max_rows_sample: int = 50000) -> Dict[str, Any]:
    """
    Profiles any dataset: dimensions, column types, missingness, and summary statistics.
    Handles large datasets safely by computing column statistics efficiently.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    memory_mb = round(float(df.memory_usage(deep=True).sum()) / (1024 * 1024), 2)

    sample_df = df.iloc[:max_rows_sample] if total_rows > max_rows_sample else df

    columns_meta: List[Dict[str, Any]] = []
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    missing_summary: List[Dict[str, Any]] = []

    for col in df.columns:
        s = df[col]
        dtype_role = infer_column_type(s)
        null_count = int(s.isnull().sum())
        null_pct = round(float((null_count / max(total_rows, 1)) * 100), 2)
        unique_count = int(s.nunique(dropna=True))
        sample_vals = [str(x) for x in s.dropna().head(4).tolist()]

        if null_count > 0:
            missing_summary.append({
                "column": str(col),
                "null_count": null_count,
                "null_pct": null_pct
            })

        col_dict: Dict[str, Any] = {
            "name": str(col),
            "dtype": dtype_role,
            "raw_dtype": str(s.dtype),
            "null_count": null_count,
            "null_pct": null_pct,
            "unique_count": unique_count,
            "sample_values": sample_vals,
            "stats": {}
        }

        if dtype_role == "numeric":
            numeric_cols.append(str(col))
            s_num = pd.to_numeric(s, errors="coerce").dropna()
            if len(s_num) > 0:
                mean_val = float(s_num.mean())
                std_val = float(s_num.std()) if len(s_num) > 1 else 0.0
                min_val = float(s_num.min())
                max_val = float(s_num.max())
                median_val = float(s_num.median())
                skew_val = float(s_num.skew()) if len(s_num) > 2 else 0.0
                col_dict["stats"] = {
                    "mean": round(mean_val, 3) if not math.isnan(mean_val) else 0.0,
                    "std": round(std_val, 3) if not math.isnan(std_val) else 0.0,
                    "min": round(min_val, 3) if not math.isnan(min_val) else 0.0,
                    "max": round(max_val, 3) if not math.isnan(max_val) else 0.0,
                    "median": round(median_val, 3) if not math.isnan(median_val) else 0.0,
                    "skewness": round(skew_val, 3) if not math.isnan(skew_val) else 0.0,
                }
        else:
            categorical_cols.append(str(col))
            top_counts = s.dropna().value_counts().head(5)
            top_cats = [
                {"category": str(k), "count": int(v), "pct": round(float(v / max(total_rows, 1)) * 100, 1)}
                for k, v in top_counts.items()
            ]
            col_dict["stats"] = {
                "top_categories": top_cats
            }

        columns_meta.append(col_dict)

    # Correlation Matrix across numeric columns (cap at 12 to prevent huge payloads)
    corr_numeric_cols = numeric_cols[:12]
    corr_data: Dict[str, Any] = {"columns": corr_numeric_cols, "matrix": [], "pairs": []}
    if len(corr_numeric_cols) >= 2:
        num_sub = sample_df[corr_numeric_cols].apply(pd.to_numeric, errors="coerce")
        corr_matrix = num_sub.corr().fillna(0.0)
        matrix_vals = []
        for i, c1 in enumerate(corr_numeric_cols):
            row_vals = []
            for j, c2 in enumerate(corr_numeric_cols):
                r_val = round(float(corr_matrix.loc[c1, c2]), 3)
                row_vals.append(r_val)
                if i < j:
                    corr_data["pairs"].append({
                        "col1": c1,
                        "col2": c2,
                        "correlation": r_val,
                        "abs_correlation": round(abs(r_val), 3)
                    })
            matrix_vals.append(row_vals)
        corr_data["matrix"] = matrix_vals
        corr_data["pairs"].sort(key=lambda x: x["abs_correlation"], reverse=True)

    # Preview sample rows (first 10)
    sample_records = sample_df.head(10).fillna("").to_dict(orient="records")

    return {
        "dimensions": {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "memory_mb": memory_mb,
            "numeric_columns_count": len(numeric_cols),
            "categorical_columns_count": len(categorical_cols),
            "total_missing_values": int(df.isnull().sum().sum()),
        },
        "columns": columns_meta,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "missing_summary": missing_summary,
        "correlations": corr_data,
        "sample_records": sample_records
    }


# =====================================================================
# 2. INTERACTIVE CHART DATA GENERATION
# =====================================================================
def apply_dataframe_filters(df: pd.DataFrame, filters: Optional[Dict[str, Any]]) -> pd.DataFrame:
    """Safely applies filters to dataframe."""
    if not filters:
        return df

    filtered_df = df
    for col, filter_rule in filters.items():
        if col not in filtered_df.columns:
            continue

        if isinstance(filter_rule, dict):
            op = filter_rule.get("op", "eq")
            val = filter_rule.get("val")
            if val is None:
                continue

            if op == "eq":
                filtered_df = filtered_df[filtered_df[col].astype(str) == str(val)]
            elif op == "neq":
                filtered_df = filtered_df[filtered_df[col].astype(str) != str(val)]
            elif op == "in" and isinstance(val, (list, tuple)):
                str_vals = [str(x) for x in val]
                filtered_df = filtered_df[filtered_df[col].astype(str).isin(str_vals)]
            elif op in ("min", "gte"):
                try:
                    num_val = float(val)
                    filtered_df = filtered_df[pd.to_numeric(filtered_df[col], errors="coerce") >= num_val]
                except (ValueError, TypeError):
                    pass
            elif op in ("max", "lte"):
                try:
                    num_val = float(val)
                    filtered_df = filtered_df[pd.to_numeric(filtered_df[col], errors="coerce") <= num_val]
                except (ValueError, TypeError):
                    pass
            elif op == "between" and isinstance(val, (list, tuple)) and len(val) == 2:
                try:
                    min_v, max_v = float(val[0]), float(val[1])
                    s_num = pd.to_numeric(filtered_df[col], errors="coerce")
                    filtered_df = filtered_df[(s_num >= min_v) & (s_num <= max_v)]
                except (ValueError, TypeError):
                    pass
        else:
            # Direct equality
            filtered_df = filtered_df[filtered_df[col].astype(str) == str(filter_rule)]

    return filtered_df


def generate_chart_data(
    df: pd.DataFrame,
    chart_type: str,
    x_col: str,
    y_col: Optional[str] = None,
    aggregation: str = "count",
    filters: Optional[Dict[str, Any]] = None,
    max_points: int = 1500
) -> Dict[str, Any]:
    """
    Renders structured chart data from real dataset values.
    Supports bar charts, line charts, histograms, scatter plots, and correlation heatmaps.
    """
    if x_col not in df.columns:
        raise ValueError(f"Column '{x_col}' does not exist in dataset.")
    if y_col and y_col not in df.columns:
        raise ValueError(f"Column '{y_col}' does not exist in dataset.")

    filtered_df = apply_dataframe_filters(df, filters)
    total_matching = len(filtered_df)

    c_type = chart_type.lower().strip()

    # 1. BAR CHART
    if c_type in ("bar", "barchart", "bar_chart", "horizontal_bar"):
        if y_col is None or aggregation == "count":
            # Frequency counts
            counts = filtered_df[x_col].dropna().value_counts()
            top_counts = counts.head(25)
            data_points = [
                {"label": str(k), "value": int(v), "count": int(v)}
                for k, v in top_counts.items()
            ]
            y_label = "Count"
        else:
            # Grouped aggregation
            df_valid = filtered_df[[x_col, y_col]].dropna()
            df_valid[y_col] = pd.to_numeric(df_valid[y_col], errors="coerce")
            df_valid = df_valid.dropna()

            agg_func = aggregation.lower()
            if agg_func == "sum":
                grouped = df_valid.groupby(x_col)[y_col].sum()
            elif agg_func in ("mean", "avg"):
                grouped = df_valid.groupby(x_col)[y_col].mean()
            elif agg_func == "min":
                grouped = df_valid.groupby(x_col)[y_col].min()
            elif agg_func == "max":
                grouped = df_valid.groupby(x_col)[y_col].max()
            else:
                grouped = df_valid.groupby(x_col)[y_col].count()

            top_grouped = grouped.sort_values(ascending=False).head(25)
            data_points = [
                {"label": str(k), "value": round(float(v), 2), "count": int(df_valid[df_valid[x_col] == k].shape[0])}
                for k, v in top_grouped.items()
            ]
            y_label = f"{aggregation.capitalize()} of {y_col}"

        return {
            "chart_type": "bar",
            "x_column": x_col,
            "y_column": y_col or "count",
            "x_label": x_col,
            "y_label": y_label,
            "aggregation": aggregation,
            "total_records": total_matching,
            "data": data_points
        }

    # 2. LINE CHART
    elif c_type in ("line", "linechart", "line_chart"):
        df_valid = filtered_df[[x_col] + ([y_col] if y_col else [])].dropna()
        if y_col is None or aggregation == "count":
            grouped = df_valid.groupby(x_col).size()
            y_label = "Count"
        else:
            df_valid[y_col] = pd.to_numeric(df_valid[y_col], errors="coerce")
            df_valid = df_valid.dropna()
            agg_func = aggregation.lower()
            if agg_func == "sum":
                grouped = df_valid.groupby(x_col)[y_col].sum()
            elif agg_func == "min":
                grouped = df_valid.groupby(x_col)[y_col].min()
            elif agg_func == "max":
                grouped = df_valid.groupby(x_col)[y_col].max()
            else:
                grouped = df_valid.groupby(x_col)[y_col].mean()
            y_label = f"{aggregation.capitalize()} of {y_col}"

        # Sort along x axis (handling numeric / string sorts)
        try:
            sorted_idx = sorted(grouped.index, key=lambda x: float(x))
            grouped = grouped.reindex(sorted_idx)
        except (ValueError, TypeError):
            try:
                sorted_idx = sorted(grouped.index)
                grouped = grouped.reindex(sorted_idx)
            except Exception:
                pass

        data_points = [
            {"label": str(k), "value": round(float(v), 2)}
            for k, v in grouped.items()
        ]
        return {
            "chart_type": "line",
            "x_column": x_col,
            "y_column": y_col or "count",
            "x_label": x_col,
            "y_label": y_label,
            "aggregation": aggregation,
            "total_records": total_matching,
            "data": data_points
        }

    # 3. HISTOGRAM
    elif c_type in ("hist", "histogram", "kde"):
        s_num = pd.to_numeric(filtered_df[x_col], errors="coerce").dropna()
        if len(s_num) == 0:
            return {
                "chart_type": "histogram",
                "x_column": x_col,
                "data": [],
                "stats": {"count": 0}
            }

        num_bins = min(max(10, int(np.sqrt(len(s_num)))), 30)
        counts, bin_edges = np.histogram(s_num, bins=num_bins)

        data_points = []
        for i in range(len(counts)):
            b_start = round(float(bin_edges[i]), 2)
            b_end = round(float(bin_edges[i + 1]), 2)
            data_points.append({
                "bin_start": b_start,
                "bin_end": b_end,
                "label": f"{b_start} - {b_end}",
                "count": int(counts[i]),
                "density": round(float(counts[i] / len(s_num)), 4)
            })

        return {
            "chart_type": "histogram",
            "x_column": x_col,
            "x_label": x_col,
            "y_label": "Frequency",
            "total_records": len(s_num),
            "data": data_points,
            "stats": {
                "count": len(s_num),
                "mean": round(float(s_num.mean()), 2),
                "median": round(float(s_num.median()), 2),
                "std": round(float(s_num.std()), 2) if len(s_num) > 1 else 0.0,
                "min": round(float(s_num.min()), 2),
                "max": round(float(s_num.max()), 2),
                "skewness": round(float(s_num.skew()), 2) if len(s_num) > 2 else 0.0
            }
        }

    # 4. SCATTER PLOT
    elif c_type in ("scatter", "scatterplot", "scatter_plot"):
        if not y_col:
            raise ValueError("Scatter plot requires both x_column and y_column.")

        df_pair = filtered_df[[x_col, y_col]].dropna()
        df_pair[x_col] = pd.to_numeric(df_pair[x_col], errors="coerce")
        df_pair[y_col] = pd.to_numeric(df_pair[y_col], errors="coerce")
        df_pair = df_pair.dropna()

        total_pts = len(df_pair)
        if total_pts == 0:
            return {
                "chart_type": "scatter",
                "x_column": x_col,
                "y_column": y_col,
                "data": [],
                "correlation": 0.0,
                "trendline": None
            }

        # Calculate Pearson correlation
        corr_val = float(df_pair[x_col].corr(df_pair[y_col]))
        corr_val = round(corr_val, 3) if not math.isnan(corr_val) else 0.0

        # Linear trendline: y = m*x + b
        trendline = None
        if total_pts > 2 and df_pair[x_col].std() > 0:
            m, b = np.polyfit(df_pair[x_col], df_pair[y_col], 1)
            x_min = float(df_pair[x_col].min())
            x_max = float(df_pair[x_col].max())
            trendline = {
                "slope": round(float(m), 4),
                "intercept": round(float(b), 4),
                "start_point": {"x": round(x_min, 2), "y": round(float(m * x_min + b), 2)},
                "end_point": {"x": round(x_max, 2), "y": round(float(m * x_max + b), 2)}
            }

        # Subsample if large to protect browser DOM & SVG rendering
        if total_pts > max_points:
            step = int(np.ceil(total_pts / max_points))
            df_sub = df_pair.iloc[::step]
        else:
            df_sub = df_pair

        data_points = [
            {"x": round(float(r[x_col]), 2), "y": round(float(r[y_col]), 2)}
            for _, r in df_sub.iterrows()
        ]

        return {
            "chart_type": "scatter",
            "x_column": x_col,
            "y_column": y_col,
            "x_label": x_col,
            "y_label": y_col,
            "total_records": total_pts,
            "sampled_records": len(data_points),
            "correlation": corr_val,
            "trendline": trendline,
            "data": data_points
        }

    # 5. CORRELATION HEATMAP
    elif c_type in ("heatmap", "correlation_heatmap", "matrix"):
        # Select numeric columns
        num_cols = [c for c in filtered_df.columns if infer_column_type(filtered_df[c]) == "numeric"][:12]
        if len(num_cols) < 2:
            return {"chart_type": "heatmap", "columns": [], "data": []}

        df_num = filtered_df[num_cols].apply(pd.to_numeric, errors="coerce")
        corr_matrix = df_num.corr().fillna(0.0)

        grid_data = []
        matrix_rows = []
        for i, c1 in enumerate(num_cols):
            row = []
            for j, c2 in enumerate(num_cols):
                val = round(float(corr_matrix.loc[c1, c2]), 3)
                row.append(val)
                grid_data.append({
                    "x": c1,
                    "y": c2,
                    "value": val,
                    "abs_val": round(abs(val), 3)
                })
            matrix_rows.append(row)

        return {
            "chart_type": "heatmap",
            "columns": num_cols,
            "matrix": matrix_rows,
            "data": grid_data,
            "total_columns": len(num_cols)
        }

    # 6. PIE / DONUT CHART
    elif c_type in ("pie", "donut", "pie_chart", "donut_chart"):
        is_donut = "donut" in c_type
        if y_col and y_col in filtered_df.columns and infer_column_type(filtered_df[y_col]) == "numeric":
            df_valid = filtered_df[[x_col, y_col]].dropna()
            df_valid[y_col] = pd.to_numeric(df_valid[y_col], errors="coerce")
            df_valid = df_valid.dropna()
            grouped = df_valid.groupby(x_col)[y_col].sum()
            total_sum = float(grouped.sum()) if len(grouped) > 0 else 1.0
            top_grouped = grouped.sort_values(ascending=False).head(10)
            data_points = [
                {
                    "label": str(k),
                    "value": round(float(v), 2),
                    "percentage": round(float(v / max(total_sum, 1e-6)) * 100, 1),
                }
                for k, v in top_grouped.items()
            ]
        else:
            counts = filtered_df[x_col].dropna().value_counts().head(10)
            total_count = int(counts.sum()) if len(counts) > 0 else 1
            data_points = [
                {
                    "label": str(k),
                    "value": int(v),
                    "percentage": round(float(v / max(total_count, 1)) * 100, 1),
                }
                for k, v in counts.items()
            ]

        return {
            "chart_type": "donut" if is_donut else "pie",
            "x_column": x_col,
            "y_column": y_col or "count",
            "x_label": x_col,
            "y_label": "Proportion",
            "total_records": total_matching,
            "data": data_points,
            "is_donut": is_donut,
        }

    # 7. AREA CHART
    elif c_type in ("area", "areachart", "area_chart"):
        df_valid = filtered_df[[x_col] + ([y_col] if y_col else [])].dropna()
        if y_col is None or aggregation == "count":
            grouped = df_valid.groupby(x_col).size()
            y_label = "Count"
        else:
            df_valid[y_col] = pd.to_numeric(df_valid[y_col], errors="coerce")
            df_valid = df_valid.dropna()
            agg_func = aggregation.lower()
            if agg_func == "sum":
                grouped = df_valid.groupby(x_col)[y_col].sum()
            elif agg_func == "min":
                grouped = df_valid.groupby(x_col)[y_col].min()
            elif agg_func == "max":
                grouped = df_valid.groupby(x_col)[y_col].max()
            else:
                grouped = df_valid.groupby(x_col)[y_col].mean()
            y_label = f"{aggregation.capitalize()} of {y_col}"

        try:
            sorted_idx = sorted(grouped.index, key=lambda x: float(x))
            grouped = grouped.reindex(sorted_idx)
        except (ValueError, TypeError):
            try:
                sorted_idx = sorted(grouped.index)
                grouped = grouped.reindex(sorted_idx)
            except Exception:
                pass

        data_points = [
            {"label": str(k), "value": round(float(v), 2)}
            for k, v in grouped.items()
        ]
        return {
            "chart_type": "area",
            "x_column": x_col,
            "y_column": y_col or "count",
            "x_label": x_col,
            "y_label": y_label,
            "aggregation": aggregation,
            "total_records": total_matching,
            "data": data_points
        }

    # 8. BOX PLOT
    elif c_type in ("box", "boxplot", "box_plot"):
        # If both categorical X and numeric Y
        if y_col and y_col in filtered_df.columns:
            cat_col, num_col = x_col, y_col
        else:
            cat_col, num_col = None, x_col

        df_valid = filtered_df.dropna(subset=[num_col])
        df_valid[num_col] = pd.to_numeric(df_valid[num_col], errors="coerce")
        df_valid = df_valid.dropna(subset=[num_col])

        box_groups = []
        if cat_col and cat_col in df_valid.columns:
            top_cats = df_valid[cat_col].value_counts().head(8).index.tolist()
            for cat in top_cats:
                vals = df_valid[df_valid[cat_col] == cat][num_col].values
                if len(vals) >= 3:
                    q1 = float(np.percentile(vals, 25))
                    med = float(np.percentile(vals, 50))
                    q3 = float(np.percentile(vals, 75))
                    iqr = q3 - q1
                    whisker_low = float(max(np.min(vals), q1 - 1.5 * iqr))
                    whisker_high = float(min(np.max(vals), q3 + 1.5 * iqr))
                    box_groups.append({
                        "label": str(cat),
                        "min": round(whisker_low, 2),
                        "q1": round(q1, 2),
                        "median": round(med, 2),
                        "q3": round(q3, 2),
                        "max": round(whisker_high, 2),
                        "mean": round(float(np.mean(vals)), 2),
                        "count": len(vals),
                    })
        else:
            vals = df_valid[num_col].values
            if len(vals) >= 3:
                q1 = float(np.percentile(vals, 25))
                med = float(np.percentile(vals, 50))
                q3 = float(np.percentile(vals, 75))
                iqr = q3 - q1
                whisker_low = float(max(np.min(vals), q1 - 1.5 * iqr))
                whisker_high = float(min(np.max(vals), q3 + 1.5 * iqr))
                box_groups.append({
                    "label": str(num_col),
                    "min": round(whisker_low, 2),
                    "q1": round(q1, 2),
                    "median": round(med, 2),
                    "q3": round(q3, 2),
                    "max": round(whisker_high, 2),
                    "mean": round(float(np.mean(vals)), 2),
                    "count": len(vals),
                })

        return {
            "chart_type": "box",
            "x_column": x_col,
            "y_column": y_col,
            "x_label": cat_col or "Metric",
            "y_label": num_col,
            "total_records": len(df_valid),
            "data": box_groups
        }

    # 9. BUBBLE CHART
    elif c_type in ("bubble", "bubblechart", "bubble_plot"):
        if not y_col:
            raise ValueError("Bubble chart requires both x_column and y_column.")
        df_pair = filtered_df[[x_col, y_col]].dropna()
        df_pair[x_col] = pd.to_numeric(df_pair[x_col], errors="coerce")
        df_pair[y_col] = pd.to_numeric(df_pair[y_col], errors="coerce")
        df_pair = df_pair.dropna()

        # Find 3rd numeric column for bubble size if available
        size_col = None
        for col in filtered_df.columns:
            if col not in (x_col, y_col) and infer_column_type(filtered_df[col]) == "numeric":
                size_col = col
                break

        if size_col:
            df_pair[size_col] = pd.to_numeric(filtered_df.loc[df_pair.index, size_col], errors="coerce").fillna(1.0)
            z_vals = df_pair[size_col].values
            z_min, z_max = float(np.min(z_vals)), float(np.max(z_vals))
            z_range = max(z_max - z_min, 1e-4)
            data_points = [
                {
                    "x": round(float(r[x_col]), 2),
                    "y": round(float(r[y_col]), 2),
                    "size": round(float(10 + 40 * ((r[size_col] - z_min) / z_range)), 1),
                    "size_val": round(float(r[size_col]), 2),
                }
                for _, r in df_pair.head(max_points).iterrows()
            ]
        else:
            data_points = [
                {"x": round(float(r[x_col]), 2), "y": round(float(r[y_col]), 2), "size": 20}
                for _, r in df_pair.head(max_points).iterrows()
            ]

        return {
            "chart_type": "bubble",
            "x_column": x_col,
            "y_column": y_col,
            "size_column": size_col or "uniform",
            "x_label": x_col,
            "y_label": y_col,
            "total_records": len(df_pair),
            "data": data_points
        }

    # 10. TREEMAP
    elif c_type in ("treemap", "tree_map"):
        # Category on X, Numeric measure on Y
        if y_col and y_col in filtered_df.columns and infer_column_type(filtered_df[y_col]) == "numeric":
            df_valid = filtered_df[[x_col, y_col]].dropna()
            df_valid[y_col] = pd.to_numeric(df_valid[y_col], errors="coerce").dropna()
            grouped = df_valid.groupby(x_col)[y_col].sum()
            top_grouped = grouped.sort_values(ascending=False).head(15)
            data_points = [
                {"name": str(k), "size": round(float(v), 2), "value": round(float(v), 2)}
                for k, v in top_grouped.items()
            ]
        else:
            counts = filtered_df[x_col].dropna().value_counts().head(15)
            data_points = [
                {"name": str(k), "size": int(v), "value": int(v)}
                for k, v in counts.items()
            ]

        return {
            "chart_type": "treemap",
            "x_column": x_col,
            "y_column": y_col or "count",
            "total_records": total_matching,
            "data": data_points
        }

    # 11. GROUPED BAR / STACKED BAR
    elif c_type in ("grouped_bar", "stacked_bar", "grouped", "stacked"):
        is_stacked = "stacked" in c_type
        # X: primary categorical, find secondary categorical for stacking if possible
        sec_cat = None
        for col in filtered_df.columns:
            if col != x_col and infer_column_type(filtered_df[col]) == "categorical":
                if filtered_df[col].nunique() <= 6:
                    sec_cat = col
                    break

        if sec_cat:
            top_x = filtered_df[x_col].value_counts().head(8).index.tolist()
            df_sub = filtered_df[filtered_df[x_col].isin(top_x)]
            if y_col and y_col in df_sub.columns and infer_column_type(df_sub[y_col]) == "numeric":
                df_sub[y_col] = pd.to_numeric(df_sub[y_col], errors="coerce")
                pivot = df_sub.pivot_table(index=x_col, columns=sec_cat, values=y_col, aggfunc="mean").fillna(0.0)
            else:
                pivot = df_sub.pivot_table(index=x_col, columns=sec_cat, aggfunc="size").fillna(0.0)

            data_points = []
            keys = pivot.columns.astype(str).tolist()
            for x_val, row in pivot.iterrows():
                row_dict = {"label": str(x_val)}
                for k in keys:
                    row_dict[k] = round(float(row[k]), 2)
                data_points.append(row_dict)

            return {
                "chart_type": "stacked_bar" if is_stacked else "grouped_bar",
                "x_column": x_col,
                "y_column": y_col or "count",
                "group_column": sec_cat,
                "group_keys": keys,
                "data": data_points,
                "total_records": total_matching,
                "is_stacked": is_stacked,
            }
        else:
            # Fall back to standard bar if secondary categorical dimension is unavailable
            return generate_chart_data(df, "bar", x_col, y_col, aggregation, filters, max_points)

    else:
        raise ValueError(
            f"Unsupported chart type '{chart_type}'. Supported: bar, line, histogram, scatter, heatmap, "
            f"pie, donut, area, box, bubble, treemap, grouped_bar, stacked_bar."
        )


# =====================================================================
# 3. VISUALIZATION CANDIDATE CATALOG GENERATION
# =====================================================================
def build_visualization_catalog(df: pd.DataFrame, client_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Constructs candidate visualization items based on dataset column signatures,
    distributions, relationships, and domain semantics.
    """
    candidates: List[Dict[str, Any]] = []

    # Classify columns
    col_types = {c: infer_column_type(df[c]) for c in df.columns}
    numeric_cols = [c for c in df.columns if col_types[c] == "numeric"]
    categorical_cols = [c for c in df.columns if col_types[c] in ("categorical", "identifier")]

    # 1. Global Correlation Heatmap Candidate (Always highly relevant initial overview)
    if len(numeric_cols) >= 3:
        candidates.append({
            "candidate_id": "cand_correlation_heatmap_overall",
            "title": "Multivariate Correlation Matrix",
            "chart_type": "heatmap",
            "x_col": numeric_cols[0],
            "y_col": numeric_cols[1],
            "aggregation": "none",
            "task_type": "correlation",
            "domain_category": "Overview",
            "description": f"Pairwise Pearson correlation heatmap across all {len(numeric_cols)} numeric attributes to identify key linear dependencies.",
            "columns_involved": numeric_cols[:10],
            "statistical_signal": {
                "signal_type": "multivariate_matrix",
                "numeric_col_count": len(numeric_cols),
                "base_suitability": 0.95
            }
        })

    # 2. Funnel Candidate (if E-Commerce behavioral funnel markers exist)
    funnel_cols = [c for c in ["added_to_cart", "purchased", "cart_abandoned"] if c in df.columns]
    if len(funnel_cols) >= 2:
        candidates.append({
            "candidate_id": "cand_funnel_progression",
            "title": "Customer Conversion & Funnel Progression",
            "chart_type": "bar",
            "x_col": "funnel_stage",
            "y_col": "count",
            "aggregation": "count",
            "task_type": "funnel",
            "domain_category": "Conversion & Funnel",
            "description": "Visualizes drop-off progression from catalog view to cart addition and purchase completion.",
            "columns_involved": funnel_cols,
            "statistical_signal": {
                "signal_type": "behavioral_funnel",
                "base_suitability": 0.94
            }
        })

    # 3. Categorical Breakdown Candidates (Top categories by interaction volume)
    for cat_c in categorical_cols[:4]:
        cardinality = df[cat_c].nunique(dropna=True)
        if 2 <= cardinality <= 30:
            candidates.append({
                "candidate_id": f"cand_bar_{cat_c}_frequency",
                "title": f"Interaction Volume by {cat_c.replace('_', ' ').title()}",
                "chart_type": "bar",
                "x_col": cat_c,
                "y_col": None,
                "aggregation": "count",
                "task_type": "comparison",
                "domain_category": "Categorical Volume",
                "description": f"Frequency distribution across {cardinality} distinct '{cat_c}' categories.",
                "columns_involved": [cat_c],
                "statistical_signal": {
                    "signal_type": "category_distribution",
                    "cardinality": cardinality,
                    "base_suitability": 0.88 if cardinality <= 15 else 0.78
                }
            })

            # If numeric columns exist, add category vs metric aggregation
            for num_c in numeric_cols[:2]:
                candidates.append({
                    "candidate_id": f"cand_bar_{cat_c}_{num_c}_agg",
                    "title": f"Average {num_c.replace('_', ' ').title()} by {cat_c.replace('_', ' ').title()}",
                    "chart_type": "bar",
                    "x_col": cat_c,
                    "y_col": num_c,
                    "aggregation": "mean",
                    "task_type": "comparison",
                    "domain_category": "Metric by Category",
                    "description": f"Compares mean '{num_c}' across '{cat_c}' groups.",
                    "columns_involved": [cat_c, num_c],
                    "statistical_signal": {
                        "signal_type": "category_metric_aggregation",
                        "cardinality": cardinality,
                        "base_suitability": 0.86
                    }
                })

    # 4. Continuous Distributions (Histograms for numeric metrics)
    for num_c in numeric_cols[:4]:
        s_num = pd.to_numeric(df[num_c], errors="coerce").dropna()
        if len(s_num) > 10 and s_num.nunique() > 10:
            skew = abs(float(s_num.skew())) if len(s_num) > 2 else 0.0
            candidates.append({
                "candidate_id": f"cand_hist_{num_c}",
                "title": f"{num_c.replace('_', ' ').title()} Spread & Density",
                "chart_type": "histogram",
                "x_col": num_c,
                "y_col": None,
                "aggregation": "frequency",
                "task_type": "distribution",
                "domain_category": "Distribution",
                "description": f"Binned histogram showing spread, variance, and skewness ({skew:.2f}) of '{num_c}'.",
                "columns_involved": [num_c],
                "statistical_signal": {
                    "signal_type": "numeric_distribution",
                    "skewness": round(skew, 2),
                    "base_suitability": 0.85
                }
            })

    # 5. Bivariate Correlation Scatter Candidates
    if len(numeric_cols) >= 2:
        df_num = df[numeric_cols[:6]].apply(pd.to_numeric, errors="coerce")
        corr = df_num.corr().fillna(0.0)

        pairs = []
        for i in range(len(numeric_cols[:6])):
            for j in range(i + 1, len(numeric_cols[:6])):
                c1 = numeric_cols[i]
                c2 = numeric_cols[j]
                r_val = float(corr.loc[c1, c2])
                if not math.isnan(r_val):
                    pairs.append((abs(r_val), r_val, c1, c2))

        pairs.sort(key=lambda x: x[0], reverse=True)
        for abs_r, r_val, c1, c2 in pairs[:4]:
            suitability = 0.70 + (abs_r * 0.28) # Higher correlation -> higher scatter suitability
            candidates.append({
                "candidate_id": f"cand_scatter_{c1}_{c2}",
                "title": f"{c1.replace('_', ' ').title()} vs {c2.replace('_', ' ').title()} Relationship",
                "chart_type": "scatter",
                "x_col": c1,
                "y_col": c2,
                "aggregation": "none",
                "task_type": "correlation",
                "domain_category": "Bivariate Relationship",
                "description": f"Scatter plot with linear trendline showing pairwise relationship (Pearson r = {r_val:.2f}).",
                "columns_involved": [c1, c2],
                "statistical_signal": {
                    "signal_type": "bivariate_correlation",
                    "correlation": round(r_val, 3),
                    "abs_correlation": round(abs_r, 3),
                    "base_suitability": round(suitability, 3)
                }
            })

    # 6. Temporal / Sequential Trend Candidates (if datetime or date-like column exists)
    time_cols = [c for c in df.columns if any(t in c.lower() for t in ["date", "time", "day", "month", "timestamp"])]
    if time_cols and numeric_cols:
        t_col = time_cols[0]
        n_col = numeric_cols[0]
        candidates.append({
            "candidate_id": f"cand_line_{t_col}_{n_col}",
            "title": f"{n_col.replace('_', ' ').title()} Trend over {t_col.replace('_', ' ').title()}",
            "chart_type": "line",
            "x_col": t_col,
            "y_col": n_col,
            "aggregation": "mean",
            "task_type": "trend",
            "domain_category": "Temporal Trend",
            "description": f"Sequential line chart illustrating chronological progression of '{n_col}'.",
            "columns_involved": [t_col, n_col],
            "statistical_signal": {
                "signal_type": "temporal_progression",
                "base_suitability": 0.89
            }
        })
        # Extended area chart candidate
        candidates.append({
            "candidate_id": f"cand_area_{t_col}_{n_col}",
            "title": f"{n_col.replace('_', ' ').title()} Cumulative Area over {t_col.replace('_', ' ').title()}",
            "chart_type": "area",
            "x_col": t_col,
            "y_col": n_col,
            "aggregation": "mean",
            "task_type": "trend",
            "domain_category": "Temporal Area",
            "description": f"Continuous area visualization showing volume and progression of '{n_col}'.",
            "columns_involved": [t_col, n_col],
            "statistical_signal": {
                "signal_type": "temporal_progression",
                "base_suitability": 0.86
            }
        })

    # 7. Low-cardinality Proportions (Pie / Donut / Treemap)
    low_card_cats = [c for c in categorical_cols if 2 <= df[c].nunique() <= 8]
    if low_card_cats:
        c_col = low_card_cats[0]
        m_col = numeric_cols[0] if numeric_cols else None
        candidates.append({
            "candidate_id": f"cand_donut_{c_col}",
            "title": f"{c_col.replace('_', ' ').title()} Share & Composition",
            "chart_type": "donut",
            "x_col": c_col,
            "y_col": m_col,
            "aggregation": "sum" if m_col else "count",
            "task_type": "part_to_whole",
            "domain_category": "Composition",
            "description": f"Donut chart breaking down proportionate shares across {c_col.replace('_', ' ')} segments.",
            "columns_involved": [c_col] + ([m_col] if m_col else []),
            "statistical_signal": {
                "signal_type": "proportional_share",
                "base_suitability": 0.88
            }
        })
        candidates.append({
            "candidate_id": f"cand_treemap_{c_col}",
            "title": f"{c_col.replace('_', ' ').title()} Hierarchical Treemap",
            "chart_type": "treemap",
            "x_col": c_col,
            "y_col": m_col,
            "aggregation": "sum" if m_col else "count",
            "task_type": "hierarchy",
            "domain_category": "Hierarchical Part-to-Whole",
            "description": f"Treemap visualization showing relative area volume for {c_col.replace('_', ' ')}.",
            "columns_involved": [c_col] + ([m_col] if m_col else []),
            "statistical_signal": {
                "signal_type": "proportional_share",
                "base_suitability": 0.82
            }
        })

    # 8. Box Plot Distributions (Grouped or Single)
    if categorical_cols and numeric_cols:
        cat_c = categorical_cols[0]
        num_c = numeric_cols[0]
        if 2 <= df[cat_c].nunique() <= 10:
            candidates.append({
                "candidate_id": f"cand_box_{cat_c}_{num_c}",
                "title": f"{num_c.replace('_', ' ').title()} Distribution by {cat_c.replace('_', ' ').title()}",
                "chart_type": "box",
                "x_col": cat_c,
                "y_col": num_c,
                "aggregation": "distribution",
                "task_type": "distribution",
                "domain_category": "Grouped Distribution",
                "description": f"Box plot displaying median, quartiles, and outliers of '{num_c}' across '{cat_c}'.",
                "columns_involved": [cat_c, num_c],
                "statistical_signal": {
                    "signal_type": "grouped_box_distribution",
                    "base_suitability": 0.87
                }
            })

    # 9. Bubble Chart Candidate (3+ Numeric Columns)
    if len(numeric_cols) >= 3:
        b_x, b_y, b_z = numeric_cols[0], numeric_cols[1], numeric_cols[2]
        candidates.append({
            "candidate_id": f"cand_bubble_{b_x}_{b_y}",
            "title": f"{b_x.replace('_', ' ').title()} vs {b_y.replace('_', ' ').title()} (Sized by {b_z.replace('_', ' ').title()})",
            "chart_type": "bubble",
            "x_col": b_x,
            "y_col": b_y,
            "aggregation": "none",
            "task_type": "multivariate_correlation",
            "domain_category": "Multivariate Bubble",
            "description": f"3D multivariate bubble plot exploring '{b_x}' and '{b_y}' with bubble sizing driven by '{b_z}'.",
            "columns_involved": [b_x, b_y, b_z],
            "statistical_signal": {
                "signal_type": "trivariate_bubble",
                "base_suitability": 0.84
            }
        })

    # 10. Stacked Bar Candidate (Two Categorical Columns)
    if len(categorical_cols) >= 2 and any(2 <= df[c].nunique() <= 6 for c in categorical_cols):
        c1, c2 = categorical_cols[0], categorical_cols[1]
        m_c = numeric_cols[0] if numeric_cols else None
        candidates.append({
            "candidate_id": f"cand_stacked_bar_{c1}_{c2}",
            "title": f"{c1.replace('_', ' ').title()} Segmented by {c2.replace('_', ' ').title()}",
            "chart_type": "stacked_bar",
            "x_col": c1,
            "y_col": m_c,
            "aggregation": "mean" if m_c else "count",
            "task_type": "composition",
            "domain_category": "Segmented Composition",
            "description": f"Stacked bar chart showing sub-segment breakdown across {c1.replace('_', ' ')}.",
            "columns_involved": [c1, c2] + ([m_c] if m_c else []),
            "statistical_signal": {
                "signal_type": "bivariate_categorical",
                "base_suitability": 0.85
            }
        })

    return candidates


# =====================================================================
# 4. CHART SIMILARITY & STATISTICAL SUITABILITY SPECIFICATION
# =====================================================================

# Defined chart families and directional similarity weights
CHART_SIMILARITY_MAP: Dict[str, List[Dict[str, Any]]] = {
    "line": [
        {"chart_type": "line", "similarity": 1.0, "relation": "exact"},
        {"chart_type": "area", "similarity": 0.85, "relation": "continuous_area_alternative"},
    ],
    "area": [
        {"chart_type": "area", "similarity": 1.0, "relation": "exact"},
        {"chart_type": "line", "similarity": 0.85, "relation": "sequential_trend_alternative"},
    ],
    "bar": [
        {"chart_type": "bar", "similarity": 1.0, "relation": "exact"},
        {"chart_type": "grouped_bar", "similarity": 0.85, "relation": "grouped_variant"},
        {"chart_type": "stacked_bar", "similarity": 0.80, "relation": "composition_variant"},
        {"chart_type": "donut", "similarity": 0.70, "relation": "proportional_variant"},
        {"chart_type": "pie", "similarity": 0.70, "relation": "part_to_whole_variant"},
        {"chart_type": "treemap", "similarity": 0.65, "relation": "hierarchical_variant"},
    ],
    "grouped_bar": [
        {"chart_type": "grouped_bar", "similarity": 1.0, "relation": "exact"},
        {"chart_type": "bar", "similarity": 0.85, "relation": "standard_bar_variant"},
        {"chart_type": "stacked_bar", "similarity": 0.80, "relation": "composition_variant"},
    ],
    "stacked_bar": [
        {"chart_type": "stacked_bar", "similarity": 1.0, "relation": "exact"},
        {"chart_type": "bar", "similarity": 0.80, "relation": "standard_bar_variant"},
        {"chart_type": "grouped_bar", "similarity": 0.80, "relation": "grouped_variant"},
    ],
    "scatter": [
        {"chart_type": "scatter", "similarity": 1.0, "relation": "exact"},
        {"chart_type": "bubble", "similarity": 0.85, "relation": "multivariate_bubble_variant"},
    ],
    "bubble": [
        {"chart_type": "bubble", "similarity": 1.0, "relation": "exact"},
        {"chart_type": "scatter", "similarity": 0.85, "relation": "bivariate_scatter_variant"},
    ],
    "histogram": [
        {"chart_type": "histogram", "similarity": 1.0, "relation": "exact"},
        {"chart_type": "box", "similarity": 0.75, "relation": "distribution_variant"},
    ],
    "box": [
        {"chart_type": "box", "similarity": 1.0, "relation": "exact"},
        {"chart_type": "histogram", "similarity": 0.75, "relation": "binned_distribution_variant"},
    ],
    "heatmap": [
        {"chart_type": "heatmap", "similarity": 1.0, "relation": "exact"},
        {"chart_type": "scatter", "similarity": 0.60, "relation": "pairwise_correlation_alternative"},
    ],
    "donut": [
        {"chart_type": "donut", "similarity": 1.0, "relation": "exact"},
        {"chart_type": "pie", "similarity": 0.90, "relation": "pie_variant"},
        {"chart_type": "bar", "similarity": 0.70, "relation": "categorical_bar_variant"},
    ],
    "pie": [
        {"chart_type": "pie", "similarity": 1.0, "relation": "exact"},
        {"chart_type": "donut", "similarity": 0.90, "relation": "donut_variant"},
        {"chart_type": "bar", "similarity": 0.70, "relation": "categorical_bar_variant"},
    ],
    "treemap": [
        {"chart_type": "treemap", "similarity": 1.0, "relation": "exact"},
        {"chart_type": "bar", "similarity": 0.65, "relation": "categorical_bar_variant"},
    ],
}


def check_chart_statistical_suitability(
    chart_type: str,
    df: pd.DataFrame,
    cand: Optional[Dict[str, Any]] = None,
    type_conf: float = 0.0
) -> Tuple[bool, str]:
    """
    Checks if a chart type is statistically suitable and compatible with the dataset.
    Strictly prevents user preference from promoting an incompatible chart type.
    """
    ct = (chart_type or "").lower()
    col_types = {c: infer_column_type(df[c]) for c in df.columns}
    numeric_cols = [c for c in df.columns if col_types[c] == "numeric"]
    categorical_cols = [c for c in df.columns if col_types[c] in ("categorical", "identifier")]
    time_cols = [
        c for c in df.columns
        if any(t in str(c).lower() for t in ["date", "time", "day", "month", "year", "timestamp", "period", "week"])
    ]

    # Rule 1: Temporal / Sequential Trend (line, area)
    if ct in ("line", "area"):
        if not time_cols:
            return False, "Dataset lacks sequential or temporal progression columns"
        if not numeric_cols:
            return False, "Dataset lacks continuous numeric metrics to evaluate progression"

    # Rule 2: Correlation / Bivariate (scatter, bubble)
    if ct in ("scatter", "bubble"):
        min_num = 3 if ct == "bubble" else 2
        if len(numeric_cols) < min_num:
            return False, f"Dataset requires at least {min_num} continuous numeric columns for {ct} plot"

    # Rule 3: Multivariate Matrix (heatmap)
    if ct == "heatmap":
        if len(numeric_cols) < 3:
            return False, "Dataset requires at least 3 numeric columns for correlation matrix heatmap"

    # Rule 4: Univariate Continuous Distribution (histogram)
    if ct == "histogram":
        if not numeric_cols:
            return False, "Dataset lacks continuous numeric columns for binned frequency distribution"

    # Rule 5: Categorical Comparisons (bar, grouped_bar, stacked_bar, donut, pie, treemap)
    if ct in ("bar", "grouped_bar", "stacked_bar", "donut", "pie", "treemap"):
        if not categorical_cols:
            return False, "Dataset lacks categorical grouping attributes"

    # Rule 6: Grouped Distribution (box)
    if ct == "box":
        if not (categorical_cols and numeric_cols):
            return False, "Dataset requires both categorical groups and numeric metrics for box plot distributions"

    # Rule 7: Candidate-level column consistency check if candidate is provided
    if cand:
        x_c = cand.get("x_col")
        y_c = cand.get("y_col")

        if ct in ("line", "area") and x_c:
            if x_c not in time_cols and infer_column_type(df[x_c]) != "numeric":
                return False, f"X-axis '{x_c}' is categorical and has no chronological or sequential ordering"

        if ct in ("scatter", "bubble"):
            if x_c and infer_column_type(df[x_c]) != "numeric":
                return False, f"X-axis '{x_c}' must be continuous numeric for scatter/bubble"
            if y_c and infer_column_type(df[y_c]) != "numeric":
                return False, f"Y-axis '{y_c}' must be continuous numeric for scatter/bubble"

        base_suit = cand.get("statistical_signal", {}).get("base_suitability", 0.5)
        if base_suit < 0.40 and type_conf < 0.10:
            return False, "Statistical suitability and MLVis confidence are both below acceptable thresholds"

    return True, "Compatible"


# =====================================================================
# 5. RECOMMENDATION RANKING: COLD-START & NEURAL MODEL INTEGRATION
# =====================================================================
def rank_visualization_recommendations(
    candidates: List[Dict[str, Any]],
    df: pd.DataFrame,
    session_context: Optional[Dict[str, Any]] = None,
    interaction_history: Optional[List[Dict[str, Any]]] = None,
    user_id: Optional[str] = None,
    client_id: Optional[str] = None,
    db_session: Optional[Any] = None,
    top_k: int = 6,
    model_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Ranks visualization candidates using multimodal fusion and context adaptation:
    - Supports model selection:
      * 'mlvis' (default): MLVis visualization neural recommender predicting chart encodings.
      * 'context_aware': Active client Two-Tower neural recommender for e-commerce.
      * 'statistical' / 'custom': Cold-start statistical suitability fallback.
    - Preserves existing client contracts and clearly distinguishes neural confidence from heuristic scores.
    """
    session_ctx = session_context or {}
    history = interaction_history or []
    history_count = len(history)

    # Resolve default model_type:
    # If client_id is given and model_type not specified, default to context_aware for client models.
    # If neither client_id nor user_id is given (anonymous / cold-start exploration), default to statistical heuristic fallback unless explicitly requested.
    if model_type is None:
        if client_id is not None:
            model_type = "context_aware"
        elif user_id is None:
            model_type = "statistical"
        else:
            model_type = "mlvis"

    current_chart = session_ctx.get("current_chart_type", "").lower()
    selected_cols = session_ctx.get("selected_columns", [])
    if isinstance(selected_cols, str):
        selected_cols = [selected_cols]

    # Check if a trained PyTorch model is available for this client
    entry = None
    active_model_version = None
    if client_id and db_session is not None:
        try:
            from backend.services.model_cache import get_model_entry
            from backend.database.models import ModelVersionModel
            rec = (
                db_session.query(ModelVersionModel)
                .filter(ModelVersionModel.client_id == client_id, ModelVersionModel.is_active == True)
                .first()
            )
            if rec:
                active_model_version = rec.version_tag
                entry = get_model_entry(client_id, rec.version_tag, db_session)
        except Exception as e:
            logger.warning(f"Could not load active model for visual recommendation: {str(e)}")
            entry = None

    # Track recently viewed chart types to encourage visual diversity
    recent_types = [h.get("chart_type", "").lower() for h in history[-5:] if h.get("chart_type")]

    # Evaluate each candidate
    ranked_candidates = []
    neural_scores_used = False
    gating_weights = None
    is_cold_start = True

    # Mode 1: MLVis Primary Visualization Recommender (v1 or v2)
    if model_type in ("mlvis", "mlvis_v1", "mlvis_v2"):
        mlvis_ver_req = "v2" if "v2" in model_type else "v1"
        mlvis_engine = get_mlvis_engine(mlvis_ver_req)
        mlvis_meta = mlvis_engine.meta
        mlvis_version = mlvis_meta.get("version", "1.0.0" if mlvis_ver_req == "v1" else "2.0.0")

        # Extract dataset-level features
        n_rows = len(df)
        n_cols = len(df.columns)
        ds_meta = {
            "ds_log_rows": float(np.log1p(n_rows) / 15.0),
            "ds_num_cols": float(min(n_cols, 100) / 100.0),
        }

        # Profile columns needed by candidates
        col_meta_cache = {}
        for c in df.columns:
            try:
                col_meta_cache[c] = compute_column_meta(df[c])
            except Exception:
                col_meta_cache[c] = {
                    "is_numeric": 0.0, "is_categorical": 1.0, "is_datetime": 0.0,
                    "unique_ratio": 0.5, "missing_ratio": 0.0, "entropy": 0.5,
                    "skewness": 0.0, "mean_magnitude": 0.0
                }

        num_numeric = sum(m["is_numeric"] for m in col_meta_cache.values())
        ds_meta["ds_numeric_ratio"] = float(num_numeric / max(n_cols, 1))
        ds_meta["ds_categorical_ratio"] = float(1.0 - ds_meta["ds_numeric_ratio"])

        # Mapping for extended visual types to the 6 canonical classes
        CANONICAL_MAP = {
            "bar": "bar",
            "grouped_bar": "bar",
            "stacked_bar": "bar",
            "pie": "bar",
            "donut": "bar",
            "treemap": "bar",
            "line": "line",
            "area": "line",
            "scatter": "scatter",
            "bubble": "scatter",
            "box": "box",
            "histogram": "histogram",
            "heatmap": "heatmap",
        }

        # Build feature vector for each candidate
        feature_rows = []
        for cand in candidates:
            x_col = cand.get("x_col") or (cand.get("columns_involved") or [None])[0]
            y_col = cand.get("y_col") or x_col
            x_m = col_meta_cache.get(x_col, {
                "is_numeric": 0.0, "is_categorical": 1.0, "is_datetime": 0.0,
                "unique_ratio": 0.5, "missing_ratio": 0.0, "entropy": 0.5,
                "skewness": 0.0, "mean_magnitude": 0.0
            })
            y_m = col_meta_cache.get(y_col, x_m)

            corr = 0.0
            if x_m["is_numeric"] > 0.5 and y_m["is_numeric"] > 0.5 and x_col in df and y_col in df:
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

        feature_matrix = np.array(feature_rows, dtype=np.float32)
        mlvis_predictions = mlvis_engine.predict_chart_types(feature_matrix)

        # Mode 1: MLVis Primary Visualization Recommender (v1 or v2)
        # Retrieve persistent user history preferences if user_id is provided
        user_pref_data = {
            "has_history": False,
            "has_relevant_preferences": False,
            "chart_type_weights": {},
            "positive_charts": [],
            "negative_charts": [],
            "selected_charts_count": {},
            "total_interactions": 0,
            "total_genuine_selections": 0
        }
        if user_id:
            try:
                from backend.services.user_history_service import get_user_chart_preferences
                cid_for_pref = client_id or "demo_ecommerce"
                user_pref_data = get_user_chart_preferences(cid_for_pref, str(user_id).strip())
            except Exception as e_pref:
                logger.warning(f"Could not load user chart preferences: {e_pref}")
                user_pref_data = {
                    "has_history": False,
                    "has_relevant_preferences": False,
                    "chart_type_weights": {},
                    "positive_charts": [],
                    "negative_charts": [],
                    "selected_charts_count": {},
                    "total_interactions": 0,
                    "total_genuine_selections": 0
                }

        is_user_warm = user_pref_data.get("has_history", False) and (user_pref_data.get("total_interactions", 0) > 0)
        is_cold_start = not is_user_warm
        has_chart_pref = is_user_warm and user_pref_data.get("has_relevant_preferences", False)

        # Detect if any strongly preferred chart types are unsuitable for this specific dataset
        unsuitable_preferred_charts = []
        if has_chart_pref:
            for pref_ct, w in user_pref_data.get("chart_type_weights", {}).items():
                if w >= 0.04:
                    is_suit, reason = check_chart_statistical_suitability(pref_ct, df)
                    if not is_suit:
                        alts = []
                        for sim_item in CHART_SIMILARITY_MAP.get(pref_ct, []):
                            alt_ct = sim_item["chart_type"]
                            if alt_ct != pref_ct:
                                alt_suit, _ = check_chart_statistical_suitability(alt_ct, df)
                                if alt_suit and alt_ct not in alts:
                                    alts.append(alt_ct)
                        if not alts:
                            for fb_ct in ["bar", "histogram", "scatter"]:
                                fb_suit, _ = check_chart_statistical_suitability(fb_ct, df)
                                if fb_suit and fb_ct not in alts:
                                    alts.append(fb_ct)
                        unsuitable_preferred_charts.append({
                            "preferred_chart": pref_ct,
                            "reason": reason,
                            "suggested_alternatives": alts[:3]
                        })

        for i, cand in enumerate(candidates):
            pred = mlvis_predictions[i]
            c_type = cand["chart_type"]
            cand_cols = cand.get("columns_involved", [cand["x_col"]])

            # Candidate canonical class lookup
            canonical_type = CANONICAL_MAP.get(c_type, "bar")
            type_probs = pred["type_probabilities"]
            type_conf = float(type_probs.get(canonical_type, pred["confidence"] * 0.5))

            # Contextual column selection alignment
            col_overlap = sum(1 for c in cand_cols if c in selected_cols)
            col_boost = min(col_overlap * 0.05, 0.10)

            # Session-level immediate feedback boost
            session_boost = 0.0
            for ev in history:
                if ev.get("event_type") == "helpful_feedback" and ev.get("chart_type") == c_type:
                    session_boost += 0.04
                elif ev.get("event_type") == "unhelpful_feedback" and ev.get("chart_type") == c_type:
                    session_boost -= 0.06

            # Persistent Warm-Start History Personalization Boost:
            # Applies bounded delta [-0.12, +0.12] derived from user's persistent selections & feedback.
            # CRITICAL CONSTRAINT: History personalization only boosts suitable charts.
            # Unsuitable charts (e.g. line charts on non-sequential data) receive strictly 0.0 boost.
            history_boost = 0.0
            pref_source = None
            pref_relation = None
            pref_label = None

            if has_chart_pref:
                is_cand_suitable, suit_reason = check_chart_statistical_suitability(c_type, df, cand, type_conf)
                if is_cand_suitable:
                    # 1. Exact match preference
                    raw_w = user_pref_data["chart_type_weights"].get(c_type, 0.0)
                    if raw_w > 0:
                        history_boost = raw_w
                        pref_source = c_type
                        pref_relation = "exact"
                        pref_label = f"Suggested based on your previous {c_type} selections"
                    elif raw_w < 0:
                        # Negative feedback penalty
                        history_boost = raw_w
                        pref_source = c_type
                        pref_relation = "negative_feedback"
                        pref_label = "Adjusted based on your previous feedback"
                    else:
                        # 2. Similar chart type preference in family
                        best_sim_boost = 0.0
                        best_sim_source = None
                        for sim_info in CHART_SIMILARITY_MAP.get(c_type, []):
                            sim_type = sim_info["chart_type"]
                            if sim_type == c_type:
                                continue
                            sim_weight = user_pref_data["chart_type_weights"].get(sim_type, 0.0)
                            if sim_weight > 0:
                                eff_boost = sim_weight * sim_info["similarity"]
                                if eff_boost > best_sim_boost:
                                    best_sim_boost = eff_boost
                                    best_sim_source = sim_type

                        if best_sim_boost > 0:
                            history_boost = round(best_sim_boost, 3)
                            pref_source = best_sim_source
                            pref_relation = "similar"
                            pref_label = f"Similar to your previous {best_sim_source} selection"

            # Combine: Base MLVis Confidence + Context Boost + Session Boost + Persistent History Personalization
            combined_score = type_conf + col_boost + session_boost + history_boost
            final_score = round(min(max(combined_score, 0.05), 0.99), 3)

            field_assignments = {
                "x_axis": cand.get("x_col"),
                "y_axis": cand.get("y_col"),
                "aggregation": cand.get("aggregation")
            }

            is_extended = c_type not in CANONICAL_CHART_TYPES
            mode_prefix = "Cold-Start" if is_cold_start else ("Warm-Start Personalized" if history_boost != 0 else "Warm-Start")

            history_explanation_tag = ""
            if history_boost != 0 and pref_label:
                history_explanation_tag = f" [{pref_label}: {f'+{history_boost:.0%}' if history_boost > 0 else f'{history_boost:.0%}'}]"
            elif not has_chart_pref and unsuitable_preferred_charts:
                un_p = unsuitable_preferred_charts[0]
                history_explanation_tag = f" [Note: Your preferred {un_p['preferred_chart']} chart is unsuitable for this dataset ({un_p['reason']}). Recommended as top statistical alternative.]"

            explanation_str = (
                f"{mode_prefix}: MLVis ({mlvis_version}) predicted canonical '{canonical_type}' ({type_conf:.1%} confidence) "
                f"{f'mapped to extended {c_type} encoding' if is_extended else ''} for "
                f"[X: {cand.get('x_col')}, Y: {cand.get('y_col')}]. {cand['description']}{history_explanation_tag}"
            )

            score_label = (
                f"MLVis Model Confidence (v{mlvis_version}) — {'Cold Start' if is_cold_start else ('Warm Personalized' if history_boost != 0 else 'Warm Start')}"
                if mlvis_engine.loaded else "Statistical Baseline"
            )

            ranked_candidates.append({
                **cand,
                "recommendation_score": final_score,
                "base_model_confidence": round(type_conf, 3),
                "history_personalization_boost": round(history_boost, 3),
                "preference_source_chart": pref_source if history_boost != 0 else None,
                "preference_relation": pref_relation if history_boost != 0 else None,
                "personalization_label": pref_label if history_boost != 0 else None,
                "preference_applied": bool(history_boost != 0),
                "score_source": f"mlvis_{mlvis_version}" if mlvis_engine.loaded else "statistical_baseline",
                "score_label": score_label,
                "explanation": explanation_str,
                "is_cold_start": is_cold_start,
                "personalization_mode": "warm_personalized" if (is_user_warm and history_boost != 0) else ("warm_start" if is_user_warm else "cold_start"),
                "model_version": f"mlvis_v{mlvis_version}",
                "model_type": f"mlvis_{mlvis_ver_req}",
                "canonical_base_type": canonical_type,
                "is_extended_type": is_extended,
                "field_assignments": field_assignments,
                "predicted_chart_type": pred["predicted_type"],
                "type_probabilities": type_probs
            })

        ranked_candidates.sort(key=lambda x: x["recommendation_score"], reverse=True)
        any_personalized = any(c.get("preference_applied") for c in ranked_candidates)
        overall_mode = "warm_personalized" if (is_user_warm and any_personalized) else ("warm_start" if is_user_warm else "cold_start")

        return {
            "client_id": client_id,
            "user_id": user_id,
            "model_type": "mlvis",
            "model_version": f"mlvis_v{mlvis_version}",
            "is_cold_start": is_cold_start,
            "has_chart_preference": has_chart_pref,
            "unsuitable_preferred_charts": unsuitable_preferred_charts,
            "history_count": user_pref_data.get("total_interactions", history_count),
            "score_source": "mlvis_model" if mlvis_engine.loaded else "statistical_baseline",
            "personalization_mode": overall_mode,
            "total_candidates": len(candidates),
            "recommendations": ranked_candidates[:top_k]
        }


    # Check user historical depth from customer profile if user_id is provided
    user_profile = None
    user_n_hist = history_count
    if entry and user_id:
        uid_clean = str(user_id).strip()
        user_profile = entry.customer_profiles.get(uid_clean)
        if user_profile is None:
            try:
                user_profile = entry.customer_profiles.get(int(uid_clean))
            except (ValueError, TypeError):
                pass
        if user_profile:
            user_n_hist = int(user_profile.get("count", history_count))

    cold_threshold = entry.cold_start_threshold if entry else 3
    is_cold_start = (user_n_hist < cold_threshold)

    # If active neural model is available and model_type == 'context_aware', compute user/session context encoding & gating weights
    if entry is not None and model_type == "context_aware":
        try:
            from backend.services.inference_service import _build_context_vector

            # Construct session context vector
            raw_ctx_dict = {
                "device_type": session_ctx.get("device_type", 1),
                "user_type": session_ctx.get("user_type", 0),
                "marketing_channel": session_ctx.get("marketing_channel", 2),
                "visit_season": session_ctx.get("visit_season", 1),
                "visit_day": session_ctx.get("visit_day", 15),
                "visit_month": session_ctx.get("visit_month", 6),
                "visit_weekday": session_ctx.get("visit_weekday", 2),
                "location": session_ctx.get("location", 50)
            }
            x_ctx = _build_context_vector(raw_ctx_dict, client_id, entry)

            # Construct behavior vector
            if user_profile and client_id == "demo_ecommerce":
                beh_scaler = entry.encoders["beh_scaler"]
                beh_num = [
                    user_profile["count"], user_profile["purchases"], user_profile["purchase_rate"],
                    user_profile["revenue"], user_profile["cart_add_rate"], user_profile["avg_pages_viewed"],
                    user_profile["avg_time_on_site"], user_profile["recency_score"]
                ]
                df_beh = pd.DataFrame([beh_num], columns=entry.encoders["beh_num_cols"])
                scaled_num = beh_scaler.transform(df_beh)
                cat_prefs = np.array(user_profile["cat_prefs"]).reshape(1, -1)
                x_beh = np.hstack([scaled_num, cat_prefs]).astype(np.float32)
            else:
                x_beh = np.zeros((1, entry.behavior_dim), dtype=np.float32)

            x_anchor = np.zeros((1, entry.content_dim), dtype=np.float32)

            with torch.no_grad():
                t_beh = torch.from_numpy(x_beh)
                t_anchor = torch.from_numpy(x_anchor)
                t_ctx = torch.from_numpy(x_ctx)
                t_hist = torch.tensor([user_n_hist], dtype=torch.long)

                _, weights, cold_mask = entry.model.encode_user_context(
                    x_beh=t_beh,
                    x_cont_anchor=t_anchor,
                    x_ctx=t_ctx,
                    hist_counts=t_hist
                )
                raw_w = weights[0].cpu().numpy().tolist()
                gating_weights = {
                    "behavior": round(float(raw_w[0]), 3),
                    "content": round(float(raw_w[1]), 3),
                    "context": round(float(raw_w[2]), 3),
                }
                neural_scores_used = True
        except Exception as e:
            logger.warning(f"Error computing neural context weights: {str(e)}")
            gating_weights = None

    for cand in candidates:
        base_suitability = float(cand.get("statistical_signal", {}).get("base_suitability", 0.75))
        c_type = cand["chart_type"]
        cand_cols = cand.get("columns_involved", [cand["x_col"]])

        # 1. Column Selection Alignment Boost
        col_overlap = sum(1 for c in cand_cols if c in selected_cols)
        col_boost = min(col_overlap * 0.08, 0.16)

        # 2. Contextual Chart Type Diversity & Relevance
        type_boost = 0.0
        if current_chart and c_type != current_chart:
            type_boost = 0.04
        if recent_types.count(c_type) >= 2:
            type_boost -= 0.05

        # 3. User feedback history boost
        feedback_boost = 0.0
        for ev in history:
            if ev.get("event_type") == "helpful_feedback" and ev.get("chart_type") == c_type:
                feedback_boost += 0.05
            elif ev.get("event_type") == "unhelpful_feedback" and ev.get("chart_type") == c_type:
                feedback_boost -= 0.08

        # Calculate final ranking score
        if neural_scores_used and gating_weights and model_type == "context_aware":
            w_beh = gating_weights["behavior"]
            w_cont = gating_weights["content"]
            w_ctx = gating_weights["context"]

            s_content = base_suitability + col_boost
            s_context = 0.70 + type_boost
            s_behavior = 0.60 + feedback_boost
            if user_profile and "top_past_category" in user_profile:
                top_cat = user_profile["top_past_category"]
                if any(str(top_cat) in str(c) for c in cand_cols):
                    s_behavior += 0.25

            final_score = (w_beh * s_behavior) + (w_cont * s_content) + (w_ctx * s_context)
            final_score = round(min(max(float(final_score), 0.1), 0.99), 3)

            score_source = "neural_model"
            score_label = f"Context-Aware Neural Personalization ({active_model_version})"
            explanation = (
                f"Ranked by {active_model_version} neural gating (Beh: {w_beh:.0%}, Cont: {w_cont:.0%}, Ctx: {w_ctx:.0%}). "
                + cand["description"]
            )
        else:
            raw_score = base_suitability + col_boost + type_boost + feedback_boost
            final_score = round(min(max(raw_score, 0.40), 0.98), 3)
            score_source = "heuristic_suitability"
            score_label = "Cold-Start Statistical Suitability"
            explanation = (
                f"Cold-start suitability ({final_score:.2f}) determined from column cardinality and statistical distribution. "
                + cand["description"]
            )

        ranked_candidates.append({
            **cand,
            "recommendation_score": final_score,
            "score_source": score_source,
            "score_label": score_label,
            "explanation": explanation,
            "is_cold_start": is_cold_start,
            "gating_weights": gating_weights,
            "model_version": active_model_version,
            "model_type": model_type,
            "field_assignments": {
                "x_axis": cand.get("x_col"),
                "y_axis": cand.get("y_col"),
                "aggregation": cand.get("aggregation")
            }
        })

    # Sort descending by recommendation score
    ranked_candidates.sort(key=lambda x: x["recommendation_score"], reverse=True)
    top_recommendations = ranked_candidates[:top_k]

    return {
        "client_id": client_id,
        "user_id": user_id,
        "model_type": model_type,
        "is_cold_start": is_cold_start,
        "history_count": user_n_hist,
        "score_source": "neural_model" if neural_scores_used else "heuristic_suitability",
        "model_version": active_model_version,
        "gating_weights": gating_weights,
        "total_candidates": len(candidates),
        "recommendations": top_recommendations
    }
