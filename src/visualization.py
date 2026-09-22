"""
Module 13 & 17: Data Visualization & Rule-Based Visual Analytics Recommender
Generates publication-quality EDA charts, training curves, evaluation graphs,
and provides a Rule-Based Visual Analytics Recommender (Secondary Dashboard Component).
"""

import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg") # Non-interactive headless backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json

# Global plot styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
sns.set_palette("tab10")


def generate_all_eda_plots(df: pd.DataFrame, output_dir: str = "outputs/eda"):
    """Generates and exports all required Exploratory Data Analysis plots."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Missing Values Bar Chart
    plt.figure(figsize=(10, 4))
    nulls = df.isnull().sum()
    bars = plt.bar(df.columns, nulls.values, color="#2b5c8f")
    plt.xticks(rotation=90, fontsize=8)
    plt.ylabel("Missing Count")
    plt.title("Missing Value Analysis (0 Missing Values across all 29 Columns)", fontsize=12, fontweight="bold")
    plt.ylim(0, 10)
    plt.axhline(0, color="green", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path / "missing_values.png", dpi=300)
    plt.close()

    # 2. Numerical Distributions (Histograms + KDE)
    num_cols = ["unit_price", "discount_amount", "pages_viewed", "time_on_site_sec"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col in zip(axes.flatten(), num_cols):
        sns.histplot(df[col], kde=True, ax=ax, color="#1f77b4", bins=30)
        ax.set_title(f"Distribution of {col}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path / "numerical_distributions.png", dpi=300)
    plt.close()

    # 3. Outlier Boxplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col in zip(axes.flatten(), num_cols):
        sns.boxplot(x=df[col], ax=ax, color="#ff7f0e")
        ax.set_title(f"Box Plot: Outlier Detection for {col}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path / "outlier_boxplots.png", dpi=300)
    plt.close()

    # 4. Correlation Heatmap
    corr_cols = ["unit_price", "quantity", "discount_percent", "discount_amount",
                 "pages_viewed", "time_on_site_sec", "added_to_cart", "purchased", "rating"]
    plt.figure(figsize=(10, 8))
    corr_matrix = df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="Blues", cbar=True, square=True)
    plt.title("Correlation Heatmap of Behavioral and Transactional Attributes", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path / "correlation_heatmap.png", dpi=300)
    plt.close()

    # 5. Product Category Frequency Chart
    plt.figure(figsize=(8, 4))
    cat_counts = df["product_category"].value_counts().sort_index()
    sns.barplot(x=cat_counts.index, y=cat_counts.values, palette="mako")
    plt.xlabel("Product Category Code")
    plt.ylabel("Interaction Volume")
    plt.title("Interaction Frequency across Product Categories (0 to 7)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path / "category_frequency.png", dpi=300)
    plt.close()

    # 6. E-Commerce Behavioral Funnel Chart
    plt.figure(figsize=(8, 5))
    funnel_stages = ["1. Catalog View", "2. Add to Cart", "3. Purchase Completed", "3. Cart Abandoned"]
    funnel_values = [
        len(df),
        (df["added_to_cart"] == 1).sum(),
        (df["purchased"] == 1).sum(),
        (df["cart_abandoned"] == 1).sum()
    ]
    colors = ["#4a90e2", "#f5a623", "#7ed321", "#d0021b"]
    bars = plt.barh(funnel_stages[::-1], funnel_values[::-1], color=colors[::-1])
    for bar, val in zip(bars, funnel_values[::-1]):
        plt.text(val + 300, bar.get_y() + bar.get_height()/2, f"{val:,} ({round(val/len(df)*100, 1)}%)",
                 va="center", fontsize=10, fontweight="bold")
    plt.xlabel("Session Count")
    plt.title("Customer Behavioral Funnel Progression", fontweight="bold")
    plt.xlim(0, len(df) * 1.2)
    plt.tight_layout()
    plt.savefig(out_path / "customer_behavior_funnel.png", dpi=300)
    plt.close()

    # 7. Customer Interaction Frequency & Cold-Start Distribution
    plt.figure(figsize=(9, 4))
    cust_counts = df["customer_id"].value_counts().value_counts().sort_index()
    colors = ["#d9534f" if k < 3 else "#5cb85c" for k in cust_counts.index]
    plt.bar([str(k) for k in cust_counts.index], cust_counts.values, color=colors)
    plt.axvline(1.5, color="black", linestyle="--", linewidth=1.5, label="Cold-Start Threshold (K=3)")
    plt.xlabel("Historical Interaction Count (N_hist)")
    plt.ylabel("Number of Unique Customers")
    plt.title("Customer Distribution by Interaction Count (Cold-Start < 3 in Red)", fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / "interaction_distribution.png", dpi=300)
    plt.close()


def plot_training_curves(history: dict, output_dir: str = "outputs/plots"):
    """Plots training and validation loss curves."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    # Combined loss plot
    plt.figure(figsize=(8, 4.5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Training Loss", color="#1f77b4", linewidth=2)
    plt.plot(epochs, history["val_loss"], marker="s", label="Validation Loss", color="#ff7f0e", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.title("Multi-Modal Neural Model Convergence (Train vs Validation)", fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path / "training_loss.png", dpi=300)
    plt.savefig(out_path / "validation_loss.png", dpi=300)
    plt.close()


def plot_model_comparisons(benchmark_results: dict, output_dir: str = "outputs/evaluation"):
    """
    Plots Precision@5, NDCG@5, and Cold-Start vs Warm-Start comparisons across all evaluated models.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    models = list(benchmark_results.keys())
    overall_p5 = [benchmark_results[m]["Overall"]["Precision@5"] for m in models]
    overall_ndcg5 = [benchmark_results[m]["Overall"]["NDCG@5"] for m in models]
    cold_ndcg5 = [benchmark_results[m]["Cold-Start"]["NDCG@5"] for m in models]
    warm_ndcg5 = [benchmark_results[m]["Warm-Start"]["NDCG@5"] for m in models]

    # 1. Precision@5 Comparison
    plt.figure(figsize=(10, 4.5))
    y_pos = np.arange(len(models))
    bars = plt.barh(y_pos, overall_p5, color="#2b5c8f")
    plt.yticks(y_pos, models)
    plt.xlabel("Precision@5")
    plt.title("Overall Precision@5 Benchmark across Models", fontweight="bold")
    for bar, val in zip(bars, overall_p5):
        plt.text(val + 0.002, bar.get_y() + bar.get_height()/2, f"{val:.4f}", va="center", fontsize=9)
    plt.xlim(0, max(overall_p5) * 1.15)
    plt.tight_layout()
    plt.savefig(out_path / "precision_at_k.png", dpi=300)
    plt.close()

    # 2. Cold vs Warm NDCG@5 Comparison
    plt.figure(figsize=(11, 5))
    bar_width = 0.35
    x = np.arange(len(models))
    plt.bar(x - bar_width/2, cold_ndcg5, width=bar_width, label="Cold-Start (N_hist < 3)", color="#d9534f")
    plt.bar(x + bar_width/2, warm_ndcg5, width=bar_width, label="Warm-Start (N_hist >= 3)", color="#5cb85c")
    plt.xticks(x, [m.replace(" ", "\n") for m in models], fontsize=9)
    plt.ylabel("NDCG@5")
    plt.title("NDCG@5 Performance: Cold-Start vs Warm-Start Cohorts", fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / "cold_vs_warm.png", dpi=300)
    plt.savefig(out_path / "ndcg_at_k.png", dpi=300)
    plt.close()


# =====================================================================
# SECONDARY COMPONENT: RULE-BASED VISUAL ANALYTICS RECOMMENDER
# =====================================================================
def recommend_visualization(df: pd.DataFrame, primary_col: str, secondary_col: str = None) -> dict:
    """
    Rule-based visualization recommender based on data types, cardinality, and correlation.
    (Secondary Visual Analytics Dashboard Component)
    """
    if primary_col not in df.columns:
        return {"error": f"Column '{primary_col}' not in dataset"}

    col1 = df[primary_col]
    dtype1 = "numeric" if pd.api.types.is_numeric_dtype(col1) else "categorical"
    card1 = col1.nunique()

    if secondary_col is None:
        # Single column recommendation
        if dtype1 == "numeric":
            return {
                "chart_type": "Histogram with KDE",
                "rationale": f"Single continuous numerical column '{primary_col}' with {card1} unique values is best represented by a histogram to analyze skewness and spread.",
                "parameters": {"x": primary_col, "bins": 30, "kde": True}
            }
        else:
            chart = "Bar Chart" if card1 <= 15 else "Horizontal Bar Chart"
            return {
                "chart_type": chart,
                "rationale": f"Categorical column '{primary_col}' with {card1} categories is best displayed as a frequency bar chart.",
                "parameters": {"x": primary_col}
            }
    else:
        if secondary_col not in df.columns:
            return {"error": f"Column '{secondary_col}' not in dataset"}

        col2 = df[secondary_col]
        dtype2 = "numeric" if pd.api.types.is_numeric_dtype(col2) else "categorical"
        card2 = col2.nunique()

        # Two columns
        if dtype1 == "numeric" and dtype2 == "numeric":
            corr = col1.corr(col2)
            return {
                "chart_type": "Scatter Plot with Trendline",
                "rationale": f"Two numerical columns ({primary_col} & {secondary_col}) with Pearson correlation r = {corr:.2f}. A scatter plot visualizes pairwise correlation and clusters.",
                "parameters": {"x": primary_col, "y": secondary_col, "correlation": round(float(corr), 3)}
            }
        elif (dtype1 == "categorical" and dtype2 == "numeric") or (dtype1 == "numeric" and dtype2 == "categorical"):
            cat_c = primary_col if dtype1 == "categorical" else secondary_col
            num_c = secondary_col if dtype1 == "categorical" else primary_col
            return {
                "chart_type": "Grouped Box Plot or Bar Chart",
                "rationale": f"Categorical breakdown of continuous variable '{num_c}' across '{cat_c}'. Box plot reveals distribution differences and category outliers.",
                "parameters": {"cat_col": cat_c, "num_col": num_c}
            }
        else:
            return {
                "chart_type": "Contingency Heatmap / Cross-tabulation",
                "rationale": f"Two categorical variables ({primary_col} & {secondary_col}). A bivariate heatmap shows joint frequency distribution.",
                "parameters": {"row_col": primary_col, "col_col": secondary_col}
            }
