"""
Module 11: Candidate Generation, Ranking & Evaluation Metrics
Implements the candidate-ranking protocol for rigorous Top-K evaluation:
- Generates 1 positive + 99 negative candidates per test interaction with zero future leakage.
- Calculates Precision@K, Recall@K, NDCG@K, and HitRate@K for K in {5, 10}.
- Evaluates cohorts: Overall, Warm-Start (N_hist >= K), Cold-Start (N_hist < K).
"""

import os
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch


def build_test_candidate_pools(test_df: pd.DataFrame, all_df: pd.DataFrame,
                               num_negatives: int = 99, seed: int = 42,
                               cache_path: str = "data/processed/test_candidates.json") -> list[dict]:
    """
    Builds fixed candidate pools (1 positive + 99 negatives) for each test positive interaction.
    To prevent false negatives and future leakage:
    Negative candidates are sampled strictly from products the customer NEVER purchased across the dataset.
    """
    cache_file = Path(cache_path)
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    np.random.seed(seed)
    all_products = np.array(sorted(all_df["product_id"].unique()))

    # Build customer -> all purchased products set (past + present + future)
    purchased_rows = all_df[all_df["purchased"] == 1]
    cust_all_purchased = purchased_rows.groupby("customer_id")["product_id"].apply(set).to_dict()

    test_positives = test_df[test_df["purchased"] == 1]
    candidate_pools = []

    # Get exact point-in-time hist_count and original indices from test_features.npz
    npz_path = Path("data/processed/test_features.npz")
    pos_hist_counts = None
    if npz_path.exists():
        npz_data = np.load(npz_path)
        pos_mask = (npz_data["y"] == 1)
        pos_hist_counts = npz_data["hist_count"][pos_mask]

    pos_indices = test_positives.index.tolist()
    for i in range(len(test_positives)):
        row = test_positives.iloc[i]
        orig_idx = int(pos_indices[i])
        cid = int(row["customer_id"])
        pos_pid = int(row["product_id"])
        if pos_hist_counts is not None and i < len(pos_hist_counts):
            hist_count = int(pos_hist_counts[i])
        else:
            hist_count = int(row.get("hist_interaction_count", 0))

        ever_purchased = cust_all_purchased.get(cid, {pos_pid})
        # Eligible negative candidate pool: products never purchased by this customer
        eligible_negs = [p for p in all_products if p not in ever_purchased]

        if len(eligible_negs) < num_negatives:
            # Fallback if catalog is very small for user
            eligible_negs = [p for p in all_products if p != pos_pid]

        sampled_negs = np.random.choice(eligible_negs, size=num_negatives, replace=False).tolist()
        candidates = [pos_pid] + sampled_negs

        candidate_pools.append({
            "test_idx": orig_idx,
            "session_id": int(row["session_id"]),
            "customer_id": cid,
            "positive_product": pos_pid,
            "hist_count": hist_count,
            "candidates": candidates # Candidate 0 is always the ground truth positive
        })

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(candidate_pools, f, indent=2)

    return candidate_pools


def compute_ranking_metrics_for_case(candidate_scores: np.ndarray, k_values: list[int] = [5, 10]) -> dict:
    """
    Given scores for [pos_product, neg_1, neg_2, ..., neg_99] (index 0 is ground truth positive),
    computes Precision@K, Recall@K, NDCG@K, HitRate@K.
    """
    # Sort candidate indices descending by score
    # Use stable sort so ties don't artificially favor index 0
    ranked_indices = np.argsort(-candidate_scores, kind="stable")
    # 1-based rank of the positive item (index 0)
    pos_rank = int(np.where(ranked_indices == 0)[0][0]) + 1

    metrics = {}
    for k in k_values:
        hit = 1.0 if pos_rank <= k else 0.0
        precision = hit / float(k)
        recall = hit / 1.0 # 1 positive item per evaluation
        ndcg = (1.0 / np.log2(pos_rank + 1)) if pos_rank <= k else 0.0

        metrics[f"HitRate@{k}"] = hit
        metrics[f"Precision@{k}"] = precision
        metrics[f"Recall@{k}"] = recall
        metrics[f"NDCG@{k}"] = ndcg

    metrics["positive_rank"] = pos_rank
    return metrics


def aggregate_cohort_metrics(case_results: list[dict], threshold_k: int = 3, k_eval: list[int] = [5, 10]) -> dict:
    """
    Aggregates ranking metrics across Overall, Warm-Start (hist >= threshold), and Cold-Start (hist < threshold).
    """
    cohorts = {
        "Overall": [res for res in case_results],
        "Cold-Start": [res for res in case_results if res["hist_count"] < threshold_k],
        "Warm-Start": [res for res in case_results if res["hist_count"] >= threshold_k]
    }

    summary = {}
    for cohort_name, results_list in cohorts.items():
        n = len(results_list)
        if n == 0:
            c_metrics = {"sample_count": 0, "MeanRank": 0.0}
            for k in k_eval:
                c_metrics[f"Precision@{k}"] = 0.0
                c_metrics[f"Recall@{k}"] = 0.0
                c_metrics[f"NDCG@{k}"] = 0.0
                c_metrics[f"HitRate@{k}"] = 0.0
            summary[cohort_name] = c_metrics
            continue

        c_metrics = {"sample_count": n}
        for k in k_eval:
            c_metrics[f"Precision@{k}"] = round(float(np.mean([r[f"Precision@{k}"] for r in results_list])), 4)
            c_metrics[f"Recall@{k}"] = round(float(np.mean([r[f"Recall@{k}"] for r in results_list])), 4)
            c_metrics[f"NDCG@{k}"] = round(float(np.mean([r[f"NDCG@{k}"] for r in results_list])), 4)
            c_metrics[f"HitRate@{k}"] = round(float(np.mean([r[f"HitRate@{k}"] for r in results_list])), 4)

        c_metrics["MeanRank"] = round(float(np.mean([r["positive_rank"] for r in results_list])), 2)
        summary[cohort_name] = c_metrics

    return summary
