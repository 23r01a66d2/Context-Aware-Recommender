"""
MLVis Model Fine-Tuning and Enhanced Evaluation Script (v2.0.0)
---------------------------------------------------------------
Evaluates and trains an enhanced MLVis Visualization Recommender (v2.0.0)
incorporating the synthetic visualization benchmark (600 records) to address
class imbalance (histograms and heatmaps) while preserving 100% backward
compatibility and holding out distinct test sets for:
1. Real-World MLVis 1K Test Set (kdd21-MLVis held-out test split)
2. Synthetic Visualization Test Set (held-out archetypes, strictly unobserved)
3. Combined Comprehensive Benchmark

Outputs:
- models/mlvis_v2/mlvis_recommender.pt (New model checkpoint)
- models/mlvis_v2/mlvis_meta.json (New model metadata)
- models/mlvis_v2/eval_report_comparison.json (Comprehensive 3-way evaluation report)
"""

import json
import logging
import os
import sys
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_score
from sklearn.model_selection import GroupShuffleSplit
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

if "." not in sys.path:
    sys.path.insert(0, ".")

from src.mlvis_preprocessing import CANONICAL_CHART_TYPES
from src.models.mlvis_recommender import (
    MLVisRecommenderNet,
    StatisticalBaselineRecommender,
)
from train_mlvis import compute_ndcg_at_k, compute_precision_at_k

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_mlvis_v2")


def evaluate_metrics(
    probs: np.ndarray, y_true: np.ndarray, model_name: str = "Model"
) -> Dict[str, Any]:
    """Computes standard multi-class classification and ranking metrics."""
    preds = np.argmax(probs, axis=1)
    acc = float(np.mean(preds == y_true))
    macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, preds, average="weighted", zero_division=0))
    p1 = compute_precision_at_k(probs, y_true, k=1)
    p3 = compute_precision_at_k(probs, y_true, k=3)
    ndcg3 = compute_ndcg_at_k(probs, y_true, k=3)

    # Per-class metrics
    class_report = classification_report(
        y_true,
        preds,
        labels=list(range(len(CANONICAL_CHART_TYPES))),
        target_names=CANONICAL_CHART_TYPES,
        output_dict=True,
        zero_division=0,
    )

    per_class = {}
    for c in CANONICAL_CHART_TYPES:
        if c in class_report:
            per_class[c] = {
                "precision": float(class_report[c]["precision"]),
                "recall": float(class_report[c]["recall"]),
                "f1_score": float(class_report[c]["f1-score"]),
                "support": int(class_report[c]["support"]),
            }

    logger.info(
        f"[{model_name}] Acc: {acc:.4f} | Macro F1: {macro_f1:.4f} | "
        f"P@1: {p1:.4f} | P@3: {p3:.4f} | NDCG@3: {ndcg3:.4f}"
    )

    return {
        "model_name": model_name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "precision_at_1": p1,
        "precision_at_3": p3,
        "ndcg_at_3": ndcg3,
        "per_class": per_class,
    }


def split_synthetic_data(
    synth_path: str = "data/mlvis/synthetic/synthetic_features.npz",
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs GroupShuffleSplit on synthetic data by archetype ID.
    Guarantees that test archetypes are never seen during training or validation.
    70% train (42 archetypes, ~420 records), 15% val (9 archetypes, ~90 records), 15% test (9 archetypes, ~90 records).
    """
    data = np.load(synth_path)
    X = data["X"]
    y = data["y"]
    groups = data["groups"]

    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    train_val_idx, test_idx = next(gss_test.split(X, y, groups))

    X_train_val, y_train_val, g_train_val = X[train_val_idx], y[train_val_idx], groups[train_val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=seed)
    train_idx, val_idx = next(gss_val.split(X_train_val, y_train_val, g_train_val))

    X_train, y_train = X_train_val[train_idx], y_train_val[train_idx]
    X_val, y_val = X_train_val[val_idx], y_train_val[val_idx]

    logger.info(
        f"Synthetic split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)} records (no archetype leakage)."
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def run_evaluation_and_training(
    real_data_path: str = "data/mlvis/processed/mlvis_splits.npz",
    synth_data_path: str = "data/mlvis/synthetic/synthetic_features.npz",
    existing_model_dir: str = "models/mlvis",
    new_model_dir: str = "models/mlvis_v2",
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 0.002,
):
    os.makedirs(new_model_dir, exist_ok=True)

    # 1. Load Real MLVis Data
    logger.info(f"Loading real MLVis splits from {real_data_path}...")
    real_data = np.load(real_data_path)
    X_real_train, y_real_train = real_data["X_train"], real_data["y_train"]
    X_real_val, y_real_val = real_data["X_val"], real_data["y_val"]
    X_real_test, y_real_test = real_data["X_test"], real_data["y_test"]

    # 2. Load and Split Synthetic Data (by archetype groups)
    X_synth_train, y_synth_train, X_synth_val, y_synth_val, X_synth_test, y_synth_test = (
        split_synthetic_data(synth_data_path)
    )

    in_features = X_real_train.shape[1]
    num_classes = len(CANONICAL_CHART_TYPES)

    # 3. Load Existing Production Checkpoint v1.0.0
    existing_ckpt_path = os.path.join(existing_model_dir, "mlvis_recommender.pt")
    model_v1 = MLVisRecommenderNet(in_features=in_features, num_classes=num_classes)
    ckpt_v1 = torch.load(existing_ckpt_path, map_location="cpu")
    model_v1.load_state_dict(ckpt_v1["model_state_dict"])
    model_v1.eval()

    # Baseline Model
    baseline = StatisticalBaselineRecommender()

    # --- STEP 1: Evaluate Existing Models on Real Test Set ---
    with torch.no_grad():
        v1_real_probs = torch.softmax(model_v1(torch.tensor(X_real_test, dtype=torch.float32)), dim=1).numpy()
    baseline_real_probs = baseline.predict_proba(X_real_test)

    eval_v1_on_real = evaluate_metrics(v1_real_probs, y_real_test, model_name="MLVis v1.0.0 on Real Test")
    eval_base_on_real = evaluate_metrics(baseline_real_probs, y_real_test, model_name="Statistical Baseline on Real Test")

    # --- Evaluate on Synthetic Test Set ---
    with torch.no_grad():
        v1_synth_probs = torch.softmax(model_v1(torch.tensor(X_synth_test, dtype=torch.float32)), dim=1).numpy()
    baseline_synth_probs = baseline.predict_proba(X_synth_test)

    eval_v1_on_synth = evaluate_metrics(v1_synth_probs, y_synth_test, model_name="MLVis v1.0.0 on Synthetic Test")
    eval_base_on_synth = evaluate_metrics(baseline_synth_probs, y_synth_test, model_name="Statistical Baseline on Synthetic Test")

    # --- STEP 2: Train Model v2.0.0 with Balanced Augmentation ---
    # Merge real training data with synthetic training records
    X_comb_train = np.vstack([X_real_train, X_synth_train])
    y_comb_train = np.concatenate([y_real_train, y_synth_train])

    X_comb_val = np.vstack([X_real_val, X_synth_val])
    y_comb_val = np.concatenate([y_real_val, y_synth_val])

    logger.info(
        f"Combined training set: {len(X_comb_train)} samples "
        f"({len(X_real_train)} real + {len(X_synth_train)} synthetic)."
    )

    # Inverse frequency weighting with smoothing
    class_counts = np.bincount(y_comb_train, minlength=num_classes)
    total_samples = len(y_comb_train)
    class_weights = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        class_weights[c] = total_samples / (num_classes * max(class_counts[c], 1))
    class_weights = np.clip(class_weights, 0.3, 4.0)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32)

    # Initialize v2 model weights from v1.0.0 (warm-start fine-tuning)
    model_v2 = MLVisRecommenderNet(in_features=in_features, num_classes=num_classes)
    model_v2.load_state_dict(ckpt_v1["model_state_dict"])

    train_dataset = TensorDataset(
        torch.tensor(X_comb_train, dtype=torch.float32),
        torch.tensor(y_comb_train, dtype=torch.int64),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_comb_val, dtype=torch.float32),
        torch.tensor(y_comb_val, dtype=torch.int64),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = optim.AdamW(model_v2.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

    best_val_f1 = 0.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        model_v2.train()
        train_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model_v2(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(by)

        train_loss /= len(train_dataset)

        # Validation evaluation
        model_v2.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for bx, by in val_loader:
                logits = model_v2(bx)
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(by.cpu().numpy())

        val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)
        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = {k: v.cpu().clone() for k, v in model_v2.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs:
            logger.info(
                f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | "
                f"Val Macro F1: {val_f1:.4f} (Best: {best_val_f1:.4f})"
            )

    if best_weights is not None:
        model_v2.load_state_dict(best_weights)

    # --- STEP 3: Final Evaluation of v2.0.0 on Separate Test Sets ---
    model_v2.eval()

    # 1. On Real Test Set (strictly real-world MLVis benchmark)
    with torch.no_grad():
        v2_real_probs = torch.softmax(model_v2(torch.tensor(X_real_test, dtype=torch.float32)), dim=1).numpy()
    eval_v2_on_real = evaluate_metrics(v2_real_probs, y_real_test, model_name="MLVis v2.0.0 on Real Test")

    # 2. On Synthetic Test Set (held-out synthetic archetypes)
    with torch.no_grad():
        v2_synth_probs = torch.softmax(model_v2(torch.tensor(X_synth_test, dtype=torch.float32)), dim=1).numpy()
    eval_v2_on_synth = evaluate_metrics(v2_synth_probs, y_synth_test, model_name="MLVis v2.0.0 on Synthetic Test")

    # 3. On Combined Test Set
    X_comb_test = np.vstack([X_real_test, X_synth_test])
    y_comb_test = np.concatenate([y_real_test, y_synth_test])
    with torch.no_grad():
        v2_comb_probs = torch.softmax(model_v2(torch.tensor(X_comb_test, dtype=torch.float32)), dim=1).numpy()
    eval_v2_on_comb = evaluate_metrics(v2_comb_probs, y_comb_test, model_name="MLVis v2.0.0 on Combined Test")

    # --- STEP 4: Save Model v2 Checkpoint, Metadata, and Comparison Report ---
    ckpt_v2_path = os.path.join(new_model_dir, "mlvis_recommender.pt")
    torch.save(
        {
            "model_state_dict": model_v2.state_dict(),
            "in_features": in_features,
            "num_classes": num_classes,
            "architecture": "MLVisRecommenderNet(26->128->64->32->6)",
            "classes": CANONICAL_CHART_TYPES,
            "version": "2.0.0",
        },
        ckpt_v2_path,
    )

    meta_v2_path = os.path.join(new_model_dir, "mlvis_meta.json")
    with open(meta_v2_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "in_features": in_features,
                "num_classes": num_classes,
                "classes": CANONICAL_CHART_TYPES,
                "version": "2.0.0",
                "metrics": eval_v2_on_real,
                "metrics_synthetic": eval_v2_on_synth,
                "metrics_combined": eval_v2_on_comb,
            },
            f,
            indent=2,
        )

    comparison_report = {
        "report_title": "MLVis Visualization Recommendation Model Evaluation & Comparison",
        "benchmark_real": "kdd21-MLVis 1K Benchmark (Real-world charts)",
        "benchmark_synthetic": "Synthetic Visualization Benchmark 600 (Documented rules, 60 archetypes)",
        "samples": {
            "real_train": int(len(X_real_train)),
            "real_test": int(len(X_real_test)),
            "synthetic_train": int(len(X_synth_train)),
            "synthetic_test": int(len(X_synth_test)),
        },
        "evaluation_on_real_test_set": {
            "statistical_baseline": eval_base_on_real,
            "mlvis_v1": eval_v1_on_real,
            "mlvis_v2": eval_v2_on_real,
            "v2_vs_v1_improvement": {
                "accuracy_delta": float(eval_v2_on_real["accuracy"] - eval_v1_on_real["accuracy"]),
                "macro_f1_delta": float(eval_v2_on_real["macro_f1"] - eval_v1_on_real["macro_f1"]),
                "ndcg_at_3_delta": float(eval_v2_on_real["ndcg_at_3"] - eval_v1_on_real["ndcg_at_3"]),
            },
        },
        "evaluation_on_synthetic_test_set": {
            "statistical_baseline": eval_base_on_synth,
            "mlvis_v1": eval_v1_on_synth,
            "mlvis_v2": eval_v2_on_synth,
            "v2_vs_v1_improvement": {
                "accuracy_delta": float(eval_v2_on_synth["accuracy"] - eval_v1_on_synth["accuracy"]),
                "macro_f1_delta": float(eval_v2_on_synth["macro_f1"] - eval_v1_on_synth["macro_f1"]),
                "ndcg_at_3_delta": float(eval_v2_on_synth["ndcg_at_3"] - eval_v1_on_synth["ndcg_at_3"]),
            },
        },
        "evaluation_on_combined_test_set": {
            "mlvis_v2": eval_v2_on_comb,
        },
    }

    report_path = os.path.join(new_model_dir, "eval_report_comparison.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(comparison_report, f, indent=2)

    logger.info(f"Model v2 saved to {ckpt_v2_path}")
    logger.info(f"Metadata saved to {meta_v2_path}")
    logger.info(f"Evaluation report saved to {report_path}")

    return comparison_report


if __name__ == "__main__":
    run_evaluation_and_training()
