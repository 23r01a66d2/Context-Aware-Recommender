"""
MLVis Model Training & Evaluation Script
----------------------------------------
Trains MLVisRecommenderNet on preprocessed MLVis splits.
Compares with Statistical Baseline across:
- Accuracy
- Macro F1
- Precision@K (K=1, 3)
- NDCG@K (K=3)
Saves model checkpoint, metadata, and evaluation report.
"""

import json
import logging
import os
import sys
from typing import Any, Dict

import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.mlvis_preprocessing import CANONICAL_CHART_TYPES
from src.models.mlvis_recommender import MLVisRecommenderNet, StatisticalBaselineRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_mlvis")


def compute_ndcg_at_k(probs: np.ndarray, y_true: np.ndarray, k: int = 3) -> float:
    """Computes Mean NDCG@K where the true class has relevance 1 and others 0."""
    ndcg_list = []
    for p, true_cls in zip(probs, y_true):
        top_k_indices = np.argsort(p)[::-1][:k]
        dcg = 0.0
        for rank, idx in enumerate(top_k_indices):
            rel = 1.0 if idx == true_cls else 0.0
            dcg += rel / np.log2(rank + 2)
        idcg = 1.0  # ideal DCG is 1.0 since only 1 correct class exists
        ndcg_list.append(dcg / idcg)
    return float(np.mean(ndcg_list))


def compute_precision_at_k(probs: np.ndarray, y_true: np.ndarray, k: int = 1) -> float:
    """Computes Top-K Accuracy / Hit-Ratio (Precision@K for single-positive items)."""
    hits = 0
    for p, true_cls in zip(probs, y_true):
        top_k_indices = np.argsort(p)[::-1][:k]
        if true_cls in top_k_indices:
            hits += 1
    return float(hits / len(y_true))


def evaluate_model(
    probs: np.ndarray, y_true: np.ndarray, model_name: str = "Model"
) -> Dict[str, Any]:
    """Computes standard classification & ranking metrics."""
    preds = np.argmax(probs, axis=1)
    acc = float(np.mean(preds == y_true))
    macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, preds, average="weighted", zero_division=0))
    p1 = compute_precision_at_k(probs, y_true, k=1)
    p3 = compute_precision_at_k(probs, y_true, k=3)
    ndcg3 = compute_ndcg_at_k(probs, y_true, k=3)

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
    }


def train_mlvis_recommender(
    data_path: str = "data/mlvis/processed/mlvis_splits.npz",
    model_dir: str = "models/mlvis",
    epochs: int = 35,
    batch_size: int = 64,
    lr: float = 0.003,
):
    os.makedirs(model_dir, exist_ok=True)
    logger.info(f"Loading preprocessed splits from {data_path}...")
    data = np.load(data_path)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    in_features = X_train.shape[1]
    num_classes = len(CANONICAL_CHART_TYPES)

    # 1. Evaluate Baseline Model on Test Set
    baseline = StatisticalBaselineRecommender()
    baseline_test_probs = baseline.predict_proba(X_test)
    baseline_metrics = evaluate_model(baseline_test_probs, y_test, model_name="Statistical Baseline")

    # 2. Train MLVis Neural Recommender
    # Calculate inverse class weights to handle imbalance
    class_counts = np.bincount(y_train, minlength=num_classes)
    total_samples = len(y_train)
    class_weights = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        class_weights[c] = total_samples / (num_classes * max(class_counts[c], 1))
    # Clip weights to avoid extreme gradients on near-zero classes
    class_weights = np.clip(class_weights, 0.2, 5.0)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.int64),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.int64),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = MLVisRecommenderNet(in_features=in_features, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

    best_val_f1 = 0.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(by)

        train_loss /= len(train_dataset)

        # Validation evaluation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for bx, by in val_loader:
                logits = model(bx)
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(by.cpu().numpy())

        val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)
        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = model.state_dict().copy()

        if epoch % 5 == 0 or epoch == epochs:
            logger.info(
                f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | "
                f"Val Macro F1: {val_f1:.4f} (Best: {best_val_f1:.4f})"
            )

    # Load best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # 3. Final Evaluation on Held-Out Test Set
    model.eval()
    with torch.no_grad():
        test_logits = model(torch.tensor(X_test, dtype=torch.float32))
        test_probs = torch.softmax(test_logits, dim=1).cpu().numpy()

    mlvis_metrics = evaluate_model(test_probs, y_test, model_name="MLVis Neural Recommender")

    # 4. Save Artifacts
    ckpt_path = os.path.join(model_dir, "mlvis_recommender.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_features": in_features,
            "num_classes": num_classes,
            "architecture": "MLVisRecommenderNet(26->128->64->32->6)",
            "classes": CANONICAL_CHART_TYPES,
        },
        ckpt_path,
    )

    report = {
        "model_version": "1.0.0",
        "dataset": "MLVis 1K Benchmark (kdd21-MLVis)",
        "task": "Visualization and Chart Type Recommendation",
        "training_samples": int(len(X_train)),
        "validation_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "mlvis_neural_model": mlvis_metrics,
        "statistical_baseline": baseline_metrics,
        "improvement": {
            "accuracy_delta": float(mlvis_metrics["accuracy"] - baseline_metrics["accuracy"]),
            "macro_f1_delta": float(mlvis_metrics["macro_f1"] - baseline_metrics["macro_f1"]),
            "ndcg_at_3_delta": float(mlvis_metrics["ndcg_at_3"] - baseline_metrics["ndcg_at_3"]),
        },
    }

    report_path = os.path.join(model_dir, "eval_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    meta_path = os.path.join(model_dir, "mlvis_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "in_features": in_features,
                "num_classes": num_classes,
                "classes": CANONICAL_CHART_TYPES,
                "version": "1.0.0",
                "metrics": mlvis_metrics,
            },
            f,
            indent=2,
        )

    logger.info(f"Model saved to {ckpt_path} and evaluation report to {report_path}")
    return report


if __name__ == "__main__":
    train_mlvis_recommender()
