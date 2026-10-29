"""
MLVis Visualization Recommendation Model
-----------------------------------------
PyTorch Neural Network and Baseline Recommenders for Visual Analytics.
Predicts chart type preferences and relevance scores from dataset meta-features
and column candidate profiles.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.mlvis_preprocessing import CANONICAL_CHART_TYPES, CHART_TYPE_TO_ID, NUMERIC_FEATURE_NAMES


class MLVisRecommenderNet(nn.Module):
    """
    Multi-layer Perceptron for chart recommendation given dataset/column meta-features.
    Outputs logit distribution over canonical chart types:
    [bar, line, scatter, box, histogram, heatmap]
    """

    def __init__(self, in_features: int = 26, num_classes: int = 6, hidden_dims: Tuple[int, ...] = (128, 64, 32)):
        super().__init__()
        layers = []
        curr_dim = in_features
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class StatisticalBaselineRecommender:
    """
    Statistical Rule-Based Baseline for visualization recommendation.
    Heuristically predicts chart suitability from column types, cardinality, and correlation.
    Used as an explicit baseline to compare against the trained neural recommender.
    """

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        X features order from NUMERIC_FEATURE_NAMES:
        0: ds_log_rows, 1: ds_num_cols, 2: ds_numeric_ratio, 3: ds_categorical_ratio
        4: x_is_num, 5: x_is_cat, 6: x_is_dt, 7: x_uniq_ratio, 8: x_miss, 9: x_ent, 10: x_skew, 11: x_mean
        12: y_is_num, 13: y_is_cat, 14: y_is_dt, 15: y_uniq_ratio, 16: y_miss, 17: y_ent, 18: y_skew, 19: y_mean
        20: xy_both_num, 21: xy_cat_num, 22: xy_num_cat, 23: xy_both_cat, 24: corr, 25: card_ratio
        """
        probs = np.zeros((len(X), len(CANONICAL_CHART_TYPES)), dtype=np.float32)

        for i, row in enumerate(X):
            # Default small prior
            score = np.array([0.2, 0.2, 0.2, 0.1, 0.1, 0.1], dtype=np.float32)

            x_num = row[4] > 0.5
            x_cat = row[5] > 0.5
            x_dt = row[6] > 0.5
            y_num = row[12] > 0.5
            both_num = row[20] > 0.5
            cat_num = row[21] > 0.5 or row[22] > 0.5
            both_cat = row[23] > 0.5
            corr = abs(row[24])

            # Rule 1: Time series / ordered -> Line chart
            if x_dt and y_num:
                score[CHART_TYPE_TO_ID["line"]] += 3.0

            # Rule 2: Both numeric with correlation or high points -> Scatter / Line
            if both_num:
                if corr > 0.4:
                    score[CHART_TYPE_TO_ID["line"]] += 2.0
                    score[CHART_TYPE_TO_ID["scatter"]] += 2.5
                else:
                    score[CHART_TYPE_TO_ID["scatter"]] += 2.5
                    score[CHART_TYPE_TO_ID["box"]] += 1.0

            # Rule 3: Categorical vs Numeric -> Bar chart or Box plot
            if cat_num:
                score[CHART_TYPE_TO_ID["bar"]] += 3.0
                score[CHART_TYPE_TO_ID["box"]] += 1.5

            # Rule 4: Both categorical -> Heatmap
            if both_cat:
                score[CHART_TYPE_TO_ID["heatmap"]] += 3.0
                score[CHART_TYPE_TO_ID["bar"]] += 1.0

            # Softmax normalize
            exp_s = np.exp(score - np.max(score))
            probs[i] = exp_s / np.sum(exp_s)

        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


class MLVisInferenceEngine:
    """
    High-level Inference Engine for the MLVis Recommender.
    Loads checkpoint, performs predictions on candidate column pairs,
    and returns ranked recommendations with explanations and confidence scores.
    """

    def __init__(self, model_dir: str = "models/mlvis"):
        self.model_dir = model_dir
        self.device = torch.device("cpu")
        self.model: Optional[MLVisRecommenderNet] = None
        self.meta: Dict[str, Any] = {}
        self.loaded = False
        self._load_model()

    def _load_model(self):
        ckpt_path = os.path.join(self.model_dir, "mlvis_recommender.pt")
        meta_path = os.path.join(self.model_dir, "mlvis_meta.json")

        if not os.path.exists(ckpt_path) or not os.path.exists(meta_path):
            return

        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        in_features = self.meta.get("in_features", len(NUMERIC_FEATURE_NAMES))
        num_classes = self.meta.get("num_classes", len(CANONICAL_CHART_TYPES))

        net = MLVisRecommenderNet(in_features=in_features, num_classes=num_classes)
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        net.load_state_dict(checkpoint["model_state_dict"])
        net.eval()
        self.model = net
        self.loaded = True

    def predict_chart_types(self, feature_vectors: np.ndarray) -> List[Dict[str, Any]]:
        """
        Takes an array of shape (N, 26) feature vectors.
        Returns list of prediction dicts:
        [{
           "predicted_type": "bar",
           "confidence": 0.82,
           "type_probabilities": {"bar": 0.82, "line": 0.10, ...}
        }, ...]
        """
        if not self.loaded or self.model is None:
            # Fall back to baseline
            baseline = StatisticalBaselineRecommender()
            probs = baseline.predict_proba(feature_vectors)
        else:
            with torch.no_grad():
                tensor_x = torch.tensor(feature_vectors, dtype=torch.float32).to(self.device)
                logits = self.model(tensor_x)
                probs = F.softmax(logits, dim=1).cpu().numpy()

        results = []
        for p in probs:
            best_idx = int(np.argmax(p))
            best_type = CANONICAL_CHART_TYPES[best_idx]
            best_conf = float(p[best_idx])
            prob_dict = {CANONICAL_CHART_TYPES[i]: float(p[i]) for i in range(len(CANONICAL_CHART_TYPES))}
            results.append({
                "predicted_type": best_type,
                "confidence": best_conf,
                "type_probabilities": prob_dict,
            })
        return results
