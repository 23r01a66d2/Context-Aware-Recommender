"""
Module 10: Recommendation Models & Baselines
Implements the Primary Research Component: Two-Tower Neural Multi-Modal Recommender with
Gated Attention and Cold-Start Adaptive Gating, alongside competitive baselines:
1. Popularity Recommender
2. Content-Based Recommender
3. Collaborative Filtering Recommender (Matrix Factorization)
4. Multi-Modal Fusion WITHOUT Cold-Start Adaptation
5. Proposed Multi-Modal Fusion WITH Cold-Start Adaptive Gating
"""

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

from src.models.behavior_encoder import BehavioralEncoder
from src.models.content_encoder import ContentEncoder
from src.models.context_encoder import ContextEncoder
from src.models.fusion_model import NeuralGatedAttentionFusion, SimpleConcatFusion
from src.models.cold_start import ColdStartAdaptiveGate


# =====================================================================
# 1. PRIMARY RESEARCH MODEL: TWO-TOWER MULTI-MODAL RECOMMENDER
# =====================================================================
class TwoTowerMultiModalRecommender(nn.Module):
    """
    Two-Tower Recommendation Architecture:
    - User/Context Tower: Encodes behavior, context, and session-anchor content via Gated Attention.
    - Product Tower: Encodes candidate product entity and metadata.
    - Scoring: Scaled dot-product similarity between User and Product representations.
    """

    def __init__(self, num_products: int = 1000, beh_dim: int = 16, ctx_dim: int = 21,
                 cont_meta_dim: int = 11, embedding_dim: int = 64, dropout: float = 0.2,
                 apply_cold_start_adaptation: bool = True, cold_threshold: int = 3,
                 behavior_penalty: float = 0.01):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.apply_cold_start_adaptation = apply_cold_start_adaptation

        # Modality Encoders
        self.behavior_encoder = BehavioralEncoder(input_dim=beh_dim, embedding_dim=embedding_dim, dropout=dropout)
        self.content_encoder = ContentEncoder(num_products=num_products, embedding_dim=embedding_dim, dropout=dropout)
        self.context_encoder = ContextEncoder(input_dim=ctx_dim, embedding_dim=embedding_dim, dropout=dropout)

        # Fusion & Cold-Start Mechanisms
        self.fusion_network = NeuralGatedAttentionFusion(embedding_dim=embedding_dim, hidden_dim=64)
        self.cold_start_gate = ColdStartAdaptiveGate(
            primary_threshold=cold_threshold,
            behavior_penalty=behavior_penalty,
            embedding_dim=embedding_dim
        )

        # Scale factor for dot product
        self.scale = math.sqrt(embedding_dim)

    def encode_user_context(self, x_beh: torch.Tensor, x_cont_anchor: torch.Tensor,
                            x_ctx: torch.Tensor, hist_counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encodes user, context, and anchor content into a single fused representation.
        Returns: (e_user, final_weights, cold_mask)
        """
        e_beh = self.behavior_encoder(x_beh)
        e_cont = self.content_encoder(x_cont_anchor)
        e_ctx = self.context_encoder(x_ctx)

        fused_raw, raw_weights = self.fusion_network(e_beh, e_cont, e_ctx)

        if self.apply_cold_start_adaptation:
            e_user, final_weights, cold_mask = self.cold_start_gate(
                e_beh, e_cont, e_ctx, raw_weights, hist_counts
            )
        else:
            e_user = fused_raw
            final_weights = raw_weights
            b_size = hist_counts.shape[0]
            cold_mask = (hist_counts < self.cold_start_gate.threshold).view(b_size, 1)

        return e_user, final_weights, cold_mask

    def encode_candidate_product(self, x_candidate: torch.Tensor) -> torch.Tensor:
        """Encodes candidate product through the Product Tower."""
        return self.content_encoder(x_candidate)

    def forward(self, x_beh: torch.Tensor, x_cont_anchor: torch.Tensor,
                x_ctx: torch.Tensor, hist_counts: torch.Tensor,
                x_candidate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes recommendation relevance logit for batch of (user, context, candidate product).
        Returns: (logits, final_weights, cold_mask)
        """
        e_user, weights, cold_mask = self.encode_user_context(x_beh, x_cont_anchor, x_ctx, hist_counts)
        e_prod = self.encode_candidate_product(x_candidate)

        # Scaled dot product
        logits = torch.sum(e_user * e_prod, dim=-1) / self.scale # [B]
        return logits, weights, cold_mask

    def score_candidates(self, e_user: torch.Tensor, x_candidates: torch.Tensor) -> torch.Tensor:
        """
        Fast scoring of M candidate products against 1 user representation.
        Args:
            e_user: [1, d] or [B, d]
            x_candidates: [M, 12]
        Returns:
            scores: [M] or [B, M] probabilities
        """
        e_prod = self.encode_candidate_product(x_candidates) # [M, d]
        logits = torch.matmul(e_user, e_prod.t()) / self.scale # [B, M]
        probs = torch.sigmoid(logits)
        return probs.squeeze(0) if e_user.shape[0] == 1 else probs


# =====================================================================
# 2. COMPETITIVE BASELINE 1: POPULARITY RECOMMENDER
# =====================================================================
class PopularityRecommender:
    """
    Ranks products globally by purchase volume and conversion rate
    calculated strictly on the training set.
    """

    def __init__(self):
        self.product_scores = {}

    def fit(self, train_df: pd.DataFrame):
        purchased_counts = train_df[train_df["purchased"] == 1]["product_id"].value_counts()
        total_counts = train_df["product_id"].value_counts()

        max_purchases = purchased_counts.max() if not purchased_counts.empty else 1.0
        for pid in total_counts.index:
            p_purch = purchased_counts.get(pid, 0)
            conversion = p_purch / total_counts[pid]
            # Combined score: normalized purchase volume * conversion rate
            self.product_scores[pid] = (p_purch / max_purchases) * 0.7 + conversion * 0.3

    def score_candidates(self, candidate_pids: list[int]) -> np.ndarray:
        return np.array([self.product_scores.get(pid, 0.0) for pid in candidate_pids], dtype=np.float32)


# =====================================================================
# 3. COMPETITIVE BASELINE 2: CONTENT-BASED RECOMMENDER
# =====================================================================
class ContentBasedRecommender:
    """
    Computes profile-product similarity using historical category preferences and price affinity.
    Falls back to popular category items for cold users.
    """

    def __init__(self, num_products: int = 1000):
        self.num_products = num_products
        self.product_catalog = {} # pid -> {'category': int, 'price': float}
        self.popular_by_category = {}

    def fit(self, train_df: pd.DataFrame):
        for _, row in train_df.iterrows():
            pid = int(row["product_id"])
            if pid not in self.product_catalog:
                self.product_catalog[pid] = {
                    "category": int(row["product_category"]),
                    "price": float(row["unit_price"])
                }
        # Popularity per category
        train_pos = train_df[train_df["purchased"] == 1]
        for cat in range(8):
            cat_pos = train_pos[train_pos["product_category"] == cat]["product_id"].value_counts()
            self.popular_by_category[cat] = cat_pos.index.tolist()

    def score_candidates(self, customer_hist_cat_prefs: np.ndarray,
                         candidate_pids: list[int], is_cold: bool) -> np.ndarray:
        scores = np.zeros(len(candidate_pids), dtype=np.float32)
        for idx, pid in enumerate(candidate_pids):
            p_info = self.product_catalog.get(pid, {"category": 0, "price": 500.0})
            cat = p_info["category"]
            if is_cold or customer_hist_cat_prefs.sum() == 0:
                # Cold user: uniform preference across categories + category rank
                rank = self.popular_by_category.get(cat, []).index(pid) if pid in self.popular_by_category.get(cat, []) else 99
                scores[idx] = 1.0 / (1.0 + rank)
            else:
                # Warm user: category preference affinity
                scores[idx] = customer_hist_cat_prefs[cat]
        return scores


# =====================================================================
# 4. COMPETITIVE BASELINE 3: COLLABORATIVE FILTERING (SVD)
# =====================================================================
class CollaborativeFilteringRecommender:
    """
    Matrix Factorization (Truncated SVD) on User-Product Purchase Matrix.
    Cold users with 0 prior interactions seamlessly fall back to popularity.
    """

    def __init__(self, n_factors: int = 32):
        self.n_factors = n_factors
        self.svd = TruncatedSVD(n_components=n_factors, random_state=42)
        self.user_to_idx = {}
        self.prod_to_idx = {}
        self.idx_to_prod = {}
        self.item_factors = None
        self.user_factors = None
        self.popularity_fallback = PopularityRecommender()

    def fit(self, train_df: pd.DataFrame):
        self.popularity_fallback.fit(train_df)

        pos_df = train_df[train_df["purchased"] == 1]
        unique_users = pos_df["customer_id"].unique()
        unique_prods = train_df["product_id"].unique()

        self.user_to_idx = {uid: i for i, uid in enumerate(unique_users)}
        self.prod_to_idx = {pid: i for i, pid in enumerate(unique_prods)}
        self.idx_to_prod = {i: pid for pid, i in self.prod_to_idx.items()}

        rows = [self.user_to_idx[uid] for uid in pos_df["customer_id"]]
        cols = [self.prod_to_idx[pid] for pid in pos_df["product_id"]]
        data = np.ones(len(rows), dtype=np.float32)

        mat = csr_matrix((data, (rows, cols)), shape=(len(unique_users), len(unique_prods)))
        self.user_factors = self.svd.fit_transform(mat)
        self.item_factors = self.svd.components_.T # [n_items, n_factors]

    def score_candidates(self, customer_id: int, candidate_pids: list[int]) -> np.ndarray:
        if customer_id not in self.user_to_idx or self.item_factors is None:
            # Cold user: fall back to popularity
            return self.popularity_fallback.score_candidates(candidate_pids)

        u_idx = self.user_to_idx[customer_id]
        u_vec = self.user_factors[u_idx] # [n_factors]

        scores = []
        for pid in candidate_pids:
            if pid in self.prod_to_idx:
                p_idx = self.prod_to_idx[pid]
                p_vec = self.item_factors[p_idx]
                scores.append(float(np.dot(u_vec, p_vec)))
            else:
                scores.append(0.0)
        return np.array(scores, dtype=np.float32)
