"""
Unit and Integration Tests for Context-Aware Multi-Modal Cold-Start Recommendation System
"""

import os
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_ingestion import ingest_data
from src.preprocessing import clean_and_split_data
from src.feature_engineering import verify_zero_leakage
from src.models.behavior_encoder import BehavioralEncoder
from src.models.content_encoder import ContentEncoder
from src.models.context_encoder import ContextEncoder
from src.models.fusion_model import NeuralGatedAttentionFusion
from src.models.cold_start import ColdStartAdaptiveGate
from src.models.recommender import TwoTowerMultiModalRecommender
from src.models.ranker import compute_ranking_metrics_for_case
from src.explainability import generate_recommendation_explanation
from src.feedback import FeedbackManager


class TestRecommenderPipeline(unittest.TestCase):

    def test_01_data_ingestion(self):
        """Verify data ingestion returns correct shape and generates metadata."""
        df = ingest_data()
        self.assertEqual(df.shape, (25000, 29))
        self.assertTrue(Path("outputs/dataset_metadata.json").exists())

    def test_02_preprocessing_and_split(self):
        """Verify clean splitting has zero duplicates and exact row totals."""
        df, train_df, val_df, test_df, outliers = clean_and_split_data()
        self.assertEqual(len(df), 25000)
        self.assertEqual(len(train_df) + len(val_df) + len(test_df), 25000)
        self.assertEqual(len(train_df), 14494)
        self.assertEqual(len(val_df), 4212)
        self.assertEqual(len(test_df), 6294)

    def test_03_leakage_audit(self):
        """Verify automated data leakage test returns 0 violations."""
        df = pd.read_csv("data/processed/cleaned_dataset.csv")
        from src.feature_engineering import build_point_in_time_behavioral_features
        df_feat = build_point_in_time_behavioral_features(df)
        audit = verify_zero_leakage(df_feat)
        self.assertEqual(audit["test_1_first_session_isolation"]["status"], "PASSED")
        self.assertEqual(audit["test_2_temporal_monotonicity"]["status"], "PASSED")
        self.assertEqual(audit["test_1_first_session_isolation"]["violations_detected"], 0)

    def test_04_encoders(self):
        """Verify tensor output dimensions for all 3 modality encoders."""
        beh_enc = BehavioralEncoder(16, 64)
        cont_enc = ContentEncoder(1000, 32, 11, 64)
        ctx_enc = ContextEncoder(21, 64)

        b_out = beh_enc(torch.randn(4, 16))
        c_in = torch.zeros(4, 12)
        c_in[:, 0] = torch.tensor([1, 2, 3, 4])
        c_out = cont_enc(c_in)
        x_out = ctx_enc(torch.randn(4, 21))

        self.assertEqual(b_out.shape, (4, 64))
        self.assertEqual(c_out.shape, (4, 64))
        self.assertEqual(x_out.shape, (4, 64))

    def test_05_cold_start_gating(self):
        """Verify cold-start adaptive gate suppresses behavioral weight when N_hist < K."""
        gate = ColdStartAdaptiveGate(primary_threshold=3, behavior_penalty=0.01)
        b = torch.randn(2, 64)
        c = torch.randn(2, 64)
        x = torch.randn(2, 64)
        weights = torch.tensor([[0.6, 0.2, 0.2], [0.6, 0.2, 0.2]])
        hist = torch.tensor([5, 0]) # User 0 is warm, User 1 is cold

        fused, adj_w, cold_m = gate(b, c, x, weights, hist)
        self.assertFalse(bool(cold_m[0].item()))
        self.assertTrue(bool(cold_m[1].item()))

        # Cold user behavioral weight must be suppressed to <= 0.01
        self.assertLessEqual(adj_w[1, 0].item(), 0.01)
        # Weights must sum to 1.0
        self.assertAlmostEqual(adj_w[1].sum().item(), 1.0, places=4)

    def test_05b_cold_warm_exact_boundary_tensor(self):
        """
        Explicit Tensor Boundary Test for ColdStartAdaptiveGate:
        Verifies N_hist in {0, 1, 2, 3, 4} with K=3 strictly produces:
        0 -> cold (True)
        1 -> cold (True)
        2 -> cold (True)
        3 -> warm (False)
        4 -> warm (False)
        """
        gate = ColdStartAdaptiveGate(primary_threshold=3, behavior_penalty=0.01)
        b = torch.randn(5, 64)
        c = torch.randn(5, 64)
        x = torch.randn(5, 64)
        weights = torch.full((5, 3), 1.0 / 3.0)
        hist = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)

        fused, adj_w, cold_m = gate(b, c, x, weights, hist)
        expected_mask = [True, True, True, False, False]
        for i, exp in enumerate(expected_mask):
            self.assertEqual(bool(cold_m[i].item()), exp, f"Failed for N_hist = {i}")
            if exp:
                # Cold cases must have behavior suppressed to <= 0.01
                self.assertLessEqual(adj_w[i, 0].item(), 0.01)
            else:
                # Warm cases preserve original weight (1/3 = ~0.33)
                self.assertGreater(adj_w[i, 0].item(), 0.3)

    def test_06_recommender_forward(self):
        """Verify Two-Tower Recommender forward pass and candidate scoring."""
        model = TwoTowerMultiModalRecommender(num_products=1000, embedding_dim=64)
        x_beh = torch.randn(2, 16)
        x_cont = torch.zeros(2, 12)
        x_cont[:, 0] = torch.tensor([10, 20])
        x_ctx = torch.randn(2, 21)
        hist = torch.tensor([4, 1])

        logits, weights, cold_m = model(x_beh, x_cont, x_ctx, hist, x_cont)
        self.assertEqual(logits.shape, (2,))
        self.assertEqual(weights.shape, (2, 3))

    def test_07_ranker_metrics(self):
        """Verify Top-K ranking metrics computation for ground truth ranks."""
        # Positive item has highest score (rank 1)
        scores_top1 = np.array([0.99] + [0.10] * 99)
        m1 = compute_ranking_metrics_for_case(scores_top1, k_values=[5, 10])
        self.assertEqual(m1["HitRate@5"], 1.0)
        self.assertEqual(m1["Precision@5"], 0.2)
        self.assertEqual(m1["NDCG@5"], 1.0)
        self.assertEqual(m1["positive_rank"], 1)

        # Positive item ranked outside top 5 (e.g. rank 6)
        scores_rank6 = np.array([0.50] + [0.90] * 5 + [0.10] * 94)
        m6 = compute_ranking_metrics_for_case(scores_rank6, k_values=[5, 10])
        self.assertEqual(m6["HitRate@5"], 0.0)
        self.assertEqual(m6["Precision@5"], 0.0)
        self.assertEqual(m6["positive_rank"], 6)

    def test_08_explainability(self):
        """Verify explainability engine produces grounded descriptions."""
        exp = generate_recommendation_explanation(
            customer_id=1024, product_id=844, product_category=2,
            hist_count=1, modality_weights=[0.01, 0.58, 0.41]
        )
        self.assertTrue(exp["is_cold_start"])
        self.assertIn("limited prior interaction history", exp["explanation"])

    def test_09_feedback_manager(self):
        """Verify recording and querying user feedback."""
        fb = FeedbackManager("outputs/test_feedback.csv")
        rec = fb.record_feedback(999, 123, 1, "Accept")
        self.assertEqual(rec["action"], "Accept")
        all_fb = fb.get_all_feedback()
        self.assertTrue(len(all_fb) > 0)
        if Path("outputs/test_feedback.csv").exists():
            Path("outputs/test_feedback.csv").unlink()


if __name__ == "__main__":
    unittest.main()
