"""
Tests for MLVis Visualization Recommender Model, Feature Schema, Preprocessing,
Baseline Comparison, and API Model Selection.
"""

import os
import json
import unittest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend.main import app
from src.mlvis_preprocessing import (
    CANONICAL_CHART_TYPES,
    NUMERIC_FEATURE_NAMES,
    compute_column_meta,
    extract_pair_features,
    resolve_chart_type,
    parse_src_tag
)
from src.models.mlvis_recommender import (
    MLVisRecommenderNet,
    StatisticalBaselineRecommender,
    MLVisInferenceEngine
)
from src.visual_analytics import (
    build_visualization_catalog,
    rank_visualization_recommendations
)


class TestMLVisRecommender(unittest.TestCase):
    """Test suite validating MLVis preprocessing, inference, baseline, and API."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.test_df = pd.DataFrame({
            "product_id": [f"P_{i}" for i in range(100)],
            "category": ["Electronics", "Clothing", "Home", "Books"] * 25,
            "unit_price": [19.99 + (i * 2.5) for i in range(100)],
            "sales_count": [int(10 + (i % 7) * 3) for i in range(100)],
            "date": pd.date_range("2026-01-01", periods=100).astype(str)
        })

    def test_01_canonical_chart_types_and_schema(self):
        """Verify schema defines all 6 canonical chart types and 26 numeric features."""
        self.assertEqual(len(CANONICAL_CHART_TYPES), 6)
        self.assertIn("bar", CANONICAL_CHART_TYPES)
        self.assertIn("scatter", CANONICAL_CHART_TYPES)
        self.assertIn("line", CANONICAL_CHART_TYPES)
        self.assertIn("histogram", CANONICAL_CHART_TYPES)
        self.assertIn("box", CANONICAL_CHART_TYPES)
        self.assertIn("heatmap", CANONICAL_CHART_TYPES)
        self.assertEqual(len(NUMERIC_FEATURE_NAMES), 26)

    def test_02_column_meta_extraction(self):
        """Verify compute_column_meta correctly distinguishes numeric, categorical, and datetime."""
        num_meta = compute_column_meta(self.test_df["unit_price"])
        self.assertEqual(num_meta["is_numeric"], 1.0)
        self.assertEqual(num_meta["is_categorical"], 0.0)

        cat_meta = compute_column_meta(self.test_df["category"])
        self.assertEqual(cat_meta["is_numeric"], 0.0)
        self.assertEqual(cat_meta["is_categorical"], 1.0)

        dt_meta = compute_column_meta(self.test_df["date"])
        self.assertEqual(dt_meta["is_datetime"], 1.0)

    def test_03_pair_feature_vector_dimension(self):
        """Verify extract_pair_features produces exact 26-dimensional vector."""
        ds_meta = {"ds_log_rows": 0.3, "ds_num_cols": 0.05, "ds_numeric_ratio": 0.5, "ds_categorical_ratio": 0.5}
        x_m = compute_column_meta(self.test_df["category"])
        y_m = compute_column_meta(self.test_df["unit_price"])
        feats = extract_pair_features(ds_meta, x_m, y_m, correlation=0.0, cardinality_ratio=0.1)

        self.assertEqual(len(feats), 26)
        self.assertTrue(all(isinstance(v, float) for v in feats))

    def test_04_statistical_baseline_predictions(self):
        """Verify StatisticalBaselineRecommender produces valid probability distributions."""
        baseline = StatisticalBaselineRecommender()
        dummy_x = np.zeros((3, 26), dtype=np.float32)
        # Simulate categorical vs numeric
        dummy_x[0, 21] = 1.0
        probs = baseline.predict_proba(dummy_x)

        self.assertEqual(probs.shape, (3, 6))
        # Probabilities should sum to 1.0 per row
        for p_row in probs:
            self.assertAlmostEqual(float(np.sum(p_row)), 1.0, places=4)

    def test_05_trained_mlvis_checkpoint_exists(self):
        """Verify trained MLVis model checkpoint and eval report exist on disk."""
        ckpt_path = "models/mlvis/mlvis_recommender.pt"
        report_path = "models/mlvis/eval_report.json"
        self.assertTrue(os.path.exists(ckpt_path), f"Checkpoint missing at {ckpt_path}")
        self.assertTrue(os.path.exists(report_path), f"Evaluation report missing at {report_path}")

        with open(report_path, "r", encoding="utf-8") as f:
            rep = json.load(f)
        self.assertIn("mlvis_neural_model", rep)
        self.assertIn("statistical_baseline", rep)
        # Test accuracy of trained model should exceed baseline
        self.assertGreater(rep["mlvis_neural_model"]["accuracy"], rep["statistical_baseline"]["accuracy"])
        self.assertGreater(rep["mlvis_neural_model"]["accuracy"], 0.80)

    def test_06_mlvis_inference_engine_prediction(self):
        """Verify MLVisInferenceEngine loads checkpoint and returns predictions."""
        engine = MLVisInferenceEngine()
        self.assertTrue(engine.loaded)

        dummy_features = np.zeros((2, 26), dtype=np.float32)
        preds = engine.predict_chart_types(dummy_features)

        self.assertEqual(len(preds), 2)
        for p in preds:
            self.assertIn(p["predicted_type"], CANONICAL_CHART_TYPES)
            self.assertGreater(p["confidence"], 0.0)
            self.assertLessEqual(p["confidence"], 1.0)
            self.assertEqual(len(p["type_probabilities"]), 6)

    def test_07_rank_recommendations_with_mlvis_model(self):
        """Verify rank_visualization_recommendations with model_type='mlvis' returns MLVis-scored candidates."""
        catalog = build_visualization_catalog(self.test_df)
        res = rank_visualization_recommendations(
            candidates=catalog,
            df=self.test_df,
            session_context={"current_chart_type": "bar"},
            interaction_history=[],
            top_k=4,
            model_type="mlvis"
        )

        self.assertEqual(res["model_type"], "mlvis")
        self.assertEqual(res["score_source"], "mlvis_model")
        self.assertEqual(len(res["recommendations"]), 4)

        top_rec = res["recommendations"][0]
        self.assertIn("field_assignments", top_rec)
        self.assertIn("predicted_chart_type", top_rec)
        self.assertIn("MLVis", top_rec["explanation"])
        self.assertIn("MLVis Model Confidence", top_rec["score_label"])

    def test_08_api_recommend_model_selector(self):
        """Verify API /visual-analytics/recommend respects model_type selector."""
        # Test MLVis selection
        resp_mlvis = self.client.post("/api/visual-analytics/recommend", json={
            "dataset_id": "demo_ecommerce",
            "model_type": "mlvis",
            "top_k": 3
        })
        self.assertEqual(resp_mlvis.status_code, 200)
        data_mlvis = resp_mlvis.json()
        self.assertEqual(data_mlvis["model_type"], "mlvis")
        self.assertEqual(data_mlvis["score_source"], "mlvis_model")

        # Test statistical cold-start selection
        resp_stat = self.client.post("/api/visual-analytics/recommend", json={
            "dataset_id": "demo_ecommerce",
            "model_type": "statistical",
            "top_k": 3
        })
        self.assertEqual(resp_stat.status_code, 200)
        data_stat = resp_stat.json()
        self.assertEqual(data_stat["model_type"], "statistical")
        self.assertEqual(data_stat["score_source"], "heuristic_suitability")

    def test_09_api_analytics_includes_mlvis_benchmark(self):
        """Verify /visual-analytics/analytics includes MLVis offline benchmark reference."""
        resp = self.client.get("/api/visual-analytics/analytics")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("mlvis_benchmark_reference", data)
        mlvis_ref = data["mlvis_benchmark_reference"]
        self.assertIn("mlvis_neural_model", mlvis_ref)
        self.assertIn("statistical_baseline", mlvis_ref)
        self.assertAlmostEqual(mlvis_ref["mlvis_neural_model"]["accuracy"], 0.9068, places=2)

    def test_10_synthetic_dataset_exists_and_valid(self):
        """Verify synthetic visualization dataset contains exactly 600 records and valid schema."""
        synth_csv = "data/mlvis/synthetic/synthetic_vis_600.csv"
        synth_schema = "data/mlvis/synthetic/synthetic_schema.json"
        synth_features = "data/mlvis/synthetic/synthetic_features.npz"

        self.assertTrue(os.path.exists(synth_csv), "Synthetic CSV missing")
        self.assertTrue(os.path.exists(synth_schema), "Synthetic schema missing")
        self.assertTrue(os.path.exists(synth_features), "Synthetic features missing")

        df = pd.read_csv(synth_csv)
        self.assertEqual(len(df), 600, "Must contain exactly 600 synthetic records")
        self.assertEqual(df["dataset_archetype_id"].nunique(), 60, "Must span 60 archetypes")
        self.assertEqual(df.isnull().sum().sum(), 0, "No missing values allowed in synthetic profiles")
        self.assertEqual(df.duplicated().sum(), 0, "No duplicate records allowed")

        with open(synth_schema, "r", encoding="utf-8") as f:
            schema = json.load(f)
        self.assertTrue(schema["is_synthetic"])
        self.assertEqual(schema["total_records"], 600)
        self.assertEqual(len(schema["feature_names"]), 26)

    def test_11_mlvis_v2_checkpoint_and_eval_report(self):
        """Verify MLVis v2 checkpoint loads correctly and multi-benchmark comparison report exists."""
        ckpt_v2_path = "models/mlvis_v2/mlvis_recommender.pt"
        report_v2_path = "models/mlvis_v2/eval_report_comparison.json"
        meta_v2_path = "models/mlvis_v2/mlvis_meta.json"

        self.assertTrue(os.path.exists(ckpt_v2_path), "v2 Checkpoint missing")
        self.assertTrue(os.path.exists(report_v2_path), "v2 Comparison report missing")
        self.assertTrue(os.path.exists(meta_v2_path), "v2 Meta missing")

        engine_v2 = MLVisInferenceEngine(model_dir="models/mlvis_v2")
        self.assertTrue(engine_v2.loaded)
        self.assertEqual(engine_v2.meta.get("version"), "2.0.0")

        with open(report_v2_path, "r", encoding="utf-8") as f:
            rep = json.load(f)
        self.assertIn("evaluation_on_real_test_set", rep)
        self.assertIn("evaluation_on_synthetic_test_set", rep)
        # Verify v2 preserves high performance on real test set (>90% accuracy)
        self.assertGreater(rep["evaluation_on_real_test_set"]["mlvis_v2"]["accuracy"], 0.90)
        # Verify v2 excels on synthetic benchmark (>90% accuracy)
        self.assertGreater(rep["evaluation_on_synthetic_test_set"]["mlvis_v2"]["accuracy"], 0.90)

    def test_12_inference_on_unseen_dataset_profiles(self):
        """Verify MLVis engine inference runs cleanly on unseen arbitrary dataset profiles."""
        engine_v1 = MLVisInferenceEngine(model_dir="models/mlvis")
        engine_v2 = MLVisInferenceEngine(model_dir="models/mlvis_v2")

        # Create 5 completely unseen synthetic dataset/column profiles
        unseen_features = np.array([
            # 1: Pure time-series trend (timestamp + numeric)
            [0.5, 0.1, 0.9, 0.1, 0, 0, 1, 0.9, 0, 0.8, 0, 0, 1, 0, 0, 0.5, 0, 0.7, 0.1, 0.4, 0, 0, 0, 0, 0.65, 0.8],
            # 2: Low-cardinality category vs numeric (categorical + numeric)
            [0.4, 0.2, 0.6, 0.4, 0, 1, 0, 0.05, 0, 0.4, 0, 0, 1, 0, 0, 0.4, 0, 0.6, 0.2, 0.5, 0, 1, 0, 0, 0.0, 0.1],
            # 3: Continuous bivariate correlation (numeric + numeric)
            [0.6, 0.3, 0.8, 0.2, 1, 0, 0, 0.8, 0, 0.7, 0, 0.4, 1, 0, 0, 0.8, 0, 0.7, 0, 0.4, 1, 0, 0, 0, 0.82, 1.0],
            # 4: Distribution of single numeric metric (histogram candidate)
            [0.5, 0.1, 0.9, 0.1, 1, 0, 0, 0.7, 0, 0.6, 0.4, 0.3, 1, 0, 0, 0.1, 0, 0.4, 0, 0.2, 1, 0, 0, 0, 0.0, 1.0],
            # 5: Bivariate grid correlation (both categorical dimensions)
            [0.3, 0.2, 0.2, 0.8, 0, 1, 0, 0.1, 0, 0.5, 0, 0, 0, 1, 0, 0.1, 0, 0.5, 0, 0, 0, 0, 0, 1, 0.0, 1.0],
        ], dtype=np.float32)

        preds_v1 = engine_v1.predict_chart_types(unseen_features)
        preds_v2 = engine_v2.predict_chart_types(unseen_features)

        self.assertEqual(len(preds_v1), 5)
        self.assertEqual(len(preds_v2), 5)

        for p in preds_v2:
            self.assertIn(p["predicted_type"], CANONICAL_CHART_TYPES)
            self.assertGreater(p["confidence"], 0.0)
            self.assertLessEqual(p["confidence"], 1.0)
            self.assertEqual(len(p["type_probabilities"]), 6)


if __name__ == "__main__":
    unittest.main()
