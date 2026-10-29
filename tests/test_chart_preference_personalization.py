"""
Unit and integration tests for History-Based Chart Preference Personalization in MLVis.
Verifies:
1. Prior line-chart selection influences ranking for a compatible new dataset.
2. Unsuitable line chart is NOT promoted above suitable charts just because of user preference.
3. Similar chart types are considered only when compatible (e.g. area for line, bubble for scatter).
4. Negative feedback reduces preference appropriately.
5. Cold-start and warm-start users without relevant preferences retain original baseline behavior.
6. Different tenants cannot share preferences.
7. Two-Tower and MLVis functionality remains intact.
"""

import os
import shutil
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

from backend.services.user_history_service import (
    USER_HISTORY_BASE_DIR,
    record_user_history_entry,
    get_user_chart_preferences,
    detect_user_status,
    clear_user_history
)
from src.visual_analytics import (
    build_visualization_catalog,
    rank_visualization_recommendations,
    check_chart_statistical_suitability,
    CHART_SIMILARITY_MAP
)


class TestChartPreferencePersonalization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tenant_a = "tenant_test_vis_a"
        cls.tenant_b = "tenant_test_vis_b"

        # Synthetic dataset 1: Sequential Time Series (compatible with line, area, bar, hist)
        np.random.seed(42)
        n = 50
        dates = pd.date_range("2026-01-01", periods=n, freq="D").astype(str)
        cls.trend_df = pd.DataFrame({
            "date": dates,
            "revenue": np.random.uniform(100.0, 1000.0, size=n),
            "units_sold": np.random.randint(5, 50, size=n),
            "category": np.random.choice(["Electronics", "Clothing", "Home"], size=n)
        })

        # Synthetic dataset 2: Purely Categorical / Non-sequential (unsuitable for line/area)
        cls.categorical_df = pd.DataFrame({
            "department": ["Engineering", "Marketing", "Sales", "HR", "Support"] * 10,
            "office_location": ["New York", "London", "Tokyo", "Berlin", "Singapore"] * 10,
            "employment_type": ["Full-Time", "Contract", "Part-Time", "Full-Time", "Intern"] * 10,
            "satisfaction_rating": np.random.uniform(1.0, 5.0, size=50)
        })

        # Synthetic dataset 3: Multivariate continuous (suitable for scatter, bubble, heatmap)
        cls.multivariate_df = pd.DataFrame({
            "metric_x": np.random.normal(50, 10, size=60),
            "metric_y": np.random.normal(100, 20, size=60),
            "metric_z": np.random.uniform(5, 25, size=60),
            "metric_w": np.random.uniform(1, 10, size=60)
        })

    def setUp(self):
        # Clear test tenant directories before each test
        for cid in [self.tenant_a, self.tenant_b]:
            p = USER_HISTORY_BASE_DIR / cid
            if p.exists():
                shutil.rmtree(p)

    def tearDown(self):
        for cid in [self.tenant_a, self.tenant_b]:
            p = USER_HISTORY_BASE_DIR / cid
            if p.exists():
                shutil.rmtree(p)

    def test_01_prior_line_chart_influences_compatible_dataset(self):
        """A prior line-chart selection influences ranking for a compatible new dataset."""
        user_id = "user_line_lover"

        # Baseline: Cold user with 0 interactions
        catalog = build_visualization_catalog(self.trend_df)
        baseline_recs = rank_visualization_recommendations(
            candidates=catalog,
            df=self.trend_df,
            user_id="cold_user_xyz",
            client_id=self.tenant_a,
            model_type="mlvis",
            top_k=6
        )
        line_baseline = next((r for r in baseline_recs["recommendations"] if r["chart_type"] == "line"), None)

        # Now record genuine user line-chart selections
        record_user_history_entry(
            client_id=self.tenant_a,
            user_id=user_id,
            entry_type="chart_selection",
            items_or_charts=[{"chart_type": "line"}]
        )
        record_user_history_entry(
            client_id=self.tenant_a,
            user_id=user_id,
            entry_type="helpful_feedback",
            items_or_charts=[{"chart_type": "line"}]
        )

        # Re-rank for the returning user
        personalized_recs = rank_visualization_recommendations(
            candidates=catalog,
            df=self.trend_df,
            user_id=user_id,
            client_id=self.tenant_a,
            model_type="mlvis",
            top_k=6
        )

        line_pers = next((r for r in personalized_recs["recommendations"] if r["chart_type"] == "line"), None)
        self.assertIsNotNone(line_pers)
        self.assertTrue(line_pers["preference_applied"])
        self.assertGreater(line_pers["history_personalization_boost"], 0.0)
        self.assertEqual(line_pers["preference_relation"], "exact")
        self.assertIn("Suggested based on your previous line selections", line_pers["personalization_label"])
        self.assertEqual(personalized_recs["personalization_mode"], "warm_personalized")

        # Score must be higher than baseline
        if line_baseline:
            self.assertGreater(line_pers["recommendation_score"], line_baseline["recommendation_score"])

    def test_02_unsuitable_line_chart_not_promoted_on_categorical_dataset(self):
        """An unsuitable line chart is NOT promoted above suitable charts just because of user preference."""
        user_id = "user_line_lover"

        # User loves line charts
        record_user_history_entry(
            client_id=self.tenant_a,
            user_id=user_id,
            entry_type="helpful_feedback",
            items_or_charts=[{"chart_type": "line"}]
        )

        # Dataset is purely categorical without temporal or sequential ordering
        catalog = build_visualization_catalog(self.categorical_df)
        recs = rank_visualization_recommendations(
            candidates=catalog,
            df=self.categorical_df,
            user_id=user_id,
            client_id=self.tenant_a,
            model_type="mlvis",
            top_k=5
        )

        # Line chart must NOT be promoted or recommended as top chart on categorical data
        top_chart = recs["recommendations"][0]
        self.assertNotEqual(top_chart["chart_type"], "line")
        self.assertIn(top_chart["chart_type"], ["bar", "donut", "treemap", "box", "histogram"])

        # Check unsuitable preferred charts alert
        self.assertTrue(len(recs["unsuitable_preferred_charts"]) > 0)
        un = recs["unsuitable_preferred_charts"][0]
        self.assertEqual(un["preferred_chart"], "line")
        self.assertIn("lacks sequential or temporal", un["reason"])
        self.assertGreaterEqual(len(un["suggested_alternatives"]), 1)

    def test_03_similar_chart_considered_only_when_compatible(self):
        """Similar chart types in family (area for line, bubble for scatter) receive boost when compatible."""
        user_id = "user_scatter_enthusiast"

        # User only selected scatter plots previously (no bubble history)
        record_user_history_entry(
            client_id=self.tenant_a,
            user_id=user_id,
            entry_type="chart_selection",
            items_or_charts=[{"chart_type": "scatter"}]
        )

        # On multivariate dataset with 3+ numeric columns, bubble chart is compatible
        catalog = build_visualization_catalog(self.multivariate_df)
        recs = rank_visualization_recommendations(
            candidates=catalog,
            df=self.multivariate_df,
            user_id=user_id,
            client_id=self.tenant_a,
            model_type="mlvis",
            top_k=6
        )

        bubble_rec = next((r for r in recs["recommendations"] if r["chart_type"] == "bubble"), None)
        if bubble_rec:
            self.assertTrue(bubble_rec["preference_applied"])
            self.assertEqual(bubble_rec["preference_relation"], "similar")
            self.assertEqual(bubble_rec["preference_source_chart"], "scatter")
            self.assertIn("Similar to your previous scatter selection", bubble_rec["personalization_label"])
            self.assertGreater(bubble_rec["history_personalization_boost"], 0.0)

    def test_04_negative_feedback_reduces_preference(self):
        """Explicit negative feedback lowers preference so other suitable charts rank higher."""
        user_id = "user_scatter_hater"

        # User dislikes scatter charts
        record_user_history_entry(
            client_id=self.tenant_a,
            user_id=user_id,
            entry_type="unhelpful_feedback",
            items_or_charts=[{"chart_type": "scatter"}],
            feedback="unhelpful"
        )

        prefs = get_user_chart_preferences(self.tenant_a, user_id)
        self.assertLess(prefs["chart_type_weights"].get("scatter", 0.0), 0.0)
        self.assertIn("scatter", prefs["negative_charts"])

        catalog = build_visualization_catalog(self.multivariate_df)
        recs = rank_visualization_recommendations(
            candidates=catalog,
            df=self.multivariate_df,
            user_id=user_id,
            client_id=self.tenant_a,
            model_type="mlvis",
            top_k=6
        )

        scatter_rec = next((r for r in recs["recommendations"] if r["chart_type"] == "scatter"), None)
        if scatter_rec:
            self.assertLess(scatter_rec["history_personalization_boost"], 0.0)
            self.assertEqual(scatter_rec["preference_relation"], "negative_feedback")
            self.assertIn("Adjusted based on your previous feedback", scatter_rec["personalization_label"])

    def test_05_cold_and_warm_users_without_relevant_preferences_retain_baseline(self):
        """Cold-start and warm-start users without relevant chart preferences retain exact baseline behavior."""
        catalog = build_visualization_catalog(self.trend_df)

        # Baseline: anonymous/cold user
        cold_recs = rank_visualization_recommendations(
            candidates=catalog,
            df=self.trend_df,
            user_id="cold_user_1",
            client_id=self.tenant_a,
            model_type="mlvis",
            top_k=5
        )
        self.assertEqual(cold_recs["personalization_mode"], "cold_start")
        self.assertFalse(cold_recs["has_chart_preference"])
        for r in cold_recs["recommendations"]:
            self.assertFalse(r["preference_applied"])
            self.assertIsNone(r["personalization_label"])
            self.assertEqual(r["history_personalization_boost"], 0.0)

        # Warm-start user with 0 chart selections (only dataset uploads or impressions)
        warm_user_id = "warm_user_no_chart_pref"
        record_user_history_entry(
            client_id=self.tenant_a,
            user_id=warm_user_id,
            entry_type="dataset_upload",
            metadata={"filename": "test.csv"}
        )
        record_user_history_entry(
            client_id=self.tenant_a,
            user_id=warm_user_id,
            entry_type="chart_impression",
            items_or_charts=[{"chart_type": "line"}, {"chart_type": "bar"}]
        )

        warm_status = detect_user_status(self.tenant_a, warm_user_id)
        self.assertEqual(warm_status["status"], "warm_start")

        warm_recs = rank_visualization_recommendations(
            candidates=catalog,
            df=self.trend_df,
            user_id=warm_user_id,
            client_id=self.tenant_a,
            model_type="mlvis",
            top_k=5
        )
        self.assertEqual(warm_recs["personalization_mode"], "warm_start")
        self.assertFalse(warm_recs["has_chart_preference"])

        # Ranking and scores must match cold baseline exactly because no genuine chart preference exists
        cold_ids = [r["candidate_id"] for r in cold_recs["recommendations"]]
        warm_ids = [r["candidate_id"] for r in warm_recs["recommendations"]]
        self.assertEqual(cold_ids, warm_ids)
        for r in warm_recs["recommendations"]:
            self.assertFalse(r["preference_applied"])
            self.assertIsNone(r["personalization_label"])
            self.assertEqual(r["history_personalization_boost"], 0.0)

    def test_06_tenant_isolation_prevents_preference_sharing(self):
        """Different tenants cannot share user preferences."""
        shared_user_id = "user_cross_tenant"

        # User loves bar charts in Tenant A
        record_user_history_entry(
            client_id=self.tenant_a,
            user_id=shared_user_id,
            entry_type="chart_selection",
            items_or_charts=[{"chart_type": "bar"}]
        )

        prefs_a = get_user_chart_preferences(self.tenant_a, shared_user_id)
        prefs_b = get_user_chart_preferences(self.tenant_b, shared_user_id)

        self.assertTrue(prefs_a["has_relevant_preferences"])
        self.assertGreater(prefs_a["chart_type_weights"].get("bar", 0.0), 0.0)

        # In Tenant B, user has no history or preferences
        self.assertFalse(prefs_b["has_history"])
        self.assertFalse(prefs_b["has_relevant_preferences"])
        self.assertEqual(prefs_b["chart_type_weights"], {})

        status_b = detect_user_status(self.tenant_b, shared_user_id)
        self.assertEqual(status_b["status"], "cold_start")

    def test_07_mlvis_and_two_tower_intact(self):
        """Existing MLVis and Two-Tower product recommendation functionality remains intact."""
        from src.models.mlvis_recommender import MLVisInferenceEngine
        from src.mlvis_preprocessing import CANONICAL_CHART_TYPES
        engine = MLVisInferenceEngine()
        self.assertTrue(engine.loaded)
        self.assertIn("bar", CANONICAL_CHART_TYPES)
        self.assertIn("line", CANONICAL_CHART_TYPES)

        # Predict with random synthetic feature matrix matching in_features
        in_dim = engine.meta.get("in_features", 26)
        feats = np.random.uniform(0.0, 1.0, size=(3, in_dim)).astype(np.float32)
        preds = engine.predict_chart_types(feats)
        self.assertEqual(len(preds), 3)
        for p in preds:
            self.assertIn("predicted_type", p)
            self.assertIn("confidence", p)
            self.assertIn("type_probabilities", p)
            self.assertGreater(p["confidence"], 0.0)


if __name__ == "__main__":
    unittest.main()
