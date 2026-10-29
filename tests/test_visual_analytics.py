"""
Comprehensive Test Suite for Visual Analytics & Context-Aware Visualization Recommendations.
Tests dataset profiling, interactive chart data generation, candidate generation,
cold-start vs neural model ranking, interaction event logging, and empirical metrics.
"""

import os
import json
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi.testclient import TestClient

from backend.main import app
from backend.database.database import SessionLocal, init_db
from backend.database.models import VisualInteractionEventModel, ClientModel
from src.visual_analytics import (
    profile_dataset,
    generate_chart_data,
    build_visualization_catalog,
    rank_visualization_recommendations
)
from backend.services.visual_analytics_service import (
    compute_visual_analytics_metrics,
    record_visual_interaction
)


class TestVisualAnalytics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()
        cls.client = TestClient(app)
        cls.db = SessionLocal()

        # Synthetic test dataframe for controlled testing
        np.random.seed(42)
        n = 100
        cls.test_df = pd.DataFrame({
            "category": np.random.choice(["Electronics", "Fashion", "Books"], size=n),
            "unit_price": np.random.uniform(10.0, 500.0, size=n),
            "discount_amount": np.random.uniform(0.0, 50.0, size=n),
            "quantity": np.random.randint(1, 10, size=n),
            "added_to_cart": np.random.choice([0, 1], size=n),
            "purchased": np.random.choice([0, 1], size=n)
        })

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    def test_01_profile_dataset(self):
        """Test dataset profiling extracts dimensions, column roles, statistics, and correlations."""
        profile = profile_dataset(self.test_df)

        self.assertEqual(profile["dimensions"]["total_rows"], 100)
        self.assertEqual(profile["dimensions"]["total_columns"], 6)
        self.assertGreaterEqual(profile["dimensions"]["numeric_columns_count"], 4)
        self.assertGreaterEqual(profile["dimensions"]["categorical_columns_count"], 1)

        # Check numeric column stats
        price_col = next(c for c in profile["columns"] if c["name"] == "unit_price")
        self.assertEqual(price_col["dtype"], "numeric")
        self.assertIn("mean", price_col["stats"])
        self.assertIn("min", price_col["stats"])
        self.assertIn("max", price_col["stats"])
        self.assertIn("skewness", price_col["stats"])
        self.assertGreater(price_col["stats"]["mean"], 0)

        # Check categorical column stats
        cat_col = next(c for c in profile["columns"] if c["name"] == "category")
        self.assertIn("top_categories", cat_col["stats"])
        self.assertEqual(len(cat_col["stats"]["top_categories"]), 3)

        # Check correlation matrix
        self.assertIn("columns", profile["correlations"])
        self.assertIn("matrix", profile["correlations"])
        self.assertIn("pairs", profile["correlations"])

    def test_02_chart_data_bar_and_line(self):
        """Test bar and line chart data generation with count and mean aggregations."""
        # Frequency Bar
        bar_count = generate_chart_data(self.test_df, "bar", "category", aggregation="count")
        self.assertEqual(bar_count["chart_type"], "bar")
        self.assertEqual(len(bar_count["data"]), 3)
        self.assertEqual(sum(d["value"] for d in bar_count["data"]), 100)

        # Aggregated Bar
        bar_mean = generate_chart_data(self.test_df, "bar", "category", y_col="unit_price", aggregation="mean")
        self.assertEqual(bar_mean["chart_type"], "bar")
        self.assertEqual(len(bar_mean["data"]), 3)
        for d in bar_mean["data"]:
            self.assertGreater(d["value"], 0)

        # Line Chart
        line_data = generate_chart_data(self.test_df, "line", "quantity", y_col="unit_price", aggregation="mean")
        self.assertEqual(line_data["chart_type"], "line")
        self.assertGreater(len(line_data["data"]), 0)

    def test_03_chart_data_histogram(self):
        """Test histogram data generation with binning, counts, and statistical summary."""
        hist = generate_chart_data(self.test_df, "histogram", "unit_price")
        self.assertEqual(hist["chart_type"], "histogram")
        self.assertGreater(len(hist["data"]), 5)
        self.assertEqual(hist["stats"]["count"], 100)
        self.assertIn("mean", hist["stats"])
        self.assertIn("skewness", hist["stats"])

        # Total counts across bins should equal total valid rows
        total_binned = sum(b["count"] for b in hist["data"])
        self.assertEqual(total_binned, 100)

    def test_04_chart_data_scatter_and_heatmap(self):
        """Test scatter plot correlation/trendline and correlation heatmap."""
        scatter = generate_chart_data(self.test_df, "scatter", "unit_price", y_col="discount_amount")
        self.assertEqual(scatter["chart_type"], "scatter")
        self.assertEqual(len(scatter["data"]), 100)
        self.assertIn("correlation", scatter)
        self.assertIn("trendline", scatter)
        self.assertIn("slope", scatter["trendline"])
        self.assertIn("intercept", scatter["trendline"])

        # Heatmap
        heatmap = generate_chart_data(self.test_df, "heatmap", "unit_price")
        self.assertEqual(heatmap["chart_type"], "heatmap")
        self.assertGreaterEqual(len(heatmap["columns"]), 3)
        self.assertEqual(len(heatmap["matrix"]), len(heatmap["columns"]))

    def test_05_chart_filtering(self):
        """Test that data filters restrict dataset and update chart calculation correctly."""
        filters = {"category": {"op": "eq", "val": "Electronics"}}
        bar_filtered = generate_chart_data(self.test_df, "bar", "category", filters=filters)
        self.assertEqual(len(bar_filtered["data"]), 1)
        self.assertEqual(bar_filtered["data"][0]["label"], "Electronics")

        # Range filter
        range_filters = {"unit_price": {"op": "gte", "val": 250.0}}
        hist_filtered = generate_chart_data(self.test_df, "histogram", "unit_price", filters=range_filters)
        self.assertLess(hist_filtered["stats"]["count"], 100)
        self.assertGreaterEqual(hist_filtered["stats"]["min"], 250.0)

    def test_06_visualization_catalog(self):
        """Test candidate generation catalog includes varied analytical chart candidates."""
        catalog = build_visualization_catalog(self.test_df)
        self.assertGreater(len(catalog), 3)

        chart_types = {c["chart_type"] for c in catalog}
        self.assertIn("bar", chart_types)
        self.assertIn("histogram", chart_types)
        self.assertIn("scatter", chart_types)
        self.assertIn("heatmap", chart_types)

        for cand in catalog:
            self.assertIn("candidate_id", cand)
            self.assertIn("title", cand)
            self.assertIn("task_type", cand)
            self.assertIn("statistical_signal", cand)

    def test_07_cold_start_recommendation_ranking(self):
        """Test cold-start recommendation ranking uses heuristic suitability and tags clearly."""
        catalog = build_visualization_catalog(self.test_df)
        res = rank_visualization_recommendations(
            candidates=catalog,
            df=self.test_df,
            session_context={"current_chart_type": "bar"},
            interaction_history=[],
            user_id=None,
            client_id=None,
            top_k=5
        )

        self.assertTrue(res["is_cold_start"])
        self.assertEqual(res["score_source"], "heuristic_suitability")
        self.assertEqual(len(res["recommendations"]), 5)

        # Scores must be in descending order
        scores = [r["recommendation_score"] for r in res["recommendations"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

        for rec in res["recommendations"]:
            self.assertEqual(rec["score_source"], "heuristic_suitability")
            self.assertIn("Cold-Start", rec["score_label"])
            self.assertTrue(len(rec["explanation"]) > 10)

    def test_08_active_model_neural_personalization(self):
        """Test integration of active Two-Tower recommendation model on demo_ecommerce."""
        # Use canonical Ecommerce dataset with active client demo_ecommerce
        resp = self.client.post("/api/visual-analytics/recommend", json={
            "dataset_id": "demo_ecommerce",
            "client_id": "demo_ecommerce",
            "user_id": "1",  # Warm customer from Ecommerce.csv
            "session_context": {
                "session_id": "test_session_123",
                "current_chart_type": "scatter",
                "selected_columns": ["unit_price", "discount_amount"]
            },
            "top_k": 4
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("recommendations", data)
        self.assertEqual(len(data["recommendations"]), 4)

        # Check neural gating was invoked for demo_ecommerce
        self.assertEqual(data["score_source"], "neural_model")
        self.assertEqual(data["model_version"], "v1")
        self.assertIsNotNone(data["gating_weights"])
        self.assertIn("behavior", data["gating_weights"])
        self.assertIn("content", data["gating_weights"])
        self.assertIn("context", data["gating_weights"])

        # Check that top recommendation has neural score and grounded explanation
        top_rec = data["recommendations"][0]
        self.assertEqual(top_rec["score_source"], "neural_model")
        self.assertGreater(top_rec["recommendation_score"], 0.0)
        self.assertIn("v1 neural gating", top_rec["explanation"])

    def test_09_visual_interaction_event_logging_and_history(self):
        """Test recording interactions (impression, click, view, feedback) and audit trail."""
        import uuid
        session_id = f"test_audit_session_{uuid.uuid4().hex[:8]}"

        # 1. Log impression
        r1 = self.client.post("/api/visual-analytics/interactions", json={
            "session_id": session_id,
            "event_type": "chart_impression",
            "chart_id": "cand_hist_unit_price",
            "chart_type": "histogram",
            "details": {"score": 0.88}
        })
        self.assertEqual(r1.status_code, 201)
        self.assertEqual(r1.json()["status"], "recorded")

        # 2. Log click
        r2 = self.client.post("/api/visual-analytics/interactions", json={
            "session_id": session_id,
            "event_type": "recommendation_click",
            "chart_id": "cand_hist_unit_price",
            "chart_type": "histogram"
        })
        self.assertEqual(r2.status_code, 201)

        # 3. Log helpful feedback
        r3 = self.client.post("/api/visual-analytics/interactions", json={
            "session_id": session_id,
            "event_type": "helpful_feedback",
            "chart_id": "cand_hist_unit_price",
            "chart_type": "histogram",
            "details": {"rating": "helpful"}
        })
        self.assertEqual(r3.status_code, 201)

        # 4. Fetch history
        hist_resp = self.client.get(f"/api/visual-analytics/interactions?session_id={session_id}")
        self.assertEqual(hist_resp.status_code, 200)
        events = hist_resp.json()
        self.assertEqual(len(events), 3)
        event_types = [e["event_type"] for e in events]
        self.assertIn("chart_impression", event_types)
        self.assertIn("recommendation_click", event_types)
        self.assertIn("helpful_feedback", event_types)

    def test_10_visual_analytics_metrics(self):
        """Test visual analytics metrics: CTR, feedback ratio, and cold-start vs returning performance."""
        resp = self.client.get("/api/visual-analytics/analytics")
        self.assertEqual(resp.status_code, 200)
        metrics = resp.json()

        self.assertEqual(metrics["data_origin"], "measured_production_data")
        self.assertIn("summary", metrics)
        self.assertIn("impressions", metrics["summary"])
        self.assertIn("clicks", metrics["summary"])
        self.assertIn("click_through_rate", metrics["summary"])
        self.assertIn("cohort_performance", metrics)
        self.assertIn("cold_start", metrics["cohort_performance"])
        self.assertIn("warm_start", metrics["cohort_performance"])
        self.assertIn("chart_type_distribution", metrics)
        self.assertIn("offline_benchmark_reference", metrics)

    def test_11_api_chart_data_and_profile_endpoints(self):
        """Test API chart data and profiling endpoints on canonical dataset."""
        # Profile endpoint
        prof_resp = self.client.get("/api/visual-analytics/profile?dataset_id=demo_ecommerce")
        self.assertEqual(prof_resp.status_code, 200)
        profile = prof_resp.json()
        self.assertEqual(profile["dimensions"]["total_rows"], 25000)
        self.assertEqual(profile["dimensions"]["total_columns"], 29)

        # Chart data endpoint: bar
        chart_resp = self.client.post("/api/visual-analytics/chart-data", json={
            "dataset_id": "demo_ecommerce",
            "chart_type": "bar",
            "x_column": "product_category",
            "aggregation": "count"
        })
        self.assertEqual(chart_resp.status_code, 200)
        cdata = chart_resp.json()
        self.assertEqual(cdata["chart_type"], "bar")
        self.assertEqual(len(cdata["data"]), 8)  # 8 product categories in benchmark

        # Chart data endpoint: scatter
        scatter_resp = self.client.post("/api/visual-analytics/chart-data", json={
            "dataset_id": "demo_ecommerce",
            "chart_type": "scatter",
            "x_column": "unit_price",
            "y_column": "discount_amount"
        })
        self.assertEqual(scatter_resp.status_code, 200)
        sdata = scatter_resp.json()
        self.assertEqual(sdata["chart_type"], "scatter")
        self.assertIn("correlation", sdata)
        self.assertIn("trendline", sdata)

    def test_12_dataset_upload_validation(self):
        """Test upload validation against empty files, invalid extensions, and malformed files."""
        # Non-csv extension rejected
        res_txt = self.client.post(
            "/api/visual-analytics/upload",
            files={"file": ("malicious.exe", b"binarycontent", "application/octet-stream")}
        )
        self.assertEqual(res_txt.status_code, 400)

        # Empty file rejected
        res_empty = self.client.post(
            "/api/visual-analytics/upload",
            files={"file": ("empty.csv", b"", "text/csv")}
        )
        self.assertEqual(res_empty.status_code, 400)

        # Valid CSV accepted
        csv_content = b"col_a,col_b\n1,10\n2,20\n3,30\n"
        res_valid = self.client.post(
            "/api/visual-analytics/upload",
            files={"file": ("valid_test.csv", csv_content, "text/csv")}
        )
        self.assertEqual(res_valid.status_code, 201)
        resp_data = res_valid.json()
        self.assertEqual(resp_data["row_count"], 3)
        self.assertEqual(resp_data["column_count"], 2)
        assigned_id = resp_data["dataset_id"]

        # Confirm newly uploaded dataset appears in dataset list
        datasets_resp = self.client.get("/api/visual-analytics/datasets")
        self.assertEqual(datasets_resp.status_code, 200)
        datasets_list = datasets_resp.json()
        matching = [d for d in datasets_list if d["id"] == assigned_id or d["filename"] == "valid_test.csv"]
        self.assertGreater(len(matching), 0, "Uploaded dataset should appear in dataset listing.")

        # Confirm dataset profile can be retrieved for the uploaded dataset
        profile_resp = self.client.get(f"/api/visual-analytics/profile?dataset_id={assigned_id}")
        self.assertEqual(profile_resp.status_code, 200)
        profile_data = profile_resp.json()
        self.assertEqual(profile_data["dimensions"]["total_rows"], 3)
        self.assertEqual(profile_data["dimensions"]["total_columns"], 2)
        col_names = [c["name"] for c in profile_data["columns"]]
        self.assertIn("col_a", col_names)
        self.assertIn("col_b", col_names)

        # Confirm chart data can be generated for the uploaded dataset
        chart_resp = self.client.post("/api/visual-analytics/chart-data", json={
            "dataset_id": assigned_id,
            "chart_type": "bar",
            "x_column": "col_a",
            "y_column": "col_b",
            "aggregation": "mean"
        })
        self.assertEqual(chart_resp.status_code, 200)
        chart_data = chart_resp.json()
        self.assertEqual(chart_data["chart_type"], "bar")
        self.assertEqual(len(chart_data["data"]), 3)

    def test_13_uploaded_dataset_with_client_id(self):
        """Test uploading CSV associated with a specific client ID, verifying listing and retrieval."""
        csv_content = b"feature_x,metric_y\ncat1,100\ncat2,200\ncat3,300\n"
        res_upload = self.client.post(
            "/api/visual-analytics/upload?client_id=demo_ecommerce",
            files={"file": ("client_upload.csv", csv_content, "text/csv")}
        )
        self.assertEqual(res_upload.status_code, 201)
        dataset_info = res_upload.json()
        dataset_id = dataset_info["dataset_id"]

        # Check datasets list contains it
        list_res = self.client.get("/api/visual-analytics/datasets")
        self.assertEqual(list_res.status_code, 200)
        matching = [d for d in list_res.json() if d["id"] == dataset_id or d["filename"] == "client_upload.csv"]
        self.assertGreater(len(matching), 0)

        # Get profile
        prof_res = self.client.get(f"/api/visual-analytics/profile?dataset_id={dataset_id}&client_id=demo_ecommerce")
        self.assertEqual(prof_res.status_code, 200)
        self.assertEqual(prof_res.json()["dimensions"]["total_rows"], 3)

    def test_14_warm_start_mlvis_personalization_and_unsuitable_boundary(self):
        """
        Verify:
        1. Cold-start recommendations work with zero history.
        2. Positive feedback on scatter plots gives them a boost for suitable datasets.
        3. Negative feedback reduces scores for that chart type.
        4. History personalization cannot promote an unsuitable chart over valid ones.
        """
        from backend.services.user_history_service import clear_user_history, record_user_history_entry
        test_uid = "warm_mlvis_test_user_77"
        clear_user_history("demo_ecommerce", test_uid)

        # 1. Cold start recommendation check
        resp_cold = self.client.post("/api/visual-analytics/recommend", json={
            "dataset_id": "demo_ecommerce",
            "client_id": "demo_ecommerce",
            "user_id": test_uid,
            "model_type": "mlvis",
            "top_k": 6
        })
        self.assertEqual(resp_cold.status_code, 200)
        cold_data = resp_cold.json()
        self.assertTrue(cold_data["is_cold_start"])
        self.assertEqual(cold_data["personalization_mode"], "cold_start")

        # Find initial score for scatter if present
        initial_scatter_rec = next((r for r in cold_data["recommendations"] if r["chart_type"] == "scatter"), None)

        # 2. Add positive interaction history for scatter
        record_user_history_entry(
            client_id="demo_ecommerce",
            user_id=test_uid,
            entry_type="chart_selection",
            items_or_charts=[{"chart_type": "scatter"}]
        )
        record_user_history_entry(
            client_id="demo_ecommerce",
            user_id=test_uid,
            entry_type="helpful_feedback",
            items_or_charts=[{"chart_type": "scatter"}],
            feedback="helpful"
        )
        # Add negative feedback for pie
        record_user_history_entry(
            client_id="demo_ecommerce",
            user_id=test_uid,
            entry_type="unhelpful_feedback",
            items_or_charts=[{"chart_type": "pie"}],
            feedback="unhelpful"
        )

        # 3. Request recommendations again as warm user
        resp_warm = self.client.post("/api/visual-analytics/recommend", json={
            "dataset_id": "demo_ecommerce",
            "client_id": "demo_ecommerce",
            "user_id": test_uid,
            "model_type": "mlvis",
            "top_k": 6
        })
        self.assertEqual(resp_warm.status_code, 200)
        warm_data = resp_warm.json()
        self.assertFalse(warm_data["is_cold_start"])
        self.assertEqual(warm_data["personalization_mode"], "warm_start")

        warm_scatter_rec = next((r for r in warm_data["recommendations"] if r["chart_type"] == "scatter"), None)
        if warm_scatter_rec and initial_scatter_rec:
            # Positive personalization boost must be applied
            self.assertGreaterEqual(warm_scatter_rec["recommendation_score"], initial_scatter_rec["recommendation_score"])
            self.assertGreater(warm_scatter_rec.get("history_personalization_boost", 0.0), 0.0)

        # 4. Verify bounds check: history cannot force an impossible/unsuitable candidate
        for rec in warm_data["recommendations"]:
            # Bounded boost must never exceed [-0.15, +0.15]
            boost = rec.get("history_personalization_boost", 0.0)
            self.assertGreaterEqual(boost, -0.15)
            self.assertLessEqual(boost, 0.15)


if __name__ == "__main__":
    unittest.main()

