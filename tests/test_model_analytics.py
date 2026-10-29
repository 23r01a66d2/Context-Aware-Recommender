"""
Unit and Integration Tests for Model Analytics & Architectural Benchmark.
Verifies:
1. Non-zero authentic evaluation metrics retrieval and percentage formatting.
2. Dynamic model enumeration (not restricted to hardcoded 5 models).
3. Fair comparison task isolation (Product Top-5 vs MLVis Top-3).
4. Tenant isolation across client contexts (demo_ecommerce, movielens_dataset, generic_client_dynamic, client_no_model).
5. Ground truth relevance explanation and N/A handling for unevaluated models.
6. API endpoint compliance for /api/models/analytics and /api/clients/{client_id}/analytics.
"""

import unittest

from backend.database.database import SessionLocal, init_db
from backend.services.model_service import get_model_analytics_summary, format_precision_pct
from backend.services.client_service import ensure_all_canonical_clients_registered
from backend.api.models import api_get_model_analytics, api_get_client_analytics


class TestModelAnalytics(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()
        db = SessionLocal()
        try:
            ensure_all_canonical_clients_registered(db)
        finally:
            db.close()

    def test_01_precision_pct_formatter(self):
        """Verifies precision percentage formatting and N/A fallback."""
        self.assertEqual(format_precision_pct(None), "N/A")
        self.assertEqual(format_precision_pct(0.0106), "1.06%")
        self.assertEqual(format_precision_pct(0.0298), "2.98%")
        self.assertEqual(format_precision_pct(0.20), "20.00%")
        self.assertEqual(format_precision_pct(0.9987), "99.87%")

    def test_02_demo_ecommerce_dynamic_models_and_nonzero_metrics(self):
        """
        Verifies demo_ecommerce returns all applicable models dynamically
        with authentic nonzero Precision@5 values and no 0.000 fallback.
        """
        data = get_model_analytics_summary("demo_ecommerce")
        self.assertEqual(data["client_id"], "demo_ecommerce")
        models = data["models_comparison"]

        # Must include at least the 5 baselines + any registered candidate models (>= 5 rows)
        self.assertGreaterEqual(len(models), 5)

        # Check Model 1: Popularity
        pop = next((m for m in models if m["model_id"] == "model_1"), None)
        self.assertIsNotNone(pop)
        self.assertEqual(pop["name"], "Model 1: Popularity Baseline")
        self.assertAlmostEqual(pop["precision_at_5"], 0.0106, places=4)
        self.assertEqual(pop["precision_formatted"], "1.06%")
        self.assertEqual(pop["evaluated_cases"], 1377)
        self.assertEqual(pop["status"], "Evaluated Baseline")
        self.assertIn("1 ground-truth positive", pop["relevance_definition"])

        # Check Model 2: Content-Based
        cb = next((m for m in models if m["model_id"] == "model_2"), None)
        self.assertIsNotNone(cb)
        self.assertAlmostEqual(cb["precision_at_5"], 0.0298, places=4)
        self.assertEqual(cb["precision_formatted"], "2.98%")
        self.assertEqual(cb["evaluated_cases"], 1377)

        # Check Model 5: Proposed Adaptive Multi-Modal
        prop = next((m for m in models if m["model_id"] == "model_5"), None)
        self.assertIsNotNone(prop)
        self.assertAlmostEqual(prop["precision_at_5"], 0.0121, places=4)
        self.assertEqual(prop["precision_formatted"], "1.21%")
        self.assertTrue(prop["is_active"])
        self.assertEqual(prop["status"], "Active Target")

        # Verify backward compatibility field aliases exist
        for m in models:
            self.assertIn("precision_at_5", m)
            self.assertIn("precision_5", m)
            self.assertIn("Precision@5", m)
            self.assertIn("recall_at_5", m)
            self.assertIn("ndcg_at_5", m)
            self.assertIn("hit_rate_at_5", m)
            self.assertIn("precision_formatted", m)

    def test_03_tenant_isolation_movielens_and_generic(self):
        """Verifies tenant isolation: different clients return only their models."""
        ml_data = get_model_analytics_summary("movielens_dataset")
        self.assertEqual(ml_data["client_id"], "movielens_dataset")
        ml_models = ml_data["models_comparison"]
        # Must only contain movielens models
        for m in ml_models:
            self.assertIn("movielens", m["model_id"])

        # Check client_no_model returns empty state
        empty_data = get_model_analytics_summary("client_no_model")
        self.assertEqual(len(empty_data["models_comparison"]), 0)

    def test_04_unevaluated_model_shows_na_and_explanation(self):
        """Verifies models without ranking test sets show N/A and explanatory reason."""
        data = get_model_analytics_summary("generic_client_dynamic")
        models = data["models_comparison"]
        if models:
            m = models[0]
            self.assertIsNone(m["precision_at_k"])
            self.assertEqual(m["precision_formatted"], "N/A")
            self.assertIsNotNone(m["unavailable_reason"])
            self.assertIn("candidate ranking test set", m["unavailable_reason"].lower())

    def test_05_visual_analytics_mlvis_benchmark_isolation(self):
        """
        Verifies MLVis chart recommendation benchmark models are provided
        under their own task protocol (Top-3) and not mixed as product models.
        """
        data = get_model_analytics_summary("demo_ecommerce")
        visual_models = data["visual_models_comparison"]
        self.assertEqual(len(visual_models), 3)

        # Baseline, MLVis v1, MLVis v2
        model_ids = [m["model_id"] for m in visual_models]
        self.assertIn("mlvis_statistical_baseline", model_ids)
        self.assertIn("mlvis_v1", model_ids)
        self.assertIn("mlvis_v2", model_ids)

        v1 = next(m for m in visual_models if m["model_id"] == "mlvis_v1")
        self.assertEqual(v1["k"], 3)
        self.assertAlmostEqual(v1["precision_at_3"], 0.9987, places=3)
        self.assertEqual(v1["precision_formatted"], "99.87%")
        self.assertEqual(v1["evaluated_cases"], 783)
        self.assertEqual(v1["task"], "Visual Analytics & Chart Recommendation")

    def test_06_api_models_analytics_endpoint(self):
        """Verifies GET /api/models/analytics endpoint handler."""
        db = SessionLocal()
        try:
            data = api_get_model_analytics(client_id="demo_ecommerce", db=db)
            self.assertEqual(data["client_id"], "demo_ecommerce")
            self.assertGreaterEqual(len(data["models_comparison"]), 5)
            self.assertIn("visual_models_comparison", data)
            self.assertIn("modality_variants", data)
        finally:
            db.close()

    def test_07_api_clients_analytics_endpoint(self):
        """Verifies GET /api/clients/{client_id}/analytics endpoint handler."""
        db = SessionLocal()
        try:
            data = api_get_client_analytics(client_id="demo_ecommerce", db=db)
            self.assertEqual(data["client_id"], "demo_ecommerce")
            self.assertEqual(data["client_name"], "Indian E-Commerce Customer Behavior & Purchase")
        finally:
            db.close()


if __name__ == "__main__":
    unittest.main()
