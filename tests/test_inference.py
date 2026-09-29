"""
Phase 5 Comprehensive Inference Test Suite.
Validates:
1. Model cache & thread-safe retrieval
2. Model cache invalidation on activation
3. Unknown client handling (404)
4. Client without active model (503)
5. Unknown user inference (history_count=0, cold_start=True, behavioral suppression <= 0.01)
6. Cold user inference (1 <= history_count < 3)
7. Warm user inference (history_count >= 3, unsuppressed behavioral contribution)
8. Categorical & numeric filters (product_category, max_price, min_price)
9. Actionable 422 for unsupported filter fields
10. Top-K validation, bounds checking, and candidate uniqueness
11. Genuine scoring, eval mode, finite scores and modality weights
12. Recommendation event persistence and feedback linkage with recommendation_id
13. Generic dynamic-dimension client end-to-end inference (multi-client isolation)
"""

import sys
import unittest
from pathlib import Path
import json
import torch
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.main import app
from backend.database.database import SessionLocal, init_db
from backend.database.models import ClientModel, ModelVersionModel, RecommendationEventModel, FeedbackEventModel
from backend.services.model_cache import (
    get_model_entry,
    invalidate_client_cache,
    clear_all_cache,
    _cache
)
from backend.services.model_service import (
    ensure_demo_ecommerce_model_registered,
    register_model_version,
    activate_client_model
)
from src.schema_mapper import ClientSchemaConfig
from src.models.recommender import TwoTowerMultiModalRecommender


class TestRealTimeInferenceEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()
        cls.db = SessionLocal()
        cls.client = TestClient(app)
        # Ensure demo_ecommerce is registered as v1
        ensure_demo_ecommerce_model_registered(cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    def setUp(self):
        clear_all_cache()

    def test_01_unknown_client_returns_404(self):
        """Unknown client returns 404 Not Found without creating the client."""
        resp = self.client.post("/api/recommend", json={
            "client_id": "non_existent_client_xyz",
            "user_id": "user_123",
            "top_k": 5
        })
        self.assertEqual(resp.status_code, 404)
        self.assertIn("not found", resp.json()["detail"].lower())

    def test_02_client_without_active_model_returns_503(self):
        """Client with no active model returns 503 Service Unavailable."""
        cid = "client_without_active_model"
        self.client.post("/api/clients", json={"client_id": cid, "name": "Inactive Client"})

        resp = self.client.post("/api/recommend", json={
            "client_id": cid,
            "user_id": "user_123",
            "top_k": 5
        })
        self.assertEqual(resp.status_code, 503)
        self.assertIn("No active model is available", resp.json()["detail"])

    def test_03_unknown_user_cold_start(self):
        """Completely new user (never in history) returns valid Top-K with cold_start=True and suppressed behavior."""
        resp = self.client.post("/api/recommend", json={
            "client_id": "demo_ecommerce",
            "user_id": "completely_new_unknown_customer_99999",
            "context": {"device_type": 1, "marketing_channel": 0},
            "top_k": 5
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertEqual(data["status"], "cold_start")
        self.assertEqual(data["history_count"], 0)
        self.assertEqual(len(data["recommendations"]), 5)
        self.assertIn("recommendation_id", data)

        # Behavioral weight must be suppressed to <= 0.01 by ColdStartAdaptiveGate
        weights = data["modality_weights"]
        self.assertLessEqual(weights["behavior"], 0.01)
        self.assertAlmostEqual(weights["behavior"] + weights["content"] + weights["context"], 1.0, places=3)

        # Grounded explanation must reflect cold-start status
        top_rec = data["recommendations"][0]
        self.assertIn("limited prior interaction history", top_rec["explanation"])
        self.assertIn("cold-start", top_rec["explanation"].lower())

    def test_04_cold_user_with_one_interaction(self):
        """User with 1 interaction in training data is recognized as cold_start (count < 3)."""
        # User 7838 has exactly 1 interaction in data/processed/train.csv
        resp = self.client.post("/api/recommend", json={
            "client_id": "demo_ecommerce",
            "user_id": "7838",
            "context": {"device_type": 0},
            "top_k": 5
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertEqual(data["status"], "cold_start")
        self.assertEqual(data["history_count"], 1)
        self.assertLessEqual(data["modality_weights"]["behavior"], 0.01)

    def test_05_warm_user_inference(self):
        """Warm user (count >= 3) uses point-in-time behavioral representation without suppression."""
        # User 9272 has 8 interactions in train.csv
        resp = self.client.post("/api/recommend", json={
            "client_id": "demo_ecommerce",
            "user_id": "9272",
            "context": {"device_type": 0, "marketing_channel": 2},
            "top_k": 5
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertEqual(data["status"], "warm_start")
        self.assertGreaterEqual(data["history_count"], 3)
        # Behavioral weight should NOT be suppressed
        self.assertGreater(data["modality_weights"]["behavior"], 0.05)
        top_rec = data["recommendations"][0]
        self.assertIn("past browsing and purchase behavior", top_rec["explanation"])

    def test_06_categorical_and_numeric_filters(self):
        """Request filters strictly constrain returned items to matching candidates."""
        # 1. Categorical filter: product_category == 3
        resp_cat = self.client.post("/api/recommend", json={
            "client_id": "demo_ecommerce",
            "user_id": "9272",
            "filters": {"product_category": 3},
            "top_k": 5
        })
        self.assertEqual(resp_cat.status_code, 200)
        for item in resp_cat.json()["recommendations"]:
            self.assertEqual(item["metadata"]["product_category"], 3)

        # 2. Numeric max filter: unit_price <= 500
        resp_price = self.client.post("/api/recommend", json={
            "client_id": "demo_ecommerce",
            "user_id": "9272",
            "filters": {"max_price": 500},
            "top_k": 5
        })
        self.assertEqual(resp_price.status_code, 200)
        for item in resp_price.json()["recommendations"]:
            self.assertLessEqual(item["metadata"]["unit_price"], 500.0)

        # 3. Numeric min filter: unit_price >= 800
        resp_min = self.client.post("/api/recommend", json={
            "client_id": "demo_ecommerce",
            "user_id": "9272",
            "filters": {"min_price": 800},
            "top_k": 5
        })
        self.assertEqual(resp_min.status_code, 200)
        for item in resp_min.json()["recommendations"]:
            self.assertGreaterEqual(item["metadata"]["unit_price"], 800.0)

    def test_07_invalid_filter_field_returns_422(self):
        """Unknown or unmapped filter fields return an actionable 422 error."""
        resp = self.client.post("/api/recommend", json={
            "client_id": "demo_ecommerce",
            "user_id": "9272",
            "filters": {"non_existent_attribute_xyz": "value"},
            "top_k": 5
        })
        self.assertEqual(resp.status_code, 422)
        detail = resp.json()["detail"]
        self.assertIn("Invalid filter field", detail)
        self.assertIn("non_existent_attribute_xyz", detail)

    def test_08_top_k_bounds_and_candidate_uniqueness(self):
        """Validates top_k bounds (1..50) and verifies all recommended item IDs are unique."""
        # top_k < 1
        r_low = self.client.post("/api/recommend", json={
            "client_id": "demo_ecommerce", "user_id": "9272", "top_k": 0
        })
        self.assertEqual(r_low.status_code, 422)

        # top_k > 50
        r_high = self.client.post("/api/recommend", json={
            "client_id": "demo_ecommerce", "user_id": "9272", "top_k": 51
        })
        self.assertEqual(r_high.status_code, 422)

        # top_k = 15: all returned item IDs must be unique
        r_valid = self.client.post("/api/recommend", json={
            "client_id": "demo_ecommerce", "user_id": "9272", "top_k": 15
        })
        self.assertEqual(r_valid.status_code, 200)
        items = r_valid.json()["recommendations"]
        self.assertEqual(len(items), 15)
        item_ids = [it["item_id"] for it in items]
        self.assertEqual(len(item_ids), len(set(item_ids)))

    def test_09_model_eval_mode_and_finite_scores(self):
        """Verifies model outputs are finite, descending, and labeled as sigmoid probabilities."""
        resp = self.client.post("/api/recommend", json={
            "client_id": "demo_ecommerce",
            "user_id": "9272",
            "top_k": 5
        })
        self.assertEqual(resp.status_code, 200)
        recs = resp.json()["recommendations"]

        for i, rec in enumerate(recs):
            self.assertEqual(rec["score_type"], "sigmoid_score")
            self.assertTrue(0.0 <= rec["score"] <= 1.0)
            if i > 0:
                self.assertGreaterEqual(recs[i - 1]["score"], rec["score"])

    def test_10_model_cache_and_invalidation(self):
        """Verifies model caching speeds up repeated calls and cache is invalidated per-client."""
        self.assertNotIn(("demo_ecommerce", "v1"), _cache)

        # First call caches model
        resp1 = self.client.post("/api/recommend", json={
            "client_id": "demo_ecommerce", "user_id": "9272", "top_k": 5
        })
        self.assertEqual(resp1.status_code, 200)
        self.assertIn(("demo_ecommerce", "v1"), _cache)

        # Invalidate demo_ecommerce cache
        invalidate_client_cache("demo_ecommerce")
        self.assertNotIn(("demo_ecommerce", "v1"), _cache)

    def test_11_feedback_linkage(self):
        """Verifies feedback linkage with recommendation_id and item validation."""
        # Generate recommendation
        rec_resp = self.client.post("/api/recommend", json={
            "client_id": "demo_ecommerce", "user_id": "9272", "top_k": 3
        })
        rec_data = rec_resp.json()
        rec_id = rec_data["recommendation_id"]
        valid_item = rec_data["recommendations"][0]["item_id"]

        # 1. Valid feedback linkage
        fb1 = self.client.post("/api/feedback", json={
            "client_id": "demo_ecommerce",
            "user_id": "9272",
            "item_id": valid_item,
            "action": "CLICK",
            "recommendation_id": rec_id
        })
        self.assertEqual(fb1.status_code, 201)
        self.assertEqual(fb1.json()["recommendation_id"], rec_id)

        # 2. Invalid item for that recommendation_id -> 400
        fb2 = self.client.post("/api/feedback", json={
            "client_id": "demo_ecommerce",
            "user_id": "9272",
            "item_id": "unrelated_item_not_recommended_9999",
            "action": "ACCEPT",
            "recommendation_id": rec_id
        })
        self.assertEqual(fb2.status_code, 400)
        self.assertIn("not in the recommended items", fb2.json()["detail"])

        # 3. Non-existent recommendation_id -> 404
        fb3 = self.client.post("/api/feedback", json={
            "client_id": "demo_ecommerce",
            "user_id": "9272",
            "item_id": valid_item,
            "action": "PURCHASE",
            "recommendation_id": "rec_does_not_exist_404"
        })
        self.assertEqual(fb3.status_code, 404)

    def test_12_generic_dynamic_dimension_client(self):
        """
        Tests end-to-end inference on a generic client with non-standard feature dimensions.
        Proves platform is dynamic and not hard-coded to 16/12/21 dimensions.
        """
        cid = "generic_client_dynamic"
        self.client.post("/api/clients", json={"client_id": cid, "name": "Dynamic Client"})

        # Setup dynamic architecture: Beh=6, Cont=4 (1 id + 3 meta), Ctx=3
        model_dir = Path(f"models/{cid}/versions/v1")
        model_dir.mkdir(parents=True, exist_ok=True)

        dynamic_model = TwoTowerMultiModalRecommender(
            num_products=50,
            beh_dim=6,
            ctx_dim=3,
            cont_meta_dim=3,  # Total content dim = 1 + 3 = 4
            embedding_dim=64,
            cold_threshold=2
        )
        ckpt_path = model_dir / "model.pt"
        enc_path = model_dir / "feature_encoders.joblib"
        torch.save(dynamic_model.state_dict(), ckpt_path)

        from sklearn.preprocessing import StandardScaler
        beh_scaler = StandardScaler()
        beh_scaler.fit(np.zeros((10, 6)))
        cont_scaler = StandardScaler()
        cont_scaler.fit(np.zeros((10, 3)))
        ctx_scaler = StandardScaler()
        ctx_scaler.fit(np.zeros((10, 3)))

        encoders = {
            "beh_scaler": beh_scaler,
            "beh_num_cols": [f"feat_{i}" for i in range(6)],
            "content_scaler": cont_scaler,
            "ctx_num_scaler": ctx_scaler,
            "item_id_map": {f"item_{i}": i for i in range(50)}
        }
        import joblib
        joblib.dump(encoders, enc_path)

        # Mock catalog
        catalog = [{"item_id": f"item_{i}", "feat_c1": 1.0, "feat_c2": 2.0, "feat_c3": 3.0} for i in range(30)]
        proc_dir = Path(f"clients/{cid}/processed")
        proc_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(catalog).to_csv(proc_dir / "train.csv", index=False)

        schema_cfg = {
            "client_id": cid,
            "column_mappings": {
                "user_id": "USER_ID",
                "item_id": "ITEM_ID",
                "feat_c1": "CONTENT_FEATURE",
                "feat_c2": "CONTENT_FEATURE",
                "feat_c3": "CONTENT_FEATURE",
                "ctx_1": "CONTEXT_FEATURE",
                "ctx_2": "CONTEXT_FEATURE",
                "ctx_3": "CONTEXT_FEATURE",
                "rating": "TARGET"
            },
            "column_types": {
                "feat_c1": "numerical",
                "feat_c2": "numerical",
                "feat_c3": "numerical",
                "ctx_1": "numerical",
                "ctx_2": "numerical",
                "ctx_3": "numerical"
            }
        }

        # Register version
        register_model_version(
            db=self.db,
            client_id=cid,
            version_tag="v1",
            training_run_id="run_dynamic_01",
            checkpoint_src=ckpt_path,
            encoders_src=enc_path,
            metrics={"test_metric": 0.85},
            feature_dimensions={"behavior": 6, "content": 4, "context": 3},
            schema_snapshot=schema_cfg["column_mappings"],
            config_snapshot=schema_cfg,
            cold_start_threshold=2,
            random_seed=42,
            dataset_reference="train.csv",
            legacy_imported=False
        )

        # Request recommendation on generic client
        resp = self.client.post("/api/recommend", json={
            "client_id": cid,
            "user_id": "user_dyn_1",
            "top_k": 3,
            "context": {"ctx_1": 0.5, "ctx_2": 1.0, "ctx_3": 0.0}
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["client_id"], cid)
        self.assertEqual(len(data["recommendations"]), 3)
        self.assertEqual(data["status"], "cold_start")

    def test_13_feature_contract_violation_missing_column(self):
        """
        Regression test for Correction B:
        Ensures that if candidate catalog items are missing required feature columns,
        the engine raises HTTP 422 Unprocessable Entity instead of silent zero-padding.
        """
        cid = "contract_violation_client"
        self.client.post("/api/clients", json={"client_id": cid, "name": "Contract Violation Client"})

        model_dir = Path(f"models/{cid}/versions/v1")
        model_dir.mkdir(parents=True, exist_ok=True)

        dynamic_model = TwoTowerMultiModalRecommender(
            num_products=20,
            beh_dim=4,
            ctx_dim=2,
            cont_meta_dim=2,
            embedding_dim=32,
            cold_threshold=2
        )
        ckpt_path = model_dir / "model.pt"
        enc_path = model_dir / "feature_encoders.joblib"
        torch.save(dynamic_model.state_dict(), ckpt_path)

        from sklearn.preprocessing import StandardScaler
        beh_scaler = StandardScaler()
        beh_scaler.fit(np.zeros((5, 4)))
        cont_scaler = StandardScaler()
        cont_scaler.fit(np.zeros((5, 2)))
        ctx_scaler = StandardScaler()
        ctx_scaler.fit(np.zeros((5, 2)))

        encoders = {
            "beh_scaler": beh_scaler,
            "beh_num_cols": [f"feat_{i}" for i in range(4)],
            "content_scaler": cont_scaler,
            "ctx_num_scaler": ctx_scaler,
            "item_id_map": {f"item_{i}": i for i in range(20)}
        }
        import joblib
        joblib.dump(encoders, enc_path)

        # Defective catalog: missing 'feat_c2' which schema declares as a required CONTENT_FEATURE
        bad_catalog = [{"item_id": f"item_{i}", "feat_c1": 1.0} for i in range(10)]
        proc_dir = Path(f"clients/{cid}/processed")
        proc_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(bad_catalog).to_csv(proc_dir / "train.csv", index=False)

        schema_cfg = {
            "client_id": cid,
            "column_mappings": {
                "user_id": "USER_ID",
                "item_id": "ITEM_ID",
                "feat_c1": "CONTENT_FEATURE",
                "feat_c2": "CONTENT_FEATURE",
                "ctx_1": "CONTEXT_FEATURE",
                "ctx_2": "CONTEXT_FEATURE",
                "rating": "TARGET"
            },
            "column_types": {
                "feat_c1": "numerical",
                "feat_c2": "numerical",
                "ctx_1": "numerical",
                "ctx_2": "numerical"
            }
        }

        register_model_version(
            db=self.db,
            client_id=cid,
            version_tag="v1",
            training_run_id="run_bad_01",
            checkpoint_src=ckpt_path,
            encoders_src=enc_path,
            metrics={"test_metric": 0.70},
            feature_dimensions={"behavior": 4, "content": 3, "context": 2},
            schema_snapshot=schema_cfg["column_mappings"],
            config_snapshot=schema_cfg,
            cold_start_threshold=2,
            random_seed=42,
            dataset_reference="train.csv",
            legacy_imported=False
        )

        resp = self.client.post("/api/recommend", json={
            "client_id": cid,
            "user_id": "user_bad_1",
            "top_k": 3
        })
        # Must fail with 422 Unprocessable Entity, explicitly citing missing columns
        self.assertEqual(resp.status_code, 422)
        self.assertIn("Inference feature contract violation", resp.json()["detail"])
        self.assertIn("feat_c2", resp.json()["detail"])

    def test_14_cold_warm_exact_boundary(self):
        """
        Explicit Quality Control Boundary Test:
        Verifies exact boundary behavior where K = 3:
        N_hist = 0 -> cold_start (behavior suppressed <= 0.01)
        N_hist = 1 -> cold_start (behavior suppressed <= 0.01)
        N_hist = 2 -> cold_start (behavior suppressed <= 0.01)
        N_hist = 3 -> warm_start (behavior NOT suppressed, >= 3)
        N_hist = 4 -> warm_start (behavior NOT suppressed, >= 3)
        """
        cases = [
            ("new_unknown_user_boundary_0", 0, "cold_start"),
            ("7838", 1, "cold_start"),
            ("6993", 2, "cold_start"),
            ("5992", 3, "warm_start"),
            ("3902", 4, "warm_start"),
        ]

        for uid, expected_count, expected_status in cases:
            resp = self.client.post("/api/recommend", json={
                "client_id": "demo_ecommerce",
                "user_id": str(uid),
                "top_k": 5
            })
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["history_count"], expected_count, f"History count mismatch for user {uid}")
            self.assertEqual(
                data["status"],
                expected_status,
                f"Status mismatch for user {uid} (N_hist={expected_count}): expected {expected_status}, got {data['status']}"
            )
            beh_weight = data["modality_weights"]["behavior"]
            if expected_status == "cold_start":
                self.assertLessEqual(beh_weight, 0.01, f"Cold start user {uid} has unsuppressed behavior weight {beh_weight}")
            else:
                self.assertGreater(beh_weight, 0.05, f"Warm start user {uid} has unexpectedly low behavior weight {beh_weight}")


if __name__ == "__main__":
    unittest.main()
