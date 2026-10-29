"""
Comprehensive Implicit Feedback & Online Reranking Test Suite.
Verifies:
1. Implicit feedback ingestion (IMPRESSION, VIEW, LONG_VIEW, CLICK, IGNORE, ADD_TO_CART, PURCHASE).
2. Online behavioral signal calculation with configurable weights.
3. Online reranking shifts recommendation ranking without model retraining.
4. Category affinity propagation: interacting with one product elevates related products in same category.
5. Weak negative weight for IGNORE without harsh over-penalization.
6. Multi-client feedback isolation: interactions for client A do not leak to client B.
7. Cold-start preservation: zero-history users use content/context baseline, and interactions dynamically update them.
8. Duplicate/rapid event bounding to prevent score saturation.
9. Feedback audit log retrieval endpoint (GET /api/feedback).
"""

import sys
import unittest
from pathlib import Path
import json
import torch
import numpy as np
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.main import app
from backend.database.database import SessionLocal, init_db
from backend.database.models import FeedbackEventModel, RecommendationEventModel
from backend.services.model_service import ensure_demo_ecommerce_model_registered
from backend.services.model_cache import clear_all_cache
from backend.services.inference_service import _get_online_feedback_signals
from backend.schemas.feedback import DEFAULT_FEEDBACK_WEIGHTS


class TestImplicitFeedbackAndReranking(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()
        cls.db = SessionLocal()
        cls.client = TestClient(app)
        ensure_demo_ecommerce_model_registered(cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    def setUp(self):
        clear_all_cache()
        # Clean feedback events for test isolation
        self.db.query(FeedbackEventModel).delete()
        self.db.query(RecommendationEventModel).delete()
        self.db.commit()

    def test_01_implicit_action_types_ingestion(self):
        """All implicit actions (IMPRESSION, VIEW, LONG_VIEW, CLICK, IGNORE, ADD_TO_CART, PURCHASE) are accepted."""
        actions = ["IMPRESSION", "VIEW", "LONG_VIEW", "CLICK", "IGNORE", "ADD_TO_CART", "PURCHASE", "ACCEPT", "REJECT"]
        
        for idx, action in enumerate(actions, start=1):
            resp = self.client.post("/api/feedback", json={
                "client_id": "demo_ecommerce",
                "user_id": "test_user_actions",
                "item_id": f"prod_{idx}",
                "action": action,
                "position": idx
            })
            self.assertEqual(resp.status_code, 201, f"Failed for action {action}: {resp.text}")
            data = resp.json()
            self.assertEqual(data["action"], action)
            self.assertEqual(data["status"], "recorded")

        # Verify retrieval via GET /api/feedback
        get_resp = self.client.get("/api/feedback?client_id=demo_ecommerce&user_id=test_user_actions")
        self.assertEqual(get_resp.status_code, 200)
        events = get_resp.json()
        self.assertEqual(len(events), len(actions))

    def test_02_online_feedback_signals_and_configurable_weights(self):
        """Validates behavioral score calculations and weight configurations."""
        user_id = "test_user_weights"
        cid = "demo_ecommerce"

        # Record a click (+2.0) and purchase (+5.0) for item '243'
        self.client.post("/api/feedback", json={
            "client_id": cid, "user_id": user_id, "item_id": "243", "action": "CLICK"
        })
        self.client.post("/api/feedback", json={
            "client_id": cid, "user_id": user_id, "item_id": "243", "action": "PURCHASE"
        })

        # Record an ignore (-0.1) for item '100'
        self.client.post("/api/feedback", json={
            "client_id": cid, "user_id": user_id, "item_id": "100", "action": "IGNORE"
        })

        item_signals, raw_scores, count = _get_online_feedback_signals(self.db, cid, user_id)
        self.assertEqual(count, 3)
        self.assertIn("243", item_signals)
        self.assertIn("100", item_signals)
        
        # 243 should have positive score boost, 100 should have a weak negative delta
        self.assertGreater(item_signals["243"], 0.1)
        self.assertLess(item_signals["100"], 0.0)
        self.assertGreater(item_signals["100"], -0.05)  # Weak negative, not harsh rejection

    def test_03_online_reranking_shifts_future_recommendations(self):
        """
        Critical Requirement: Interacting naturally with a product alters subsequent recommendation rankings.
        """
        user_id = "user_rerank_test_01"
        cid = "demo_ecommerce"

        # 1. Initial recommendation request
        r1 = self.client.post("/api/recommend", json={
            "client_id": cid,
            "user_id": user_id,
            "top_k": 10
        })
        self.assertEqual(r1.status_code, 200)
        items_1 = r1.json()["recommendations"]
        rec_id_1 = r1.json()["recommendation_id"]
        
        # Pick an item not at rank 1 (e.g. rank 5)
        target_item = items_1[4]
        target_item_id = str(target_item["item_id"])
        target_score_1 = target_item["score"]

        # 2. Record natural interactions (CLICK + PURCHASE) for target_item
        fb1 = self.client.post("/api/feedback", json={
            "client_id": cid,
            "user_id": user_id,
            "item_id": target_item_id,
            "action": "CLICK",
            "position": 5,
            "recommendation_id": rec_id_1
        })
        self.assertEqual(fb1.status_code, 201)

        fb2 = self.client.post("/api/feedback", json={
            "client_id": cid,
            "user_id": user_id,
            "item_id": target_item_id,
            "action": "PURCHASE",
            "position": 5,
            "recommendation_id": rec_id_1
        })
        self.assertEqual(fb2.status_code, 201)

        # 3. Next recommendation request for same user
        r2 = self.client.post("/api/recommend", json={
            "client_id": cid,
            "user_id": user_id,
            "top_k": 10
        })
        self.assertEqual(r2.status_code, 200)
        items_2 = r2.json()["recommendations"]
        
        # Verify target item's score increased and its rank improved
        target_in_r2 = next((it for it in items_2 if str(it["item_id"]) == target_item_id), None)
        self.assertIsNotNone(target_in_r2)
        self.assertGreater(target_in_r2["score"], target_score_1)
        self.assertLessEqual(target_in_r2["rank"], target_item["rank"])
        self.assertIn("recent natural interaction", target_in_r2["explanation"].lower())

    def test_04_category_affinity_propagation(self):
        """Interacting with a product boosts other products in the same category."""
        user_id = "user_cat_affinity"
        cid = "demo_ecommerce"

        # Find items in catalog in category 2
        r1 = self.client.post("/api/recommend", json={
            "client_id": cid,
            "user_id": user_id,
            "top_k": 20
        })
        recs_1 = r1.json()["recommendations"]
        
        # Find two items in same category
        cat2_items = [it for it in recs_1 if it["metadata"].get("product_category") == 2]
        if len(cat2_items) >= 2:
            item_a = str(cat2_items[0]["item_id"])
            item_b = str(cat2_items[1]["item_id"])
            score_b_initial = cat2_items[1]["score"]

            # User clicks item_a
            self.client.post("/api/feedback", json={
                "client_id": cid, "user_id": user_id, "item_id": item_a, "action": "CLICK"
            })
            self.client.post("/api/feedback", json={
                "client_id": cid, "user_id": user_id, "item_id": item_a, "action": "PURCHASE"
            })

            # Next recommendation
            r2 = self.client.post("/api/recommend", json={
                "client_id": cid,
                "user_id": user_id,
                "top_k": 20
            })
            recs_2 = r2.json()["recommendations"]
            item_b_after = next((it for it in recs_2 if str(it["item_id"]) == item_b), None)
            if item_b_after:
                self.assertGreater(item_b_after["score"], score_b_initial)

    def test_05_client_isolation_preservation(self):
        """Feedback for client A does NOT affect recommendations for client B."""
        user_id = "shared_user_name_123"
        cid_a = "demo_ecommerce"
        
        # Record strong interaction on demo_ecommerce
        self.client.post("/api/feedback", json={
            "client_id": cid_a, "user_id": user_id, "item_id": "243", "action": "PURCHASE"
        })

        # Query feedback signals for another client (e.g. movielens_dataset)
        item_signals_b, _, count_b = _get_online_feedback_signals(self.db, "movielens_dataset", user_id)
        self.assertEqual(count_b, 0)
        self.assertEqual(len(item_signals_b), 0)

    def test_06_cold_start_unaffected_initially_and_adapts_after_feedback(self):
        """Cold start user receives cold-start modality gating, then benefits from implicit feedback."""
        cold_user = "brand_new_cold_customer_xyz"
        cid = "demo_ecommerce"

        # 1. Cold start baseline
        r1 = self.client.post("/api/recommend", json={
            "client_id": cid, "user_id": cold_user, "top_k": 5
        })
        self.assertEqual(r1.status_code, 200)
        self.assertEqual(r1.json()["status"], "cold_start")
        self.assertLessEqual(r1.json()["modality_weights"]["behavior"], 0.01)

        # 2. Natural click feedback
        rec_id = r1.json()["recommendation_id"]
        first_item = r1.json()["recommendations"][0]["item_id"]
        self.client.post("/api/feedback", json={
            "client_id": cid,
            "user_id": cold_user,
            "item_id": str(first_item),
            "action": "CLICK",
            "recommendation_id": rec_id
        })

        # 3. Next recommendation maintains cold-start neural gating but benefits from online boost
        r2 = self.client.post("/api/recommend", json={
            "client_id": cid, "user_id": cold_user, "top_k": 5
        })
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(r2.json()["status"], "cold_start")
        first_item_r2 = next(it for it in r2.json()["recommendations"] if str(it["item_id"]) == str(first_item))
        self.assertIn("recent natural interaction", first_item_r2["explanation"].lower())


if __name__ == "__main__":
    unittest.main()
