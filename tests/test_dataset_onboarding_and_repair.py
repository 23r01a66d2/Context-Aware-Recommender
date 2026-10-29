"""
Comprehensive Unit & Integration Tests for:
- Automatic ID generation and schema repair
- Generated-field labeling
- Image column/path detection and rendering
- Human-readable display-name resolution
- Missing behavioral history handling (Content-Only / Cold-Start classification)
- Adaptive Representation Fusion weights computation
- Multi-client model isolation
- Fashion dataset onboarding, training, and real-time candidate inference
"""

import os
import unittest
import pandas as pd
import numpy as np
import torch
from pathlib import Path

from src.schema_mapper import (
    ColumnType,
    SchemaRole,
    infer_column_type,
    repair_and_augment_dataset_dataframe,
    compute_dataset_capabilities,
    ClientSchemaConfig
)
from src.dataset_onboarding import detect_dataset_schema, onboard_fashion_dataset
from backend.database.database import SessionLocal
from backend.schemas.recommendation import RecommendationRequest
from backend.services.inference_service import (
    generate_recommendations,
    _resolve_item_display_name,
    _resolve_item_image
)
from backend.services.client_service import get_clients_overview


class TestDatasetOnboardingAndRepair(unittest.TestCase):

    def setUp(self):
        self.db = SessionLocal()

    def tearDown(self):
        self.db.close()

    def test_automatic_id_generation(self):
        """Test automatic stable ID generation for datasets lacking item_id."""
        raw_data = {
            "title": ["Silk Dress", "Running Shoes", "Leather Jacket"],
            "category": ["Apparel", "Footwear", "Apparel"],
            "price": [120.0, 85.0, 240.0]
        }
        df = pd.DataFrame(raw_data)
        repaired_df, generated_fields = repair_and_augment_dataset_dataframe(df, client_id="test_store")

        self.assertIn("item_id", repaired_df.columns)
        self.assertEqual(len(repaired_df), 3)
        self.assertEqual(repaired_df["item_id"].tolist(), ["test_store_000001", "test_store_000002", "test_store_000003"])
        self.assertIn("item_id", generated_fields)

    def test_generated_fields_labeling(self):
        """Test that schema detection explicitly tracks generated fields."""
        raw_data = {
            "name": ["Espresso", "Latte"],
            "category": ["Beverage", "Beverage"]
        }
        df = pd.DataFrame(raw_data)
        repaired_df, generated = repair_and_augment_dataset_dataframe(df, client_id="coffee_shop")
        self.assertIn("item_id", generated)

        # Ensure display name is resolved and preserved
        self.assertIn("name", repaired_df.columns)

    def test_image_column_and_path_detection(self):
        """Test automatic detection of image columns and image file paths."""
        series_img = pd.Series(["img_01.jpg", "img_02.png", "img_03.webp"], name="picture")
        col_type = infer_column_type(series_img)
        self.assertEqual(col_type, ColumnType.IMAGE.value)

        cand_item = {"picture": "18872.jpg", "category": "Footwear"}
        img_url = _resolve_item_image(cand_item, "fashion_client", "18872")
        self.assertIsNotNone(img_url)
        self.assertTrue(img_url.startswith("/api/images/fashion_client/"))

    def test_display_name_resolution(self):
        """Test display name resolution across multiple domain types without returning raw IDs alone."""
        # 1. Title/Name
        item1 = {"movie_title": "Toy Story (1995)", "item_id": 1}
        self.assertEqual(_resolve_item_display_name(item1, "movielens_dataset", "1"), "Toy Story (1995)")

        # 2. Location/AR
        item2 = {"recommended_content_type": "Historical_Monument", "location_id": "L007", "cultural_interest_category": "Temple"}
        self.assertEqual(_resolve_item_display_name(item2, "ar_client", "10"), "Historical Monument: Temple (Location L007)")

        # 3. Category + SKU Fallback
        item3 = {"product_category": 1, "unit_price": 49.99}
        name3 = _resolve_item_display_name(item3, "demo_ecommerce", "52")
        self.assertIn("Fashion & Apparel", name3)
        self.assertIn("SKU-52", name3)

    def test_missing_behavioral_fields_classification(self):
        """Test that catalogs without interaction logs are classified as content-only / cold-start capable."""
        column_mappings = {
            "item_id": SchemaRole.ITEM_ID.value,
            "category": SchemaRole.CONTENT_FEATURE.value,
            "price": SchemaRole.CONTENT_FEATURE.value
        }
        column_types = {
            "item_id": ColumnType.ID.value,
            "category": ColumnType.CATEGORICAL.value,
            "price": ColumnType.NUMERIC.value
        }
        caps = compute_dataset_capabilities(column_mappings, column_types)
        self.assertTrue(caps.get("content_only_cold_start"))
        self.assertFalse(caps.get("has_behavior_interactions"))

    def test_adaptive_fusion_weights_cold_start(self):
        """Test that the live recommender generates adaptive fusion weights and gates behavior on cold users."""
        req = RecommendationRequest(
            client_id="demo_ecommerce",
            user_id="cold_user_999999",
            top_k=3,
            context={"device_type": "Mobile", "marketing_channel": "Direct"}
        )
        res = generate_recommendations(req, self.db)
        self.assertIsNotNone(res.modality_weights)
        self.assertIn("behavior", res.modality_weights)
        self.assertIn("content", res.modality_weights)
        self.assertIn("context", res.modality_weights)
        # Cold start user should have behavior weight penalized/gated down close to 0
        self.assertLessEqual(res.modality_weights["behavior"], 0.05)
        self.assertGreater(res.modality_weights["content"] + res.modality_weights["context"], 0.90)

    def test_client_and_model_isolation(self):
        """Test that multiple clients maintain separate catalogs, encoders, and checkpoints."""
        req_demo = RecommendationRequest(client_id="demo_ecommerce", user_id="9272", top_k=2, context={})
        res_demo = generate_recommendations(req_demo, self.db)

        req_ml = RecommendationRequest(client_id="movielens_dataset", user_id="1", top_k=2, context={})
        res_ml = generate_recommendations(req_ml, self.db)

        self.assertNotEqual(res_demo.recommendations[0].display_name, res_ml.recommendations[0].display_name)
        self.assertIn("SKU", res_demo.recommendations[0].display_name)

    def test_fashion_client_inference_ready(self):
        """Test that fashion_client is inference ready and returns valid candidate recommendations."""
        req = RecommendationRequest(
            client_id="fashion_client",
            user_id="anon_user_42",
            top_k=5,
            context={"device_type": "Desktop"}
        )
        res = generate_recommendations(req, self.db)
        self.assertEqual(len(res.recommendations), 5)
        for item in res.recommendations:
            self.assertIsNotNone(item.display_name)
            self.assertFalse(item.display_name.isdigit())
            self.assertGreater(item.score, 0.0)


if __name__ == "__main__":
    unittest.main()
