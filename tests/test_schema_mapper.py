"""
Unit and Integration Tests for Schema Mapping Engine & Generic Client Data Processing (Phase 2)
Tests:
1. Role definitions & auto-suggestion heuristics
2. Schema validation & leakage warnings
3. Client configuration serialization (YAML / dict)
4. Backward compatibility with demo_ecommerce (guaranteeing exact 16/12/21 dimensions)
5. Generic schema execution on a synthetic second client with dynamic feature dimensions
"""

import os
import sys
import unittest
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.schema_mapper import (
    SchemaRole,
    ColumnType,
    ClientSchemaConfig,
    auto_suggest_mappings,
    validate_schema,
    create_demo_ecommerce_schema,
    get_or_create_client_config
)
from src.preprocessing import clean_and_split_data
from src.feature_engineering import (
    build_point_in_time_behavioral_features,
    verify_zero_leakage,
    prepare_multimodal_features
)
from src.models.recommender import TwoTowerMultiModalRecommender


class TestSchemaMapper(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path("outputs/test_clients")
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_auto_suggestion_heuristics(self):
        """Verify heuristic rule detection on varied column names."""
        df_sample = pd.DataFrame({
            "buyer_id": [1, 2, 3],
            "article_id": [101, 102, 103],
            "event_time": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "is_ordered": [0, 1, 0],
            "order_revenue": [0.0, 49.9, 0.0],
            "pages_viewed": [3, 12, 1],
            "unit_price": [25.0, 49.9, 15.0],
            "device_type": ["Mobile", "Desktop", "Mobile"]
        })

        mappings, inferred_types = auto_suggest_mappings(df_sample)

        self.assertEqual(mappings["buyer_id"], SchemaRole.USER_ID.value)
        self.assertEqual(mappings["article_id"], SchemaRole.ITEM_ID.value)
        self.assertEqual(mappings["event_time"], SchemaRole.TIMESTAMP.value)
        self.assertEqual(mappings["is_ordered"], SchemaRole.TARGET.value)
        self.assertEqual(mappings["order_revenue"], SchemaRole.IGNORE.value)
        self.assertEqual(mappings["pages_viewed"], SchemaRole.BEHAVIOR_FEATURE.value)
        self.assertEqual(mappings["unit_price"], SchemaRole.CONTENT_FEATURE.value)
        self.assertEqual(mappings["device_type"], SchemaRole.CONTEXT_FEATURE.value)

    def test_02_schema_validation(self):
        """Verify schema validation rejects invalid configs and warns on leakage."""
        # Missing USER_ID
        invalid_mappings_no_user = {
            "item_id": SchemaRole.ITEM_ID.value,
            "timestamp": SchemaRole.TIMESTAMP.value,
            "target": SchemaRole.TARGET.value,
            "feat": SchemaRole.CONTENT_FEATURE.value
        }
        is_valid, errors, warnings = validate_schema(invalid_mappings_no_user)
        self.assertFalse(is_valid)
        self.assertTrue(any("USER_ID" in e for e in errors))

        # Missing ITEM_ID
        invalid_mappings_no_item = {
            "user_id": SchemaRole.USER_ID.value,
            "timestamp": SchemaRole.TIMESTAMP.value,
            "target": SchemaRole.TARGET.value,
            "feat": SchemaRole.CONTENT_FEATURE.value
        }
        is_valid, errors, warnings = validate_schema(invalid_mappings_no_item)
        self.assertFalse(is_valid)
        self.assertTrue(any("ITEM_ID" in e for e in errors))

        # Potential leakage warning
        leakage_mappings = {
            "user_id": SchemaRole.USER_ID.value,
            "item_id": SchemaRole.ITEM_ID.value,
            "timestamp": SchemaRole.TIMESTAMP.value,
            "target": SchemaRole.TARGET.value,
            "checkout_revenue": SchemaRole.BEHAVIOR_FEATURE.value  # Leakage!
        }
        is_valid, errors, warnings = validate_schema(leakage_mappings)
        self.assertTrue(is_valid)
        self.assertTrue(any("LEAKAGE" in w for w in warnings))

    def test_02b_schema_capabilities_no_timestamp_no_target(self):
        """Verify policy: missing TIMESTAMP or TARGET does not block registration but toggles capabilities."""
        from src.schema_mapper import compute_dataset_capabilities, inspect_and_validate_schema

        # Case A: Missing TIMESTAMP (valid for upload, but temporal disabled)
        mappings_no_ts = {
            "user_id": SchemaRole.USER_ID.value,
            "item_id": SchemaRole.ITEM_ID.value,
            "target": SchemaRole.TARGET.value,
            "feat": SchemaRole.CONTENT_FEATURE.value
        }
        is_valid, errors, warnings = validate_schema(mappings_no_ts)
        self.assertTrue(is_valid, "Schema without timestamp should be valid for upload")
        self.assertEqual(len(errors), 0)
        self.assertTrue(any("TIMESTAMP" in w for w in warnings))

        caps_no_ts = compute_dataset_capabilities(mappings_no_ts)
        self.assertTrue(caps_no_ts["recommendation_compatible"])
        self.assertTrue(caps_no_ts["supervised_training"])
        self.assertFalse(caps_no_ts["temporal_training"])
        self.assertFalse(caps_no_ts["cold_start_history"])

        # Case B: Missing TARGET (valid for upload/inference, but supervised training disabled)
        mappings_no_tgt = {
            "user_id": SchemaRole.USER_ID.value,
            "item_id": SchemaRole.ITEM_ID.value,
            "timestamp": SchemaRole.TIMESTAMP.value,
            "feat": SchemaRole.CONTENT_FEATURE.value
        }
        is_valid, errors, warnings = validate_schema(mappings_no_tgt)
        self.assertTrue(is_valid, "Schema without target should be valid for upload")
        self.assertEqual(len(errors), 0)
        self.assertTrue(any("TARGET" in w for w in warnings))

        caps_no_tgt = compute_dataset_capabilities(mappings_no_tgt)
        self.assertTrue(caps_no_tgt["recommendation_compatible"])
        self.assertFalse(caps_no_tgt["supervised_training"])
        self.assertTrue(caps_no_tgt["temporal_training"])

        # Case C: inspect_and_validate_schema helper returns complete contract
        res = inspect_and_validate_schema(mappings_no_ts)
        self.assertTrue(res["valid"])
        self.assertIn("capabilities", res)
        self.assertIn("detected_roles", res)
        self.assertFalse(res["capabilities"]["temporal_training"])

    def test_03_client_config_serialization(self):
        """Verify ClientSchemaConfig correctly saves and loads from YAML."""
        cfg = create_demo_ecommerce_schema()
        test_yaml_path = self.test_dir / "demo_test_config.yaml"
        cfg.to_yaml(test_yaml_path)

        loaded_cfg = ClientSchemaConfig.from_yaml(test_yaml_path)
        self.assertEqual(loaded_cfg.client_id, "demo_ecommerce")
        self.assertEqual(loaded_cfg.get_user_id_col(), "customer_id")
        self.assertEqual(loaded_cfg.get_item_id_col(), "product_id")
        self.assertEqual(loaded_cfg.get_timestamp_col(), "visit_date")
        self.assertEqual(loaded_cfg.get_target_col(), "purchased")
        self.assertEqual(len(loaded_cfg.column_mappings), 29)

    def test_04_demo_ecommerce_exact_feature_dimensions(self):
        """
        Verify reference demo_ecommerce reproduces EXACT validated feature dimensions:
        Behavior = 16, Content = 12, Context = 21.
        """
        demo_cfg = create_demo_ecommerce_schema()
        temp_out = self.test_dir / "demo_processed"
        temp_models = self.test_dir / "demo_models"

        df_feat, summary = prepare_multimodal_features(
            schema_config=demo_cfg,
            output_dir=temp_out,
            models_dir=temp_models
        )

        self.assertEqual(summary["behavioral_dims"], 16)
        self.assertEqual(summary["content_dims"], 12)
        self.assertEqual(summary["context_dims"], 21)
        self.assertEqual(summary["content_meta_dims"], 11)
        self.assertEqual(summary["leakage_audit"]["test_1_first_session_isolation"]["status"], "PASSED")
        self.assertEqual(summary["leakage_audit"]["test_2_temporal_monotonicity"]["status"], "PASSED")
        self.assertEqual(summary["leakage_audit"]["test_1_first_session_isolation"]["violations_detected"], 0)

    def test_05_synthetic_second_client_dynamic_dimensions(self):
        """
        Verify generic schema engine executes cleanly on a SECOND CLIENT with different column names,
        producing valid dynamic feature dimensions and instantiating a functional Two-Tower model.
        """
        # 1. Create realistic synthetic interaction dataset for "retail_client_b"
        np.random.seed(42)
        n_samples = 300
        n_users = 30
        n_articles = 50

        user_ids = [f"BUYER_{i:04d}" for i in np.random.randint(1, n_users + 1, n_samples)]
        article_ids = [f"ART_{i:04d}" for i in np.random.randint(1, n_articles + 1, n_samples)]
        dates = pd.date_range("2024-01-01", periods=n_samples, freq="h").strftime("%Y-%m-%d %H:%M:%S")

        categories = np.random.choice(["Apparel", "Footwear", "Accessories"], size=n_samples)
        prices = np.round(np.random.uniform(10.0, 150.0, size=n_samples), 2)
        dwell_secs = np.random.randint(5, 600, size=n_samples)
        clicks = np.random.randint(1, 20, size=n_samples)
        platforms = np.random.choice(["iOS", "Android", "Web"], size=n_samples)
        countries = np.random.choice(["US", "UK", "DE"], size=n_samples)
        is_ordered = np.random.choice([0, 1], p=[0.8, 0.2], size=n_samples)
        order_revenue = np.where(is_ordered == 1, prices, 0.0)

        synthetic_df = pd.DataFrame({
            "buyer_uuid": user_ids,
            "article_code": article_ids,
            "event_time": dates,
            "is_ordered": is_ordered,
            "dwell_sec": dwell_secs,
            "clicks_count": clicks,
            "article_category": categories,
            "price_usd": prices,
            "device_platform": platforms,
            "country_code": countries,
            "order_revenue": order_revenue
        })

        # 2. Configure schema for retail_client_b
        client_b_mappings = {
            "buyer_uuid": SchemaRole.USER_ID.value,
            "article_code": SchemaRole.ITEM_ID.value,
            "event_time": SchemaRole.TIMESTAMP.value,
            "is_ordered": SchemaRole.TARGET.value,
            "dwell_sec": SchemaRole.BEHAVIOR_FEATURE.value,
            "clicks_count": SchemaRole.BEHAVIOR_FEATURE.value,
            "article_category": SchemaRole.CONTENT_FEATURE.value,
            "price_usd": SchemaRole.CONTENT_FEATURE.value,
            "device_platform": SchemaRole.CONTEXT_FEATURE.value,
            "country_code": SchemaRole.CONTEXT_FEATURE.value,
            "order_revenue": SchemaRole.IGNORE.value
        }

        client_b_types = {
            "buyer_uuid": ColumnType.ID.value,
            "article_code": ColumnType.ID.value,
            "event_time": ColumnType.DATETIME.value,
            "is_ordered": ColumnType.NUMERIC.value,
            "dwell_sec": ColumnType.NUMERIC.value,
            "clicks_count": ColumnType.NUMERIC.value,
            "article_category": ColumnType.CATEGORICAL.value,
            "price_usd": ColumnType.NUMERIC.value,
            "device_platform": ColumnType.CATEGORICAL.value,
            "country_code": ColumnType.CATEGORICAL.value,
            "order_revenue": ColumnType.NUMERIC.value
        }

        client_b_cfg = ClientSchemaConfig(
            client_id="retail_client_b",
            client_name="Retail Client B Fashion Store",
            column_mappings=client_b_mappings,
            column_types=client_b_types,
            timestamp_format="%Y-%m-%d %H:%M:%S",
            split_config={"strategy": "chronological", "split_ratios": [0.6, 0.2, 0.2]},
            outlier_columns=["dwell_sec", "price_usd"],
            cold_start_threshold=3,
            evaluated_thresholds=[1, 2, 3, 5]
        )

        # 3. Validate schema
        is_valid, errors, warnings = validate_schema(client_b_cfg.column_mappings, list(synthetic_df.columns))
        self.assertTrue(is_valid, f"Validation failed: {errors}")

        # 4. Run generic preprocessing & feature engineering
        b_out = self.test_dir / "client_b_processed"
        b_models = self.test_dir / "client_b_models"

        df_feat, summary = prepare_multimodal_features(
            schema_config=client_b_cfg,
            raw_df=synthetic_df,
            output_dir=b_out,
            models_dir=b_models
        )

        # Check feature summary
        beh_dim = summary["behavioral_dims"]
        cont_dim = summary["content_dims"]
        ctx_dim = summary["context_dims"]
        cont_meta_dim = summary["content_meta_dims"]

        self.assertGreater(beh_dim, 0)
        self.assertGreater(cont_dim, 1) # Item ID + metadata
        self.assertGreater(ctx_dim, 0)
        self.assertEqual(summary["client_id"], "retail_client_b")
        self.assertEqual(summary["leakage_audit"]["test_1_first_session_isolation"]["status"], "PASSED")
        self.assertEqual(summary["leakage_audit"]["test_2_temporal_monotonicity"]["status"], "PASSED")
        self.assertEqual(summary["leakage_audit"]["test_1_first_session_isolation"]["violations_detected"], 0)

        # 5. Verify dynamic PyTorch Two-Tower model instantiation with dynamic dimensions
        model = TwoTowerMultiModalRecommender(
            num_products=n_articles + 10,
            beh_dim=beh_dim,
            ctx_dim=ctx_dim,
            cont_meta_dim=cont_meta_dim,
            embedding_dim=64
        )

        # Load generated train features to verify actual forward pass
        train_npz = np.load(b_out / "train_features.npz")
        x_beh = torch.tensor(train_npz["X_beh"][:4], dtype=torch.float32)
        x_cont = torch.tensor(train_npz["X_cont"][:4], dtype=torch.float32)
        x_ctx = torch.tensor(train_npz["X_ctx"][:4], dtype=torch.float32)
        hist = torch.tensor(train_npz["hist_count"][:4], dtype=torch.int64)

        logits, weights, cold_mask = model(x_beh, x_cont, x_ctx, hist, x_cont)

        self.assertEqual(logits.shape, (4,))
        self.assertEqual(weights.shape, (4, 3))
        self.assertEqual(cold_mask.shape, (4, 1))
        # Ensure weights sum to 1
        np.testing.assert_allclose(weights.sum(dim=-1).detach().numpy(), np.ones(4), atol=1e-4)

    def test_08_canonical_schema_roles_roundtrip(self):
        """
        Quality Control Regression Test (Requirement 1):
        Verifies all 8 canonical semantic roles:
        USER_ID, ITEM_ID, TIMESTAMP, TARGET, BEHAVIOR_FEATURE,
        CONTENT_FEATURE, CONTEXT_FEATURE, IGNORE
        are validated, serialized to YAML, and reloaded identically without mutation.
        """
        canonical_mappings = {
            "cust_id": SchemaRole.USER_ID.value,
            "prod_sku": SchemaRole.ITEM_ID.value,
            "event_time": SchemaRole.TIMESTAMP.value,
            "is_bought": SchemaRole.TARGET.value,
            "past_views": SchemaRole.BEHAVIOR_FEATURE.value,
            "category_name": SchemaRole.CONTENT_FEATURE.value,
            "device_model": SchemaRole.CONTEXT_FEATURE.value,
            "post_cart_revenue": SchemaRole.IGNORE.value,
        }
        canonical_types = {
            "cust_id": ColumnType.ID.value,
            "prod_sku": ColumnType.ID.value,
            "event_time": ColumnType.DATETIME.value,
            "is_bought": ColumnType.NUMERIC.value,
            "past_views": ColumnType.NUMERIC.value,
            "category_name": ColumnType.CATEGORICAL.value,
            "device_model": ColumnType.CATEGORICAL.value,
            "post_cart_revenue": ColumnType.NUMERIC.value,
        }

        # 1. Validation test
        is_valid, errors, warnings = validate_schema(canonical_mappings)
        self.assertTrue(is_valid, f"Canonical roles failed validation: {errors}")
        self.assertEqual(len(errors), 0)

        # 2. Config creation & helper checks
        cfg = ClientSchemaConfig(
            client_id="canonical_test_client",
            client_name="Canonical Test Client",
            column_mappings=canonical_mappings,
            column_types=canonical_types,
            cold_start_threshold=3
        )

        self.assertEqual(cfg.get_user_id_col(), "cust_id")
        self.assertEqual(cfg.get_item_id_col(), "prod_sku")
        self.assertEqual(cfg.get_timestamp_col(), "event_time")
        self.assertEqual(cfg.get_target_col(), "is_bought")
        self.assertEqual(cfg.get_behavior_cols(), ["past_views"])
        self.assertEqual(cfg.get_content_cols(), ["category_name"])
        self.assertEqual(cfg.get_context_cols(), ["device_model"])
        self.assertEqual(cfg.get_ignored_cols(), ["post_cart_revenue"])

        # 3. Serialization to YAML & reload roundtrip
        yaml_path = self.test_dir / "canonical_schema.yaml"
        cfg.to_yaml(yaml_path)
        reloaded_cfg = ClientSchemaConfig.from_yaml(yaml_path)

        for col, role in canonical_mappings.items():
            self.assertEqual(
                reloaded_cfg.column_mappings[col],
                role,
                f"Role for column {col} changed from {role} to {reloaded_cfg.column_mappings[col]}"
            )

        caps = reloaded_cfg.get_capabilities()
        self.assertTrue(caps["recommendation_compatible"])
        self.assertTrue(caps["supervised_training"])
        self.assertTrue(caps["temporal_training"])
        self.assertTrue(caps["cold_start_history"])


if __name__ == "__main__":
    unittest.main()
