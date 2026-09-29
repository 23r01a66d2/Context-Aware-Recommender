"""
Integration Test Suite for FastAPI Backend API (Phase 3).
Tests:
- health endpoint
- create client
- duplicate client
- list clients
- get client
- unknown client
- invalid client ID
- path traversal client ID
- CSV upload
- invalid extension
- empty upload
- dataset preview
- schema suggestion
- schema save
- schema validation
- dataset without timestamp
- dataset without target
- client isolation
- training run status
- recommendation contract (503 without fake recommendations)
- feedback endpoint
"""

import os
import io
import sys
import shutil
import unittest
from pathlib import Path
import pandas as pd
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.main import app
from backend.database.database import init_db, SessionLocal
from backend.database.models import ClientModel
from src.schema_mapper import SchemaRole


class TestBackendAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Initialize database tables and register canonical demo_ecommerce client
        init_db()
        from backend.services.client_service import ensure_demo_ecommerce_registered
        cls.test_clients = ["test_client_alpha", "test_client_beta"]
        db = SessionLocal()
        try:
            for cid in cls.test_clients:
                existing = db.query(ClientModel).filter(ClientModel.client_id == cid).first()
                if existing:
                    db.delete(existing)
            db.commit()
            ensure_demo_ecommerce_registered(db)
        finally:
            db.close()

        # Clean up any leftover folders
        for cid in cls.test_clients:
            cpath = Path(f"clients/{cid}")
            if cpath.exists():
                shutil.rmtree(cpath, ignore_errors=True)
            mpath = Path(f"models/{cid}")
            if mpath.exists():
                shutil.rmtree(mpath, ignore_errors=True)

        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        # Clean up test client DB records
        db = SessionLocal()
        try:
            for cid in cls.test_clients:
                existing = db.query(ClientModel).filter(ClientModel.client_id == cid).first()
                if existing:
                    db.delete(existing)
            db.commit()
        finally:
            db.close()

        # Clean up any created test client folders
        for cid in cls.test_clients:
            cpath = Path(f"clients/{cid}")
            if cpath.exists():
                shutil.rmtree(cpath, ignore_errors=True)
            mpath = Path(f"models/{cid}")
            if mpath.exists():
                shutil.rmtree(mpath, ignore_errors=True)

    def test_01_health_endpoint(self):
        """Verify GET /api/health returns exact specification."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["service"], "context-aware-recommender")

    def test_02_create_client(self):
        """Verify POST /api/clients creates client and workspace."""
        payload = {
            "client_id": "test_client_alpha",
            "name": "Alpha Retail Store",
            "description": "Integration test client"
        }
        response = self.client.post("/api/clients", json=payload)
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertEqual(data["client_id"], "test_client_alpha")
        self.assertEqual(data["name"], "Alpha Retail Store")

        # Verify workspace directories created
        self.assertTrue(Path("clients/test_client_alpha/raw").exists())
        self.assertTrue(Path("clients/test_client_alpha/processed").exists())
        self.assertTrue(Path("models/test_client_alpha").exists())

    def test_03_duplicate_client(self):
        """Verify POST /api/clients rejects duplicate client_id with 409 Conflict."""
        payload = {
            "client_id": "test_client_alpha",
            "name": "Alpha Duplicate",
            "description": "Should fail"
        }
        response = self.client.post("/api/clients", json=payload)
        self.assertEqual(response.status_code, 409)
        self.assertIn("already exists", response.json()["detail"])

    def test_04_list_clients(self):
        """Verify GET /api/clients lists all clients including demo_ecommerce."""
        response = self.client.get("/api/clients")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreaterEqual(data["total"], 2)
        client_ids = [c["client_id"] for c in data["clients"]]
        self.assertIn("demo_ecommerce", client_ids)
        self.assertIn("test_client_alpha", client_ids)

    def test_05_get_client(self):
        """Verify GET /api/clients/{client_id} returns client metadata."""
        response = self.client.get("/api/clients/test_client_alpha")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["client_id"], "test_client_alpha")

    def test_06_unknown_client(self):
        """Verify GET /api/clients/{client_id} returns 404 for unknown client."""
        response = self.client.get("/api/clients/unknown_client_xyz")
        self.assertEqual(response.status_code, 404)

    def test_07_invalid_client_id(self):
        """Verify invalid client IDs (spaces, invalid chars) are rejected."""
        # Query parameter with invalid chars
        response = self.client.get("/api/clients/bad%20id%20with%20spaces")
        self.assertEqual(response.status_code, 400)

        # Body with invalid chars
        payload = {"client_id": "bad client!", "name": "Bad Client"}
        response = self.client.post("/api/clients", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_08_path_traversal_client_id(self):
        """Verify path traversal sequences are rejected."""
        response = self.client.get("/api/clients/..%2F..%2Fetc")
        self.assertIn(response.status_code, [400, 404])

        payload = {"client_id": "../evil_client", "name": "Evil Client"}
        response = self.client.post("/api/clients", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_09_csv_upload(self):
        """Verify POST /api/clients/{client_id}/dataset uploads and inspects CSV."""
        csv_content = (
            "buyer_id,item_code,order_date,is_bought,price,platform\n"
            "U101,P201,2024-01-01,1,29.99,Mobile\n"
            "U102,P202,2024-01-02,0,49.99,Desktop\n"
            "U101,P203,2024-01-03,1,19.99,Mobile\n"
        )
        file_obj = io.BytesIO(csv_content.encode("utf-8"))
        files = {"file": ("sample_orders.csv", file_obj, "text/csv")}

        response = self.client.post("/api/clients/test_client_alpha/dataset", files=files)
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertEqual(data["client_id"], "test_client_alpha")
        self.assertEqual(data["rows"], 3)
        self.assertEqual(data["columns"], 6)
        self.assertEqual(data["filename"], "sample_orders.csv")

        # Verify saved file in isolated raw directory
        saved_file = Path("clients/test_client_alpha/raw/sample_orders.csv")
        self.assertTrue(saved_file.exists())

    def test_10_invalid_extension_upload(self):
        """Verify non-CSV files are rejected."""
        file_obj = io.BytesIO(b"binary content")
        files = {"file": ("malicious.exe", file_obj, "application/octet-stream")}
        response = self.client.post("/api/clients/test_client_alpha/dataset", files=files)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Only CSV files", response.json()["detail"])

    def test_11_empty_upload(self):
        """Verify empty files are rejected."""
        file_obj = io.BytesIO(b"")
        files = {"file": ("empty.csv", file_obj, "text/csv")}
        response = self.client.post("/api/clients/test_client_alpha/dataset", files=files)
        self.assertEqual(response.status_code, 400)
        self.assertIn("empty", response.json()["detail"].lower())

    def test_12_dataset_preview(self):
        """Verify GET /api/clients/{client_id}/dataset/preview returns metadata and sample rows."""
        response = self.client.get("/api/clients/test_client_alpha/dataset/preview?limit=2")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["client_id"], "test_client_alpha")
        self.assertEqual(data["total_columns"], 6)
        self.assertEqual(len(data["sample_data"]), 2)
        col_names = [c["name"] for c in data["columns"]]
        self.assertIn("buyer_id", col_names)
        self.assertIn("price", col_names)

    def test_13_schema_suggestion(self):
        """Verify GET /api/clients/{client_id}/schema auto-suggests roles from uploaded dataset."""
        response = self.client.get("/api/clients/test_client_alpha/schema")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        mappings = data["column_mappings"]
        self.assertEqual(mappings.get("buyer_id"), SchemaRole.USER_ID.value)
        self.assertEqual(mappings.get("item_code"), SchemaRole.ITEM_ID.value)
        self.assertEqual(mappings.get("order_date"), SchemaRole.TIMESTAMP.value)
        self.assertEqual(mappings.get("is_bought"), SchemaRole.TARGET.value)
        self.assertEqual(mappings.get("price"), SchemaRole.CONTENT_FEATURE.value)
        self.assertEqual(mappings.get("platform"), SchemaRole.CONTEXT_FEATURE.value)

    def test_14_schema_validation(self):
        """Verify POST /api/clients/{client_id}/validate checks mappings without persisting."""
        payload = {
            "column_mappings": {
                "buyer_id": SchemaRole.USER_ID.value,
                "item_code": SchemaRole.ITEM_ID.value,
                "price": SchemaRole.CONTENT_FEATURE.value
            }
        }
        response = self.client.post("/api/clients/test_client_alpha/validate", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["valid"])
        self.assertTrue(data["capabilities"]["recommendation_compatible"])
        self.assertFalse(data["capabilities"]["temporal_training"])

    def test_15_schema_save(self):
        """Verify PUT /api/clients/{client_id}/schema saves mapping to config.yaml and DB."""
        payload = {
            "column_mappings": {
                "buyer_id": SchemaRole.USER_ID.value,
                "item_code": SchemaRole.ITEM_ID.value,
                "order_date": SchemaRole.TIMESTAMP.value,
                "is_bought": SchemaRole.TARGET.value,
                "price": SchemaRole.CONTENT_FEATURE.value,
                "platform": SchemaRole.CONTEXT_FEATURE.value
            },
            "timestamp_format": "%Y-%m-%d",
            "cold_start_threshold": 3
        }
        response = self.client.put("/api/clients/test_client_alpha/schema", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["is_valid"])
        self.assertTrue(data["capabilities"]["temporal_training"])
        self.assertTrue(data["capabilities"]["supervised_training"])

        # Check config.yaml written
        self.assertTrue(Path("clients/test_client_alpha/config.yaml").exists())

    def test_16_dataset_without_timestamp(self):
        """
        Verify policy: dataset without TIMESTAMP is valid for registration and inspection,
        with clear warning and temporal_training=False.
        """
        payload = {
            "column_mappings": {
                "buyer_id": SchemaRole.USER_ID.value,
                "item_code": SchemaRole.ITEM_ID.value,
                "is_bought": SchemaRole.TARGET.value,
                "price": SchemaRole.CONTENT_FEATURE.value
            }
        }
        response = self.client.post("/api/clients/test_client_alpha/validate", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["valid"])
        self.assertFalse(data["capabilities"]["temporal_training"])
        self.assertFalse(data["capabilities"]["cold_start_history"])
        self.assertTrue(any("TIMESTAMP" in w for w in data["warnings"]))

    def test_17_dataset_without_target(self):
        """
        Verify policy: dataset without TARGET is valid for registration and mapping,
        with supervised_training=False.
        """
        payload = {
            "column_mappings": {
                "buyer_id": SchemaRole.USER_ID.value,
                "item_code": SchemaRole.ITEM_ID.value,
                "order_date": SchemaRole.TIMESTAMP.value,
                "price": SchemaRole.CONTENT_FEATURE.value
            }
        }
        response = self.client.post("/api/clients/test_client_alpha/validate", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["valid"])
        self.assertFalse(data["capabilities"]["supervised_training"])
        self.assertTrue(data["capabilities"]["temporal_training"])
        self.assertTrue(any("TARGET" in w for w in data["warnings"]))

    def test_18_client_isolation(self):
        """Verify client_alpha and client_beta have mutually isolated files and records."""
        # Create client beta
        resp_b = self.client.post("/api/clients", json={"client_id": "test_client_beta", "name": "Beta Client"})
        self.assertEqual(resp_b.status_code, 201)

        # Upload dataset for beta
        csv_b = "cust_code,sku,sales\nC1,S1,10.0\n"
        file_b = io.BytesIO(csv_b.encode("utf-8"))
        self.client.post("/api/clients/test_client_beta/dataset", files={"file": ("beta_data.csv", file_b, "text/csv")})

        # Verify alpha files are NOT present in beta
        alpha_files = list(Path("clients/test_client_alpha/raw").glob("*"))
        beta_files = list(Path("clients/test_client_beta/raw").glob("*"))

        alpha_names = [f.name for f in alpha_files]
        beta_names = [f.name for f in beta_files]

        self.assertIn("sample_orders.csv", alpha_names)
        self.assertNotIn("beta_data.csv", alpha_names)
        self.assertIn("beta_data.csv", beta_names)
        self.assertNotIn("sample_orders.csv", beta_names)

    def test_19_training_status_foundation(self):
        """Verify training trigger and status endpoints return real states (no fake percentages)."""
        # Trigger training for alpha
        resp = self.client.post("/api/clients/test_client_alpha/train", json={"epochs": 5})
        self.assertEqual(resp.status_code, 202)
        data = resp.json()
        self.assertEqual(data["status"], "NOT_STARTED")

        # Query status
        resp_stat = self.client.get("/api/clients/test_client_alpha/training-status")
        self.assertEqual(resp_stat.status_code, 200)
        stat_data = resp_stat.json()
        self.assertIn(stat_data["status"], ["NOT_STARTED", "PREPROCESSING", "TRAINING", "COMPLETED"])

    def test_20_recommendation_contract(self):
        """
        Verify recommendation contract:
        - Validates top_k bounds (returns 422 for >50)
        - Returns 503 Service Unavailable when client has no active trained model
        - Returns 200 OK with genuine recommendations for active client (demo_ecommerce)
        """
        # Invalid top_k (>50)
        invalid_payload = {
            "client_id": "demo_ecommerce",
            "user_id": "cust_123",
            "top_k": 999
        }
        resp = self.client.post("/api/recommend", json=invalid_payload)
        self.assertEqual(resp.status_code, 422)

        # Client without active model returns 503
        self.client.post("/api/clients", json={"client_id": "client_no_model", "name": "No Model Client"})
        no_model_payload = {
            "client_id": "client_no_model",
            "user_id": "cust_123",
            "top_k": 5
        }
        resp_no_model = self.client.post("/api/recommend", json=no_model_payload)
        self.assertEqual(resp_no_model.status_code, 503)
        self.assertIn("No active model is available", resp_no_model.json()["detail"])

        # Valid payload for demo_ecommerce (active model v1): returns 200 OK with genuine recommendations
        valid_payload = {
            "client_id": "demo_ecommerce",
            "user_id": "9272",
            "context": {"device_type": 0},
            "top_k": 5
        }
        resp = self.client.post("/api/recommend", json=valid_payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("recommendation_id", data)
        self.assertEqual(len(data["recommendations"]), 5)
        self.assertIn("modality_weights", data)

    def test_21_feedback_endpoint(self):
        """Verify feedback endpoint records events without retraining."""
        payload = {
            "client_id": "demo_ecommerce",
            "user_id": "cust_101",
            "item_id": "prod_505",
            "action": "ACCEPT",
            "position": 1
        }
        resp = self.client.post("/api/feedback", json=payload)
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertEqual(data["action"], "ACCEPT")
        self.assertEqual(data["status"], "recorded")

        # Unknown client should return 404
        bad_client_payload = {
            "client_id": "nonexistent_client",
            "user_id": "u1",
            "item_id": "i1",
            "action": "CLICK"
        }
        resp_bad = self.client.post("/api/feedback", json=bad_client_payload)
        self.assertEqual(resp_bad.status_code, 404)


if __name__ == "__main__":
    unittest.main()
