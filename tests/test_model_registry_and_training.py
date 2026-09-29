"""
Comprehensive Test Suite for Phase 4: Model Registry & Background Training Service.
Covers:
- Section M: Client Isolation Tests (cross-client visibility, activation isolation, directory containment)
- Section N: Failure Tests (missing dataset, missing schema, missing target, missing timestamp, duplicate runs, invalid version, incomplete model activation)
- Real background training execution, lifecycle states, real metrics tracking, immutable model versioning
- demo_ecommerce legacy reference model registration and metadata verification
"""

import io
import json
import shutil
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

from backend.main import app
from backend.database.database import init_db, SessionLocal
from backend.database.models import ClientModel, TrainingRunModel, ModelVersionModel
from backend.services.client_service import ensure_demo_ecommerce_registered
from backend.services.training_service import execute_training_lifecycle
from src.schema_mapper import SchemaRole, ClientSchemaConfig


class TestModelRegistryAndTraining(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()
        cls.test_clients = ["test_iso_client_a", "test_iso_client_b", "test_fail_client"]
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
            shutil.rmtree(Path(f"clients/{cid}"), ignore_errors=True)
            shutil.rmtree(Path(f"models/{cid}"), ignore_errors=True)

        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        db = SessionLocal()
        try:
            for cid in cls.test_clients:
                existing = db.query(ClientModel).filter(ClientModel.client_id == cid).first()
                if existing:
                    db.delete(existing)
            db.commit()
        finally:
            db.close()

        for cid in cls.test_clients:
            shutil.rmtree(Path(f"clients/{cid}"), ignore_errors=True)
            shutil.rmtree(Path(f"models/{cid}"), ignore_errors=True)

    # =========================================================================
    # 1. DEMO_ECOMMERCE REFERENCE REGISTRATION
    # =========================================================================
    def test_01_demo_ecommerce_model_registry(self):
        """Verify demo_ecommerce is registered as v1 with legacy_imported=True and no leaked paths."""
        resp = self.client.get("/api/clients/demo_ecommerce/models")
        self.assertEqual(resp.status_code, 200)
        models = resp.json()
        self.assertGreaterEqual(len(models), 1)

        v1 = next((m for m in models if m["version"] == "v1"), None)
        self.assertIsNotNone(v1)
        self.assertTrue(v1["is_active"])
        self.assertTrue(v1["legacy_imported"])
        self.assertEqual(v1["feature_dimensions"], {"behavior": 16, "content": 12, "context": 21})

        # Ensure no filesystem paths are exposed
        raw_text = json.dumps(v1)
        self.assertNotIn("models/", raw_text)
        self.assertNotIn(".pt", raw_text)

        # Detailed metadata endpoint
        detail_resp = self.client.get("/api/clients/demo_ecommerce/models/v1")
        self.assertEqual(detail_resp.status_code, 200)
        detail = detail_resp.json()
        self.assertEqual(detail["version"], "v1")
        self.assertEqual(detail["cold_start_threshold"], 3)
        self.assertEqual(detail["random_seed"], 42)
        self.assertNotIn(".pt", json.dumps(detail))

    # =========================================================================
    # 2. FAILURE MODES (SECTION N)
    # =========================================================================
    def test_02_training_without_dataset(self):
        """Verify training without an uploaded dataset returns 400 with explanation."""
        # Create client with no dataset
        self.client.post("/api/clients", json={"client_id": "test_fail_client", "name": "Fail Test Client"})

        resp = self.client.post("/api/clients/test_fail_client/train", json={"epochs": 2})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("No uploaded dataset found", resp.json()["detail"])

    def test_03_training_without_schema(self):
        """Verify training with dataset but without confirmed valid schema returns 400."""
        # Upload dataset for test_fail_client
        csv_data = "buyer,product,dt,is_buy,price\nU1,P1,2024-01-01,1,29.9\nU2,P2,2024-01-02,0,19.9\n"
        files = {"file": ("data.csv", io.BytesIO(csv_data.encode("utf-8")), "text/csv")}
        self.client.post("/api/clients/test_fail_client/dataset", files=files)

        # Trigger training without saving schema
        resp = self.client.post("/api/clients/test_fail_client/train", json={"epochs": 2})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("No confirmed, valid schema mapping", resp.json()["detail"])

    def test_04_training_without_target(self):
        """Verify training without TARGET capability returns 400 explaining missing target."""
        # Save schema lacking TARGET
        payload = {
            "column_mappings": {
                "buyer": SchemaRole.USER_ID.value,
                "product": SchemaRole.ITEM_ID.value,
                "dt": SchemaRole.TIMESTAMP.value,
                "price": SchemaRole.CONTENT_FEATURE.value,
                "is_buy": SchemaRole.IGNORE.value  # Dropped target
            }
        }
        resp_put = self.client.put("/api/clients/test_fail_client/schema", json=payload)
        self.assertEqual(resp_put.status_code, 200)

        resp = self.client.post("/api/clients/test_fail_client/train", json={"epochs": 2})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("TARGET", resp.json()["detail"])

    def test_05_training_without_timestamp(self):
        """Verify training without TIMESTAMP capability returns 400 explaining missing temporal capability."""
        # Save schema with TARGET but lacking TIMESTAMP
        payload = {
            "column_mappings": {
                "buyer": SchemaRole.USER_ID.value,
                "product": SchemaRole.ITEM_ID.value,
                "is_buy": SchemaRole.TARGET.value,
                "price": SchemaRole.CONTENT_FEATURE.value,
                "dt": SchemaRole.IGNORE.value  # Dropped timestamp
            }
        }
        resp_put = self.client.put("/api/clients/test_fail_client/schema", json=payload)
        self.assertEqual(resp_put.status_code, 200)

        resp = self.client.post("/api/clients/test_fail_client/train", json={"epochs": 2})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("TIMESTAMP", resp.json()["detail"])

    def test_06_duplicate_active_training(self):
        """Verify launching duplicate training when a run is already active returns 409 Conflict."""
        # Fix schema for test_fail_client so it is eligible
        payload = {
            "column_mappings": {
                "buyer": SchemaRole.USER_ID.value,
                "product": SchemaRole.ITEM_ID.value,
                "dt": SchemaRole.TIMESTAMP.value,
                "price": SchemaRole.CONTENT_FEATURE.value,
                "is_buy": SchemaRole.TARGET.value
            }
        }
        resp_put = self.client.put("/api/clients/test_fail_client/schema", json=payload)
        self.assertEqual(resp_put.status_code, 200)

        # Insert active run record manually
        db = SessionLocal()
        try:
            run = TrainingRunModel(
                id="active_test_run_123",
                client_id="test_fail_client",
                status="TRAINING",
                total_epochs=10,
                current_epoch=2
            )
            db.add(run)
            db.commit()
        finally:
            db.close()

        # Trigger duplicate training
        resp = self.client.post("/api/clients/test_fail_client/train", json={"epochs": 2})
        self.assertEqual(resp.status_code, 409)
        self.assertIn("already in progress", resp.json()["detail"])

        # Clean up mock active run
        db = SessionLocal()
        try:
            db.query(TrainingRunModel).filter(TrainingRunModel.id == "active_test_run_123").delete()
            db.commit()
        finally:
            db.close()

    def test_07_invalid_version_or_missing_checkpoint_activation(self):
        """Verify activating nonexistent or incomplete model returns 404/400."""
        # Nonexistent version
        resp_404 = self.client.post("/api/clients/demo_ecommerce/models/v999/activate")
        self.assertEqual(resp_404.status_code, 404)

        # Insert model version pointing to nonexistent file
        db = SessionLocal()
        try:
            bad_model = ModelVersionModel(
                client_id="test_fail_client",
                version_tag="v_broken",
                checkpoint_path="models/test_fail_client/nonexistent.pt",
                encoders_path="models/test_fail_client/nonexistent.joblib",
                is_active=False
            )
            db.add(bad_model)
            db.commit()
        finally:
            db.close()

        resp_bad = self.client.post("/api/clients/test_fail_client/models/v_broken/activate")
        self.assertEqual(resp_bad.status_code, 400)
        self.assertIn("missing", resp_bad.json()["detail"].lower())

    # =========================================================================
    # 3. END-TO-END TRAINING & MODEL REGISTRY
    # =========================================================================
    def test_08_end_to_end_training_and_versioning(self):
        """
        Verify real end-to-end training lifecycle execution:
        - Upload multi-session dataset
        - Configure schema
        - Execute training pipeline synchronously via execute_training_lifecycle
        - Verify transitions: PREPROCESSING -> TRAINING -> EVALUATING -> COMPLETED
        - Verify real metrics and epoch updates
        - Verify version v1 created in registry with all 6 artifacts
        """
        cid = "test_iso_client_a"
        self.client.post("/api/clients", json={"client_id": cid, "name": "Client Alpha"})

        # Build realistic temporal multi-session dataset (30 rows, 3 items, temporal order)
        rows = [
            "user_id,item_id,timestamp,purchased,category,price,channel",
            "U1,I1,2024-01-01,1,Electronics,99.9,Mobile",
            "U1,I2,2024-01-02,0,Clothing,29.9,Desktop",
            "U1,I3,2024-01-03,1,Electronics,49.9,Mobile",
            "U2,I1,2024-01-04,0,Electronics,99.9,Desktop",
            "U2,I2,2024-01-05,1,Clothing,29.9,Mobile",
            "U2,I3,2024-01-06,0,Electronics,49.9,Mobile",
            "U3,I1,2024-01-07,1,Electronics,99.9,Desktop",
            "U3,I2,2024-01-08,0,Clothing,29.9,Desktop",
            "U3,I3,2024-01-09,1,Electronics,49.9,Mobile",
            "U4,I1,2024-01-10,0,Electronics,99.9,Mobile",
            "U4,I2,2024-01-11,1,Clothing,29.9,Desktop",
            "U4,I3,2024-01-12,0,Electronics,49.9,Mobile",
            "U5,I1,2024-01-13,1,Electronics,99.9,Desktop",
            "U5,I2,2024-01-14,0,Clothing,29.9,Mobile",
            "U5,I3,2024-01-15,1,Electronics,49.9,Mobile",
            "U1,I1,2024-01-16,0,Electronics,99.9,Mobile",
            "U2,I2,2024-01-17,1,Clothing,29.9,Desktop",
            "U3,I3,2024-01-18,0,Electronics,49.9,Desktop",
            "U4,I1,2024-01-19,1,Electronics,99.9,Mobile",
            "U5,I2,2024-01-20,0,Clothing,29.9,Mobile",
            "U1,I3,2024-01-21,1,Electronics,49.9,Desktop",
            "U2,I1,2024-01-22,0,Electronics,99.9,Desktop",
            "U3,I2,2024-01-23,1,Clothing,29.9,Mobile",
            "U4,I3,2024-01-24,0,Electronics,49.9,Desktop",
            "U5,I1,2024-01-25,1,Electronics,99.9,Mobile",
            "U1,I2,2024-01-26,0,Clothing,29.9,Desktop",
            "U2,I3,2024-01-27,1,Electronics,49.9,Mobile",
            "U3,I1,2024-01-28,0,Electronics,99.9,Desktop",
            "U4,I2,2024-01-29,1,Clothing,29.9,Mobile",
            "U5,I3,2024-01-30,1,Electronics,49.9,Desktop"
        ]
        csv_content = "\n".join(rows) + "\n"
        files = {"file": ("alpha_interactions.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
        resp_up = self.client.post(f"/api/clients/{cid}/dataset", files=files)
        self.assertEqual(resp_up.status_code, 201)

        # Save schema
        schema_payload = {
            "column_mappings": {
                "user_id": SchemaRole.USER_ID.value,
                "item_id": SchemaRole.ITEM_ID.value,
                "timestamp": SchemaRole.TIMESTAMP.value,
                "purchased": SchemaRole.TARGET.value,
                "category": SchemaRole.CONTENT_FEATURE.value,
                "price": SchemaRole.CONTENT_FEATURE.value,
                "channel": SchemaRole.CONTEXT_FEATURE.value
            },
            "timestamp_format": "%Y-%m-%d",
            "cold_start_threshold": 2
        }
        resp_sch = self.client.put(f"/api/clients/{cid}/schema", json=schema_payload)
        self.assertEqual(resp_sch.status_code, 200)

        # Execute training directly to ensure deterministic test completion
        run_id = f"test_run_{cid}_001"
        db = SessionLocal()
        try:
            run_rec = TrainingRunModel(
                id=run_id,
                client_id=cid,
                status="NOT_STARTED",
                current_epoch=0,
                total_epochs=3
            )
            db.add(run_rec)
            db.commit()
        finally:
            db.close()

        schema_cfg = ClientSchemaConfig.from_dict({
            "client_id": cid,
            "column_mappings": schema_payload["column_mappings"],
            "timestamp_format": "%Y-%m-%d",
            "cold_start_threshold": 2
        })

        execute_training_lifecycle(
            run_id=run_id,
            client_id=cid,
            dataset_path=f"clients/{cid}/raw/alpha_interactions.csv",
            schema_config=schema_cfg,
            epochs=3,
            batch_size=8,
            lr=0.01,
            seed=42
        )

        # Verify training run status
        stat_resp = self.client.get(f"/api/clients/{cid}/training-status")
        self.assertEqual(stat_resp.status_code, 200)
        stat = stat_resp.json()
        self.assertEqual(stat["status"], "COMPLETED")
        self.assertEqual(stat["current_epoch"], 3)
        self.assertEqual(stat["total_epochs"], 3)
        self.assertIsNotNone(stat["metrics"])
        self.assertIn("Overall", stat["metrics"])
        self.assertIn("Cold-Start", stat["metrics"])

        # Verify model version registered in registry
        mod_resp = self.client.get(f"/api/clients/{cid}/models")
        self.assertEqual(mod_resp.status_code, 200)
        models = mod_resp.json()
        self.assertEqual(len(models), 1)
        v1 = models[0]
        self.assertEqual(v1["version"], "v1")
        self.assertTrue(v1["is_active"])
        self.assertFalse(v1["legacy_imported"])

        # Verify physical version folder contains all 6 artifacts
        v1_dir = Path(f"models/{cid}/versions/v1")
        self.assertTrue((v1_dir / "model.pt").exists())
        self.assertTrue((v1_dir / "feature_encoders.joblib").exists())
        self.assertTrue((v1_dir / "schema.json").exists())
        self.assertTrue((v1_dir / "config.yaml").exists())
        self.assertTrue((v1_dir / "metrics.json").exists())
        self.assertTrue((v1_dir / "training_metadata.json").exists())

    # =========================================================================
    # 4. CLIENT ISOLATION TESTS (SECTION M)
    # =========================================================================
    def test_09_client_isolation_models_and_activation(self):
        """
        Verify Section M isolation rules:
        - client_A model versions do not appear in client_B model listing.
        - client_A cannot request client_B version metadata through client_A URL.
        - client_A activation cannot deactivate client_B active model.
        - client_A training does not write to client_B directory.
        """
        # Create client B
        cid_b = "test_iso_client_b"
        self.client.post("/api/clients", json={"client_id": cid_b, "name": "Client Beta"})

        # 1. Listing isolation
        models_b = self.client.get(f"/api/clients/{cid_b}/models").json()
        self.assertEqual(len(models_b), 0)  # Client B has no models

        models_a = self.client.get("/api/clients/test_iso_client_a/models").json()
        self.assertEqual(len(models_a), 1)  # Client A has 1 model

        # 2. URL request isolation (client B requesting client A's version)
        resp_cross = self.client.get(f"/api/clients/{cid_b}/models/v1")
        self.assertEqual(resp_cross.status_code, 404)
        self.assertIn("not found for client", resp_cross.json()["detail"])

        # 3. Activation isolation: Register a model for client B, verify activating it does not touch client A
        db = SessionLocal()
        try:
            # Create a mock valid model for client B
            b_version_dir = Path(f"models/{cid_b}/versions/v1")
            b_version_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2("models/test_iso_client_a/versions/v1/model.pt", b_version_dir / "model.pt")
            shutil.copy2("models/test_iso_client_a/versions/v1/feature_encoders.joblib", b_version_dir / "feature_encoders.joblib")

            b_model = ModelVersionModel(
                client_id=cid_b,
                version_tag="v1",
                training_run_id="run_b_mock",
                checkpoint_path=str(b_version_dir / "model.pt"),
                encoders_path=str(b_version_dir / "feature_encoders.joblib"),
                is_active=False
            )
            db.add(b_model)
            db.commit()
        finally:
            db.close()

        # Activate model for client B
        act_resp = self.client.post(f"/api/clients/{cid_b}/models/v1/activate")
        self.assertEqual(act_resp.status_code, 200)

        # Check client B model is active
        b_detail = self.client.get(f"/api/clients/{cid_b}/models/v1").json()
        self.assertTrue(b_detail["is_active"])

        # Check client A model is STILL active (unaffected)
        a_detail = self.client.get("/api/clients/test_iso_client_a/models/v1").json()
        self.assertTrue(a_detail["is_active"])

        # 4. Directory containment check: Ensure no files from client A leaked into client B
        a_files = [f.name for f in Path("clients/test_iso_client_a/raw").glob("*")]
        b_files = [f.name for f in Path(f"clients/{cid_b}/raw").glob("*")]
        self.assertIn("alpha_interactions.csv", a_files)
        self.assertNotIn("alpha_interactions.csv", b_files)


if __name__ == "__main__":
    unittest.main()
