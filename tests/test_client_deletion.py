"""
Test Suite for Generic Client Deletion (Admin Feature).
Covers:
- Scenario A: Delete temporary client (DB and API list updated)
- Scenario B: Delete non-existent client returns 404
- Scenario C: Dependent datasets and schema mappings cleaned up
- Scenario D: Dependent model versions and training runs cleaned up
- Scenario E: Dependent recommendation and feedback events cleaned up
- Scenario F: Protected reference client 'demo_ecommerce' returns 403 Forbidden
- Scenario G: Any client marked is_system=True returns 403 Forbidden
- Scenario H: Client with active training run returns 409 Conflict
- Scenario I: Physical files preserved when delete_physical_files=False
- Scenario J: Physical files removed when delete_physical_files=True
- Scenario K: Model cache invalidated on client deletion
- Scenario L: Path traversal client ID rejected with 400 Bad Request
"""

import sys
import shutil
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.main import app
from backend.database.database import init_db, SessionLocal
from backend.database.models import (
    ClientModel,
    DatasetModel,
    SchemaMappingModel,
    ModelVersionModel,
    TrainingRunModel,
    RecommendationEventModel,
    FeedbackEventModel
)
from backend.services.client_service import ensure_demo_ecommerce_registered
from backend.services.model_cache import _cache, ModelCacheEntry


class TestClientDeletion(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()
        db = SessionLocal()
        try:
            ensure_demo_ecommerce_registered(db)
        finally:
            db.close()
        cls.client = TestClient(app)

    def tearDown(self):
        # Clean up any leftover test directories
        test_cids = [
            "test_del_temp_1",
            "test_del_dep_1",
            "test_del_models_1",
            "test_del_events_1",
            "test_del_sys_1",
            "test_del_train_1",
            "test_del_preserve_1",
            "test_del_remove_1",
            "test_del_cache_1"
        ]
        db = SessionLocal()
        try:
            for cid in test_cids:
                c = db.query(ClientModel).filter(ClientModel.client_id == cid).first()
                if c:
                    db.query(FeedbackEventModel).filter(FeedbackEventModel.client_id == cid).delete()
                    db.query(RecommendationEventModel).filter(RecommendationEventModel.client_id == cid).delete()
                    db.query(ModelVersionModel).filter(ModelVersionModel.client_id == cid).delete()
                    db.query(TrainingRunModel).filter(TrainingRunModel.client_id == cid).delete()
                    db.query(SchemaMappingModel).filter(SchemaMappingModel.client_id == cid).delete()
                    db.query(DatasetModel).filter(DatasetModel.client_id == cid).delete()
                    db.delete(c)
            db.commit()
        finally:
            db.close()

        for cid in test_cids:
            p_client = Path(f"clients/{cid}")
            if p_client.exists():
                shutil.rmtree(p_client, ignore_errors=True)
            p_model = Path(f"models/{cid}")
            if p_model.exists():
                shutil.rmtree(p_model, ignore_errors=True)

    def test_a_delete_temporary_client_success(self):
        cid = "test_del_temp_1"
        # 1. Create client
        res = self.client.post("/api/clients", json={"client_id": cid, "name": "Temp Test Client"})
        self.assertEqual(res.status_code, 201)

        # Verify client is present in list
        list_res = self.client.get("/api/clients")
        cids = [c["client_id"] for c in list_res.json()["clients"]]
        self.assertIn(cid, cids)

        # 2. Delete client
        del_res = self.client.delete(f"/api/clients/{cid}")
        self.assertEqual(del_res.status_code, 200)
        data = del_res.json()
        self.assertEqual(data["client_id"], cid)
        self.assertIn("successfully deleted", data["message"])
        self.assertFalse(data["physical_files_deleted"])

        # 3. Verify client is gone from DB and list
        get_res = self.client.get(f"/api/clients/{cid}")
        self.assertEqual(get_res.status_code, 404)

        list_res2 = self.client.get("/api/clients")
        cids2 = [c["client_id"] for c in list_res2.json()["clients"]]
        self.assertNotIn(cid, cids2)

    def test_b_delete_nonexistent_client_404(self):
        res = self.client.delete("/api/clients/non_existent_client_xyz987")
        self.assertEqual(res.status_code, 404)
        self.assertIn("not found", res.json()["detail"].lower())

    def test_c_delete_client_with_dataset_and_schema(self):
        cid = "test_del_dep_1"
        self.client.post("/api/clients", json={"client_id": cid, "name": "Dependent Test Client"})

        db = SessionLocal()
        try:
            # Add dataset
            ds = DatasetModel(
                client_id=cid,
                filename="data.csv",
                file_path=f"clients/{cid}/raw/data.csv",
                file_size_bytes=1024,
                row_count=100,
                column_count=5
            )
            db.add(ds)
            # Add schema mapping
            sm = SchemaMappingModel(
                client_id=cid,
                mappings_json="{}",
                types_json="{}",
                is_valid=True,
                version=1
            )
            db.add(sm)
            db.commit()
        finally:
            db.close()

        # Delete client
        del_res = self.client.delete(f"/api/clients/{cid}")
        self.assertEqual(del_res.status_code, 200)
        deleted_records = del_res.json()["deleted_records"]
        self.assertEqual(deleted_records["datasets"], 1)
        self.assertEqual(deleted_records["schema_mappings"], 1)

        # Verify DB records are gone
        db = SessionLocal()
        try:
            self.assertEqual(db.query(DatasetModel).filter(DatasetModel.client_id == cid).count(), 0)
            self.assertEqual(db.query(SchemaMappingModel).filter(SchemaMappingModel.client_id == cid).count(), 0)
            self.assertIsNone(db.query(ClientModel).filter(ClientModel.client_id == cid).first())
        finally:
            db.close()

    def test_d_delete_client_with_models_and_runs(self):
        cid = "test_del_models_1"
        self.client.post("/api/clients", json={"client_id": cid, "name": "Models Test Client"})

        db = SessionLocal()
        try:
            # Add completed training run
            run = TrainingRunModel(
                id="run_test_1",
                client_id=cid,
                status="COMPLETED",
                current_epoch=5,
                total_epochs=5
            )
            db.add(run)
            # Add model version
            mv = ModelVersionModel(
                client_id=cid,
                version_tag="v1",
                training_run_id="run_test_1",
                checkpoint_path="dummy.pt",
                encoders_path="dummy.joblib",
                is_active=True
            )
            db.add(mv)
            db.commit()
        finally:
            db.close()

        # Delete client
        del_res = self.client.delete(f"/api/clients/{cid}")
        self.assertEqual(del_res.status_code, 200)
        deleted_records = del_res.json()["deleted_records"]
        self.assertEqual(deleted_records["training_runs"], 1)
        self.assertEqual(deleted_records["model_versions"], 1)

        # Verify DB records are gone
        db = SessionLocal()
        try:
            self.assertEqual(db.query(TrainingRunModel).filter(TrainingRunModel.client_id == cid).count(), 0)
            self.assertEqual(db.query(ModelVersionModel).filter(ModelVersionModel.client_id == cid).count(), 0)
        finally:
            db.close()

    def test_e_delete_client_with_recs_and_feedback(self):
        cid = "test_del_events_1"
        self.client.post("/api/clients", json={"client_id": cid, "name": "Events Test Client"})

        db = SessionLocal()
        try:
            rec = RecommendationEventModel(
                id="rec_test_1",
                client_id=cid,
                user_id="U100",
                recommended_items_json="[1, 2, 3]",
                latency_ms=12.5
            )
            db.add(rec)
            fb = FeedbackEventModel(
                client_id=cid,
                user_id="U100",
                item_id="1",
                action="CLICK",
                recommendation_id="rec_test_1"
            )
            db.add(fb)
            db.commit()
        finally:
            db.close()

        # Delete client
        del_res = self.client.delete(f"/api/clients/{cid}")
        self.assertEqual(del_res.status_code, 200)
        deleted_records = del_res.json()["deleted_records"]
        self.assertEqual(deleted_records["recommendation_events"], 1)
        self.assertEqual(deleted_records["feedback_events"], 1)

        # Verify DB records are gone
        db = SessionLocal()
        try:
            self.assertEqual(db.query(RecommendationEventModel).filter(RecommendationEventModel.client_id == cid).count(), 0)
            self.assertEqual(db.query(FeedbackEventModel).filter(FeedbackEventModel.client_id == cid).count(), 0)
        finally:
            db.close()

    def test_f_protected_system_client_demo_ecommerce_forbidden(self):
        res = self.client.delete("/api/clients/demo_ecommerce")
        self.assertEqual(res.status_code, 403)
        self.assertIn("protected", res.json()["detail"].lower())

        # Verify demo_ecommerce is still intact
        db = SessionLocal()
        try:
            c = db.query(ClientModel).filter(ClientModel.client_id == "demo_ecommerce").first()
            self.assertIsNotNone(c)
            self.assertTrue(c.is_system)
        finally:
            db.close()

    def test_g_protected_custom_system_client_forbidden(self):
        cid = "test_del_sys_1"
        db = SessionLocal()
        try:
            c = ClientModel(client_id=cid, name="System Protected Client", is_system=True)
            db.add(c)
            db.commit()
        finally:
            db.close()

        res = self.client.delete(f"/api/clients/{cid}")
        self.assertEqual(res.status_code, 403)
        self.assertIn("protected", res.json()["detail"].lower())

    def test_h_active_training_blocks_deletion_409(self):
        cid = "test_del_train_1"
        self.client.post("/api/clients", json={"client_id": cid, "name": "Active Training Client"})

        db = SessionLocal()
        try:
            run = TrainingRunModel(
                id="run_active_1",
                client_id=cid,
                status="TRAINING",
                current_epoch=2,
                total_epochs=10
            )
            db.add(run)
            db.commit()
        finally:
            db.close()

        res = self.client.delete(f"/api/clients/{cid}")
        self.assertEqual(res.status_code, 409)
        self.assertIn("active", res.json()["detail"].lower())

        # Now mark run as FAILED and verify deletion succeeds
        db = SessionLocal()
        try:
            r = db.query(TrainingRunModel).filter(TrainingRunModel.id == "run_active_1").first()
            r.status = "FAILED"
            db.commit()
        finally:
            db.close()

        res2 = self.client.delete(f"/api/clients/{cid}")
        self.assertEqual(res2.status_code, 200)

    def test_i_delete_preserves_physical_files_by_default(self):
        cid = "test_del_preserve_1"
        self.client.post("/api/clients", json={"client_id": cid, "name": "Preserve Files Client"})

        client_dir = Path(f"clients/{cid}")
        model_dir = Path(f"models/{cid}")
        client_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        dummy_data = client_dir / "sample.txt"
        dummy_data.write_text("client raw data")
        dummy_model = model_dir / "weights.pt"
        dummy_model.write_text("model weights")

        # Delete with delete_physical_files=False (default)
        res = self.client.delete(f"/api/clients/{cid}")
        self.assertEqual(res.status_code, 200)
        self.assertFalse(res.json()["physical_files_deleted"])

        # DB record removed
        self.assertEqual(self.client.get(f"/api/clients/{cid}").status_code, 404)

        # Files MUST still exist
        self.assertTrue(client_dir.exists())
        self.assertTrue(dummy_data.exists())
        self.assertTrue(model_dir.exists())
        self.assertTrue(dummy_model.exists())

    def test_j_delete_removes_physical_files_when_requested(self):
        cid = "test_del_remove_1"
        self.client.post("/api/clients", json={"client_id": cid, "name": "Remove Files Client"})

        client_dir = Path(f"clients/{cid}")
        model_dir = Path(f"models/{cid}")
        client_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        dummy_data = client_dir / "sample.txt"
        dummy_data.write_text("client raw data")
        dummy_model = model_dir / "weights.pt"
        dummy_model.write_text("model weights")

        # Delete with delete_physical_files=True
        res = self.client.delete(f"/api/clients/{cid}?delete_physical_files=true")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["physical_files_deleted"])

        # DB record removed
        self.assertEqual(self.client.get(f"/api/clients/{cid}").status_code, 404)

        # Files MUST be removed
        self.assertFalse(client_dir.exists())
        self.assertFalse(model_dir.exists())

    def test_k_model_cache_invalidated_on_deletion(self):
        cid = "test_del_cache_1"
        self.client.post("/api/clients", json={"client_id": cid, "name": "Cache Invalidation Client"})

        # Manually insert dummy cache entry
        cache_key = (cid, "v1")
        _cache[cache_key] = "dummy_entry"
        self.assertIn(cache_key, _cache)

        # Delete client
        res = self.client.delete(f"/api/clients/{cid}")
        self.assertEqual(res.status_code, 200)

        # Cache key MUST be removed
        self.assertNotIn(cache_key, _cache)

    def test_l_path_traversal_client_id_rejected(self):
        res1 = self.client.delete("/api/clients/..%2F..%2Fetc")
        self.assertIn(res1.status_code, [400, 404, 405])

        res2 = self.client.delete("/api/clients/invalid..client")
        self.assertEqual(res2.status_code, 400)
        self.assertIn("invalid client id", res2.json()["detail"].lower())


if __name__ == "__main__":
    unittest.main()
