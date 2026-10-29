"""
Unit tests for Persistent User History Service,
Cold Start vs Warm Start Detection, Tenant Isolation, and Corrupted Recovery.
"""

import os
import shutil
import unittest
from pathlib import Path

from backend.services.user_history_service import (
    USER_HISTORY_BASE_DIR,
    validate_user_id,
    sanitize_id,
    detect_user_status,
    get_user_history,
    record_user_history_entry,
    clear_user_history
)


class TestUserHistoryService(unittest.TestCase):
    def setUp(self):
        # Use a temporary test client namespace
        self.client_a = "test_tenant_alpha"
        self.client_b = "test_tenant_beta"
        self.user_1 = "user_1001"
        self.user_2 = "user_2002"

        # Clean any existing test directories
        for cid in [self.client_a, self.client_b]:
            p = USER_HISTORY_BASE_DIR / cid
            if p.exists():
                shutil.rmtree(p)

    def tearDown(self):
        for cid in [self.client_a, self.client_b]:
            p = USER_HISTORY_BASE_DIR / cid
            if p.exists():
                shutil.rmtree(p)

    def test_user_id_validation(self):
        """Validates proper validation of empty, whitespace, and path traversal IDs."""
        valid, _ = validate_user_id("user_42")
        self.assertTrue(valid)

        valid, msg = validate_user_id("")
        self.assertFalse(valid)
        self.assertIn("empty", msg)

        valid, msg = validate_user_id("   ")
        self.assertFalse(valid)

        valid, msg = validate_user_id(None)
        self.assertFalse(valid)

        valid, msg = validate_user_id("../traversal")
        self.assertFalse(valid)

        valid, msg = validate_user_id("user/slash")
        self.assertFalse(valid)

    def test_cold_start_initial_status(self):
        """A user with 0 saved interactions must be classified as Cold Start."""
        status = detect_user_status(self.client_a, self.user_1)
        self.assertEqual(status["status"], "cold_start")
        self.assertTrue(status["is_cold_start"])
        self.assertEqual(status["history_count"], 0)
        self.assertIsNone(status["last_active"])
        self.assertEqual(status["recent_datasets"], [])

    def test_warm_start_after_recording_event(self):
        """After recording at least 1 interaction, user must be classified as Warm Start."""
        # Initial cold start
        status_init = detect_user_status(self.client_a, self.user_1)
        self.assertEqual(status_init["status"], "cold_start")

        # Record visual recommendation event
        res = record_user_history_entry(
            client_id=self.client_a,
            user_id=self.user_1,
            entry_type="visual_recommendation",
            dataset_id="demo_ecommerce",
            items_or_charts=[{"chart_type": "bar", "x": "category", "y": "sales"}],
            metadata={"model": "mlvis"}
        )
        self.assertEqual(res["status"], "recorded")
        self.assertEqual(res["total_history_count"], 1)

        # Re-check status -> now warm_start
        status_after = detect_user_status(self.client_a, self.user_1)
        self.assertEqual(status_after["status"], "warm_start")
        self.assertFalse(status_after["is_cold_start"])
        self.assertEqual(status_after["history_count"], 1)
        self.assertIsNotNone(status_after["last_active"])
        self.assertIn("demo_ecommerce", status_after["recent_datasets"])

        # History retrieval
        hist = get_user_history(self.client_a, self.user_1)
        self.assertEqual(hist["total_records"], 1)
        self.assertEqual(hist["history"][0]["entry_type"], "visual_recommendation")
        self.assertEqual(hist["history"][0]["dataset_id"], "demo_ecommerce")

    def test_tenant_isolation(self):
        """Interactions in tenant A must never bleed into tenant B."""
        # Record interaction for user in tenant A
        record_user_history_entry(
            client_id=self.client_a,
            user_id=self.user_1,
            entry_type="chart_click",
            dataset_id="dataset_alpha"
        )

        status_a = detect_user_status(self.client_a, self.user_1)
        self.assertEqual(status_a["status"], "warm_start")
        self.assertEqual(status_a["history_count"], 1)

        # Same user_id in tenant B must still be cold_start
        status_b = detect_user_status(self.client_b, self.user_1)
        self.assertEqual(status_b["status"], "cold_start")
        self.assertEqual(status_b["history_count"], 0)

        hist_b = get_user_history(self.client_b, self.user_1)
        self.assertEqual(hist_b["total_records"], 0)

    def test_upload_or_dataset_switch_does_not_reset_warm_status(self):
        """Uploading or switching datasets should NOT reset a warm user back to cold start."""
        # Warm user up on dataset 1
        record_user_history_entry(
            client_id=self.client_a,
            user_id=self.user_1,
            entry_type="chart_click",
            dataset_id="dataset_1"
        )
        self.assertEqual(detect_user_status(self.client_a, self.user_1)["status"], "warm_start")

        # Now simulate dataset upload event
        record_user_history_entry(
            client_id=self.client_a,
            user_id=self.user_1,
            entry_type="dataset_upload",
            dataset_id="uploaded_custom_data.csv"
        )

        status = detect_user_status(self.client_a, self.user_1)
        self.assertEqual(status["status"], "warm_start")
        self.assertEqual(status["history_count"], 2)
        self.assertIn("uploaded_custom_data.csv", status["recent_datasets"])
        self.assertIn("dataset_1", status["recent_datasets"])

    def test_corrupted_json_recovery(self):
        """Service must gracefully recover from a corrupted or invalid JSON file without crashing."""
        # Write corrupted content to user's JSON file
        path = USER_HISTORY_BASE_DIR / self.client_a / f"{self.user_1}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("{ INVALID MALFORMED JSON !!!")

        # Should recover safely to empty history / cold start
        status = detect_user_status(self.client_a, self.user_1)
        self.assertEqual(status["status"], "cold_start")
        self.assertEqual(status["history_count"], 0)

        # Now recording a new entry should work normally
        record_user_history_entry(
            client_id=self.client_a,
            user_id=self.user_1,
            entry_type="recovery_test_event"
        )
        status_new = detect_user_status(self.client_a, self.user_1)
        self.assertEqual(status_new["status"], "warm_start")
        self.assertEqual(status_new["history_count"], 1)

    def test_warm_start_chart_preferences_and_bounded_weights(self):
        """Verify positive and negative feedback adjust chart preferences within bounded range [-0.15, +0.15]."""
        from backend.services.user_history_service import get_user_chart_preferences

        # 1. Cold user has no chart preferences
        cold_prefs = get_user_chart_preferences(self.client_a, "cold_user_99")
        self.assertFalse(cold_prefs["has_history"])
        self.assertEqual(cold_prefs["chart_type_weights"], {})

        # 2. User selects scatter plot repeatedly and provides helpful feedback
        record_user_history_entry(
            client_id=self.client_a,
            user_id=self.user_1,
            entry_type="chart_selection",
            items_or_charts=[{"chart_type": "scatter"}]
        )
        record_user_history_entry(
            client_id=self.client_a,
            user_id=self.user_1,
            entry_type="helpful_feedback",
            items_or_charts=[{"chart_type": "scatter"}],
            feedback="helpful"
        )

        # 3. User leaves unhelpful feedback on pie chart
        record_user_history_entry(
            client_id=self.client_a,
            user_id=self.user_1,
            entry_type="unhelpful_feedback",
            items_or_charts=[{"chart_type": "pie"}],
            feedback="unhelpful"
        )

        warm_prefs = get_user_chart_preferences(self.client_a, self.user_1)
        self.assertTrue(warm_prefs["has_history"])
        weights = warm_prefs["chart_type_weights"]

        # Scatter should receive positive weight, pie should receive negative weight
        self.assertGreater(weights.get("scatter", 0.0), 0.0)
        self.assertLess(weights.get("pie", 0.0), 0.0)

        # Bounds check: weights must never exceed [-0.15, 0.15]
        for ct, w in weights.items():
            self.assertGreaterEqual(w, -0.15)
            self.assertLessEqual(w, 0.15)


if __name__ == "__main__":
    unittest.main()

