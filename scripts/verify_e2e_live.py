"""
End-to-End Quality Control Verification Script (Requirement 5).
Validates:
1. Admin Dashboard, Dataset, Schema, Training, Models, Analytics, Feedback HTML routes.
2. Models page has v1 active and saved metrics.
3. Live Recommendations across all 7 scenarios:
   - 1. Completely new user (N_hist = 0 -> cold_start)
   - 2. Cold known user (N_hist = 1 -> cold_start)
   - 3. User with exactly 3 interactions (N_hist = 3 -> warm_start because 3 >= K=3)
   - 4. Warm user with >3 interactions (N_hist = 8 -> warm_start)
   - 5. Category filter applied
   - 6. Maximum price filter applied
   - 7. Feedback event submission linked to recommendation_id
"""

import sys
import json
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

print("=" * 60)
print("RUNNING END-TO-END QC VERIFICATION")
print("=" * 60)

# A. Admin routes check
routes = [
    ("/", 200, "Multi-Modal Context-Aware Recommender Platform"),
    ("/admin", 200, "<div id=\"root\"></div>"),
    ("/admin/clients", 200, "<div id=\"root\"></div>"),
    ("/admin/clients/demo_ecommerce/dataset", 200, "<div id=\"root\"></div>"),
    ("/admin/clients/demo_ecommerce/schema", 200, "<div id=\"root\"></div>"),
    ("/admin/clients/demo_ecommerce/training", 200, "<div id=\"root\"></div>"),
    ("/admin/clients/demo_ecommerce/models", 200, "<div id=\"root\"></div>"),
    ("/admin/clients/demo_ecommerce/analytics", 200, "<div id=\"root\"></div>"),
    ("/admin/clients/demo_ecommerce/feedback", 200, "<div id=\"root\"></div>"),
    ("/recommend", 200, "<div id=\"root\"></div>"),
]

for route, expected_code, expected_snippet in routes:
    res = client.get(route)
    assert res.status_code == expected_code, f"Route {route} returned {res.status_code}"
    assert expected_snippet in res.text, f"Snippet missing from {route}"
    print(f"[PASS] Route '{route}' OK (Status {res.status_code})")

# B. Verify active model v1 and analytics
res_models = client.get("/api/clients/demo_ecommerce/models")
assert res_models.status_code == 200
models = res_models.json()
active_model = next((m for m in models if m["is_active"]), None)
assert active_model is not None, "No active model found for demo_ecommerce"
assert active_model["version"] == "v1", f"Active model is {active_model['version']} instead of v1"
print(f"[PASS] Active model for demo_ecommerce verified: v1 (Run: {active_model.get('training_run_id')})")
print(f"  Feature Dimensions: {active_model.get('feature_dimensions')}")
print(f"  Saved Metrics: {active_model.get('metrics')}")

# C. Live Recommendations Scenario Tests
# 1. Completely new user
res_new = client.post("/api/recommend", json={
    "client_id": "demo_ecommerce",
    "user_id": "completely_new_user_qc_001",
    "top_k": 5
})
assert res_new.status_code == 200
d_new = res_new.json()
assert d_new["status"] == "cold_start", f"Expected cold_start, got {d_new['status']}"
assert d_new["history_count"] == 0, f"Expected 0 history, got {d_new['history_count']}"
assert d_new["modality_weights"]["behavior"] <= 0.01, f"Expected suppressed beh weight, got {d_new['modality_weights']['behavior']}"
print(f"[PASS] Scenario 1: Completely new user -> status='{d_new['status']}', N_hist={d_new['history_count']}, beh_weight={d_new['modality_weights']['behavior']:.4f}")

# 2. Cold known user (User 7838 has 1 interaction)
res_cold = client.post("/api/recommend", json={
    "client_id": "demo_ecommerce",
    "user_id": "7838",
    "top_k": 5
})
assert res_cold.status_code == 200
d_cold = res_cold.json()
assert d_cold["status"] == "cold_start", f"Expected cold_start, got {d_cold['status']}"
assert d_cold["history_count"] == 1, f"Expected 1 history, got {d_cold['history_count']}"
assert d_cold["modality_weights"]["behavior"] <= 0.01
print(f"[PASS] Scenario 2: Cold known user (7838) -> status='{d_cold['status']}', N_hist={d_cold['history_count']}, beh_weight={d_cold['modality_weights']['behavior']:.4f}")

# 3. User with EXACTLY 3 interactions (User 5992 has 3 interactions)
res_exact3 = client.post("/api/recommend", json={
    "client_id": "demo_ecommerce",
    "user_id": "5992",
    "top_k": 5
})
assert res_exact3.status_code == 200
d_exact3 = res_exact3.json()
assert d_exact3["history_count"] == 3, f"Expected exactly 3 history, got {d_exact3['history_count']}"
assert d_exact3["status"] == "warm_start", f"CRITICAL: User with 3 interactions must be warm_start (3 >= 3), got {d_exact3['status']}"
assert d_exact3["modality_weights"]["behavior"] > 0.05, f"Warm user should not have suppressed behavior weight, got {d_exact3['modality_weights']['behavior']}"
print(f"[PASS] Scenario 3: EXACTLY 3 interactions (5992) -> status='{d_exact3['status']}', N_hist={d_exact3['history_count']}, beh_weight={d_exact3['modality_weights']['behavior']:.4f} [WARM START CONFIRMED]")

# 4. Warm user with >3 interactions (User 9272 has 8 interactions)
res_warm = client.post("/api/recommend", json={
    "client_id": "demo_ecommerce",
    "user_id": "9272",
    "top_k": 5
})
assert res_warm.status_code == 200
d_warm = res_warm.json()
assert d_warm["status"] == "warm_start", f"Expected warm_start, got {d_warm['status']}"
assert d_warm["history_count"] == 8, f"Expected 8 history, got {d_warm['history_count']}"
assert d_warm["modality_weights"]["behavior"] > 0.05
print(f"[PASS] Scenario 4: Warm user (9272) -> status='{d_warm['status']}', N_hist={d_warm['history_count']}, beh_weight={d_warm['modality_weights']['behavior']:.4f}")

# 5. Category filter
res_cat = client.post("/api/recommend", json={
    "client_id": "demo_ecommerce",
    "user_id": "9272",
    "top_k": 5,
    "filters": {"category": "1"}
})
assert res_cat.status_code == 200
d_cat = res_cat.json()
for rec in d_cat["recommendations"]:
    cat_val = str(rec["metadata"].get("product_category"))
    assert cat_val == "1", f"Category mismatch: expected '1', got '{cat_val}'"
print(f"[PASS] Scenario 5: Category filter ('1') -> all {len(d_cat['recommendations'])} items match category '1'")

# 6. Maximum price filter
res_price = client.post("/api/recommend", json={
    "client_id": "demo_ecommerce",
    "user_id": "9272",
    "top_k": 5,
    "filters": {"max_price": 75.0}
})
assert res_price.status_code == 200
d_price = res_price.json()
for rec in d_price["recommendations"]:
    price_val = float(rec["metadata"].get("unit_price", 999.0))
    assert price_val <= 75.0, f"Price violation: expected <= 75.0, got {price_val}"
print(f"[PASS] Scenario 6: Max price filter (<= 75.0) -> all {len(d_price['recommendations'])} items have price <= $75.0")

# 7. Feedback submission linked to recommendation_id
rec_id = d_exact3["recommendation_id"]
recommended_item = d_exact3["recommendations"][0]["item_id"]
res_fb = client.post("/api/feedback", json={
    "client_id": "demo_ecommerce",
    "user_id": "5992",
    "item_id": recommended_item,
    "action": "PURCHASE",
    "recommendation_id": rec_id
})
assert res_fb.status_code == 201
d_fb = res_fb.json()
assert d_fb["recommendation_id"] == rec_id
assert d_fb["action"] == "PURCHASE"
assert d_fb["item_id"] == recommended_item
print(f"[PASS] Scenario 7: Feedback submission linked -> Event ID #{d_fb['id']}, action={d_fb['action']}, rec_id={d_fb['recommendation_id']}")

print("=" * 60)
print("ALL 7 END-TO-END SCENARIOS & ROUTE CHECKS VERIFIED SUCCESSFULLY!")
print("=" * 60)
