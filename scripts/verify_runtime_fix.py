import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from fastapi.testclient import TestClient
from backend.main import app

def run_tests():
    client = TestClient(app)

    print("=== 1. VERIFY CLIENT SELECTOR METADATA ===")
    res = client.get("/api/clients")
    print("Status:", res.status_code)
    clients_data = res.json().get("clients", [])
    ar_client = next((c for c in clients_data if c["client_id"] == "context_aware_ar_interaction"), None)
    print("context_aware_ar_interaction client info:")
    print(json.dumps(ar_client, indent=2))
    assert ar_client is not None, "Client context_aware_ar_interaction not found"
    assert ar_client.get("active_model_version") == "v2", f"Expected active_model_version 'v2', got {ar_client.get('active_model_version')}"
    assert ar_client.get("has_model") is True

    print("\n=== 2. VERIFY U153 LIVE RECOMMENDATION ===")
    u153_payload = {
        "client_id": "context_aware_ar_interaction",
        "user_id": "U153",
        "top_k": 5,
        "context": {"weather": "Clear", "temperature_celsius": 24.5, "crowd_density": "Medium"},
        "filters": {},
        "include_cold_start_weights": True
    }
    res_u153 = client.post("/api/recommend", json=u153_payload)
    print("Status:", res_u153.status_code)
    d_u153 = res_u153.json()
    print("rec_id:", d_u153.get("recommendation_id"))
    print("status:", d_u153.get("status"))
    print("model_version:", d_u153.get("model_version"))
    print("latency_ms:", d_u153.get("latency_ms"))
    print("modality_weights:", d_u153.get("modality_weights"))
    print("recommendations count:", len(d_u153.get("recommendations", [])))
    for r in d_u153.get("recommendations", []):
        print(f"  Rank {r['rank']}: {r['item_id']} (score={r['score']:.4f}) metadata={r['metadata']}")
    assert d_u153.get("status") == "warm_start", f"Expected warm_start, got {d_u153.get('status')}"
    assert d_u153.get("model_version") == "v2", f"Expected v2, got {d_u153.get('model_version')}"
    assert len(d_u153.get("recommendations", [])) == 5
    assert all(r["item_id"].startswith("L") for r in d_u153.get("recommendations", []))

    print("\n=== 3. VERIFY CONFIRMED KNOWN USER (U058) ===")
    u058_payload = {
        "client_id": "context_aware_ar_interaction",
        "user_id": "U058",
        "top_k": 5,
        "context": {},
        "filters": {},
        "include_cold_start_weights": True
    }
    res_u058 = client.post("/api/recommend", json=u058_payload)
    print("Status:", res_u058.status_code)
    d_u058 = res_u058.json()
    print("status:", d_u058.get("status"))
    print("model_version:", d_u058.get("model_version"))
    print("modality_weights:", d_u058.get("modality_weights"))
    assert d_u058.get("status") == "warm_start", f"Expected warm_start, got {d_u058.get('status')}"
    assert d_u058.get("model_version") == "v2"

    print("\n=== 4. VERIFY NEW COLD-START USER (NEW_COLD_USER_4521) ===")
    cold_payload = {
        "client_id": "context_aware_ar_interaction",
        "user_id": "NEW_COLD_USER_4521",
        "top_k": 5,
        "context": {"weather": "Rainy", "temperature_celsius": 18.0},
        "filters": {},
        "include_cold_start_weights": True
    }
    res_cold = client.post("/api/recommend", json=cold_payload)
    print("Status:", res_cold.status_code)
    d_cold = res_cold.json()
    print("status:", d_cold.get("status"))
    print("model_version:", d_cold.get("model_version"))
    print("modality_weights:", d_cold.get("modality_weights"))
    print("recommendations count:", len(d_cold.get("recommendations", [])))
    for r in d_cold.get("recommendations", []):
        print(f"  Rank {r['rank']}: {r['item_id']} (score={r['score']:.4f})")
    assert d_cold.get("status") == "cold_start", f"Expected cold_start, got {d_cold.get('status')}"
    assert d_cold.get("modality_weights", {}).get("behavior", 1.0) < 0.01, f"Expected behavior weight < 0.01, got {d_cold.get('modality_weights', {}).get('behavior')}"

    print("\n=== 5. VERIFY DEMO ECOMMERCE IS UNTOUCHED AND WORKS ===")
    demo_payload = {
        "client_id": "demo_ecommerce",
        "user_id": "9272",
        "top_k": 5,
        "context": {
            "device_type": "Mobile",
            "marketing_channel": "Direct",
            "visit_season": "Autumn",
            "location": 42
        },
        "filters": {},
        "include_cold_start_weights": True
    }
    res_demo = client.post("/api/recommend", json=demo_payload)
    print("Demo status:", res_demo.status_code)
    d_demo = res_demo.json()
    print("Demo model version:", d_demo.get("model_version"))
    print("Demo recommendations count:", len(d_demo.get("recommendations", [])))
    assert d_demo.get("model_version") == "v1"
    assert len(d_demo.get("recommendations", [])) == 5

    print("\nALL VERIFICATIONS PASSED!")

if __name__ == "__main__":
    run_tests()
