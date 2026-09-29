import requests
import json

def test_live():
    print("=== 1. HEALTH CHECK ===")
    r_health = requests.get("http://127.0.0.1:8000/api/health")
    print("Status:", r_health.status_code, r_health.json())
    assert r_health.status_code == 200

    print("\n=== 2. CLIENTS LIST ===")
    r_clients = requests.get("http://127.0.0.1:8000/api/clients")
    print("Status:", r_clients.status_code)
    clients = r_clients.json().get("clients", [])
    for c in clients:
        print(f"  Client: {c['client_id']} | Active Model: {c.get('active_model_version')} | Has Model: {c.get('has_model')}")

    ar_client = next((c for c in clients if c["client_id"] == "context_aware_ar_interaction"), None)
    assert ar_client is not None, "Client not found"
    assert ar_client.get("active_model_version") == "v2", f"Expected v2, got {ar_client.get('active_model_version')}"

    print("\n=== 3. LIVE RECOMMENDATION FOR U153 ===")
    payload_u153 = {
        "client_id": "context_aware_ar_interaction",
        "user_id": "U153",
        "top_k": 5,
        "context": {"weather": "Clear", "temperature_celsius": 24.5, "crowd_density": "Medium"},
        "filters": {},
        "include_cold_start_weights": True
    }
    r_u153 = requests.post("http://127.0.0.1:8000/api/recommend", json=payload_u153)
    print("Status:", r_u153.status_code)
    data_u153 = r_u153.json()
    print("rec_id:", data_u153.get("recommendation_id"))
    print("status:", data_u153.get("status"))
    print("model_version:", data_u153.get("model_version"))
    print("latency_ms:", data_u153.get("latency_ms"))
    print("modality_weights:", data_u153.get("modality_weights"))
    print("recommendations count:", len(data_u153.get("recommendations", [])))
    for rec in data_u153.get("recommendations", []):
        print(f"  Rank {rec['rank']}: {rec['item_id']} (score={rec['score']:.4f}) metadata={rec['metadata']}")

    assert r_u153.status_code == 200
    assert data_u153.get("status") == "warm_start"
    assert data_u153.get("model_version") == "v2"
    assert len(data_u153.get("recommendations", [])) == 5
    assert all(rec["item_id"].startswith("L") for rec in data_u153.get("recommendations", []))

    print("\n=== 4. CONFIRMED KNOWN USER U058 ===")
    payload_u058 = {
        "client_id": "context_aware_ar_interaction",
        "user_id": "U058",
        "top_k": 5,
        "context": {},
        "filters": {},
        "include_cold_start_weights": True
    }
    r_u058 = requests.post("http://127.0.0.1:8000/api/recommend", json=payload_u058)
    print("Status:", r_u058.status_code)
    data_u058 = r_u058.json()
    print("status:", data_u058.get("status"), "model_version:", data_u058.get("model_version"))
    assert r_u058.status_code == 200
    assert data_u058.get("status") == "warm_start"
    assert data_u058.get("model_version") == "v2"

    print("\n=== 5. NEW COLD-START USER ===")
    payload_cold = {
        "client_id": "context_aware_ar_interaction",
        "user_id": "NEW_COLD_USER_4521",
        "top_k": 5,
        "context": {"weather": "Rainy", "temperature_celsius": 18.0},
        "filters": {},
        "include_cold_start_weights": True
    }
    r_cold = requests.post("http://127.0.0.1:8000/api/recommend", json=payload_cold)
    print("Status:", r_cold.status_code)
    data_cold = r_cold.json()
    print("status:", data_cold.get("status"), "model_version:", data_cold.get("model_version"))
    print("modality_weights:", data_cold.get("modality_weights"))
    assert r_cold.status_code == 200
    assert data_cold.get("status") == "cold_start"
    assert data_cold.get("model_version") == "v2"
    assert data_cold.get("modality_weights", {}).get("behavior", 1.0) < 0.01

    print("\n=== 6. SPA ROUTE /recommend ===")
    r_spa = requests.get("http://127.0.0.1:8000/recommend")
    print("Status:", r_spa.status_code, "Content length:", len(r_spa.text))
    assert r_spa.status_code == 200
    assert '<div id="root"></div>' in r_spa.text

    print("\n=== 7. DEMO ECOMMERCE CANONICAL ROUTE ===")
    payload_demo = {
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
    r_demo = requests.post("http://127.0.0.1:8000/api/recommend", json=payload_demo)
    print("Status:", r_demo.status_code)
    data_demo = r_demo.json()
    print("demo status:", data_demo.get("status"), "demo model_version:", data_demo.get("model_version"))
    assert r_demo.status_code == 200
    assert data_demo.get("model_version") == "v1"

    print("\n==========================================")
    print("ALL LIVE HTTP TESTS PASSED AGAINST ACTUAL RUNNING SERVER!")
    print("==========================================")

if __name__ == "__main__":
    test_live()
