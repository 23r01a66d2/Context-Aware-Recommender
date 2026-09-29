import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from backend.main import app

def run():
    client = TestClient(app)

    print("=== 1. Test MovieLens User 1 (Known) ===")
    res1 = client.post('/api/recommend', json={
        'client_id': 'movielens_dataset',
        'user_id': '1',
        'top_k': 5,
        'context': {},
        'filters': {},
        'include_cold_start_weights': True
    })
    print("Status 1:", res1.status_code)
    d1 = res1.json()
    print("Status:", d1.get("status"), "Model:", d1.get("model_version"))
    print("History count:", d1.get("history_count"))
    print("Candidate pool size:", d1.get("candidate_pool_size"))
    print("Latency ms:", d1.get("latency_ms"))
    print("Modality weights:", d1.get("modality_weights"))
    print("Recommendations:")
    for r in d1.get("recommendations", []):
        print(f"  Rank {r['rank']}: {r['item_id']} score={r['score']:.4f} meta={r.get('metadata')}")

    print("\n=== 2. Test MovieLens User 2 (Known) ===")
    res2 = client.post('/api/recommend', json={
        'client_id': 'movielens_dataset',
        'user_id': '2',
        'top_k': 5,
        'context': {},
        'filters': {},
        'include_cold_start_weights': True
    })
    print("Status 2:", res2.status_code)
    d2 = res2.json()
    print("Status:", d2.get("status"), "Model:", d2.get("model_version"))
    print("History count:", d2.get("history_count"))
    print("Candidate pool size:", d2.get("candidate_pool_size"))
    print("Latency ms:", d2.get("latency_ms"))
    print("Modality weights:", d2.get("modality_weights"))
    print("Recommendations:")
    for r in d2.get("recommendations", []):
        print(f"  Rank {r['rank']}: {r['item_id']} score={r['score']:.4f}")

    print("\n=== 3. Test MovieLens Unknown User 99999 (Cold-Start) ===")
    res3 = client.post('/api/recommend', json={
        'client_id': 'movielens_dataset',
        'user_id': '99999',
        'top_k': 5,
        'context': {},
        'filters': {},
        'include_cold_start_weights': True
    })
    print("Status 3:", res3.status_code)
    d3 = res3.json()
    print("Status:", d3.get("status"), "Model:", d3.get("model_version"))
    print("History count:", d3.get("history_count"))
    print("Candidate pool size:", d3.get("candidate_pool_size"))
    print("Latency ms:", d3.get("latency_ms"))
    print("Modality weights:", d3.get("modality_weights"))
    print("Recommendations:")
    for r in d3.get("recommendations", []):
        print(f"  Rank {r['rank']}: {r['item_id']} score={r['score']:.4f}")

if __name__ == "__main__":
    run()
