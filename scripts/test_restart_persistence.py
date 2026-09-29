import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import requests
import subprocess
import os
import signal

def test_restart():
    venv_python = str(Path(__file__).resolve().parent.parent / ".venv" / "Scripts" / "python.exe")
    cmd = [venv_python, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"]

    print("Step 1: Launching FastAPI server (Process 1)...")
    proc1 = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        # Wait for server to come online
        online = False
        for _ in range(30):
            try:
                r = requests.get("http://127.0.0.1:8000/api/health", timeout=1)
                if r.status_code == 200:
                    online = True
                    break
            except Exception:
                time.sleep(0.5)
        assert online, "Server 1 failed to start within 15 seconds"
        print("Server 1 is ONLINE!")

        # Check clients and active_model_version
        r_clients = requests.get("http://127.0.0.1:8000/api/clients").json()
        ar_c = next((c for c in r_clients.get("clients", []) if c["client_id"] == "context_aware_ar_interaction"), None)
        print("Server 1 - context_aware_ar_interaction active_model_version:", ar_c.get("active_model_version"))
        assert ar_c.get("active_model_version") == "v2"

        # Check recommendation for U153
        payload = {
            "client_id": "context_aware_ar_interaction",
            "user_id": "U153",
            "top_k": 5,
            "context": {"weather": "Clear"},
            "filters": {},
            "include_cold_start_weights": True
        }
        r_rec1 = requests.post("http://127.0.0.1:8000/api/recommend", json=payload).json()
        print("Server 1 - rec status:", r_rec1.get("status"), "model_version:", r_rec1.get("model_version"))
        assert r_rec1.get("model_version") == "v2"
        assert len(r_rec1.get("recommendations", [])) == 5

    finally:
        print("\nStep 2: Completely terminating Server 1...")
        proc1.terminate()
        proc1.kill()
        proc1.wait()
        time.sleep(2)

    # Verify server is down
    down = False
    try:
        requests.get("http://127.0.0.1:8000/api/health", timeout=1)
    except Exception:
        down = True
    print("Server 1 is fully DOWN:", down)
    assert down, "Server did not shut down properly"

    print("\nStep 3: Launching NEW FastAPI server instance (Process 2 - Cold Restart)...")
    proc2 = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        online = False
        for _ in range(30):
            try:
                r = requests.get("http://127.0.0.1:8000/api/health", timeout=1)
                if r.status_code == 200:
                    online = True
                    break
            except Exception:
                time.sleep(0.5)
        assert online, "Server 2 failed to start within 15 seconds"
        print("Server 2 is ONLINE!")

        # Check clients again after restart
        r_clients2 = requests.get("http://127.0.0.1:8000/api/clients").json()
        ar_c2 = next((c for c in r_clients2.get("clients", []) if c["client_id"] == "context_aware_ar_interaction"), None)
        print("Server 2 - context_aware_ar_interaction active_model_version:", ar_c2.get("active_model_version"))
        assert ar_c2.get("active_model_version") == "v2", "Model v2 was not persisted across restart!"

        # Check recommendations after restart
        r_rec2 = requests.post("http://127.0.0.1:8000/api/recommend", json=payload).json()
        print("Server 2 - rec status:", r_rec2.get("status"), "model_version:", r_rec2.get("model_version"))
        assert r_rec2.get("model_version") == "v2"
        assert len(r_rec2.get("recommendations", [])) == 5
        print("Server 2 successfully loaded model v2 from disk and served recommendations!")

    finally:
        print("\nStep 4: Cleaning up Process 2...")
        proc2.terminate()
        proc2.kill()
        proc2.wait()
        time.sleep(1)

    print("\nRESTART PERSISTENCE VERIFICATION PASSED COMPLETELY!")

if __name__ == "__main__":
    test_restart()
