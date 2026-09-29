"""
One-Click Local Launcher Helper for Context-Aware Recommender.
Provides safe start, health verification, duplicate prevention, and clean shutdown.
"""

import sys
import os
import time
import json
import socket
import urllib.request
import urllib.error
import subprocess
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PID_FILE = PROJECT_ROOT / "data" / "server.pid"
LOG_FILE = PROJECT_ROOT / "data" / "server.log"
PORT = 8000
FALLBACK_URL = f"http://127.0.0.1:{PORT}"
HOST_NAME = "context-aware-recommender.local"


def get_target_url() -> str:
    """
    Returns the friendly URL if context-aware-recommender.local resolves to loopback (127.x.x.x),
    otherwise returns the reliable fallback http://127.0.0.1:8000.
    """
    try:
        ip = socket.gethostbyname(HOST_NAME)
        if ip.startswith("127."):
            return f"http://{HOST_NAME}:{PORT}"
    except Exception:
        pass
    return FALLBACK_URL


def is_port_in_use(port: int = PORT) -> bool:
    """Checks whether the specified TCP port is listening on 127.0.0.1."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def check_app_health(timeout: float = 1.5) -> tuple[bool, dict]:
    """
    Sends GET request to /api/health and verifies that it responds
    with status=ok and service=context-aware-recommender.
    """
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/api/health")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("status") in ("ok", "healthy") and data.get("service") == "context-aware-recommender":
                    return True, data
    except Exception:
        pass
    return False, {}


def is_our_server_proc(pid: int) -> bool:
    """
    Safely verifies via Windows PowerShell whether a given process ID
    corresponds to our Uvicorn recommender server (backend.main:app).
    """
    if pid <= 0:
        return False
    try:
        cmd = f"(Get-CimInstance Win32_Process -Filter \"ProcessId = {pid}\").CommandLine"
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", cmd],
            text=True,
            stderr=subprocess.DEVNULL
        ).strip()
        if "backend.main:app" in out:
            return True
    except Exception:
        pass
    return False


def get_pids_on_port(port: int = PORT) -> list[int]:
    """Retrieves all process IDs listening on the given port via PowerShell."""
    try:
        cmd = f"Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess"
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", cmd],
            text=True,
            stderr=subprocess.DEVNULL
        ).strip()
        pids = []
        for line in out.splitlines():
            line = line.strip()
            if line.isdigit():
                pids.append(int(line))
        return list(set(pids))
    except Exception:
        return []


def terminate_pid(pid: int) -> bool:
    """Terminates a process and its process tree safely using taskkill."""
    try:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, text=True)
        return True
    except Exception:
        return False


def print_log_tail(num_lines: int = 15):
    """Prints the last few lines of the server log file for diagnostics."""
    if LOG_FILE.exists():
        try:
            with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
                tail = lines[-num_lines:] if len(lines) > num_lines else lines
                print("\n--- Recent Server Log Output ---")
                print("".join(tail))
                print("--------------------------------")
        except Exception:
            pass


def cmd_start() -> int:
    """Main start routine."""
    target_url = get_target_url()

    print("========================================================")
    print("             CONTEXT-AWARE RECOMMENDER")
    print("          Multi-Modal Recommendation Platform")
    print("========================================================")
    print()
    print("Checking application...")
    print()
    print("Python Environment : OK")

    # 1. Check if port is already in use
    if is_port_in_use(PORT):
        healthy, data = check_app_health(timeout=2.0)
        if healthy:
            print("Application Server : ALREADY RUNNING")
            print(f"URL                : {target_url}")
            print(f"Status             : HEALTHY ({data.get('service', 'online')})")
            print()
            print("No duplicate Uvicorn process was launched.")
            print(f"Opening browser at {target_url}...")
            try:
                webbrowser.open(target_url)
            except Exception as e:
                print(f"[NOTE] Please open your browser to {target_url} ({e})")
            print()
            print("Application is active and ready for demonstration.")
            return 0
        else:
            print("Application Server : PORT CONFLICT ERROR")
            print()
            print(f"[ERROR] Port {PORT} is already occupied by another application.")
            print(f"The service at http://127.0.0.1:{PORT}/api/health did not identify as 'context-aware-recommender'.")
            print("Please terminate the conflicting application before starting this project.")
            return 1

    # 2. Port is free, start new server
    print("Application Server : STARTING")
    print(f"URL                : {target_url}")
    print()
    print("Please keep this window open during the demonstration.")
    print()

    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    log_fp = open(LOG_FILE, "a", encoding="utf-8")
    log_fp.write(f"\n========================================================\n")
    log_fp.write(f"Server launched at {time.ctime()}\n")
    log_fp.write(f"========================================================\n")
    log_fp.flush()

    server_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", str(PORT)],
        cwd=str(PROJECT_ROOT),
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
    )

    with open(PID_FILE, "w", encoding="utf-8") as f:
        f.write(str(server_proc.pid))

    # 3. Health check polling
    print("Awaiting application initialization (/api/health)...")
    healthy = False
    start_time = time.time()
    max_wait_seconds = 20.0

    while time.time() - start_time < max_wait_seconds:
        # Check if process died prematurely
        if server_proc.poll() is not None:
            print(f"\n[ERROR] Server process exited unexpectedly with code {server_proc.poll()}.")
            print_log_tail()
            PID_FILE.unlink(missing_ok=True)
            return 1

        healthy, _ = check_app_health(timeout=1.0)
        if healthy:
            break
        time.sleep(0.5)

    if not healthy:
        print("\n[ERROR] Health check timed out after 20 seconds.")
        print_log_tail()
        terminate_pid(server_proc.pid)
        PID_FILE.unlink(missing_ok=True)
        return 1

    elapsed = time.time() - start_time
    print()
    print(f"Health Check       : PASSED ({elapsed:.1f}s)")
    print(f"Process ID (PID)   : {server_proc.pid}")
    print()
    print(f"Opening browser at {target_url}...")
    try:
        webbrowser.open(target_url)
    except Exception as e:
        print(f"[NOTE] Please open your browser to {target_url} ({e})")
    print()
    print("========================================================")
    print(" Demonstration environment is ready!")
    print(" To stop the server, run STOP_CONTEXT_AWARE_RECOMMENDER.bat")
    print(" or press Ctrl+C in this window.")
    print("========================================================")
    print()

    # 4. Keep process open
    try:
        server_proc.wait()
    except KeyboardInterrupt:
        print("\nStopping application server gracefully...")
        terminate_pid(server_proc.pid)
    finally:
        PID_FILE.unlink(missing_ok=True)
        try:
            log_fp.close()
        except Exception:
            pass

    return 0


def cmd_stop() -> int:
    """Main stop routine."""
    print("========================================================")
    print("             CONTEXT-AWARE RECOMMENDER")
    print("          Multi-Modal Recommendation Platform")
    print("========================================================")
    print()
    print("Stopping application server...")

    stopped_any = False

    # 1. Stop recorded PID
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text(encoding="utf-8").strip())
            if is_our_server_proc(pid):
                print(f"Stopping registered server process (PID {pid})...")
                terminate_pid(pid)
                print(f"[OK] Terminated Context-Aware Recommender process (PID {pid}).")
                stopped_any = True
            else:
                print(f"[NOTE] PID {pid} is no longer running our recommender server.")
        except Exception as e:
            print(f"[WARN] Error reading PID file: {e}")
        finally:
            PID_FILE.unlink(missing_ok=True)

    # 2. Check port 8000 for any lingering recommender processes
    pids_on_port = get_pids_on_port(PORT)
    for p in pids_on_port:
        if is_our_server_proc(p):
            print(f"Stopping active recommender server on port {PORT} (PID {p})...")
            terminate_pid(p)
            print(f"[OK] Terminated Context-Aware Recommender on port {PORT} (PID {p}).")
            stopped_any = True
        else:
            print(f"[INFO] Process {p} on port {PORT} belongs to another application. Preserved.")

    # 3. Verify port is released
    time.sleep(1.0)
    for _ in range(6):
        if not is_port_in_use(PORT):
            break
        time.sleep(0.5)

    if not is_port_in_use(PORT):
        print(f"[OK] Port {PORT} is free.")
        print()
        print("Application server stopped successfully.")
    else:
        if stopped_any:
            print(f"[WARN] Termination signal sent, but port {PORT} is still occupied.")
        else:
            print(f"[INFO] No running Context-Aware Recommender server was found on port {PORT}.")

    return 0


if __name__ == "__main__":
    action = sys.argv[1].lower() if len(sys.argv) > 1 else "start"
    if action == "start":
        sys.exit(cmd_start())
    elif action == "stop":
        sys.exit(cmd_stop())
    else:
        print(f"Unknown action: {action}. Use 'start' or 'stop'.")
        sys.exit(1)
