# System Run & Execution Guide

This guide provides step-by-step instructions for launching, testing, and evaluating the **Real-Time Context-Aware Multi-Client Recommendation Platform** on Windows from a clean terminal.

---

## 1. Prerequisites & Environment

- **Operating System:** Windows 10/11 (PowerShell or Command Prompt)
- **Python:** Python 3.10+ (Tested on Python 3.13)
- **Node.js:** Node.js 18+ and npm (for frontend development and testing)
- **Repository Root:** `context_aware_recommender/`

All commands below assume your terminal is located at the project root directory:
```powershell
cd path\to\context_aware_recommender
```

---

## 2. Environment Activation

### PowerShell
```powershell
# If execution policy requires it:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```

### Windows Command Prompt (CMD)
```cmd
.\.venv\Scripts\activate.bat
```

### Direct Python Invocation (No Activation Required)
All commands can also be executed directly using the virtual environment's Python executable without activating the shell:
```powershell
.\.venv\Scripts\python.exe <command>
```

---

## 3. Starting the Backend API Server

The backend is built with FastAPI and runs on Uvicorn.

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

- **Backend Base URL:** `http://127.0.0.1:8000`
- **Interactive Swagger Documentation:** `http://127.0.0.1:8000/docs`
- **Alternative ReDoc Documentation:** `http://127.0.0.1:8000/redoc`
- **API Health Check:** `http://127.0.0.1:8000/api/health`

---

## 4. Production Single-Process Mode (Recommended for Demos)

In production single-process mode, FastAPI serves both the REST API endpoints and the pre-built React Single Page Application (SPA) on a single port (`8000`).

### Step 1: Build the React Frontend (One-time or after frontend changes)
```powershell
cd frontend
npm run build
cd ..
```
*Note: A pre-compiled `frontend/dist/` build is already included with the project repository.*

### Step 2: Launch the Unified Server
```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

### Step 3: Access the Application
Open your web browser and navigate to:
- **Interactive Recommendation Console:** `http://127.0.0.1:8000/recommend`
- **Management & Admin Dashboard:** `http://127.0.0.1:8000/admin`
- **Root Landing Page:** `http://127.0.0.1:8000/`

---

## 5. Development Mode (Hot-Reloading Frontend)

If you are actively modifying the React frontend, run Vite in development mode alongside the backend.

### Terminal 1: Backend Server
```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### Terminal 2: Frontend Vite Dev Server
```powershell
cd frontend
npm run dev
```

- **Vite Dev Server URL:** `http://localhost:5173`
- API calls made by the Vite development server are automatically proxied to `http://127.0.0.1:8000` via the configuration in `frontend/vite.config.js`.

---

## 6. Exploratory Streamlit Dashboard

The original exploratory Streamlit research application has been fully preserved.

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/app.py
```

- **Streamlit URL:** `http://localhost:8501`
- Includes interactive data exploration, cold-start cohort diagnostics, and multi-modal feature distribution plots.

---

## 7. Running Automated Test Suites

### Backend Unit Tests (65 Tests)
The backend test suite covers the complete data pipeline, leakage safeguards, multi-modal neural architecture, model registry, inference engine, REST endpoints, and closed-loop feedback persistence.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```
*Expected Result:* `Ran 65 tests in ~35s ... OK`

### Frontend Component Tests (10 Tests)
Frontend tests use Vitest and React Testing Library to verify form rendering, canonical schema roles, cold-start badge indicators, and training status displays.

```powershell
cd frontend
npm test
cd ..
```
*Expected Result:* `Test Files: 3 passed (3), Tests: 10 passed (10)`

---

## 8. Verifying the Production Build

To verify that the frontend compiles cleanly with zero syntax, JSX, or bundling errors:

```powershell
cd frontend
npm run build
cd ..
```
*Expected Result:* Clean generation of static chunks in `frontend/dist/`.

---

## 9. Common Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `Address already in use` (Port 8000) | A previous Uvicorn process is still running. | In PowerShell run: `Get-Process python \| Stop-Process` or find the PID using `netstat -ano \| findstr :8000` and kill it. |
| `No module named ...` | Virtual environment is not being referenced. | Always use `.\.venv\Scripts\python.exe` or execute `.\.venv\Scripts\Activate.ps1` before running Python commands. |
| `database is locked` | Concurrent SQLite write lock during rapid testing. | Stop running servers and remove any stale lock or run tests sequentially. |
| Frontend displays blank page on `http://127.0.0.1:8000` | `frontend/dist/` has not been compiled. | Run `cd frontend; npm run build; cd ..` to regenerate the production bundle. |
| `Failed to fetch` on Live Recommender | Backend server is not running on port 8000. | Ensure `uvicorn backend.main:app` is running and accessible at `http://127.0.0.1:8000/api/health`. |
