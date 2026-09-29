# Live System Demonstration Checklist & Test User Guide

This checklist ensures flawless preparation and execution when demonstrating the **Real-Time Context-Aware Multi-Client Recommendation Platform** to evaluators, professors, or stakeholders.

---

## 1. Pre-Demonstration Verification Checklist

Complete these verification checks at least 15 minutes prior to the live presentation:

- [ ] **Python Virtual Environment:**
  - Verify that `.\.venv\Scripts\python.exe` is functional.
  - Quick test: `.\.venv\Scripts\python.exe --version`
- [ ] **Pre-compiled Frontend Assets:**
  - Ensure `frontend/dist/index.html` exists.
  - If missing, compile immediately: `cd frontend; npm run build; cd ..`
- [ ] **Reference Client & Model Artifacts:**
  - Verify that `data/raw/Ecommerce.csv` is present.
  - Verify that `models/demo_ecommerce/v1/checkpoint.pt` is present (frozen PyTorch model).
  - Verify that `models/demo_ecommerce/v1/feature_encoders.joblib` is present.
  - Verify that `data/platform.db` contains client `demo_ecommerce` with active model `v1`.
- [ ] **Automated Test Health:**
  - Backend tests: `.\.venv\Scripts\python.exe -m unittest discover -s tests` (65 tests pass).
  - Frontend tests: `cd frontend; npm test` (10 tests pass).
- [ ] **Port Availability:**
  - Ensure port `8000` is free: `netstat -ano | findstr :8000` returns empty.
- [ ] **Launch Unified Application Server:**
  - Start server: `.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000`
  - Verify API health in browser: `http://127.0.0.1:8000/api/health` returns `{"status": "ok"}`.
- [ ] **Pre-open Browser Tabs:**
  - **Tab 1 (Live Recommender):** `http://127.0.0.1:8000/recommend`
  - **Tab 2 (Admin & Analytics Console):** `http://127.0.0.1:8000/admin`
  - **Tab 3 (Interactive API Docs):** `http://127.0.0.1:8000/docs`

---

## 2. Verified Demonstration Users (From Reference Dataset)

The following customer profiles are pre-verified against the reference dataset (`Ecommerce.csv`, 25,000 interactions, 8,442 unique customers) and reflect exact cold-start and warm-start behaviors under the primary gating threshold $K = 3$:

| User Type | Customer ID | Interaction Depth ($N_{\text{hist}}$) | Gating Cohort | Behavioral Weight ($\alpha_{\text{beh}}$) | Expected UI Display | Recommended Talking Point |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **Completely New User** | `new_customer_demo_001` *(or any unobserved string)* | **0** | **Cold Start** ($N_{\text{hist}} < 3$) | **$\le 0.002$** (Suppressed) | Blue Badge: `Cold Start (0 interactions)` | "Zero prior history. Gating gate completely suppresses behavioral noise; recommendation is driven by item content and environmental context." |
| **Sparse History User** | `7838` | **1** | **Cold Start** ($N_{\text{hist}} < 3$) | **$\le 0.002$** (Suppressed) | Blue Badge: `Cold Start (1 interaction)` | "A single prior interaction is insufficient for reliable profiling. The adaptive gate keeps behavioral weight suppressed." |
| **Sparse History User** | `6993` | **2** | **Cold Start** ($N_{\text{hist}} < 3$) | **$\le 0.002$** (Suppressed) | Blue Badge: `Cold Start (2 interactions)` | "Two past interactions—still below threshold $K=3$. Safeguards against early-session overfitting." |
| **Exact Boundary User** | `5992` | **3** | **Warm Start** ($N_{\text{hist}} \ge 3$) | **$\approx 0.145$** (Activated) | Emerald Badge: `Warm Start (3 interactions)` | "Crucial demonstration point: Exactly at $N_{\text{hist}} = 3$, the gate crosses threshold $K=3$. Behavioral modality activates and blends with content/context." |
| **Known Warm User** | `9272` | **8** | **Warm Start** ($N_{\text{hist}} \ge 3$) | **$\approx 0.142$** (Activated) | Emerald Badge: `Warm Start (8 interactions)` | "Established customer with 8 prior transactions. Full personal history, category affinity, and purchase velocity are actively utilized." |

---

## 3. Demonstration Flow Checklist (5–8 Minutes)

- [ ] **1. Executive Problem Statement (1 min):**
  - Highlight the 44.34% cold-start challenge in e-commerce ($N_{\text{hist}} < 3$).
  - State the solution: Multi-modal fusion with adaptive gating ($K=3$).
- [ ] **2. Admin Workspace & Isolation (1.5 min):**
  - Navigate to `/admin` -> `demo_ecommerce`.
  - Show dataset statistics: 25,000 interactions, 8,442 users, 899 products.
  - Show chronological split: Train (Jan–Jul 2024), Val (Aug–Sep 2024), Test (Oct–Dec 2024).
- [ ] **3. Semantic Schema Mapping (1 min):**
  - Show canonical 8 roles: `USER_ID`, `ITEM_ID`, `TIMESTAMP`, `TARGET`, `BEHAVIOR_FEATURE`, `CONTENT_FEATURE`, `CONTEXT_FEATURE`, `IGNORE`.
  - Highlight post-recommendation leakage fields set to `IGNORE` (`revenue`, `added_to_cart`, `rating`).
- [ ] **4. Model Registry & Offline Metrics (1 min):**
  - Show active model `v1` in Model Registry.
  - Open Analytics: Explain the **Sampled Offline Ranking Evaluation** (1 positive + 99 unpurchased negatives).
  - Explicitly acknowledge baseline comparisons (Content-Based NDCG@5 = 0.1374; Multi-Modal No Adaptation = 0.0401; Proposed Adaptive = 0.0352).
- [ ] **5. Live Recommender: Cold User vs Warm User (2 min):**
  - Navigate to `/recommend`.
  - Input `new_customer_demo_001` -> Click **Generate Recommendations**.
    - Verify: Blue Badge `Cold Start (0 interactions)`, $\alpha_{\text{beh}} \le 0.002$, Content & Context dominate.
  - Input `5992` (Boundary, 3 interactions) or `9272` (Warm, 8 interactions) -> Click **Generate Recommendations**.
    - Verify: Emerald Badge `Warm Start`, $\alpha_{\text{beh}} \approx 0.14$.
- [ ] **6. Live Candidate Filtering & Feedback (1 min):**
  - Toggle **Advanced Filters** -> Set Category `1` and Max Price `$75.00`.
  - Verify that returned candidates strictly match constraints.
  - Click **Purchase** feedback button on recommendation #1 -> Verify green confirmation toast.
- [ ] **7. Q&A and Boundary Clarifications (1 min):**
  - Reiterate scope: Product recommendation is learned; visual analytics are deterministic.
  - Scores are relative relevance scores, not calibrated purchase probabilities.
  - Candidate scoring runs exact catalog matrix multiplication (899 items) under 55ms.

---

## 4. Startup Troubleshooting

| Scenario | Symptom | Action |
| :--- | :--- | :--- |
| **Port 8000 Conflict** | `[Errno 10048] error while attempting to bind on address ('127.0.0.1', 8000)` | Stop any lingering Python processes: In PowerShell: `Get-Process python \| Stop-Process -Force` |
| **Virtual Environment Missing** | Command `python` points to system Python without dependencies | Invoke explicitly via path: `.\.venv\Scripts\python.exe -m uvicorn backend.main:app --port 8000` |
| **White Screen on Browser** | `http://127.0.0.1:8000` loads a blank page | Build assets were missing or moved: Run `cd frontend; npm run build; cd ..` and refresh browser. |
| **Database Lock Error** | `sqlite3.OperationalError: database is locked` | Ensure no multiple background worker processes are writing simultaneously. Restart Uvicorn server. |
| **Recommendation Fails (503)** | `Active model not found for client demo_ecommerce` | Ensure `data/platform.db` has model `v1` activated. Check via Swagger: `POST /api/clients/demo_ecommerce/models/v1/activate`. |
