# Final Project Manifest: Multi-Client Context-Aware Recommendation Platform

This document provides a comprehensive inventory of all architectural, operational, experimental, and documentation components comprising the final project submission.

---

## 1. System Inventory

```
context_aware_recommender/
├── backend/                             # Production FastAPI Backend Service
│   ├── api/                             # REST API route handlers
│   │   ├── clients.py                   # /api/clients endpoints
│   │   ├── datasets.py                  # /api/clients/{id}/dataset endpoints
│   │   ├── feedback.py                  # /api/feedback logging endpoints
│   │   ├── models.py                    # /api/clients/{id}/models endpoints
│   │   ├── recommendations.py           # /api/recommend live scoring endpoint
│   │   ├── schemas.py                   # /api/clients/{id}/schema endpoints
│   │   └── training.py                  # /api/clients/{id}/train lifecycle endpoints
│   ├── database/                        # Persistence Layer
│   │   ├── database.py                  # SQLAlchemy engine & session maker (data/platform.db)
│   │   └── models.py                    # SQLAlchemy ORM models (Client, ModelVersion, Job, Feedback)
│   ├── schemas/                         # Pydantic validation models & contracts
│   │   ├── client.py, dataset.py, feedback.py, model.py
│   │   └── recommendation.py, schema_map.py, training.py
│   ├── services/                        # Business logic & domain services
│   │   ├── client_service.py            # Client creation & directory isolation
│   │   ├── dataset_service.py           # Dataset ingestion & profiling
│   │   ├── inference_service.py         # Real-time candidate generation & scoring
│   │   ├── model_cache.py               # In-memory thread-safe model and catalog cache
│   │   ├── model_service.py             # Model registry & activation management
│   │   ├── schema_service.py            # Canonical schema mapping service
│   │   └── training_service.py          # Background training worker thread
│   ├── main.py                          # FastAPI application, route mounting & static SPA serving
│   └── __init__.py
│
├── frontend/                            # Production React Single-Page Application (SPA)
│   ├── dist/                            # Production-compiled static assets (served by FastAPI)
│   │   ├── index.html
│   │   └── assets/                      # Bundled JS and CSS chunks
│   ├── public/                          # Public assets (icons, static resources)
│   ├── src/
│   │   ├── api/                         # Axios REST API client & endpoints mapping
│   │   ├── components/                  # Reusable UI components (Navbar, Sidebar, Badges, Cards)
│   │   ├── context/                     # Global state (ClientContext provider)
│   │   ├── pages/                       # 8 application views
│   │   │   ├── OverviewPage.jsx         # Executive system summary and architecture highlights
│   │   │   ├── ClientsPage.jsx          # Multi-tenant client creation & management
│   │   │   ├── SchemaMappingPage.jsx    # Canonical 8-role schema configuration interface
│   │   │   ├── TrainingPage.jsx         # Background training launch & live epoch telemetry
│   │   │   ├── ModelRegistryPage.jsx    # Model version lifecycle & activation management
│   │   │   ├── AnalyticsPage.jsx        # Offline evaluation metrics, baselines & ablations
│   │   │   ├── RecommendationPage.jsx   # Live interactive recommendation & feedback console
│   │   │   └── SystemHealthPage.jsx     # Service status, DB connectivity & latency health
│   │   ├── tests/                       # Vitest component test suites (10 passing tests)
│   │   ├── App.jsx                      # React Router configuration & layout wrapping
│   │   ├── main.jsx                     # Application bootstrap
│   │   └── index.css                    # Tailwind CSS styling definitions
│   ├── package.json                     # Node.js dependencies & scripts
│   ├── vite.config.js                   # Vite builder & dev server proxy configuration
│   ├── tailwind.config.js               # Tailwind CSS theme configuration
│   └── postcss.config.js
│
├── src/                                 # Core Machine Learning & Data Pipeline Library
│   ├── __init__.py
│   ├── data_ingestion.py                # Dataset loading, validation, encoding detection & profiling
│   ├── preprocessing.py                 # Data cleaning, outlier capping & chronological splitting
│   ├── feature_engineering.py           # Point-in-time behavioral, content & context transformations
│   ├── schema_engine.py                 # Canonical 8-role schema mapping & validation engine
│   ├── models.py                        # PyTorch multi-modal neural architecture & adaptive gate
│   ├── train.py                         # Offline PyTorch training loops & validation checkpoints
│   ├── evaluate.py                      # Sampled offline ranking protocol (NDCG@5, HR@5, MRR@5)
│   └── utils.py                         # Logging setup, seed setting & helper routines
│
├── app/                                 # Exploratory Streamlit Dashboard (Preserved)
│   └── app.py                           # Multi-page interactive Streamlit analysis interface
│
├── data/                                # Dataset & Database Storage
│   ├── raw/
│   │   └── Ecommerce.csv                # Reference e-commerce interaction dataset (25,000 rows)
│   ├── processed/                       # Chronological split CSVs
│   │   ├── cleaned_dataset.csv          # 25,000 cleaned interaction records
│   │   ├── train.csv                    # Train split (14,494 rows, Jan 1 – Jul 31, 2024)
│   │   ├── validation.csv               # Validation split (4,212 rows, Aug 1 – Sep 30, 2024)
│   │   └── test.csv                     # Test split (6,294 rows, Oct 1 – Dec 30, 2024)
│   └── platform.db                      # Active SQLite metadata repository
│
├── clients/                             # Tenant Isolation Storage Directory
│   └── demo_ecommerce/                  # Isolated workspace for reference client
│       ├── config.yaml                  # Confirmed schema mapping & feature configuration
│       ├── dataset.csv                  # Ingested client interaction data
│       └── profiles.parquet             # Pre-aggregated historical behavioral profiles
│
├── models/                              # Serialized Model Checkpoints & Preprocessors
│   └── demo_ecommerce/
│       └── v1/
│           ├── checkpoint.pt            # Frozen PyTorch model weights (Best Val Loss: 0.5284)
│           ├── feature_encoders.joblib  # Fitted scikit-learn encoders & scalers
│           ├── metrics.json             # Training history, validation loss & evaluation metrics
│           └── processed_tensors.pt     # Pre-processed feature tensor caches
│
├── outputs/                             # Research & Evaluation Artifacts
│   ├── final_results/                   # Approved experimental research outputs
│   │   ├── baseline_results.csv         # Comparative ranking metrics across 5 models
│   │   ├── ablation_results.csv         # Ablation study metrics across 8 model configurations
│   │   ├── cold_start_sensitivity.csv   # Threshold sensitivity metrics across K in {1, 2, 3, 5}
│   │   ├── final_metrics.json           # Machine-readable evaluation metrics repository
│   │   ├── experiment_config.json       # Exact experimental parameters, seeds & dimensions
│   │   ├── latency_benchmark.json       # Live candidate scoring latency distribution
│   │   ├── RESULTS_INDEX.md             # Detailed documentation of research outputs & protocol
│   │   └── figures/                     # 6 Publication-ready research figures (300 DPI)
│   │       ├── figure_a_baselines.png
│   │       ├── figure_b_cohort_performance.png
│   │       ├── figure_c_threshold_sensitivity.png
│   │       ├── figure_d_ablation_comparison.png
│   │       ├── figure_e_modality_weights.png
│   │       └── figure_f_inference_latency.png
│   ├── dataset_metadata.json            # Ingested dataset schema and statistical summary
│   ├── pipeline_metrics.json            # Data processing pipeline telemetry
│   └── temporal_summary.json            # Chronological splitting & timestamp audit summary
│
├── tests/                               # Comprehensive Automated Test Suite (65 Unit Tests)
│   ├── __init__.py
│   ├── test_api_endpoints.py            # FastAPI REST endpoint integration & contracts
│   ├── test_data_pipeline.py            # Ingestion, cleaning & outlier handling verification
│   ├── test_feedback.py                 # Closed-loop feedback persistence & validation
│   ├── test_inference_engine.py         # Real-time candidate generation, filtering & ranking
│   ├── test_model_registry.py           # Model versioning, metadata storage & activation
│   ├── test_multi_client_isolation.py   # Multi-tenant directory & database isolation
│   ├── test_multimodal_architecture.py  # PyTorch encoders, gating function & forward pass
│   ├── test_point_in_time_leakage.py    # Zero future leakage verification & chronology tests
│   └── test_schema_engine.py            # Canonical 8-role schema mapping & validation
│
├── docs/                                # Technical Documentation & Demonstration Assets
│   ├── FINAL_PROJECT_MANIFEST.md        # Complete system inventory and submission guide (this file)
│   ├── RUN_GUIDE.md                     # Step-by-step setup and execution instructions
│   ├── DEMO_CHECKLIST.md                # Demonstration checklist and verified test users
│   ├── DEMO_SCRIPT.md                   # 5–8 minute timed demonstration walkthrough
│   ├── PROJECT_REPORT_NOTES.md          # Comprehensive academic report notes and analysis
│   └── PROJECT_FACT_SHEET.md            # Verified numerical facts, tables, and specifications
│
├── README.md                            # Primary project documentation and quickstart
└── requirements.txt                     # Pinned Python package dependencies
```

---

## 2. Submission Recommendations

When packaging this project for final academic submission, archiving, or transfer to an evaluation team, use the following classifications:

### A. Required Source Files (MUST SUBMIT)
These files constitute the primary code, architecture, and configuration of the platform:
- `backend/` — All Python backend source files (under `api/`, `database/`, `schemas/`, `services/`, and `main.py`).
- `src/` — All core ML modules (`data_ingestion.py`, `preprocessing.py`, `feature_engineering.py`, `schema_engine.py`, `models.py`, `train.py`, `evaluate.py`, `utils.py`).
- `frontend/src/` — All React application source code, components, context, pages, and tests.
- `frontend/package.json`, `frontend/vite.config.js`, `frontend/tailwind.config.js`, `frontend/postcss.config.js`, `frontend/index.html`.
- `app/app.py` — Exploratory Streamlit application.
- `tests/` — Complete Python test suite (all 9 test files).
- `requirements.txt` — Python dependencies specification.
- `README.md` — Project overview and quickstart guide.

### B. Required Model & Research Artifacts (MUST SUBMIT)
These files provide the empirical evidence, pre-trained weights, and reproducible evaluation data:
- `models/demo_ecommerce/v1/` — Frozen reference model checkpoint (`checkpoint.pt`), preprocessor transformers (`feature_encoders.joblib`), and metadata.
- `outputs/final_results/` — All 6 research artifact data files (`baseline_results.csv`, `ablation_results.csv`, `cold_start_sensitivity.csv`, `final_metrics.json`, `experiment_config.json`, `latency_benchmark.json`, `RESULTS_INDEX.md`).
- `outputs/final_results/figures/` — All 6 publication charts (`figure_a` through `figure_f`).
- `data/raw/Ecommerce.csv` — Primary reference interaction dataset.
- `clients/demo_ecommerce/` — Reference client configuration and ingested dataset.

### C. Technical Documentation (MUST SUBMIT)
All markdown documents supporting evaluation and reproducibility:
- `docs/FINAL_PROJECT_MANIFEST.md`
- `docs/RUN_GUIDE.md`
- `docs/DEMO_CHECKLIST.md`
- `docs/DEMO_SCRIPT.md`
- `docs/PROJECT_REPORT_NOTES.md`
- `docs/PROJECT_FACT_SHEET.md`

### D. Optional for Submission (Included for Immediate Out-of-the-Box Execution)
- `frontend/dist/` — Pre-built production single-page application. Including this directory allows the evaluator to run `python -m uvicorn backend.main:app` and access the complete React dashboard immediately without needing Node.js or `npm install`.
- `data/platform.db` — Pre-initialized SQLite database containing the registered `demo_ecommerce` client, `v1` model registration, schema mapping, and sample feedback.
- `data/processed/` — Pre-generated chronological split CSVs.

### E. Regeneratable / Excluded Files (DO NOT SUBMIT IN SOURCE ARCHIVE)
To keep the archive clean, portable, and within file-size limits, the following directories should be excluded from the final ZIP or Git repository:
- `frontend/node_modules/` — Regenerated via `npm install`.
- `.venv/` — Virtual environment regenerated via `python -m venv .venv` and `pip install -r requirements.txt`.
- `**/__pycache__/` — Automatically compiled Python bytecode.
- `*.pyc`, `*.pyo`, `.pytest_cache/` — Temporary test execution caches.
- `outputs/test_clients/` — Temporary directories generated by unit tests.

