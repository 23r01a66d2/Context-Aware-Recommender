# Context-Aware Recommender Architecture & Workspace Structure

```
context_aware_recommender/
├── backend/                             # FastAPI Backend Service & Routing
│   ├── api/                             # Endpoint Routers (clients, datasets, inference, models, schemas, feedback)
│   ├── database/                        # SQLAlchemy Models & SQLite DB Session
│   ├── schemas/                         # Pydantic Schemas & DTOs
│   ├── services/                        # Business Logic (inference, client readiness, caching, training)
│   └── main.py                          # FastAPI Entrypoint & Static Image Serving
├── frontend/                            # React 18 + Vite + Tailwind CSS Single-Page App
│   ├── src/
│   │   ├── components/                  # Reusable UI (ModalityWeightBar, DatasetClientsShowcase, Sidebar, Navbar)
│   │   ├── context/                     # Client & App State Management
│   │   ├── pages/                       # User & Admin Views (RecommendationPage, AnalyticsPage, ModelsPage, etc.)
│   │   └── services/api.js              # Centralized Fetch Client
│   ├── package.json
│   └── vite.config.js
├── src/                                 # Research Engine & Model Architecture
│   ├── models/                          # TwoTowerMultiModalRecommender & Sub-Encoders (Behavior, Content, Context, Gating)
│   ├── schema_mapper.py                 # Multi-Domain Schema Detection, Auto-Repair & Capabilities
│   ├── dataset_onboarding.py            # Automated CSV + Image Catalog Onboarding Flow
│   ├── feature_engineering.py           # Temporal Point-in-Time Behavioral Profiles & Leakage Audit
│   ├── preprocessing.py                 # Chronological Split & Cleaning Pipeline
│   └── explainability.py                # Grounded Natural Language Explanations
├── clients/                             # Multi-Tenant Client Workspaces
│   ├── demo_ecommerce/                  # Canonical Indian E-Commerce Benchmark Workspace
│   ├── movielens_dataset/               # MovieLens 100k Benchmark Workspace
│   ├── context_aware_ar_interaction/    # Augmented Reality Cultural Tourism Workspace
│   ├── fashion_client/                  # Multimodal Fashion Catalog Workspace (Visual + Text)
│   └── contract_violation_client/       # Validation / Schema Test Client
├── data/                                # Data Repository
│   ├── raw/                             # Raw Ingestion Files (Ecommerce.csv)
│   ├── processed/                       # Processed Chronological Splits (train.csv, validation.csv, test.csv)
│   └── datasets/                        # Standard Domain Catalog Artifacts
├── models/                              # Versioned Model Checkpoints & Preprocessor Artifacts
│   ├── demo_ecommerce/versions/
│   ├── movielens_dataset/versions/
│   ├── context_aware_ar_interaction/versions/
│   └── fashion_client/versions/
├── outputs/                             # Offline Artifacts & Analytics
│   ├── evaluations/                     # Offline Benchmark Matrices & Metrics Snapshots
│   └── plots/                           # Loss Curves, Ablation Visualizations & Cohort Plots
├── tests/                               # Backend PyUnit & Integration Test Suite
│   ├── test_dataset_onboarding_and_repair.py
│   ├── test_inference.py
│   ├── test_model_registry_and_training.py
│   ├── test_schema_mapper.py
│   ├── test_implicit_feedback.py
│   ├── test_client_deletion.py
│   └── test_backend_api.py
├── scripts/                             # Operational & Benchmarking Automation
│   ├── onboard_fashion.py               # Reusable Fashion Onboarding Runner
│   ├── generate_final_research_artifacts.py
│   └── launcher.py
├── docs/                                # Documentation & Specifications
├── notebooks/                           # Research Jupyter Notebooks
├── config.yaml                          # Default Platform Configuration
├── requirements.txt                     # Python Dependencies
├── README.md                            # Main System Overview
└── train.py                             # Platform Offline Training Pipeline
```

## Client Readiness Classification
1. **Ready for Inference**:
   - `demo_ecommerce` (Canonical E-Commerce, Multi-Modal v1 & v2 active)
   - `movielens_dataset` (MovieLens 100k, Multi-Modal v1 & v2 active)
   - `context_aware_ar_interaction` (AR Tourism, Context-Aware Multi-Modal active)
   - `fashion_client` (Fashion Catalog, Image + Text + Metadata Multimodal v1 active)
2. **Ready for Training**:
   - Generic clients with valid uploaded datasets & mapped schemas.
3. **Needs Schema Mapping**:
   - Unmapped or incomplete raw catalogs (supported with 1-click **Auto-Fix / Map Schema**).
4. **Validation/Test Clients**:
   - `contract_violation_client` and synthetic test fixtures.
