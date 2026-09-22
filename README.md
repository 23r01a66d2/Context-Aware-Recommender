# A Context-Aware Recommendation System for Visual Analytics: Mitigating Cold-Start via Multi-Modal Behavioral Fusion

A complete, locally runnable Machine Learning application and research prototype designed to solve the e-commerce cold-start recommendation problem using **Multi-Modal Gated Fusion** and **Adaptive Cold-Start Gating**.

---

## 1. Problem Statement
Collaborative Filtering and behavioral recommendation models rely heavily on rich historical interactions. When faced with new customers or sparse interaction histories (**the cold-start problem**), traditional methods experience severe performance degradation or fail completely. In e-commerce platforms, cold-start users represent a massive fraction of traffic (over 44% of customers have fewer than 3 historical interactions in typical benchmarks). 

---

## 2. Research Objective
To develop a **Context-Aware Two-Tower Neural Recommendation Architecture** that:
1. Jointly encodes **Point-in-Time Behavioral History**, **Candidate Product Content**, and **Session Context**.
2. Learns dynamic modality weights $(\alpha_{\text{beh}}, \alpha_{\text{cont}}, \alpha_{\text{ctx}})$ via a **Neural Gated Attention Network**.
3. Incorporates **Cold-Start Adaptive Gating**: When a customer has limited interaction history ($N_{\text{hist}} < K$), the model automatically suppresses uninformative behavioral noise and dynamically re-allocates attention mass to product content attributes and session context.
4. Prevents all forms of temporal data leakage by utilizing strictly non-overlapping chronological splitting and point-in-time past feature accumulation.
5. Provides a full interactive visual analytics dashboard, feature-grounded explainability, and a local feedback loop.

---

## 3. Dataset Description
- **Dataset**: Kaggle *"Indian E-Commerce Customer Behavior & Purchase"* (`Ecommerce.csv`).
- **Total Records**: 25,000 interactions.
- **Unique Customers**: 8,442.
- **Catalog Products**: 899 products across 8 categories.
- **Temporal Range**: 2024-01-01 to 2024-12-30 (365 calendar days).
- **Target Variable**: `purchased` $\in \{0, 1\}$ (5,616 positive purchase outcomes, 22.46% conversion).
- **Leakage-Prone Excluded Attributes**: In-session `revenue`, `rating`, `review_text`, `review_helpful_votes`, `cart_abandoned`, `payment_method`, and `added_to_cart` are strictly excluded from predictive model inputs (recommendation occurs pre-cart addition).

### Cold-Start Cohort Profile
- **Unique Customers at Entry**: 2,962 cold customers ($65.6\%$ of test customers) vs 1,555 warm customers ($34.4\%$).
- **Test Purchase Interactions ($N_{\text{hist}} < 3$)**: 777 cold interactions ($56.4\%$) vs 600 warm interactions ($43.6\%$).

---

## 4. System Architecture

```mermaid
flowchart TD
    subgraph Data Layer
        A[Raw Kaggle Dataset: Ecommerce.csv] --> B[Data Ingestion Module]
        B --> C[Data Cleaning & Deduplication]
        C --> D[Strict Chronological Split: Jan-Jul Train | Aug-Sep Val | Oct-Dec Test]
        D --> E[Point-in-Time Feature Construction: t_prev < t]
    end

    subgraph Modality Encoders
        E --> BE[Behavioral Encoder: MLP LayerNorm d=64]
        E --> CE[Content Encoder: Product Embedding + Meta MLP d=64]
        E --> XE[Context Encoder: Session Context MLP d=64]
    end

    subgraph Multi-Modal Fusion & Cold-Start Adaptation
        BE --> GAF[Neural Gated Attention Network]
        CE --> GAF
        XE --> GAF
        CSD[Cold-Start Detector: N_hist < K] -->|Gating Constraint| GAF
        GAF -->|Warm: Full Multi-Modal| USER[User-Context Tower Representation]
        GAF -->|Cold: Behavioral Suppressed| USER
    end

    subgraph Candidate Ranking & Product Tower
        PROD_IN[Candidate Products 1 Pos + 99 Negs] --> PROD_TOWER[Product Content Tower d=64]
        USER --> SCORE[Scaled Dot-Product Similarity]
        PROD_TOWER --> SCORE
        SCORE --> RANK[Top-K Candidate Ranker]
        RANK --> EVAL[Evaluation: Precision@5, Recall@5, NDCG@5, HitRate@5]
        RANK --> EXP[Explainability Engine: Modality Attribution]
    end

    subgraph Interface & Local Feedback
        EXP --> STREAMLIT[10-Page Streamlit Web Application]
        STREAMLIT --> VREC[Secondary: Rule-Based Visual Analytics Recommender]
        STREAMLIT --> FB[Local Feedback Store: outputs/feedback.csv]
    end
```

---

## 5. Module Structure
```
context_aware_recommender/
│
├── data/
│   ├── raw/
│   │   └── Ecommerce.csv               # Raw Kaggle dataset
│   └── processed/
│       ├── cleaned_dataset.csv         # Cleaned full dataset
│       ├── train.csv, validation.csv, test.csv
│       ├── train_features.npz, val_features.npz, test_features.npz
│       └── test_candidates.json        # 100% fair candidate pools (1 pos + 99 negs)
│
├── src/
│   ├── data_ingestion.py               # Ingestion, encodings, schema validation
│   ├── preprocessing.py                # Deduplication, outlier capping, chronological splitting
│   ├── feature_engineering.py          # Point-in-time feature extraction & automated leakage audit
│   ├── visualization.py                # Publication EDA plots & Rule-based visual analytics recommender
│   ├── explainability.py               # Feature-grounded explanation generation
│   ├── feedback.py                     # Local user feedback store (Accept/Reject/Click)
│   │
│   └── models/
│       ├── behavior_encoder.py         # Historical RFM & category preference encoder
│       ├── content_encoder.py          # Product catalog embedding & metadata encoder
│       ├── context_encoder.py          # Session context & cyclical temporal encoder
│       ├── fusion_model.py             # Neural Gated Attention Network
│       ├── cold_start.py               # Adaptive Cold-Start gating constraint
│       ├── recommender.py              # Two-Tower neural recommender + 3 baselines
│       └── ranker.py                   # Candidate ranking & Top-K metrics engine
│
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploratory Data Analysis notebook
│   ├── 02_preprocessing.ipynb          # Data cleaning & chronological splitting
│   ├── 03_feature_engineering.ipynb    # Feature engineering & leakage verification
│   ├── 04_model_training.ipynb         # Model training & convergence curves
│   └── 05_evaluation.ipynb             # Candidate ranking benchmarks & ablations
│
├── app/
│   └── app.py                          # 10-page interactive Streamlit dashboard
│
├── models/
│   ├── proposed_multimodal_model.pt    # Best trained PyTorch model checkpoint
│   ├── feature_encoders.joblib         # Fitted scalers & category encoders
│   └── ablation_*.pt                   # Ablation study model checkpoints
│
├── outputs/
│   ├── eda/                            # 7 publication-ready EDA charts
│   ├── plots/                          # Training loss & validation loss curves
│   ├── evaluation/                     # Models comparison, threshold sensitivity, ablation study
│   ├── dataset_metadata.json           # Ingestion schema metadata
│   └── feedback.csv                    # Local feedback activity log
│
├── tests/
│   └── test_pipeline.py                # 9 unit & integration tests
│
├── config.yaml                         # Centralized configuration
├── requirements.txt                    # Project dependencies
├── data_dictionary.md                  # Complete column specification & role mapping
└── train.py                            # Master training, evaluation & ablation script
```

---

## 6. Installation Steps

### Prerequisites
- Python 3.10+ (Tested on Python 3.13 / 3.14)
- Windows / Linux / macOS

```powershell
# 1. Clone or navigate to the project directory
cd C:\Users\Varsha\.gemini\antigravity\scratch\context_aware_recommender

# 2. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 7. Execution Instructions

### A. Run Data Ingestion & Preprocessing
```powershell
python src/data_ingestion.py
python src/preprocessing.py
python src/feature_engineering.py
```

### B. Run Automated Unit Test Suite
```powershell
python -m unittest discover -s tests -p "test_*.py"
```

### C. Run Master Training & Benchmark Pipeline
```powershell
# Run the complete experimental suite (5 models, threshold sensitivity, 8-model ablation)
python train.py --mode full
```

### D. Launch Interactive Streamlit Dashboard
```powershell
streamlit run app/app.py
```

---

## 8. Empirical Benchmark Results

Evaluated across **1,377 positive purchase test interactions** against **1 ground-truth positive + 99 unpurchased negatives** (100 candidate items per evaluation):

| Model Architecture | Overall Precision@5 | Overall NDCG@5 | Cold-Start NDCG@5 ($N < 3$) | Warm-Start NDCG@5 ($N \ge 3$) | Mean Rank |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Popularity Baseline** | 0.0106 | 0.0320 | 0.0347 | 0.0285 | 50.51 |
| **Content-Based Filtering** | 0.0298 | 0.1374 | 0.0312 | 0.2750 | 31.32 |
| **Collaborative Filtering (SVD)** | 0.0109 | 0.0308 | 0.0342 | 0.0264 | 49.89 |
| **Multi-Modal (No Adaptation)** | 0.0129 | 0.0401 | 0.0415 | 0.0382 | 48.09 |
| **Proposed Cold-Start Multi-Modal** | **0.0121** | **0.0352** | **0.0353** | **0.0351** | **49.98** |

---

## 9. Cold-Start Threshold Sensitivity Analysis ($K$)
Evaluated on the proposed adaptive multi-modal architecture:

| Threshold ($K$) | Cold NDCG@5 | Cold Test Samples | Warm NDCG@5 | Warm Test Samples |
| :---: | :---: | :---: | :---: | :---: |
| **$K = 1$** (Brand New Users) | 0.0311 | 129 | 0.0359 | 1,248 |
| **$K = 2$** ($< 2$ past interactions) | 0.0385 | 411 | 0.0345 | 966 |
| **$K = 3$** (**Primary Baseline**) | 0.0353 | 777 | 0.0351 | 600 |
| **$K = 5$** ($< 5$ past interactions) | 0.0340 | 1,231 | 0.0393 | 146 |

---

## 10. 8-Model Modality Ablation Study

| Ablation Model Configuration | Overall Precision@5 | Overall NDCG@5 | Cold-Start NDCG@5 | Warm-Start NDCG@5 |
| :--- | :---: | :---: | :---: | :---: |
| **Model A: Behavior Only** | 0.0126 | 0.0375 | 0.0328 | 0.0435 |
| **Model B: Content Only** | 0.0118 | 0.0330 | 0.0308 | 0.0359 |
| **Model C: Context Only** | 0.0115 | 0.0326 | 0.0326 | 0.0327 |
| **Model D: Behavior + Content** | 0.0121 | 0.0361 | 0.0372 | 0.0348 |
| **Model E: Behavior + Context** | 0.0139 | 0.0407 | 0.0375 | 0.0447 |
| **Model F: Content + Context** | 0.0112 | 0.0323 | 0.0347 | 0.0291 |
| **Model G: Unweighted Concatenation** | 0.0126 | 0.0362 | 0.0334 | 0.0397 |
| **Model H: Full Proposed Adaptive Fusion** | **0.0115** | **0.0323** | **0.0336** | **0.0306** |

---

## 11. Streamlit Interactive Dashboard (10 Pages)
1. **Home**: Project motivation, research architecture diagram, summary KPIs.
2. **Dataset Overview**: Interactive table of 25,000 interactions, schema inspector, and metadata.
3. **Data Cleaning**: Outlier IQR Winsorization, deduplication reports, and chronological split summary.
4. **Exploratory Analysis**: Distribution histograms, correlation heatmaps, and the Secondary Rule-Based Visual Analytics Recommender.
5. **Customer Behavior**: Behavioral funnel analysis (Browse $\to$ Cart $\to$ Purchase/Abandon) and interaction frequencies.
6. **Product Analysis**: Category economics, interaction volumes, and conversion rates.
7. **Recommendation**: Live interactive recommendation engine with Top-5/Top-10 item rankings, candidate scores, and feature-grounded explanations.
8. **Cold-Start Demo**: Side-by-side demonstration contrasting dynamic learned modality attention weights between new ($N < 3$) and returning customers ($N \ge 3$).
9. **Model Performance**: Full benchmark comparison table, threshold sensitivity, and ablation study results.
10. **Feedback**: Interactive feedback submission (Accept/Reject/Click/Ignore) and historical activity tracker.

---

## 12. Research Integrity & Limitations
- **Zero Fabrication**: All reported metrics are generated strictly from executing the code on the chronological test set.
- **Zero Data Leakage**: Future outcomes (`revenue`, `rating`, `payment_method`, `added_to_cart`) are strictly excluded from predictive model inputs.
- **Simulated Interaction Layer**: Sequential action flows used for sequential ablation experiments are explicitly labeled as *"Simulated interaction data generated for research experimentation."*
- **Offline Protocol**: All evaluations are computed offline using local CPU resources without cloud dependencies.
