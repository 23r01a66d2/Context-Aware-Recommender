# Project Report Notes: Real-Time Multi-Client Context-Aware Recommendation Platform

---

## 1. Abstract
Recommender systems in e-commerce frequently encounter the *user cold-start problem*, where sparse or nonexistent interaction history impairs recommendation quality. In real-world customer bases, a large proportion of users (over 44% in our reference dataset) have fewer than three lifetime interactions. This project presents a feature-complete, multi-client recommendation platform designed to mitigate cold-start degradation. The system integrates a multi-modal neural architecture with a dynamic **Cold-Start Adaptive Gate** that conditions modality fusion weights on historical interaction depth $N_{\text{hist}}$ relative to a threshold $K=3$. When $N_{\text{hist}} < K$, unreliable behavioral features are automatically suppressed in favor of content attributes and session context. The ML engine is operationalized through a decoupled **FastAPI** backend supporting schema-driven multi-tenant onboarding, point-in-time leakage prevention, versioned model registry, sub-55ms live inference, and a production **React** SPA dashboard with closed-loop feedback logging. Controlled offline evaluation on a chronological split demonstrates the empirical behavior of the gating mechanism across cold and warm user cohorts.

---

## 2. Introduction
Modern online retail environments require personalized item ranking while simultaneously handling continuous streams of first-time visitors and sparse profiles. Conventional Collaborative Filtering (CF) algorithms struggle in these settings due to extreme user-item matrix sparsity. While multi-modal models combining behavioral sequences, item content, and session context provide a promising avenue, standard fusion mechanisms often suffer from noise propagation when behavioral features derived from sparse histories are treated as reliable indicators of preference. This project implements an end-to-end multi-tenant platform that formalizes cold-start adaptation into an automated engineering and serving architecture.

---

## 3. Problem Statement
Given an e-commerce platform with $U$ users, $I$ items, and continuous interaction streams:
1. **The User Cold-Start Problem:** When a customer interacts for the first time ($N_{\text{hist}} = 0$) or has very few interactions ($N_{\text{hist}} < 3$), collaborative and historical behavioral signals are uninformative or misleading.
2. **Data Leakage Risk:** Naive offline pipelines often introduce future leakage by calculating customer aggregate profiles using data observed *after* the recommendation event, or by including post-decision attributes (e.g., `added_to_cart`, `revenue`, `rating`).
3. **Multi-Client Disconnect:** Many ML recommendation codebases are tightly coupled to a single hardcoded CSV schema, preventing dynamic onboarding of new enterprise tenants with differing column names and feature dimensions.
4. **Online vs. Offline Divergence:** Offline protocols that evaluate against known positive items cannot be deployed directly to real-time serving without a robust candidate generation and filtering layer.

---

## 4. Objectives
- **Zero Future Leakage:** Guarantee strict point-in-time feature calculation where features for timestamp $t$ depend exclusively on observations prior to $t$.
- **Cold-Start Mitigation:** Implement a learned adaptive gating network that suppresses behavioral features for $N_{\text{hist}} < K$ without manual heuristic switching.
- **Multi-Tenant Schema Mapping:** Enable automated client onboarding with dynamic role inference across 8 canonical roles and client-isolated model registries.
- **Sub-100ms Live Inference:** Provide candidate pre-filtering, two-tower neural scoring, Top-K ranking, and latency guarantees on standard hardware.
- **Empirical Rigor:** Evaluate all baseline and ablation models under identical, reproducible chronological test protocols with fixed negative candidate pools.

---

## 5. Literature & Conceptual Background
*(Note: Literature citations are marked with standard bibliographic placeholders)*
- **Collaborative Filtering & Sparsity:** Matrix factorization methods (e.g., Sarwar et al., 2001; Koren et al., 2009 `[CITATION-KOREN-2009]`) effectively capture collaborative signals in dense interaction regimes but collapse under extreme sparsity ($N_{\text{hist}} \to 0$).
- **Multi-Modal Recommendation:** Integrating content metadata (descriptions, categories, pricing) and session context (device, time, location) supplements sparse collaborative data (e.g., Zhang et al., 2017 `[CITATION-ZHANG-2017]`).
- **Gated Attention & Mixture-of-Experts:** Gating mechanisms (e.g., Ma et al., 2018 Multi-gate Mixture-of-Experts `[CITATION-MA-2018]`) allow networks to dynamically allocate representational capacity across competing input modalities.
- **Leakage in Temporal Evaluation:** Ji et al. (2020 `[CITATION-JI-2020]`) and Sun et al. (2020 `[CITATION-SUN-2020]`) emphasize that non-chronological splitting or post-target feature contamination leads to severely inflated offline ranking metrics that fail in production.

---

## 6. Dataset Specification

The reference implementation is evaluated on `Ecommerce.csv`:
- **Total Records:** 25,000 interactions
- **Original Columns:** 29 columns
- **Memory Footprint:** 5.94 MB
- **Unique Customers:** 8,442
- **Unique Products:** 899 catalog products
- **Product Categories:** 8 discrete categories
- **Overall Purchase Target Distribution:**
  - Non-purchased (0): 19,384 (77.54%)
  - Purchased (1): 5,616 (22.46%)
- **Customer Interaction Depth Distribution:**
  - Mean interactions per customer: 2.96 ($\sigma = 1.55$)
  - Quartiles: 25% = 2, Median = 3, 75% = 4, Max = 11
  - Customer-level Cold Users ($N_{\text{lifetime}} < 3$): 3,743 customers (**44.34%**)
  - Customer-level Warm Users ($N_{\text{lifetime}} \ge 3$): 4,699 customers (**55.66%**)

### Chronological Dataset Splits
To reflect real-world deployment, splitting is performed strictly chronologically:
- **Train Set:** 14,494 rows (2024-01-01 to 2024-07-31) | 3,258 purchases (22.48%)
- **Validation Set:** 4,212 rows (2024-08-01 to 2024-09-30) | 981 purchases (23.29%)
- **Test Set:** 6,294 rows (2024-10-01 to 2024-12-30) | 1,377 purchases (21.88%)

---

## 7. Data Preprocessing & Leakage-Safety Protocol

### Excluded Post-Outcome Fields
The following fields are strictly excluded from recommendation inputs:
- `added_to_cart`: Direct downstream conversion action
- `revenue`, `revenue_normalized`: Financial realization post-purchase
- `rating`: Customer satisfaction rating entered post-delivery
- `review_text`, `review_helpful_votes`: Text feedback generated post-consumption
- `cart_abandoned`: Downstream session termination state
- `payment_method`: Selected at checkout completion

### Current-Session Exclusions
Current-session duration (`time_on_site_sec`) and pages viewed (`pages_viewed`) are excluded as predictors of current-session recommendations; only prior-session historical aggregates are permitted.

### Point-in-Time Behavioral Computation
For any interaction occurring at timestamp $t_i$ for customer $c$:
$$\text{History}(c, t_i) = \{ (x_j, y_j, t_j) \in \mathcal{D} \mid \text{customer}(x_j) = c \land t_j < t_i \}$$
All historical counts, purchase velocities, and category affinities are computed strictly over $\text{History}(c, t_i)$. An automated unit test (`tests/test_pipeline.py`) audits that no future row indices are accessed.

---

## 8. Final Feature Architecture (Reference Client)

| Modality | Dimension | Features Included |
| :--- | :---: | :--- |
| **Behavioral** | **16** | `hist_interaction_count`, `hist_purchase_count`, `hist_purchase_rate`, `hist_revenue`, `hist_cart_add_rate`, `hist_avg_pages_viewed`, `hist_avg_time_on_site`, `recency_score`, `hist_cat_pref_0` through `hist_cat_pref_7` (8 categories) |
| **Content** | **12** | Product ID embedding (dim 8), one-hot category embedding (dim 8/projected), `unit_price` (standardized), `discount_percent`, `discount_amount` |
| **Context** | **21** | Device type (3 one-hot), user type (2 one-hot), marketing channel (6 one-hot), season (4 one-hot), cyclical day ($\sin/\cos$), cyclical month ($\sin/\cos$), cyclical weekday ($\sin/\cos$), location ID |

---

## 9. Model Architecture & Cold-Start Adaptive Gating

```
User Behavioral Features [16]   --> [Behavior Encoder (MLP 64->32)]  --> h_beh [32]
Item Content Features    [12]   --> [Content Encoder (MLP 64->32)]   --> h_cont [32]
Session Context Features [21]   --> [Context Encoder (MLP 64->32)]   --> h_ctx [32]
                                               |
                                     [Gated Multi-Modal Fusion]
                                               |
                                    alpha_beh, alpha_cont, alpha_ctx
                                               |
                                    [ColdStartAdaptiveGate]
                                  (Suppresses alpha_beh if N_hist < 3)
                                               |
                                      h_fused = sum(alpha_m * h_m)
                                               |
                                    [Candidate Scoring Tower]
                                               |
                                          Sigmoid Score
```

### Fusion & Adaptive Gate Formulation
1. **Raw Modality Projections:**
   $$\mathbf{h}_m = \text{LayerNorm}(\text{ReLU}(\mathbf{W}_m \mathbf{x}_m + \mathbf{b}_m)), \quad m \in \{\text{beh}, \text{cont}, \text{ctx}\}$$
2. **Context-Dependent Gating:**
   $$\mathbf{s} = \text{Softmax}(\mathbf{W}_g [\mathbf{h}_{\text{beh}} \,\|\, \mathbf{h}_{\text{cont}} \,\|\, \mathbf{h}_{\text{ctx}}] + \mathbf{b}_g) \in \mathbb{R}^3$$
3. **Adaptive Cold-Start Gate:**
   Given historical interaction count $N_{\text{hist}}$ and threshold $K=3$:
   $$\text{mask}_{\text{cold}} = \mathbb{I}(N_{\text{hist}} < K)$$
   $$\alpha_{\text{beh}} = s_{\text{beh}} \cdot (1 - \text{mask}_{\text{cold}}) + \epsilon \cdot \text{mask}_{\text{cold}} \quad (\epsilon \le 0.002)$$
   $$\alpha_{\text{cont}}, \alpha_{\text{ctx}} = \text{Normalize}(s_{\text{cont}}, s_{\text{ctx}} \mid \alpha_{\text{beh}})$$
4. **Scoring:** The fused representation $\mathbf{h}_{\text{fused}} = \sum_m \alpha_m \mathbf{h}_m$ is projected via a dense scoring head to produce a scalar relevance score $\hat{y} \in (0, 1)$.

---

## 10. Training Protocol
- **Objective Function:** Binary Cross-Entropy with Logits Loss ($\mathcal{L}_{\text{BCE}}$)
- **Optimizer:** AdamW ($\text{lr} = 10^{-3}$, weight decay $= 10^{-4}$)
- **Batch Size:** 64
- **Epochs:** 15 epochs with early stopping on validation loss (patience = 3)
- **Execution:** Managed asynchronously via FastAPI `BackgroundTasks` with live SQLite state transitions (`NOT_STARTED` $\to$ `PREPROCESSING` $\to$ `TRAINING` $\to$ `COMPLETED`).

## 11. Sampled Offline Ranking Evaluation Protocol

To evaluate ranking performance without full catalog exposure during offline benchmarking:
- **Protocol Type:** **Sampled Offline Ranking Evaluation**.
- **Evaluation Set:** All positive test events in the chronological test period ($N = 1,377$).
- **Candidate Pool:** Exactly **1 held-out ground truth positive item** + **99 sampled unpurchased negative items** per event.
- **Negative Sampling Rule:** Negatives are sampled strictly from products the customer **never purchased** across the entire dataset (past, present, or future).
- **Scope Note:** This sampled protocol evaluates ranking against 100 candidate items per event and does NOT rank against the entire 899-item catalog. (Live inference remains distinct, scoring eligible items from the real client catalog).
- **Ranking:** Stable descending sort by candidate relevance score.
- **Metrics Computed:**
  - $\text{HitRate}@K = \mathbb{I}(\text{rank}_{\text{pos}} \le K)$
  - $\text{Precision}@K = \text{HitRate}@K / K$
  - $\text{Recall}@K = \text{HitRate}@K / 1.0$
  - $\text{NDCG}@K = \frac{1}{\log_2(\text{rank}_{\text{pos}} + 1)}$ if $\text{rank}_{\text{pos}} \le K$ else $0$
  - $\text{MeanRank} = \frac{1}{N} \sum_{i=1}^N \text{rank}_{\text{pos}, i}$

---

## 12. Final Empirical Results

### A. Baseline Comparison (Chronological Test Set, Sampled Ranking Protocol)

| Model | Cohort | Sample Count | Precision@5 | Recall@5 | NDCG@5 | HitRate@5 | NDCG@10 | MeanRank |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Popularity** | Overall | 1,377 | 0.0106 | 0.0530 | 0.0320 | 0.0530 | 0.0458 | 50.51 |
| | Cold-Start ($N < 3$) | 777 | 0.0111 | 0.0553 | 0.0347 | 0.0553 | 0.0484 | 50.50 |
| | Warm-Start ($N \ge 3$) | 600 | 0.0100 | 0.0500 | 0.0285 | 0.0500 | 0.0424 | 50.53 |
| **Content-Based** | Overall | 1,377 | 0.0298 | 0.1489 | 0.1374 | 0.1489 | 0.1518 | 31.32 |
| | Cold-Start ($N < 3$) | 777 | 0.0103 | 0.0515 | 0.0312 | 0.0515 | 0.0499 | 36.01 |
| | Warm-Start ($N \ge 3$) | 600 | **0.0550** | **0.2750** | **0.2750** | **0.2750** | **0.2836** | **25.25** |
| **Collaborative Filtering** | Overall | 1,377 | 0.0109 | 0.0545 | 0.0308 | 0.0545 | 0.0464 | 49.89 |
| | Cold-Start ($N < 3$) | 777 | 0.0118 | 0.0592 | 0.0342 | 0.0592 | 0.0505 | 49.86 |
| | Warm-Start ($N \ge 3$) | 600 | 0.0097 | 0.0483 | 0.0264 | 0.0483 | 0.0411 | 49.94 |
| **Multi-Modal (No Adaptation)** | Overall | 1,377 | 0.0129 | 0.0646 | 0.0401 | 0.0646 | 0.0568 | 48.09 |
| | Cold-Start ($N < 3$) | 777 | 0.0136 | 0.0682 | 0.0415 | 0.0682 | 0.0580 | 48.18 |
| | Warm-Start ($N \ge 3$) | 600 | 0.0120 | 0.0600 | 0.0382 | 0.0600 | 0.0553 | 47.98 |
| **Proposed Adaptive Model** | Overall | 1,377 | 0.0121 | 0.0603 | 0.0352 | 0.0603 | 0.0493 | 49.98 |
| | Cold-Start ($N < 3$) | 777 | 0.0124 | 0.0618 | 0.0353 | 0.0618 | 0.0510 | 49.59 |
| | Warm-Start ($N \ge 3$) | 600 | 0.0117 | 0.0583 | 0.0351 | 0.0583 | 0.0470 | 50.49 |

### Empirical Interpretation of Baseline Results
The empirical evidence does **not** establish superior ranking accuracy for the Proposed Adaptive Model:
- **Overall NDCG@5:** Content-Based (0.1374) > Multi-Modal No Adaptation (0.0401) > Proposed Adaptive (0.0352).
- **Cold-Start NDCG@5:** Multi-Modal No Adaptation (0.0415) > Proposed Adaptive (0.0353) > Content-Based (0.0312).
- **Content-Based Result Discussion:** Content-Based ranking is particularly strong for warm users in this dataset/evaluation setup (Warm NDCG@5 = 0.2750), while its cold-start performance declines substantially (Cold NDCG@5 = 0.0312). Repetitive category affinity is a possible interpretation of this pattern, though it has not been separately tested as an established causal factor.
- **Proposed Model Conclusion:** The adaptive fusion mechanism successfully enables recommendation for users with sparse or zero behavioral history by suppressing the behavioral modality when history is insufficient. Under the sampled ranking evaluation used in this study, however, adaptive gating did not outperform all comparison methods in ranking accuracy. Its most notable observed characteristic was stable ranking performance across cold- and warm-start cohorts (NDCG@5 = 0.0353 for cold vs. 0.0351 for warm).

---

## 13. Ablation Study

| Ablation Model Configuration | Overall NDCG@5 | Cold-Start NDCG@5 | Warm-Start NDCG@5 | Overall Precision@5 |
| :--- | :---: | :---: | :---: | :---: |
| **Model A:** Behavior Only | 0.0375 | 0.0328 | 0.0435 | 0.0126 |
| **Model B:** Content Only | 0.0330 | 0.0308 | 0.0359 | 0.0118 |
| **Model C:** Context Only | 0.0326 | 0.0326 | 0.0327 | 0.0115 |
| **Model D:** Behavior + Content | 0.0361 | 0.0372 | 0.0348 | 0.0121 |
| **Model E:** Behavior + Context | 0.0407 | 0.0375 | 0.0447 | 0.0139 |
| **Model F:** Content + Context | 0.0323 | 0.0347 | 0.0291 | 0.0112 |
| **Model G:** All Three (Unweighted Uniform) | 0.0362 | 0.0334 | 0.0397 | 0.0126 |
| **Model H:** Full + Adaptive Cold-Start Gate | 0.0352 | 0.0353 | 0.0351 | 0.0121 |

### Empirical Interpretation of Ablation Results
- Model E (Behavior + Context) achieves the highest NDCG@5 in the ablation set: Overall NDCG@5 = 0.0407, Cold NDCG@5 = 0.0375.
- Model H (Full Adaptive Model) achieves Overall NDCG@5 = 0.0352 and Cold NDCG@5 = 0.0353.
- Therefore, the ablation study does not demonstrate that adding every modality plus adaptive gating maximizes ranking accuracy. This is documented as a legitimate experimental finding.

---

## 14. Cold-Start Sensitivity Analysis

| Threshold $K$ | Cold Test Count | Warm Test Count | Cold NDCG@5 | Warm NDCG@5 | Overall NDCG@5 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$K = 1$** | 129 (9.37%) | 1,248 (90.63%) | 0.0311 | 0.0359 | 0.0354 |
| **$K = 2$** | 411 (29.85%) | 966 (70.15%) | 0.0385 | 0.0345 | 0.0357 |
| **$K = 3$** (Primary) | 777 (56.43%) | 600 (43.57%) | 0.0353 | 0.0351 | 0.0352 |
| **$K = 5$** | 1,231 (89.40%) | 146 (10.60%) | 0.0340 | 0.0393 | 0.0346 |

### Exact Threshold Sensitivity Procedure
In this experiment, predictions and model gating were rerun separately for each threshold $K \in \{1, 2, 3, 5\}$ by calling `proposed_model.cold_start_gate.set_threshold(K)`. This dynamically modified the boolean cold mask inside the PyTorch forward pass, modifying the neural candidate scoring and rankings for affected users across the test set. This explains why overall NDCG@5 varied across $K$ (0.0354 at $K=1$, 0.0357 at $K=2$, 0.0352 at $K=3$, and 0.0346 at $K=5$).

*Cohort Denominator Notice:* The percentages above represent the distribution over the **1,377 positive test events**. In contrast, across the **8,442 unique lifetime customers**, 3,743 (44.34%) have $< 3$ total interactions and 4,699 (55.66%) have $\ge 3$ interactions.

---

## 15. Research Question Conclusions

- **RQ1: Can the system generate recommendations for users with sparse or zero behavioral history?**
  - **Answer from evidence:** **Yes, operationally.** The implemented cold-start pathway successfully supports zero-history and sparse-history inference using content and context signals without runtime failures or cold-start crashes.
- **RQ2: Does the adaptive cold-start gate improve ranking accuracy over the comparison methods on this dataset?**
  - **Answer from evidence:** **Not consistently under the reported sampled offline ranking protocol.** While adaptive gating provides balanced ranking across cold and warm cohorts, it did not outperform all comparison baselines (such as Content-Based or unadapted Multi-Modal) in offline ranking metrics.

---

## 16. Real-Time Inference Performance

Latency benchmarked on local development environment (Windows, Python 3.13, 899 catalog products, in-memory model cache, 50 repetitions):

| Cohort | Catalog Size | Repetitions | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Std (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cold-Start ($N_{\text{hist}} = 0$)** | 899 | 50 | 48.74 | 45.47 | 31.03 | 128.91 | 16.48 |
| **Warm-Start ($N_{\text{hist}} = 8$)** | 899 | 50 | 50.29 | 48.67 | 42.91 | 69.48 | 5.34 |

---

## 17. Multi-Client Dynamic Dimension Validation
The platform was validated with an alternate synthetic client (`retail_client_b`) using non-reference feature dimensions:
- **Reference Client (`demo_ecommerce`):** Behavior = 16, Content = 12, Context = 21
- **Alternate Client (`retail_client_b`):** Behavior = 9, Content = 5, Context = 6
The schema transformation and PyTorch encoder instantiation dynamically adapted to the client configuration without manual codebase modifications.

---

## 18. Scope of Visual Analytics Component
- **Primary Learned ML Component:** Multi-modal context-aware product recommendation scoring and ranking.
- **Secondary Visual Analytics Component:** Deterministic, rule-based selection and rendering of analytics visualizations (Recharts in React, Altair in Streamlit) driven by dataset properties and query contexts.
- **Boundary Clarification:** The e-commerce dataset does *not* contain learned visualization preference labels; visualization choice is purely heuristic.

---

## 19. Limitations
1. **Local Prototype Storage:** Relies on SQLite (`platform.db`) and local filesystem workspaces rather than cloud object storage (S3/GCS).
2. **Background Task Queue:** Asynchronous training uses FastAPI `BackgroundTasks`, which is memory-bound and lacks durable task retries found in Celery/Redis.
3. **Exact Neural Scoring:** Candidate ranking evaluates the entire catalog via dense PyTorch matrix operations; web-scale catalogs ($>10^5$ items) require two-stage retrieval with ANN vector indices (e.g., FAISS, ScaNN).
4. **Uncalibrated Sigmoid Scores:** Model outputs represent monotonic relevance ranking scores rather than calibrated true purchase probabilities.
5. **Non-Causal Modality Weights:** The learned attention weights indicate representational blending coefficients, not causal feature importances.
6. **No Production Authentication:** Built for academic and local prototyping without JWT/OAuth role-based access control.

---

## 20. Future Work
- Integration of an Approximate Nearest Neighbor (ANN) vector retrieval stage for million-item catalogs.
- Implementation of isotonic regression or Platt scaling for true purchase probability calibration.
- Multi-armed bandit exploration strategies (e.g., Thompson Sampling) to actively acquire interactions for cold users.
- Migration to Redis-backed Celery worker pools for distributed model training and artifact storage on cloud object stores.

---

## 21. Conclusion
This project developed and validated a full-stack, multi-client recommendation platform with dynamic cold-start adaptation. Operationally, the system successfully handles both zero-interaction and sparse-history users by dynamically suppressing the behavioral modality and scoring items via item content and session context. Under the sampled offline ranking evaluation protocol, the adaptive gating mechanism did not outperform all comparison baselines in ranking accuracy, but demonstrated stable, consistent performance across cold- and warm-start user cohorts. The complete platform provides a leakage-safe foundation for multi-tenant recommendation serving.
