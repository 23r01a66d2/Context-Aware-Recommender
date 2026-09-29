# Project Fact Sheet & Verification Reference

This document serves as the single source of truth for all quantitative metrics, architectural specifications, empirical evaluation numbers, and research findings of the **Real-Time Context-Aware Multi-Client Recommendation Platform**.

---

## 1. Reference Dataset & Partitioning

| Parameter | Value | Source / Notes |
| :--- | :--- | :--- |
| **Dataset Name** | Reference E-Commerce Interaction Log (`Ecommerce.csv`) | `data/raw/Ecommerce.csv` |
| **Total Interaction Records** | **25,000** rows | Raw CSV row count |
| **Total Features (Columns)** | **29** original columns | Tabular schema |
| **Unique Customer Entities** | **8,442** unique customers | `customer_id` |
| **Unique Catalog Products** | **899** unique products | `product_id` |
| **Product Categories** | **8** categories | Categories 1 through 8 |
| **Target Label (`purchase`)** | Binary ($1 = \text{Purchased}, 0 = \text{Browse/Abandon}$) | Supervised recommendation target |
| **Class Distribution** | **5,616** purchases (22.46%) / **19,384** non-purchases (77.54%) | Imbalance ratio $\approx 1:3.45$ |
| **Cold-Start Customers** ($N_{\text{hist}} < 3$) | **3,743** customers (**44.34%** of user base) | Severe cold-start condition |
| **Warm-Start Customers** ($N_{\text{hist}} \ge 3$) | **4,699** customers (**55.66%** of user base) | Established history |

### Chronological Splits (Zero Future Leakage Partitioning)
| Split | Date Range | Row Count | Percentage | Positive Purchases | Positive Rate |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Train Set** | 2024-01-01 to 2024-07-31 | **14,494** | 57.98% | **3,258** | 22.48% |
| **Validation Set** | 2024-08-01 to 2024-09-30 | **4,212** | 16.85% | **981** | 23.29% |
| **Test Set** | 2024-10-01 to 2024-12-30 | **6,294** | 25.18% | **1,377** | 21.88% |

---

## 2. Feature Modalities & Dimensions

The confirmed schema for the reference client `demo_ecommerce` defines 49 total input feature dimensions across three distinct modalities:

| Modality | Dimension | Key Engineered Features | Encoding / Transformation |
| :--- | :---: | :--- | :--- |
| **Behavioral Modality** | **16** | $N_{\text{hist}}$ (past interaction count), past purchase velocity, historical cart addition rate, customer lifetime days, preferred historical category affinities. | Point-in-time rolling aggregation ($t < t_i$), fitted exclusively on Train split; MinMaxScaler. |
| **Content Modality** | **12** | Product base price, discount percentage, category one-hot vector (8 classes), seasonal suitability flags. | OneHotEncoder for category, robust scaler for continuous prices. |
| **Context Modality** | **21** | Session hour, day-of-week one-hot, device type (Desktop, Mobile, Tablet), traffic channel, geographical region, local weather context. | Cyclical sine/cosine transformation for time, OneHotEncoder for device & channel. |
| **Total Input Vector** | **49** | Concatenated multi-modal feature vector | $[x_{\text{beh}} (16) \parallel x_{\text{cont}} (12) \parallel x_{\text{ctx}} (21)]$ |

### Leakage Prevention Policy
- **Excluded Columns (Canonical Role: `IGNORE`):** `added_to_cart`, `revenue`, `rating`, `review_text`, `cart_abandoned`.
- **Rationale:** These variables occur concurrently with or subsequent to the recommendation event; including them introduces post-decision target leakage.

---

## 3. Deep Learning Architecture Specifications

| Layer / Component | Specification | Details |
| :--- | :--- | :--- |
| **Behavioral Encoder** | $\text{Linear}(16 \to 64) \to \text{ReLU} \to \text{Dropout}(0.2) \to \text{Linear}(64 \to 32)$ | Produces 32-dim latent behavioral embedding $h_{\text{beh}}$ |
| **Content Encoder** | $\text{Linear}(12 \to 64) \to \text{ReLU} \to \text{Dropout}(0.2) \to \text{Linear}(64 \to 32)$ | Produces 32-dim latent content embedding $h_{\text{cont}}$ |
| **Context Encoder** | $\text{Linear}(21 \to 64) \to \text{ReLU} \to \text{Dropout}(0.2) \to \text{Linear}(64 \to 32)$ | Produces 32-dim latent context embedding $h_{\text{ctx}}$ |
| **Gated Fusion Layer** | Softmax projection across concatenated latent representations | Computes dynamic modality logits $[\hat{\alpha}_{\text{beh}}, \hat{\alpha}_{\text{cont}}, \hat{\alpha}_{\text{ctx}}]$ |
| **Cold-Start Adaptive Gate** | $g_{\text{cold}} = \sigma(\beta \cdot (N_{\text{hist}} - K))$ with $K = 3, \beta = 2.0$ | If $N_{\text{hist}} < 3 \implies \alpha_{\text{beh}} \le 0.002$; if $N_{\text{hist}} \ge 3 \implies \alpha_{\text{beh}} \approx 0.14$ |
| **Fused Representation** | $h_{\text{fused}} = \alpha_{\text{beh}} h_{\text{beh}} + \alpha_{\text{cont}} h_{\text{cont}} + \alpha_{\text{ctx}} h_{\text{ctx}}$ | 32-dim unified multi-modal representation |
| **Ranking Head** | $\text{Linear}(32 \to 16) \to \text{ReLU} \to \text{Linear}(16 \to 1) \to \sigma$ | Outputs scalar `sigmoid_score` $\in (0, 1)$ |
| **Loss Function** | Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`) | Optimized using Adam (`lr = 0.001`, batch size = 128) |
| **Best Validation Loss** | **0.5284** (Epoch 10) | Saved at `models/demo_ecommerce/v1/checkpoint.pt` |

---

## 4. Empirical Evaluation Results

### A. Baseline Ranking Performance (Sampled Protocol: 1 Pos + 99 Negs)
Metrics evaluated across all 1,377 positive purchase events in the chronological test split:

| Model | Overall NDCG@5 | Cold NDCG@5 ($N_{\text{hist}} < 3$) | Warm NDCG@5 ($N_{\text{hist}} \ge 3$) | Overall HR@5 | Cold HR@5 | Warm HR@5 | Overall MRR@5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Popularity** | 0.0320 | 0.0347 | 0.0285 | 0.0617 | 0.0645 | 0.0581 | 0.0218 |
| **Collaborative Filtering** | 0.0308 | 0.0342 | 0.0264 | 0.0588 | 0.0633 | 0.0531 | 0.0212 |
| **Content-Based** | **0.1374** | 0.0312 | **0.2750** | **0.2033** | 0.0543 | **0.3965** | **0.1147** |
| **Multi-Modal (No Adaptation)** | 0.0401 | **0.0415** | 0.0382 | 0.0763 | **0.0792** | 0.0724 | 0.0276 |
| **Proposed Adaptive Model** | 0.0352 | 0.0353 | 0.0351 | 0.0675 | 0.0673 | 0.0678 | 0.0242 |

### B. Architectural Ablation Study (Models A through H)
| Model ID | Active Modalities | Gating Mechanism | Overall NDCG@5 | Cold NDCG@5 | Warm NDCG@5 | Overall HR@5 |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **Model A** | Behavior Only | None | 0.0340 | 0.0348 | 0.0329 | 0.0646 |
| **Model B** | Content Only | None | 0.0325 | 0.0319 | 0.0332 | 0.0625 |
| **Model C** | Context Only | None | 0.0371 | 0.0357 | 0.0389 | 0.0712 |
| **Model D** | Behavior + Content | Unweighted Concatenation | 0.0356 | 0.0360 | 0.0350 | 0.0683 |
| **Model E** | Behavior + Context | Unweighted Concatenation | **0.0407** | **0.0375** | **0.0447** | **0.0770** |
| **Model F** | Content + Context | Unweighted Concatenation | 0.0368 | 0.0362 | 0.0376 | 0.0704 |
| **Model G** | Behavior + Content + Context | Static Softmax Fusion | 0.0401 | 0.0415 | 0.0382 | 0.0763 |
| **Model H** | Behavior + Content + Context | **Adaptive Sigmoid Gate ($K=3$)** | 0.0352 | 0.0353 | 0.0351 | 0.0675 |

### C. Threshold Sensitivity Analysis across Interaction Depth $K$
| Threshold $K$ | Cold Cutoff Rule | Overall NDCG@5 | Cold NDCG@5 | Warm NDCG@5 | Overall HR@5 |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **$K = 1$** | $N_{\text{hist}} < 1$ | 0.0354 | 0.0339 | 0.0356 | 0.0675 |
| **$K = 2$** | $N_{\text{hist}} < 2$ | **0.0357** | 0.0348 | 0.0361 | **0.0690** |
| **$K = 3$** | $N_{\text{hist}} < 3$ (Primary) | 0.0352 | 0.0353 | 0.0351 | 0.0675 |
| **$K = 5$** | $N_{\text{hist}} < 5$ | 0.0346 | 0.0346 | 0.0345 | 0.0668 |

---

## 5. Operational Latency & Production Benchmarks

- **Target Candidate Catalog:** 899 candidate items scored per request.
- **Latency Benchmark Protocol:** 100 simulated requests under production Uvicorn runtime.
- **Cold-Start Cohort Latency:**
  - Mean: **48.74 ms**
  - Median (p50): **45.47 ms**
  - Min: **31.03 ms**
  - Max: **128.91 ms**
- **Warm-Start Cohort Latency:**
  - Mean: **50.29 ms**
  - Median (p50): **48.67 ms**
  - Min: **42.91 ms**
  - Max: **69.48 ms**
- **SLA Compliance:** 100% of median response times meet the $< 100\text{ ms}$ real-time recommendation threshold.

---

## 6. Research Questions (RQ) Conclusions

- **Research Question 1 (Operational Feasibility):**
  > *Can a multi-modal neural architecture dynamically support cold-start and warm-start users in a real-time web application?*
  > **Conclusion: Supported.** The platform reliably serves zero-history and sparse-history users using item content and session context, suppressing behavioral noise when $N_{\text{hist}} < 3$ without operational errors or throughput bottlenecks (mean latency $< 51\text{ ms}$).

- **Research Question 2 (Ranking Accuracy):**
  > *Does the proposed cold-start adaptive gating mechanism outperform comparison baselines in ranking accuracy on the reference dataset?*
  > **Conclusion: Not consistently supported on this dataset under the reported sampled offline ranking protocol.** While the adaptive model exhibits highly consistent and balanced ranking performance across cold ($0.0353$) and warm ($0.0351$) cohorts, Content-Based filtering achieves significantly higher ranking accuracy for warm users ($0.2750$), and unadapted multi-modal fusion achieves a marginally higher overall score ($0.0401$). The primary empirical property of the adaptive gate is therefore **stability and smooth degradation prevention**, rather than universal ranking superiority.

---

## 7. Scope Boundaries & Architectural Constraints

1. **Exact Matrix Scoring:** In this prototype, all 899 catalog candidates are scored simultaneously via PyTorch matrix multiplication. Industrial deployment with $\ge 100,000$ products requires a two-stage pipeline with an Approximate Nearest Neighbor (ANN) index (e.g., FAISS, ScaNN) for sub-linear retrieval.
2. **Relevance Scores vs Calibrated Probabilities:** Model outputs are sigmoid transformed activations representing relative recommendation relevance (`relevance_score` / `sigmoid_score`). They must not be interpreted as calibrated percentage purchase probabilities.
3. **Modality Weights Interpretation:** Modality weights ($\alpha_{\text{beh}}, \alpha_{\text{cont}}, \alpha_{\text{ctx}}$) are learned parameters of the internal gating network; they are not causal feature attributions.
4. **Visual Analytics Separation:** The primary learned machine learning artifact is product recommendation; visual analytics generation in the dashboard is auxiliary and rule-based.
