# Live System Demonstration Script (5–8 Minutes)

This script provides an exact, step-by-step walkthrough for demonstrating the **Real-Time Context-Aware Multi-Client Recommendation Platform** to evaluators, stakeholders, or examiners.

---

## ⏱ Demonstration Overview
- **Target Duration:** 5–8 minutes
- **Environment:** Local Development Prototype
- **Services Running:**
  - FastAPI Backend: `http://127.0.0.1:8000`
  - React Dashboard SPA: `http://127.0.0.1:8000` (or `http://localhost:5173`)
- **Key Concepts Highlighted:**
  - Multi-client tenant isolation
  - Schema-driven ingestion and mapping
  - Point-in-time leakage prevention
  - Multi-modal cold-start adaptive gating ($K = 3$)
  - Real-time catalog filtering and ranking
  - Explicit feedback recording

---

## 🎬 Step-by-Step Walkthrough

### 1. Introduction & The Core Problem (0:00 – 0:45)
- **Speaker:** 
  > *"Welcome. Today we are presenting our Real-Time Multi-Client Context-Aware Recommendation Platform. Standard collaborative filtering recommenders suffer severely from the user cold-start problem when sparse or no interaction history exists. In e-commerce, over 44% of users interact fewer than 3 times. Our system introduces a multi-modal neural architecture with a dynamic cold-start adaptive gate that automatically suppresses unreliable historical behavioral signals when a user's interaction count is below threshold $K=3$, seamlessly falling back to item content attributes and environmental session context."*

### 2. High-Level Architecture (0:45 – 1:30)
- **Action:** Open the **Architecture Diagram** in the documentation or browser.
- **Speaker:**
  > *"The platform is divided into two distinct pipelines:
  > 1. **Offline Lifecycle:** Handles dataset validation, semantic schema mapping, strict point-in-time feature extraction with zero future leakage, background PyTorch model training, offline ranking evaluation, and model artifact registration.
  > 2. **Online Inference Engine:** Serves live candidate recommendations via FastAPI by fetching active client models, evaluating user history depth ($N_{\text{hist}}$), querying the in-memory item catalog, applying client-side filters, and performing two-tower neural scoring in under 55 milliseconds."*

### 3. Multi-Client Isolation & Dataset Preview (1:30 – 2:30)
- **Action:** Navigate to `http://127.0.0.1:8000/admin` (or `/admin/clients`). Click on **`demo_ecommerce`**, then click **Dataset**.
- **Speaker:**
  > *"Here in the Admin Workspace, we see multi-tenant isolation. Each client has an isolated database entry and directory workspace. Looking at the reference client `demo_ecommerce`, we inspect its 25,000 interaction records spanning 8,442 unique customers and 899 catalog products across 8 categories. Notice the chronological split: Train (Jan–Jul 2024), Validation (Aug–Sep 2024), and Test (Oct–Dec 2024). All target purchases and timestamps are validated."*

### 4. Semantic Schema Mapping (2:30 – 3:15)
- **Action:** Click on **Schema Mapping** in the sidebar.
- **Speaker:**
  > *"New clients don't have to conform to hardcoded database schemas. Our schema engine automatically inspects uploaded tabular files and infers column roles across 8 canonical types: `USER_ID`, `ITEM_ID`, `TIMESTAMP`, `TARGET`, `BEHAVIOR_FEATURE`, `CONTENT_FEATURE`, `CONTEXT_FEATURE`, and `IGNORE`.
  > Notice that post-recommendation leakage fields—such as `added_to_cart`, `revenue`, `rating`, `review_text`, and `cart_abandoned`—are explicitly flagged and marked as `IGNORE` to guarantee zero future leakage at inference time."*

### 5. Model Registry & Offline Metrics (3:15 – 4:00)
- **Action:** Click on **Model Registry**, then **Analytics**.
- **Speaker:**
  > *"In the Model Registry, versioned model artifacts are stored with full reproducibility metadata—random seeds, feature vector dimensions (Behavior: 16, Content: 12, Context: 21), and validation metrics. Model `v1` is currently active.
  > Under Analytics, we see the strictly controlled **Sampled Offline Ranking Evaluation** metrics computed against 1 held-out positive + 99 sampled unpurchased negatives. For the primary threshold $K=3$, the model scores across Overall, Cold-Start ($N_{\text{hist}} < 3$), and Warm-Start ($N_{\text{hist}} \ge 3$) cohorts are displayed alongside baseline models.
  > We explicitly note that under this sampled offline ranking protocol, the adaptive model did not outperform all comparison baselines in ranking accuracy (Content-Based achieved higher warm-cohort scores, and unadapted multi-modal had a slightly higher overall score). Rather, the primary observed property of the adaptive gate is stable, balanced ranking performance across cold and warm cohorts (NDCG@5 of 0.0353 cold vs 0.0351 warm)."*

### 6. Live Recommendation: Completely New User (Cold-Start) (4:00 – 4:45)
- **Action:** Click **Live Recommender** in the top navigation bar (`/recommend`).
  - Client ID: `demo_ecommerce`
  - User ID: Enter `new_customer_999`
  - Top-K: `5`
  - Click **Generate Recommendations**.
- **Speaker:**
  > *"Let's generate recommendations for a brand-new customer who has never visited the site ($N_{\text{hist}} = 0$). 
  > Notice the result:
  > - **Status Badge:** Displays a blue `Cold Start (0 interactions)` tag.
  > - **Modality Weights:** The behavioral weight $\alpha_{\text{beh}}$ is completely suppressed ($\le 0.002$). The recommendation is driven almost entirely by item content features ($\alpha_{\text{cont}} \approx 0.65$) and context features ($\alpha_{\text{ctx}} \approx 0.35$).
  > - **Latency:** Generated in ~45 ms for 899 catalog items."*

### 7. Live Recommendation: Warm User & Weight Shift (4:45 – 5:30)
- **Action:** In the User ID field, enter `9272` ($N_{\text{hist}} = 8$). Click **Generate Recommendations**.
- **Speaker:**
  > *"Now let's test a known warm customer, User `9272`, who has 8 historical interactions in the system.
  > Observe the immediate shift:
  > - **Status Badge:** Changes to an emerald `Warm Start (8 interactions)` tag.
  > - **Modality Weights:** The adaptive gate activates the behavioral encoder! $\alpha_{\text{beh}}$ rises to ~0.14, actively integrating the user's historical category preferences, cart addition rates, and past purchase velocity with item content and context."*

### 8. Live Filters & Real-Time Feedback (5:30 – 6:30)
- **Action:**
  - Toggle **Advanced Filters**.
  - Select **Category:** `1` (or desired category).
  - Enter **Max Price:** `75.00`.
  - Click **Generate Recommendations**.
  - Observe candidate filtering: All 5 recommendations match category 1 with prices under \$75.00.
  - On the first recommended card, click the **Purchase** feedback button.
- **Speaker:**
  > *"The live inference engine supports real-time candidate pre-filtering. Here we filtered for Category 1 items under \$75.00. All returned items strictly satisfy these business constraints.
  > Finally, when the user interacts with a recommendation, clicking **Purchase** immediately dispatches a feedback event linked to this exact recommendation ID via `POST /api/feedback`, persisting the signal for future continuous learning."*

### 9. Research Questions & Conclusion (6:30 – 7:15)
- **Speaker:**
  > *"To summarize our research findings:
  > - **Research Question 1 (Operational Feasibility):** Can the system generate recommendations for users with sparse or zero behavioral history? **Yes, operationally.** The platform seamlessly supports zero-history and sparse-history inference via item content and session context without runtime degradation.
  > - **Research Question 2 (Ranking Accuracy):** Does adaptive gating improve ranking accuracy over comparison methods on this dataset? **Not consistently under the reported sampled offline ranking protocol.** While it provides balanced ranking between cold and warm cohorts, it did not outperform all comparison baselines in ranking metrics.
  >
  > Finally, we emphasize key prototype boundaries:
  > 1. Candidate scoring runs exact neural matrix multiplication across the 899-item catalog; web-scale deployment would require Approximate Nearest Neighbor (ANN) index retrieval (e.g., FAISS).
  > 2. Sigmoid outputs represent relative recommendation relevance scores, not calibrated purchase probabilities.
  > 3. Modality weights reflect learned gating parameters rather than causal feature attributions.
  > 4. The primary learned ML component is product recommendation; visual analytics generation is deterministic and rule-based.
  > Thank you. We are happy to take any questions."*
