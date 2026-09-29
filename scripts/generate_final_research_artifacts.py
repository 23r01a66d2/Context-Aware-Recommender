"""
Phase 7 Research Artifacts Generation Script.
Processes verified offline evaluation results, runs local prototype latency measurements,
exports machine-readable tables to outputs/final_results/, and generates research charts
to outputs/final_results/figures/.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

OUTPUTS_DIR = ROOT_DIR / "outputs" / "final_results"
FIGURES_DIR = OUTPUTS_DIR / "figures"
EVAL_DIR = ROOT_DIR / "outputs" / "evaluation"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# 1. PROCESS BASELINE RESULTS
# -------------------------------------------------------------
with open(EVAL_DIR / "models_comparison.json", "r", encoding="utf-8") as f:
    models_comparison = json.load(f)

baseline_rows = []
for model_name, cohorts in models_comparison.items():
    for cohort_name, metrics in cohorts.items():
        row = {
            "Model": model_name,
            "Cohort": cohort_name,
            "Sample_Count": metrics.get("sample_count", 0),
            "Precision@5": metrics.get("Precision@5", 0.0),
            "Recall@5": metrics.get("Recall@5", 0.0),
            "NDCG@5": metrics.get("NDCG@5", 0.0),
            "HitRate@5": metrics.get("HitRate@5", 0.0),
            "Precision@10": metrics.get("Precision@10", 0.0),
            "Recall@10": metrics.get("Recall@10", 0.0),
            "NDCG@10": metrics.get("NDCG@10", 0.0),
            "HitRate@10": metrics.get("HitRate@10", 0.0),
            "MeanRank": metrics.get("MeanRank", 0.0)
        }
        baseline_rows.append(row)

baseline_df = pd.DataFrame(baseline_rows)
baseline_df.to_csv(OUTPUTS_DIR / "baseline_results.csv", index=False)
print(f"[OK] Saved {OUTPUTS_DIR / 'baseline_results.csv'}")

# -------------------------------------------------------------
# 2. PROCESS ABLATION RESULTS
# -------------------------------------------------------------
with open(EVAL_DIR / "ablation_study.json", "r", encoding="utf-8") as f:
    ablation_study = json.load(f)

ablation_rows = []
for model_name, cohorts in ablation_study.items():
    for cohort_name, metrics in cohorts.items():
        row = {
            "Ablation_Model": model_name,
            "Cohort": cohort_name,
            "Sample_Count": metrics.get("sample_count", 0),
            "Precision@5": metrics.get("Precision@5", 0.0),
            "Recall@5": metrics.get("Recall@5", 0.0),
            "NDCG@5": metrics.get("NDCG@5", 0.0),
            "HitRate@5": metrics.get("HitRate@5", 0.0),
            "Precision@10": metrics.get("Precision@10", 0.0),
            "Recall@10": metrics.get("Recall@10", 0.0),
            "NDCG@10": metrics.get("NDCG@10", 0.0),
            "HitRate@10": metrics.get("HitRate@10", 0.0),
            "MeanRank": metrics.get("MeanRank", 0.0)
        }
        ablation_rows.append(row)

ablation_df = pd.DataFrame(ablation_rows)
ablation_df.to_csv(OUTPUTS_DIR / "ablation_results.csv", index=False)
print(f"[OK] Saved {OUTPUTS_DIR / 'ablation_results.csv'}")

# -------------------------------------------------------------
# 3. PROCESS SENSITIVITY RESULTS
# -------------------------------------------------------------
with open(EVAL_DIR / "threshold_sensitivity.json", "r", encoding="utf-8") as f:
    sensitivity_study = json.load(f)

sensitivity_rows = []
for k_threshold, cohorts in sensitivity_study.items():
    for cohort_name, metrics in cohorts.items():
        row = {
            "Threshold_K": k_threshold,
            "Cohort": cohort_name,
            "Sample_Count": metrics.get("sample_count", 0),
            "Precision@5": metrics.get("Precision@5", 0.0),
            "Recall@5": metrics.get("Recall@5", 0.0),
            "NDCG@5": metrics.get("NDCG@5", 0.0),
            "HitRate@5": metrics.get("HitRate@5", 0.0),
            "Precision@10": metrics.get("Precision@10", 0.0),
            "Recall@10": metrics.get("Recall@10", 0.0),
            "NDCG@10": metrics.get("NDCG@10", 0.0),
            "HitRate@10": metrics.get("HitRate@10", 0.0),
            "MeanRank": metrics.get("MeanRank", 0.0)
        }
        sensitivity_rows.append(row)

sensitivity_df = pd.DataFrame(sensitivity_rows)
sensitivity_df.to_csv(OUTPUTS_DIR / "cold_start_sensitivity.csv", index=False)
print(f"[OK] Saved {OUTPUTS_DIR / 'cold_start_sensitivity.csv'}")

# -------------------------------------------------------------
# 4. EXPORT FINAL METRICS & EXPERIMENT CONFIG JSON
# -------------------------------------------------------------
final_metrics = {
    "evaluation_type": "OFFLINE RANKING EVALUATION",
    "protocol": "1 ground-truth positive + 99 unpurchased negatives per test positive interaction",
    "total_positive_test_samples": 1377,
    "cold_start_threshold": 3,
    "cold_start_test_samples": 777,
    "warm_start_test_samples": 600,
    "proposed_model_metrics": models_comparison.get("Proposed Cold-Start Multi-Modal", {}),
    "popularity_baseline_metrics": models_comparison.get("Popularity", {}),
    "content_based_baseline_metrics": models_comparison.get("Content-Based", {}),
    "collaborative_filtering_baseline_metrics": models_comparison.get("Collaborative Filtering", {}),
    "unadapted_multimodal_metrics": models_comparison.get("Multi-Modal (No Adaptation)", {})
}

with open(OUTPUTS_DIR / "final_metrics.json", "w", encoding="utf-8") as f:
    json.dump(final_metrics, f, indent=2)
print(f"[OK] Saved {OUTPUTS_DIR / 'final_metrics.json'}")

experiment_config = {
    "random_seed": 42,
    "model_version": "v1",
    "client_id": "demo_ecommerce",
    "dataset": {
        "raw_file": "data/raw/Ecommerce.csv",
        "total_rows": 25000,
        "total_columns": 29,
        "unique_customers": 8442,
        "unique_products": 899,
        "product_categories": 8,
        "splits": {
            "train": {"rows": 14494, "date_range": "2024-01-01 to 2024-07-31", "purchases": 3258},
            "validation": {"rows": 4212, "date_range": "2024-08-01 to 2024-09-30", "purchases": 981},
            "test": {"rows": 6294, "date_range": "2024-10-01 to 2024-12-30", "purchases": 1377}
        }
    },
    "feature_dimensions": {
        "behavior": 16,
        "content": 12,
        "context": 21
    },
    "candidate_ranking_protocol": {
        "num_negatives": 99,
        "negative_sampling_policy": "Strictly from products never purchased by customer across entire dataset",
        "sorting": "Stable sort descending by candidate score",
        "metrics": ["Precision@5", "Recall@5", "NDCG@5", "HitRate@5", "Precision@10", "Recall@10", "NDCG@10", "HitRate@10", "MeanRank"]
    },
    "cold_start_thresholds_evaluated": [1, 2, 3, 5],
    "primary_threshold": 3,
    "timestamp": "2026-09-28T20:55:00Z"
}

with open(OUTPUTS_DIR / "experiment_config.json", "w", encoding="utf-8") as f:
    json.dump(experiment_config, f, indent=2)
print(f"[OK] Saved {OUTPUTS_DIR / 'experiment_config.json'}")

# -------------------------------------------------------------
# 5. MEASURE LOCAL PROTOTYPE LATENCY
# -------------------------------------------------------------
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

# Warmup call
client.post("/api/recommend", json={"client_id": "demo_ecommerce", "user_id": "9272", "top_k": 5})

repetitions = 50
cold_latencies = []
warm_latencies = []

# Cold User (N_hist = 0)
for _ in range(repetitions):
    t0 = time.perf_counter()
    res = client.post("/api/recommend", json={
        "client_id": "demo_ecommerce",
        "user_id": "latency_bench_cold_user",
        "top_k": 5
    })
    t1 = time.perf_counter()
    assert res.status_code == 200
    cold_latencies.append((t1 - t0) * 1000.0) # ms

# Warm User (User 9272, N_hist = 8)
for _ in range(repetitions):
    t0 = time.perf_counter()
    res = client.post("/api/recommend", json={
        "client_id": "demo_ecommerce",
        "user_id": "9272",
        "top_k": 5
    })
    t1 = time.perf_counter()
    assert res.status_code == 200
    warm_latencies.append((t1 - t0) * 1000.0) # ms

cold_stats = {
    "cohort": "Cold Start (N_hist = 0)",
    "catalog_size": 899,
    "cache_state": "Warm Model & Catalog Cache",
    "repetitions": repetitions,
    "mean_ms": round(float(np.mean(cold_latencies)), 2),
    "median_ms": round(float(np.median(cold_latencies)), 2),
    "min_ms": round(float(np.min(cold_latencies)), 2),
    "max_ms": round(float(np.max(cold_latencies)), 2),
    "std_ms": round(float(np.std(cold_latencies)), 2)
}

warm_stats = {
    "cohort": "Warm Start (N_hist = 8)",
    "catalog_size": 899,
    "cache_state": "Warm Model & Catalog Cache",
    "repetitions": repetitions,
    "mean_ms": round(float(np.mean(warm_latencies)), 2),
    "median_ms": round(float(np.median(warm_latencies)), 2),
    "min_ms": round(float(np.min(warm_latencies)), 2),
    "max_ms": round(float(np.max(warm_latencies)), 2),
    "std_ms": round(float(np.std(warm_latencies)), 2)
}

latency_benchmark = {
    "environment": "Local Prototype (Windows, Python 3.13, FastAPI TestClient)",
    "catalog_size": 899,
    "cold_start": cold_stats,
    "warm_start": warm_stats
}

with open(OUTPUTS_DIR / "latency_benchmark.json", "w", encoding="utf-8") as f:
    json.dump(latency_benchmark, f, indent=2)
print(f"[OK] Saved {OUTPUTS_DIR / 'latency_benchmark.json'}")
print(f"Cold Latency (ms): Mean={cold_stats['mean_ms']}, Median={cold_stats['median_ms']}, Min={cold_stats['min_ms']}, Max={cold_stats['max_ms']}")
print(f"Warm Latency (ms): Mean={warm_stats['mean_ms']}, Median={warm_stats['median_ms']}, Min={warm_stats['min_ms']}, Max={warm_stats['max_ms']}")

# -------------------------------------------------------------
# 6. GENERATE RESEARCH FIGURES (A THROUGH F)
# -------------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Figure A: Baseline Model Comparison (NDCG@5 and HitRate@5, Overall Cohort)
models = ["Popularity", "Content-Based", "Collaborative Filtering", "Multi-Modal (No Adaptation)", "Proposed Cold-Start Multi-Modal"]
short_names = ["Popularity", "Content-Based", "Collab. Filtering", "Multi-Modal (No Adapt)", "Proposed Multi-Modal"]
ndcg5_vals = [models_comparison[m]["Overall"]["NDCG@5"] for m in models]
hr5_vals = [models_comparison[m]["Overall"]["HitRate@5"] for m in models]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
rects1 = ax.bar(x - width/2, ndcg5_vals, width, label='NDCG@5', color='#3b82f6')
rects2 = ax.bar(x + width/2, hr5_vals, width, label='HitRate@5', color='#10b981')

ax.set_ylabel('Score (Sampled Offline Ranking)', fontsize=11)
ax.set_title('Figure A: Baseline Model Comparison (Sampled Offline Ranking: 1 Pos + 99 Negs, Overall N=1,377)', fontsize=11, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(short_names, fontsize=9, rotation=15, ha='right')
ax.legend(frameon=True)
ax.set_ylim(0, max(hr5_vals) * 1.25)

# Value labels on bars
for rect in rects1:
    height = rect.get_height()
    ax.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
for rect in rects2:
    height = rect.get_height()
    ax.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "figure_a_baselines.png")
plt.close()
print(f"[OK] Saved {FIGURES_DIR / 'figure_a_baselines.png'}")

# Figure B: Overall vs Cold vs Warm Performance for Proposed Model
cohort_labels = ["Overall (N=1,377)", "Cold-Start (N=777, N_hist < 3)", "Warm-Start (N=600, N_hist >= 3)"]
prop_model = models_comparison["Proposed Cold-Start Multi-Modal"]
p5 = [prop_model["Overall"]["Precision@5"], prop_model["Cold-Start"]["Precision@5"], prop_model["Warm-Start"]["Precision@5"]]
r5 = [prop_model["Overall"]["Recall@5"], prop_model["Cold-Start"]["Recall@5"], prop_model["Warm-Start"]["Recall@5"]]
ndcg5 = [prop_model["Overall"]["NDCG@5"], prop_model["Cold-Start"]["NDCG@5"], prop_model["Warm-Start"]["NDCG@5"]]
hr5 = [prop_model["Overall"]["HitRate@5"], prop_model["Cold-Start"]["HitRate@5"], prop_model["Warm-Start"]["HitRate@5"]]

fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
x_b = np.arange(len(cohort_labels))
w_b = 0.2

ax.bar(x_b - 1.5*w_b, p5, w_b, label='Precision@5', color='#6366f1')
ax.bar(x_b - 0.5*w_b, r5, w_b, label='Recall@5', color='#0ea5e9')
ax.bar(x_b + 0.5*w_b, ndcg5, w_b, label='NDCG@5', color='#f59e0b')
ax.bar(x_b + 1.5*w_b, hr5, w_b, label='HitRate@5', color='#10b981')

ax.set_ylabel('Metric Value (Sampled Offline Ranking)', fontsize=11)
ax.set_title('Figure B: Proposed Model Cohort Performance (Sampled Offline Ranking, Threshold K=3)', fontsize=11, fontweight='bold')
ax.set_xticks(x_b)
ax.set_xticklabels(cohort_labels, fontsize=10)
ax.legend(frameon=True)
ax.set_ylim(0, max(r5) * 1.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "figure_b_cohort_performance.png")
plt.close()
print(f"[OK] Saved {FIGURES_DIR / 'figure_b_cohort_performance.png'}")

# Figure C: Cold-Start Threshold Sensitivity (K = 1, 2, 3, 5)
k_keys = ["K=1", "K=2", "K=3", "K=5"]
cold_ndcg = [sensitivity_study[k]["Cold-Start"]["NDCG@5"] for k in k_keys]
warm_ndcg = [sensitivity_study[k]["Warm-Start"]["NDCG@5"] for k in k_keys]
overall_ndcg = [sensitivity_study[k]["Overall"]["NDCG@5"] for k in k_keys]
cold_samples = [sensitivity_study[k]["Cold-Start"]["sample_count"] for k in k_keys]
warm_samples = [sensitivity_study[k]["Warm-Start"]["sample_count"] for k in k_keys]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

ax1.plot(k_keys, cold_ndcg, marker='o', linewidth=2, color='#ef4444', label='Cold-Start (N_hist < K)')
ax1.plot(k_keys, warm_ndcg, marker='s', linewidth=2, color='#3b82f6', label='Warm-Start (N_hist >= K)')
ax1.plot(k_keys, overall_ndcg, marker='^', linewidth=2, linestyle='--', color='#64748b', label='Overall (N=1,377)')
ax1.set_ylabel('NDCG@5 (Sampled Offline Ranking)', fontsize=11)
ax1.set_xlabel('Adaptive Gate Threshold K (Rerun Gating)', fontsize=11)
ax1.set_title('Ranking Metric (NDCG@5) vs Threshold K', fontsize=11, fontweight='bold')
ax1.legend(frameon=True)
ax1.grid(True, linestyle=':', alpha=0.6)

# Cohort size distribution
bar_x = np.arange(len(k_keys))
ax2.bar(bar_x - 0.2, cold_samples, 0.4, label='Cold Samples (N_hist < K)', color='#fca5a5')
ax2.bar(bar_x + 0.2, warm_samples, 0.4, label='Warm Samples (N_hist >= K)', color='#93c5fd')
ax2.set_ylabel('Positive Test Interaction Count', fontsize=11)
ax2.set_xlabel('Threshold K', fontsize=11)
ax2.set_title('Test Sample Cohort Distribution (Total N=1,377)', fontsize=11, fontweight='bold')
ax2.set_xticks(bar_x)
ax2.set_xticklabels(k_keys)
ax2.legend(frameon=True)

plt.suptitle('Figure C: Cold-Start Gate Sensitivity Analysis (Sampled Offline Ranking, Rerun Gating for K in {1, 2, 3, 5})', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / "figure_c_threshold_sensitivity.png")
plt.close()
print(f"[OK] Saved {FIGURES_DIR / 'figure_c_threshold_sensitivity.png'}")

# Figure D: Modality Ablation Comparison
ablation_names = [
    "A: Behavior",
    "B: Content",
    "C: Context",
    "D: Beh + Cont",
    "E: Beh + Ctx",
    "F: Cont + Ctx",
    "G: All (Unweighted)",
    "H: Full + Adaptive Gate"
]
ablation_keys = list(ablation_study.keys())
abl_overall_ndcg = [ablation_study[k]["Overall"]["NDCG@5"] for k in ablation_keys]
abl_cold_ndcg = [ablation_study[k]["Cold-Start"]["NDCG@5"] for k in ablation_keys]
abl_warm_ndcg = [ablation_study[k]["Warm-Start"]["NDCG@5"] for k in ablation_keys]

fig, ax = plt.subplots(figsize=(12, 5.5), dpi=300)
x_d = np.arange(len(ablation_names))
w_d = 0.25

ax.bar(x_d - w_d, abl_overall_ndcg, w_d, label='Overall NDCG@5', color='#475569')
ax.bar(x_d, abl_cold_ndcg, w_d, label='Cold-Start NDCG@5', color='#f87171')
ax.bar(x_d + w_d, abl_warm_ndcg, w_d, label='Warm-Start NDCG@5', color='#60a5fa')

ax.set_ylabel('NDCG@5 (Sampled Offline Ranking)', fontsize=11)
ax.set_title('Figure D: Modality Ablation Study Across User Cohorts (Sampled Offline Ranking, K=3)', fontsize=11, fontweight='bold')
ax.set_xticks(x_d)
ax.set_xticklabels(ablation_names, fontsize=9, rotation=20, ha='right')
ax.legend(frameon=True)
ax.set_ylim(0, max(abl_warm_ndcg + abl_overall_ndcg) * 1.25)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "figure_d_ablation_comparison.png")
plt.close()
print(f"[OK] Saved {FIGURES_DIR / 'figure_d_ablation_comparison.png'}")

# Figure E: Example Modality Weights (New vs Cold Known vs Warm)
# Get real live weights from API
res_new = client.post("/api/recommend", json={"client_id": "demo_ecommerce", "user_id": "new_user_fig_e", "top_k": 5}).json()
res_cold = client.post("/api/recommend", json={"client_id": "demo_ecommerce", "user_id": "7838", "top_k": 5}).json()
res_warm = client.post("/api/recommend", json={"client_id": "demo_ecommerce", "user_id": "9272", "top_k": 5}).json()

users_e = [
    f"Completely New User\n(N_hist=0, Cold)",
    f"Cold Known User 7838\n(N_hist=1, Cold)",
    f"Warm User 9272\n(N_hist=8, Warm)"
]
beh_weights = [
    res_new["modality_weights"]["behavior"],
    res_cold["modality_weights"]["behavior"],
    res_warm["modality_weights"]["behavior"]
]
cont_weights = [
    res_new["modality_weights"]["content"],
    res_cold["modality_weights"]["content"],
    res_warm["modality_weights"]["content"]
]
ctx_weights = [
    res_new["modality_weights"]["context"],
    res_cold["modality_weights"]["context"],
    res_warm["modality_weights"]["context"]
]

fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
x_e = np.arange(len(users_e))
w_e = 0.25

ax.bar(x_e - w_e, beh_weights, w_e, label='Behavioral Weight (alpha_beh)', color='#ef4444')
ax.bar(x_e, cont_weights, w_e, label='Content Weight (alpha_cont)', color='#3b82f6')
ax.bar(x_e + w_e, ctx_weights, w_e, label='Context Weight (alpha_ctx)', color='#10b981')

ax.set_ylabel('Learned Gate Modality Weight', fontsize=11)
ax.set_title('Figure E: Dynamic Modality Weights Across User Cold-Start States (Live API Inference)', fontsize=11, fontweight='bold')
ax.set_xticks(x_e)
ax.set_xticklabels(users_e, fontsize=10)
ax.legend(frameon=True)
ax.set_ylim(0, max(cont_weights + ctx_weights) * 1.25)

for i in range(len(users_e)):
    ax.annotate(f"{beh_weights[i]:.4f}", xy=(x_e[i] - w_e, beh_weights[i]), xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
    ax.annotate(f"{cont_weights[i]:.4f}", xy=(x_e[i], cont_weights[i]), xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
    ax.annotate(f"{ctx_weights[i]:.4f}", xy=(x_e[i] + w_e, ctx_weights[i]), xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "figure_e_modality_weights.png")
plt.close()
print(f"[OK] Saved {FIGURES_DIR / 'figure_e_modality_weights.png'}")

# Figure F: Local Prototype Latency Summary (Warm vs Cold Live Inference)
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
latency_data = [cold_latencies, warm_latencies]
box = ax.boxplot(latency_data, tick_labels=['Cold-Start (N_hist=0)', 'Warm-Start (N_hist=8)'], patch_artist=True)
colors = ['#fee2e2', '#dbeafe']
edge_colors = ['#ef4444', '#3b82f6']
for patch, color, ec in zip(box['boxes'], colors, edge_colors):
    patch.set_facecolor(color)
    patch.set_edgecolor(ec)

ax.set_ylabel('Inference Latency (Milliseconds)', fontsize=11)
ax.set_title(f'Figure F: Local Prototype Latency Distribution (Live Catalog Scoring N=899, Repetitions={repetitions})', fontsize=11, fontweight='bold')
ax.grid(True, linestyle=':', alpha=0.6)

# Annotate median values
med_cold = float(np.median(cold_latencies))
med_warm = float(np.median(warm_latencies))
ax.annotate(f"Median: {med_cold:.2f} ms\n(Mean: {cold_stats['mean_ms']:.2f} ms)", xy=(1, med_cold), xytext=(15, 10), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color='#ef4444'), fontsize=9)
ax.annotate(f"Median: {med_warm:.2f} ms\n(Mean: {warm_stats['mean_ms']:.2f} ms)", xy=(2, med_warm), xytext=(15, 10), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color='#3b82f6'), fontsize=9)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "figure_f_inference_latency.png")
plt.close()
print(f"[OK] Saved {FIGURES_DIR / 'figure_f_inference_latency.png'}")

print("\n" + "="*60)
print("ALL PHASE 7 ARTIFACTS AND FIGURES GENERATED SUCCESSFULLY!")
print("="*60)
