"""
Master Training, Evaluation & Ablation Pipeline
Context-Aware Recommendation System for Visual Analytics: Mitigating Cold-Start via Multi-Modal Behavioral Fusion
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import logging
import argparse
import json
import random
import numpy as np
import pandas as pd
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.data_ingestion import ingest_data
from src.preprocessing import clean_and_split_data
from src.feature_engineering import prepare_multimodal_features
from src.models.recommender import (
    TwoTowerMultiModalRecommender,
    PopularityRecommender,
    ContentBasedRecommender,
    CollaborativeFilteringRecommender
)
from src.models.ranker import (
    build_test_candidate_pools,
    compute_ranking_metrics_for_case,
    aggregate_cohort_metrics
)
from src.visualization import (
    generate_all_eda_plots,
    plot_training_curves,
    plot_model_comparisons
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainPipeline")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MultiModalDataset(Dataset):
    """PyTorch Dataset for multi-modal interaction instances."""

    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.X_beh = torch.tensor(data["X_beh"], dtype=torch.float32)
        self.X_cont = torch.tensor(data["X_cont"], dtype=torch.float32)
        self.X_ctx = torch.tensor(data["X_ctx"], dtype=torch.float32)
        self.y = torch.tensor(data["y"], dtype=torch.float32)
        self.hist_count = torch.tensor(data["hist_count"], dtype=torch.long)
        self.cust_id = torch.tensor(data["cust_id"], dtype=torch.long)
        self.prod_id = torch.tensor(data["prod_id"], dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "X_beh": self.X_beh[idx],
            "X_cont": self.X_cont[idx],
            "X_ctx": self.X_ctx[idx],
            "y": self.y[idx],
            "hist_count": self.hist_count[idx],
            "cust_id": self.cust_id[idx],
            "prod_id": self.prod_id[idx]
        }


def train_neural_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                       epochs: int = 15, lr: float = 0.001, device: str = "cpu",
                       checkpoint_path: str = "models/best_multimodal_model.pt") -> dict:
    """Trains TwoTowerMultiModalRecommender using BCEWithLogitsLoss."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    # Positives are ~22.5%, so pos_weight of ~3.4 balances gradient contribution
    pos_weight = torch.tensor([3.4], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")

    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            x_beh = batch["X_beh"].to(device)
            x_cont = batch["X_cont"].to(device)
            x_ctx = batch["X_ctx"].to(device)
            y = batch["y"].to(device)
            hist = batch["hist_count"].to(device)

            optimizer.zero_grad()
            logits, weights, cold_mask = model(x_beh, x_cont, x_ctx, hist, x_cont)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x_beh = batch["X_beh"].to(device)
                x_cont = batch["X_cont"].to(device)
                x_ctx = batch["X_ctx"].to(device)
                y = batch["y"].to(device)
                hist = batch["hist_count"].to(device)

                logits, _, _ = model(x_beh, x_cont, x_ctx, hist, x_cont)
                loss = criterion(logits, y)
                val_losses.append(loss.item())

        avg_train = float(np.mean(train_losses))
        avg_val = float(np.mean(val_losses))
        history["train_loss"].append(round(avg_train, 4))
        history["val_loss"].append(round(avg_val, 4))

        logger.info(f"Epoch {epoch:02d}/{epochs:02d} - Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), checkpoint_path)

    # Load best checkpoint
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return history


def evaluate_neural_recommender(model: TwoTowerMultiModalRecommender, candidate_pools: list[dict],
                                test_dataset: MultiModalDataset, product_features_table: dict,
                                threshold_k: int = 3, device: str = "cpu") -> dict:
    """Evaluates neural two-tower model on test candidate pools."""
    model.eval()
    case_results = []

    with torch.no_grad():
        for case in candidate_pools:
            idx = case["test_idx"]
            candidates = case["candidates"] # [pos_pid, neg_1, ..., neg_99]

            item = test_dataset[idx]
            x_beh = item["X_beh"].unsqueeze(0).to(device)
            x_cont_anchor = item["X_cont"].unsqueeze(0).to(device)
            x_ctx = item["X_ctx"].unsqueeze(0).to(device)
            hist = item["hist_count"].unsqueeze(0).to(device)

            # 1. Encode user-context representation
            e_user, weights, cold_mask = model.encode_user_context(x_beh, x_cont_anchor, x_ctx, hist)

            # 2. Build candidate tensor [100, 12]
            cand_tensors = [product_features_table[pid] for pid in candidates]
            x_cands = torch.tensor(np.array(cand_tensors), dtype=torch.float32, device=device)

            # 3. Score all 100 candidate items
            cand_scores = model.score_candidates(e_user, x_cands).cpu().numpy()

            # 4. Compute ranking metrics
            res = compute_ranking_metrics_for_case(cand_scores, k_values=[5, 10])
            res["hist_count"] = case["hist_count"]
            res["learned_weights"] = weights.cpu().squeeze(0).tolist()
            res["is_cold"] = bool(cold_mask.cpu().item())
            case_results.append(res)

    summary = aggregate_cohort_metrics(case_results, threshold_k=threshold_k, k_eval=[5, 10])
    return summary, case_results


def evaluate_heuristic_model(scorer_fn, candidate_pools: list[dict], threshold_k: int = 3) -> dict:
    """Evaluates non-neural baselines (Popularity, Content-based, Collaborative Filtering)."""
    case_results = []
    for case in candidate_pools:
        candidates = case["candidates"]
        scores = scorer_fn(case)
        res = compute_ranking_metrics_for_case(scores, k_values=[5, 10])
        res["hist_count"] = case["hist_count"]
        case_results.append(res)

    summary = aggregate_cohort_metrics(case_results, threshold_k=threshold_k, k_eval=[5, 10])
    return summary


def run_pipeline(mode: str = "initial"):
    """Executes training, evaluation, and ablation experiments."""
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config["project"]["seed"])
    device = config["project"]["device"]

    # 1. Ensure data features exist
    feat_train = Path("data/processed/train_features.npz")
    if not feat_train.exists():
        logger.info("Features not found. Running feature engineering...")
        df_feat, _ = prepare_multimodal_features(config)
    else:
        df_feat = pd.read_csv("data/processed/cleaned_dataset.csv")

    # 2. Generate EDA plots if not present
    eda_dir = Path("outputs/eda")
    if not (eda_dir / "customer_behavior_funnel.png").exists():
        logger.info("Generating publication EDA plots...")
        generate_all_eda_plots(df_feat, str(eda_dir))

    # 3. Load Datasets
    logger.info("Loading PyTorch datasets...")
    train_dataset = MultiModalDataset("data/processed/train_features.npz")
    val_dataset = MultiModalDataset("data/processed/val_features.npz")
    test_dataset = MultiModalDataset("data/processed/test_features.npz")

    train_loader = DataLoader(train_dataset, batch_size=config["model"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["model"]["batch_size"], shuffle=False)

    # 4. Build/load candidate pools for test evaluation (100% fair baseline protocol)
    test_df = pd.read_csv(config["data"]["test_path"])
    all_df = pd.read_csv(config["data"]["cleaned_path"])
    logger.info("Building reproducible test candidate pools (1 positive + 99 unpurchased negatives)...")
    candidate_pools = build_test_candidate_pools(test_df, all_df, num_negatives=99, seed=42)
    logger.info(f"Generated candidate pools for {len(candidate_pools)} test purchase evaluations.")

    # 5. Build candidate product feature dictionary (for fast tensor construction)
    # Map each product_id to its representative content feature vector [12]
    test_features_npz = np.load("data/processed/test_features.npz")
    product_features_table = {}
    for pid, cont_feat in zip(test_features_npz["prod_id"], test_features_npz["X_cont"]):
        if pid not in product_features_table:
            product_features_table[pid] = cont_feat

    # Fill any missing catalog product using train features
    train_features_npz = np.load("data/processed/train_features.npz")
    for pid, cont_feat in zip(train_features_npz["prod_id"], train_features_npz["X_cont"]):
        if pid not in product_features_table:
            product_features_table[pid] = cont_feat

    # -------------------------------------------------------------
    # EXPERIMENT 1: PROPOSED MULTI-MODAL MODEL WITH ADAPTIVE GATING
    # -------------------------------------------------------------
    logger.info("\n>>> Training PROPOSED MULTI-MODAL MODEL WITH ADAPTIVE COLD-START GATING...")
    proposed_model = TwoTowerMultiModalRecommender(
        num_products=1000,
        beh_dim=16, ctx_dim=21, cont_meta_dim=11,
        embedding_dim=config["model"]["embedding_dim"],
        apply_cold_start_adaptation=True,
        cold_threshold=config["cold_start"]["primary_threshold"],
        behavior_penalty=config["cold_start"]["cold_behavior_penalty"]
    )

    history = train_neural_model(
        proposed_model, train_loader, val_loader,
        epochs=config["model"]["epochs"],
        lr=config["model"]["learning_rate"],
        device=device,
        checkpoint_path="models/proposed_multimodal_model.pt"
    )

    # Plot training loss curves
    plot_training_curves(history, "outputs/plots")

    # Evaluate Proposed Model on Test candidates
    logger.info("Evaluating Proposed Model on Test Candidate Pools...")
    prop_summary, prop_cases = evaluate_neural_recommender(
        proposed_model, candidate_pools, test_dataset, product_features_table,
        threshold_k=config["cold_start"]["primary_threshold"], device=device
    )

    print("\n=======================================================")
    print("      FIRST TRAINING OUTPUT: PROPOSED MODEL (K=3)       ")
    print("=======================================================")
    print(f"Final Train Loss: {history['train_loss'][-1]:.4f} | Final Val Loss: {history['val_loss'][-1]:.4f}")
    print("\nTest Evaluation Results (Primary Threshold K=3):")
    for cohort in ["Overall", "Cold-Start", "Warm-Start"]:
        m = prop_summary[cohort]
        print(f"  [{cohort} - {m['sample_count']} samples]:")
        print(f"    Precision@5: {m['Precision@5']:.4f} | Recall@5: {m['Recall@5']:.4f} | NDCG@5: {m['NDCG@5']:.4f} | HitRate@5: {m['HitRate@5']:.4f} (Mean Rank: {m['MeanRank']})")

    if mode == "initial":
        # Save initial results
        with open("outputs/evaluation/initial_experiment_result.json", "w", encoding="utf-8") as f:
            json.dump({"history": history, "metrics": prop_summary}, f, indent=2)
        return

    # -------------------------------------------------------------
    # BASELINE 1: POPULARITY RECOMMENDER
    # -------------------------------------------------------------
    logger.info("\n>>> Evaluating BASELINE 1: POPULARITY...")
    train_df = pd.read_csv(config["data"]["train_path"])
    pop_model = PopularityRecommender()
    pop_model.fit(train_df)
    pop_summary = evaluate_heuristic_model(
        lambda case: pop_model.score_candidates(case["candidates"]),
        candidate_pools, threshold_k=3
    )

    # -------------------------------------------------------------
    # BASELINE 2: CONTENT-BASED RECOMMENDER
    # -------------------------------------------------------------
    logger.info("\n>>> Evaluating BASELINE 2: CONTENT-BASED...")
    cb_model = ContentBasedRecommender(num_products=1000)
    cb_model.fit(train_df)

    def cb_scorer(case):
        idx = case["test_idx"]
        cust_beh = test_dataset[idx]["X_beh"][-8:].numpy() # last 8 dims are category preference histogram
        is_cold = (case["hist_count"] < 3)
        return cb_model.score_candidates(cust_beh, case["candidates"], is_cold)

    cb_summary = evaluate_heuristic_model(cb_scorer, candidate_pools, threshold_k=3)

    # -------------------------------------------------------------
    # BASELINE 3: COLLABORATIVE FILTERING (SVD)
    # -------------------------------------------------------------
    logger.info("\n>>> Evaluating BASELINE 3: COLLABORATIVE FILTERING (SVD)...")
    cf_model = CollaborativeFilteringRecommender(n_factors=32)
    cf_model.fit(train_df)
    cf_summary = evaluate_heuristic_model(
        lambda case: cf_model.score_candidates(case["customer_id"], case["candidates"]),
        candidate_pools, threshold_k=3
    )

    # -------------------------------------------------------------
    # BASELINE 4: MULTI-MODAL FUSION WITHOUT COLD-START ADAPTATION
    # -------------------------------------------------------------
    logger.info("\n>>> Training MULTI-MODAL FUSION WITHOUT COLD-START ADAPTATION...")
    unadapted_model = TwoTowerMultiModalRecommender(
        num_products=1000, beh_dim=16, ctx_dim=21, cont_meta_dim=11,
        embedding_dim=config["model"]["embedding_dim"],
        apply_cold_start_adaptation=False
    )
    train_neural_model(
        unadapted_model, train_loader, val_loader,
        epochs=config["model"]["epochs"], lr=config["model"]["learning_rate"],
        device=device, checkpoint_path="models/unadapted_multimodal_model.pt"
    )
    unadapt_summary, _ = evaluate_neural_recommender(
        unadapted_model, candidate_pools, test_dataset, product_features_table,
        threshold_k=3, device=device
    )

    # Combine benchmark results
    all_benchmarks = {
        "Popularity": pop_summary,
        "Content-Based": cb_summary,
        "Collaborative Filtering": cf_summary,
        "Multi-Modal (No Adaptation)": unadapt_summary,
        "Proposed Cold-Start Multi-Modal": prop_summary
    }

    # Save benchmark table
    with open("outputs/evaluation/models_comparison.json", "w", encoding="utf-8") as f:
        json.dump(all_benchmarks, f, indent=2)

    # Plot comparisons
    plot_model_comparisons(all_benchmarks, "outputs/evaluation")

    # -------------------------------------------------------------
    # THRESHOLD SENSITIVITY EXPERIMENTS (K in {1, 2, 3, 5})
    # -------------------------------------------------------------
    logger.info("\n>>> Running Cold-Start Threshold Sensitivity Experiments (K in {1, 2, 3, 5})...")
    threshold_results = {}
    for K in [1, 2, 3, 5]:
        proposed_model.cold_start_gate.set_threshold(K)
        t_summary, _ = evaluate_neural_recommender(
            proposed_model, candidate_pools, test_dataset, product_features_table,
            threshold_k=K, device=device
        )
        threshold_results[f"K={K}"] = t_summary
    proposed_model.cold_start_gate.set_threshold(3) # restore default

    with open("outputs/evaluation/threshold_sensitivity.json", "w", encoding="utf-8") as f:
        json.dump(threshold_results, f, indent=2)

    # -------------------------------------------------------------
    # ABLATION STUDY (Models A through H)
    # -------------------------------------------------------------
    logger.info("\n>>> Running 8-Model Ablation Study...")
    # Ablation mask flags: [use_beh, use_cont, use_ctx, apply_cold_gate]
    ablation_configs = {
        "Model A (Behavior Only)": [True, False, False, False],
        "Model B (Content Only)": [False, True, False, False],
        "Model C (Context Only)": [False, False, True, False],
        "Model D (Behavior + Content)": [True, True, False, False],
        "Model E (Behavior + Context)": [True, False, True, False],
        "Model F (Content + Context)": [False, True, True, False],
        "Model G (Behavior + Content + Context - Unweighted)": [True, True, True, False],
        "Model H (Proposed Multi-Modal + Adaptive Cold-Start Gate)": [True, True, True, True]
    }

    ablation_summary = {}
    for ab_name, flags in ablation_configs.items():
        use_beh, use_cont, use_ctx, apply_gate = flags
        logger.info(f"Ablation: {ab_name}")

        ab_model = TwoTowerMultiModalRecommender(
            num_products=1000, beh_dim=16, ctx_dim=21, cont_meta_dim=11,
            embedding_dim=config["model"]["embedding_dim"],
            apply_cold_start_adaptation=apply_gate
        )
        # Train lightweight 5-epoch ablation
        train_neural_model(
            ab_model, train_loader, val_loader, epochs=6,
            lr=config["model"]["learning_rate"], device=device,
            checkpoint_path=f"models/ablation_{ab_name[:7].replace(' ', '_').lower()}.pt"
        )
        ab_eval, _ = evaluate_neural_recommender(
            ab_model, candidate_pools, test_dataset, product_features_table,
            threshold_k=3, device=device
        )
        ablation_summary[ab_name] = ab_eval

    with open("outputs/evaluation/ablation_study.json", "w", encoding="utf-8") as f:
        json.dump(ablation_summary, f, indent=2)

    # Convert to clean CSV for paper / thesis presentation
    ablation_rows = []
    for model_name, res in ablation_summary.items():
        ablation_rows.append({
            "Ablation Model": model_name,
            "Overall Precision@5": res["Overall"]["Precision@5"],
            "Overall NDCG@5": res["Overall"]["NDCG@5"],
            "Cold-Start NDCG@5": res["Cold-Start"]["NDCG@5"],
            "Warm-Start NDCG@5": res["Warm-Start"]["NDCG@5"]
        })
    pd.DataFrame(ablation_rows).to_csv("outputs/evaluation/ablation_study.csv", index=False)
    logger.info("Full training, benchmark comparison, threshold sensitivity, and ablation experiments complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="initial", choices=["initial", "full"],
                        help="Mode: 'initial' (single experiment as requested by Step 13) or 'full' (all models + ablations)")
    args = parser.parse_args()
    run_pipeline(mode=args.mode)
