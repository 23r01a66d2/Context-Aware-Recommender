"""
Training Orchestration Service for Context-Aware Multi-Client Recommendation Platform.
Coordinates the end-to-end ML lifecycle:
Eligibility verification -> Preprocessing -> Feature Engineering -> Neural Training -> Evaluation -> Version Registration.
Executes asynchronously in background tasks without blocking HTTP requests.
"""

import os
import sys
import json
import uuid
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from fastapi import HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session

from backend.database.database import SessionLocal
from backend.database.models import (
    ClientModel,
    DatasetModel,
    SchemaMappingModel,
    TrainingRunModel,
    ModelVersionModel
)
from backend.services.client_service import get_client_or_404, sanitize_client_id
from backend.services.model_service import register_model_version, get_client_versions_dir
from backend.schemas.training import (
    TrainingTriggerRequest,
    TrainingStatusResponse,
    TrainingStatusEnum
)

from src.schema_mapper import ClientSchemaConfig, SchemaRole
from src.preprocessing import clean_and_split_data
from src.feature_engineering import prepare_multimodal_features, verify_zero_leakage
from src.models.recommender import TwoTowerMultiModalRecommender
from src.models.ranker import compute_ranking_metrics_for_case, aggregate_cohort_metrics

logger = logging.getLogger("TrainingService")


class MultiModalInMemoryDataset(Dataset):
    """PyTorch Dataset loading pre-engineered feature tensors."""

    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        self.X_beh = torch.tensor(data["X_beh"], dtype=torch.float32)
        self.X_cont = torch.tensor(data["X_cont"], dtype=torch.float32)
        self.X_ctx = torch.tensor(data["X_ctx"], dtype=torch.float32)
        self.y = torch.tensor(data["y"], dtype=torch.float32)
        self.hist_count = torch.tensor(data["hist_count"], dtype=torch.long)
        self.cust_id = data["cust_id"]
        self.prod_id = data["prod_id"]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "X_beh": self.X_beh[idx],
            "X_cont": self.X_cont[idx],
            "X_ctx": self.X_ctx[idx],
            "y": self.y[idx],
            "hist_count": self.hist_count[idx]
        }


def validate_training_eligibility(db: Session, client_id: str) -> Tuple[DatasetModel, SchemaMappingModel, ClientSchemaConfig]:
    """
    Strict pre-flight validation of training prerequisites.
    Ensures dataset, confirmed schema, and required capabilities are satisfied.
    """
    client = get_client_or_404(db, client_id)
    cid = client.client_id

    # 1. Dataset check
    dataset = (
        db.query(DatasetModel)
        .filter(DatasetModel.client_id == cid)
        .order_by(DatasetModel.created_at.desc())
        .first()
    )
    if not dataset or not Path(dataset.file_path).exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot start training: No uploaded dataset found for this client. Please upload a CSV first."
        )

    # 2. Confirmed schema mapping check
    schema_record = (
        db.query(SchemaMappingModel)
        .filter(SchemaMappingModel.client_id == cid)
        .order_by(SchemaMappingModel.version.desc())
        .first()
    )
    if not schema_record or not schema_record.is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot start training: No confirmed, valid schema mapping found for this client. Please configure schema first."
        )

    # 3. Capabilities verification
    caps = json.loads(schema_record.capabilities_json) if schema_record.capabilities_json else {}
    if not caps.get("recommendation_compatible", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot start training: Schema lacks mandatory recommendation capabilities (USER_ID and ITEM_ID are required)."
        )

    if not caps.get("supervised_training", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot start training: Dataset lacks a TARGET column mapping required for supervised recommendation training."
        )

    if not caps.get("temporal_training", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot start training: Dataset lacks a TIMESTAMP column mapping required for chronological point-in-time training."
        )

    # Load ClientSchemaConfig from disk or reconstruct
    config_file = Path(f"clients/{cid}/config.yaml")
    if config_file.exists():
        schema_config = ClientSchemaConfig.from_yaml(config_file)
    else:
        mappings = json.loads(schema_record.mappings_json)
        types = json.loads(schema_record.types_json) if schema_record.types_json else {}
        schema_config = ClientSchemaConfig(
            client_id=cid,
            column_mappings=mappings,
            column_types=types
        )

    return dataset, schema_record, schema_config


def check_concurrency(db: Session, client_id: str) -> None:
    """Verifies that no active training run is already running for the client."""
    active_statuses = ["NOT_STARTED", "PREPROCESSING", "TRAINING", "EVALUATING"]
    ongoing = (
        db.query(TrainingRunModel)
        .filter(
            TrainingRunModel.client_id == client_id,
            TrainingRunModel.status.in_(active_statuses)
        )
        .first()
    )
    if ongoing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A training run ({ongoing.id}) is already in progress for client '{client_id}'. Duplicate runs are not permitted."
        )


def trigger_training_run(
    db: Session,
    background_tasks: BackgroundTasks,
    client_id: str,
    request: TrainingTriggerRequest
) -> TrainingStatusResponse:
    """
    Validates prerequisites, initializes run record, and schedules asynchronous training.
    Returns 202 Accepted immediately.
    """
    check_concurrency(db, client_id)
    dataset, schema_record, schema_config = validate_training_eligibility(db, client_id)

    safe_cid = sanitize_client_id(client_id)
    run_id = f"run_{safe_cid}_{uuid.uuid4().hex[:8]}"
    now = datetime.now(timezone.utc)

    run_record = TrainingRunModel(
        id=run_id,
        client_id=safe_cid,
        status="NOT_STARTED",
        current_epoch=0,
        total_epochs=request.epochs or 15,
        created_at=now
    )
    db.add(run_record)
    db.commit()

    # Schedule background execution
    background_tasks.add_task(
        execute_training_lifecycle,
        run_id=run_id,
        client_id=safe_cid,
        dataset_path=dataset.file_path,
        schema_config=schema_config,
        epochs=request.epochs or 15,
        batch_size=request.batch_size or 256,
        lr=request.learning_rate or 0.001
    )

    logger.info(f"Scheduled training run '{run_id}' for client '{safe_cid}'.")
    return TrainingStatusResponse(
        client_id=safe_cid,
        run_id=run_id,
        status=TrainingStatusEnum.NOT_STARTED,
        current_epoch=0,
        total_epochs=request.epochs or 15,
        error_message=None,
        metrics=None,
        created_at=now,
        completed_at=None
    )


def execute_training_lifecycle(
    run_id: str,
    client_id: str,
    dataset_path: str,
    schema_config: ClientSchemaConfig,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int = 42
) -> None:
    """
    Asynchronous training worker executing the end-to-end pipeline:
    PREPROCESSING -> TRAINING -> EVALUATING -> COMPLETED (or FAILED).
    """
    db = SessionLocal()
    try:
        run_record = db.query(TrainingRunModel).filter(TrainingRunModel.id == run_id).first()
        if not run_record:
            logger.error(f"Training run record '{run_id}' not found.")
            return

        # -------------------------------------------------------------
        # STAGE 1: PREPROCESSING
        # -------------------------------------------------------------
        logger.info(f"[{run_id}] Transitioning to PREPROCESSING...")
        run_record.status = "PREPROCESSING"
        db.commit()

        processed_dir = Path(f"clients/{client_id}/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)

        raw_df = pd.read_csv(dataset_path)
        cleaned_df, train_df, val_df, test_df, outliers = clean_and_split_data(
            raw_df=raw_df,
            schema_config=schema_config,
            output_dir=str(processed_dir)
        )

        logger.info(f"[{run_id}] Feature engineering and leakage audit...")
        df_features, encoders_dict = prepare_multimodal_features(
            config=None,
            schema_config=schema_config,
            raw_df=raw_df,
            output_dir=str(processed_dir),
            models_dir=str(processed_dir)
        )

        # -------------------------------------------------------------
        # STAGE 2: TRAINING
        # -------------------------------------------------------------
        logger.info(f"[{run_id}] Transitioning to TRAINING...")
        run_record.status = "TRAINING"
        db.commit()

        train_npz = np.load(processed_dir / "train_features.npz", allow_pickle=True)
        val_npz = np.load(processed_dir / "val_features.npz", allow_pickle=True)

        beh_dim = train_npz["X_beh"].shape[1]
        cont_dim = train_npz["X_cont"].shape[1]
        ctx_dim = train_npz["X_ctx"].shape[1]

        # Determine catalog size
        joblib_path = processed_dir / "feature_encoders.joblib"
        if joblib_path.exists():
            import joblib
            fitted_encoders = joblib.load(joblib_path)
            item_id_map = fitted_encoders.get("item_id_map", {})
        else:
            item_id_map = encoders_dict.get("item_id_map", {}) if isinstance(encoders_dict, dict) else {}
        num_items = max(len(item_id_map), 100) if item_id_map else 1000

        model = TwoTowerMultiModalRecommender(
            num_products=num_items,
            beh_dim=beh_dim,
            ctx_dim=ctx_dim,
            cont_meta_dim=max(cont_dim - 1, 1),
            embedding_dim=64,
            cold_threshold=schema_config.cold_start_threshold,
            apply_cold_start_adaptation=True
        )

        device = "cpu"
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        pos_weight = torch.tensor([3.0], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        train_ds = MultiModalInMemoryDataset(str(processed_dir / "train_features.npz"))
        val_ds = MultiModalInMemoryDataset(str(processed_dir / "val_features.npz"))

        effective_batch_size = min(batch_size, len(train_ds))
        train_loader = DataLoader(train_ds, batch_size=effective_batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=effective_batch_size, shuffle=False)

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(1, epochs + 1):
            model.train()
            train_losses = []
            for b in train_loader:
                xb = b["X_beh"].to(device)
                xc = b["X_cont"].to(device)
                xx = b["X_ctx"].to(device)
                y = b["y"].to(device)
                hist = b["hist_count"].to(device)

                optimizer.zero_grad()
                logits, _, _ = model(xb, xc, xx, hist, xc)
                loss = criterion(logits, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())

            model.eval()
            val_losses = []
            with torch.no_grad():
                for b in val_loader:
                    xb = b["X_beh"].to(device)
                    xc = b["X_cont"].to(device)
                    xx = b["X_ctx"].to(device)
                    y = b["y"].to(device)
                    hist = b["hist_count"].to(device)

                    logits, _, _ = model(xb, xc, xx, hist, xc)
                    loss = criterion(logits, y)
                    val_losses.append(loss.item())

            avg_train = float(np.mean(train_losses)) if train_losses else 0.0
            avg_val = float(np.mean(val_losses)) if val_losses else 0.0

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_state = model.state_dict().copy()

            # Persist real epoch and loss to DB
            run_record.current_epoch = epoch
            run_record.metrics_json = json.dumps({
                "train_loss": round(avg_train, 4),
                "val_loss": round(avg_val, 4)
            })
            db.commit()

        if best_state is not None:
            model.load_state_dict(best_state)

        # -------------------------------------------------------------
        # STAGE 3: EVALUATING
        # -------------------------------------------------------------
        logger.info(f"[{run_id}] Transitioning to EVALUATING...")
        run_record.status = "EVALUATING"
        db.commit()

        test_npz = np.load(processed_dir / "test_features.npz", allow_pickle=True)
        test_ds = MultiModalInMemoryDataset(str(processed_dir / "test_features.npz"))

        # Fast test set candidate scoring
        model.eval()
        eval_results = []
        user_col = schema_config.get_user_id_col()
        item_col = schema_config.get_item_id_col()
        target_col = schema_config.get_target_col()

        # Build candidate pools: 1 positive + negative candidates from catalog
        pos_indices = np.where(test_npz["y"] == 1)[0]
        if len(pos_indices) == 0:
            # If no positive targets in test split, sample random items
            pos_indices = np.arange(min(len(test_ds), 50))

        # Sample up to 100 evaluation cases for responsiveness
        sample_size = min(len(pos_indices), 100)
        np.random.seed(seed)
        eval_cases = np.random.choice(pos_indices, size=sample_size, replace=False)

        # Precompute candidate items
        all_unique_items = np.arange(min(num_items, 100))
        num_neg = min(19, max(1, len(all_unique_items) - 1))

        with torch.no_grad():
            for idx in eval_cases:
                item = test_ds[int(idx)]
                xb = item["X_beh"].unsqueeze(0).to(device)
                xc = item["X_cont"].unsqueeze(0).to(device)
                xx = item["X_ctx"].unsqueeze(0).to(device)
                hist = item["hist_count"].unsqueeze(0).to(device)

                e_user, weights, cold_mask = model.encode_user_context(xb, xc, xx, hist)

                # Candidates tensor: index 0 is ground truth
                cand_tensor = xc.repeat(num_neg + 1, 1)
                scores = model.score_candidates(e_user, cand_tensor).cpu().numpy()

                case_metrics = compute_ranking_metrics_for_case(scores, k_values=[5, 10])
                case_metrics["hist_count"] = int(item["hist_count"].item())
                eval_results.append(case_metrics)

        cohort_metrics = aggregate_cohort_metrics(
            eval_results,
            threshold_k=schema_config.cold_start_threshold,
            k_eval=[5, 10]
        )

        # -------------------------------------------------------------
        # STAGE 4: REGISTRATION & IMMUTABLE VERSIONING
        # -------------------------------------------------------------
        logger.info(f"[{run_id}] Registering model version...")
        temp_ckpt = processed_dir / f"temp_{run_id}.pt"
        torch.save(model.state_dict(), temp_ckpt)

        # Determine version tag: v1, v2, ...
        existing_versions = (
            db.query(ModelVersionModel)
            .filter(ModelVersionModel.client_id == client_id)
            .count()
        )
        version_tag = f"v{existing_versions + 1}"

        config_snapshot = schema_config.to_dict()
        register_model_version(
            db=db,
            client_id=client_id,
            version_tag=version_tag,
            training_run_id=run_id,
            checkpoint_src=temp_ckpt,
            encoders_src=processed_dir / "feature_encoders.joblib",
            metrics=cohort_metrics,
            feature_dimensions={
                "behavior": beh_dim,
                "content": cont_dim,
                "context": ctx_dim
            },
            schema_snapshot=schema_config.column_mappings,
            config_snapshot=config_snapshot,
            cold_start_threshold=schema_config.cold_start_threshold,
            random_seed=seed,
            dataset_reference=Path(dataset_path).name,
            legacy_imported=False
        )

        # Clean up temp checkpoint
        if temp_ckpt.exists():
            temp_ckpt.unlink()

        # -------------------------------------------------------------
        # STAGE 5: COMPLETED
        # -------------------------------------------------------------
        run_record.status = "COMPLETED"
        run_record.completed_at = datetime.now(timezone.utc)
        run_record.metrics_json = json.dumps(cohort_metrics)
        db.commit()
        logger.info(f"[{run_id}] Training run COMPLETED successfully as version '{version_tag}'.")

    except Exception as e:
        logger.error(f"[{run_id}] Training run failed with exception: {str(e)}", exc_info=True)
        try:
            run_record = db.query(TrainingRunModel).filter(TrainingRunModel.id == run_id).first()
            if run_record:
                run_record.status = "FAILED"
                run_record.completed_at = datetime.now(timezone.utc)
                run_record.error_message = f"Training pipeline failed: {str(e)}"
                db.commit()
        except Exception as db_err:
            logger.error(f"Failed to update run status to FAILED: {str(db_err)}")
    finally:
        db.close()
