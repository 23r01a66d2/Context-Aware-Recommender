"""
Model Registry Service for Context-Aware Multi-Client Recommendation Platform.
Manages immutable versioned model artifacts, metadata snapshots, active states,
and safe multi-client isolation.
"""

import os
import shutil
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from fastapi import HTTPException, status
from sqlalchemy.orm import Session
import yaml

from backend.database.models import ClientModel, ModelVersionModel
from backend.services.client_service import get_client_or_404, sanitize_client_id
from backend.schemas.model import (
    ModelVersionSummaryResponse,
    ModelVersionDetailResponse,
    ModelActivationResponse
)

logger = logging.getLogger("ModelService")


def get_client_models_dir(client_id: str) -> Path:
    """Returns the isolated base models directory for a client."""
    safe_cid = sanitize_client_id(client_id)
    return Path("models") / safe_cid


def get_client_versions_dir(client_id: str) -> Path:
    """Returns the version registry directory for a client."""
    return get_client_models_dir(client_id) / "versions"


def list_client_models(db: Session, client_id: str) -> List[ModelVersionSummaryResponse]:
    """
    Lists registered model versions for a client.
    Sanitized: does not expose filesystem paths to clients.
    """
    client = get_client_or_404(db, client_id)
    cid = client.client_id

    records = (
        db.query(ModelVersionModel)
        .filter(ModelVersionModel.client_id == cid)
        .order_by(ModelVersionModel.created_at.desc())
        .all()
    )

    results = []
    for r in records:
        dims = json.loads(r.feature_dimensions_json) if r.feature_dimensions_json else None
        metrics = json.loads(r.metrics_json) if r.metrics_json else None
        results.append(
            ModelVersionSummaryResponse(
                client_id=cid,
                version=r.version_tag,
                is_active=r.is_active,
                legacy_imported=r.legacy_imported,
                feature_dimensions=dims,
                metrics=metrics,
                training_run_id=r.training_run_id,
                created_at=r.created_at
            )
        )
    return results


def get_client_model_version(db: Session, client_id: str, version: str) -> ModelVersionDetailResponse:
    """
    Retrieves detailed metadata for a specific model version of a client.
    Enforces isolation: returns 404 if version does not belong to this client.
    """
    client = get_client_or_404(db, client_id)
    cid = client.client_id

    record = (
        db.query(ModelVersionModel)
        .filter(
            ModelVersionModel.client_id == cid,
            ModelVersionModel.version_tag == version
        )
        .first()
    )

    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model version '{version}' not found for client '{client_id}'."
        )

    dims = json.loads(record.feature_dimensions_json) if record.feature_dimensions_json else None
    metrics = json.loads(record.metrics_json) if record.metrics_json else None
    schema_snap = json.loads(record.schema_snapshot_json) if record.schema_snapshot_json else None

    return ModelVersionDetailResponse(
        client_id=cid,
        version=record.version_tag,
        is_active=record.is_active,
        legacy_imported=record.legacy_imported,
        feature_dimensions=dims,
        metrics=metrics,
        training_run_id=record.training_run_id,
        cold_start_threshold=record.cold_start_threshold,
        random_seed=record.random_seed,
        dataset_reference=record.dataset_reference,
        schema_snapshot=schema_snap,
        created_at=record.created_at
    )


def activate_client_model(db: Session, client_id: str, version: str) -> ModelActivationResponse:
    """
    Activates a specific model version for a client.
    Deactivates any previous version for that client only.
    Verifies artifacts exist before activation.
    """
    client = get_client_or_404(db, client_id)
    cid = client.client_id

    target_version = (
        db.query(ModelVersionModel)
        .filter(
            ModelVersionModel.client_id == cid,
            ModelVersionModel.version_tag == version
        )
        .first()
    )

    if not target_version:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model version '{version}' not found for client '{client_id}'."
        )

    # Verify artifacts physically exist on disk
    ckpt_path = Path(target_version.checkpoint_path)
    enc_path = Path(target_version.encoders_path)
    if not ckpt_path.exists() or not enc_path.exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot activate version '{version}': model artifacts are missing on disk."
        )

    # Deactivate existing active versions for THIS client only
    db.query(ModelVersionModel).filter(
        ModelVersionModel.client_id == cid,
        ModelVersionModel.is_active == True
    ).update({"is_active": False})

    # Activate requested version
    target_version.is_active = True
    db.commit()

    # Invalidate model cache for this client
    from backend.services.model_cache import invalidate_client_cache
    invalidate_client_cache(cid)

    logger.info(f"Model version '{version}' activated for client '{cid}'.")
    return ModelActivationResponse(
        client_id=cid,
        version=version,
        is_active=True,
        message=f"Model version '{version}' activated successfully for client '{cid}'."
    )


def register_model_version(
    db: Session,
    client_id: str,
    version_tag: str,
    training_run_id: str,
    checkpoint_src: Path,
    encoders_src: Path,
    metrics: Dict[str, Any],
    feature_dimensions: Dict[str, int],
    schema_snapshot: Dict[str, Any],
    config_snapshot: Dict[str, Any],
    cold_start_threshold: int = 3,
    random_seed: int = 42,
    dataset_reference: Optional[str] = None,
    legacy_imported: bool = False
) -> ModelVersionModel:
    """
    Registers an immutable model version in the registry.
    Creates version directory, saves artifacts and reproducibility snapshots.
    """
    safe_cid = sanitize_client_id(client_id)
    version_dir = get_client_versions_dir(safe_cid) / version_tag
    version_dir.mkdir(parents=True, exist_ok=True)

    # Copy / save artifacts into version directory
    dest_ckpt = version_dir / "model.pt"
    dest_enc = version_dir / "feature_encoders.joblib"

    if checkpoint_src.resolve() != dest_ckpt.resolve():
        shutil.copy2(checkpoint_src, dest_ckpt)
    if encoders_src.resolve() != dest_enc.resolve():
        shutil.copy2(encoders_src, dest_enc)

    # Save snapshots
    with open(version_dir / "schema.json", "w", encoding="utf-8") as f:
        json.dump(schema_snapshot, f, indent=2)

    with open(version_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config_snapshot, f, sort_keys=False)

    with open(version_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    metadata = {
        "client_id": safe_cid,
        "model_version": version_tag,
        "training_run_id": training_run_id,
        "dataset_reference": dataset_reference,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "feature_dimensions": feature_dimensions,
        "cold_start_threshold": cold_start_threshold,
        "random_seed": random_seed,
        "legacy_imported": legacy_imported
    }
    with open(version_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Deactivate any previous active version for this client
    db.query(ModelVersionModel).filter(
        ModelVersionModel.client_id == safe_cid,
        ModelVersionModel.is_active == True
    ).update({"is_active": False})

    # Insert DB record
    new_version = ModelVersionModel(
        client_id=safe_cid,
        version_tag=version_tag,
        training_run_id=training_run_id,
        checkpoint_path=str(dest_ckpt),
        encoders_path=str(dest_enc),
        metrics_json=json.dumps(metrics),
        feature_dimensions_json=json.dumps(feature_dimensions),
        schema_snapshot_json=json.dumps(schema_snapshot),
        cold_start_threshold=cold_start_threshold,
        random_seed=random_seed,
        dataset_reference=dataset_reference,
        is_active=True,
        legacy_imported=legacy_imported
    )
    db.add(new_version)
    db.commit()
    db.refresh(new_version)

    # Invalidate model cache for this client upon new version registration
    from backend.services.model_cache import invalidate_client_cache
    invalidate_client_cache(safe_cid)

    logger.info(f"Registered model version '{version_tag}' for client '{safe_cid}'.")
    return new_version


def ensure_demo_ecommerce_model_registered(db: Session) -> None:
    """
    Registers the validated reference demo_ecommerce checkpoint as version v1 in the registry.
    Does not destroy original models/proposed_multimodal_model.pt.
    Marks legacy_imported = True and records benchmark metrics.
    """
    existing = (
        db.query(ModelVersionModel)
        .filter(
            ModelVersionModel.client_id == "demo_ecommerce",
            ModelVersionModel.version_tag == "v1"
        )
        .first()
    )
    if existing:
        return

    # Check if reference checkpoint exists
    ref_ckpt = Path("models/proposed_multimodal_model.pt")
    ref_enc = Path("models/feature_encoders.joblib")
    if not ref_ckpt.exists() or not ref_enc.exists():
        logger.warning("Reference demo_ecommerce artifacts not found in models/; skipping registration.")
        return

    # Read verified reference benchmark metrics if available
    metrics_file = Path("outputs/evaluation/models_comparison.json")
    if metrics_file.exists():
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                all_metrics = json.load(f)
            ref_metrics = all_metrics.get("Proposed Cold-Start Multi-Modal", {})
        except Exception:
            ref_metrics = {"status": "validated_baseline"}
    else:
        ref_metrics = {"status": "validated_baseline"}

    schema_file = Path("clients/demo_ecommerce/config.yaml")
    config_dict = {}
    if schema_file.exists():
        with open(schema_file, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or {}

    register_model_version(
        db=db,
        client_id="demo_ecommerce",
        version_tag="v1",
        training_run_id="legacy_run_baseline",
        checkpoint_src=ref_ckpt,
        encoders_src=ref_enc,
        metrics=ref_metrics,
        feature_dimensions={"behavior": 16, "content": 12, "context": 21},
        schema_snapshot=config_dict.get("column_mappings", {}),
        config_snapshot=config_dict,
        cold_start_threshold=3,
        random_seed=42,
        dataset_reference="Ecommerce.csv",
        legacy_imported=True
    )
    logger.info("Successfully registered demo_ecommerce reference model version 'v1' (legacy_imported=True).")
