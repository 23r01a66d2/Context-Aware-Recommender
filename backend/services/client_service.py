"""
Service for client management, isolation enforcement, and demo_ecommerce bootstrapping.
"""

import re
import json
import logging
import shutil
from pathlib import Path
from typing import List, Optional
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from backend.database.models import (
    ClientModel,
    DatasetModel,
    SchemaMappingModel,
    ModelVersionModel,
    TrainingRunModel,
    RecommendationEventModel,
    FeedbackEventModel
)
from backend.schemas.client import ClientCreate
from src.schema_mapper import create_demo_ecommerce_schema

logger = logging.getLogger("ClientService")

CLIENT_ID_SAFE_REGEX = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def validate_client_id_security(client_id: str) -> str:
    """
    Validates client_id to prevent path traversal, injection, and invalid filesystem characters.
    Raises HTTPException(400) if invalid.
    """
    if not client_id or not isinstance(client_id, str):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Client ID is required."
        )

    cid = client_id.strip()
    if not CLIENT_ID_SAFE_REGEX.match(cid):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid client ID format. Use only alphanumeric characters, hyphens, and underscores (1-64 chars)."
        )

    # Explicit path traversal defense
    if ".." in cid or "/" in cid or "\\" in cid or ":" in cid or "\0" in cid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Path traversal sequences are strictly forbidden."
        )

    return cid


sanitize_client_id = validate_client_id_security


def get_client_workspace(client_id: str) -> Path:
    """Returns the isolated workspace directory for a validated client."""
    safe_cid = validate_client_id_security(client_id)
    workspace = Path(f"clients/{safe_cid}").resolve()
    workspace_root = Path("clients").resolve()
    # Path traversal check
    if not str(workspace).startswith(str(workspace_root)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid workspace resolution."
        )
    return workspace


def create_client(db: Session, client_in: ClientCreate) -> ClientModel:
    """Creates a new client record in DB and provisions its isolated workspace."""
    cid = validate_client_id_security(client_in.client_id)

    # Check existence
    existing = db.query(ClientModel).filter(ClientModel.client_id == cid).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Client '{cid}' already exists."
        )

    # 1. Create DB record
    client_obj = ClientModel(
        client_id=cid,
        name=client_in.name.strip(),
        description=(client_in.description or "").strip()
    )
    db.add(client_obj)
    db.commit()
    db.refresh(client_obj)

    # 2. Provision isolated filesystem directories
    workspace = Path(f"clients/{cid}")
    (workspace / "raw").mkdir(parents=True, exist_ok=True)
    (workspace / "processed").mkdir(parents=True, exist_ok=True)

    models_dir = Path(f"models/{cid}")
    models_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Successfully created client '{cid}' and provisioned workspace at {workspace}")
    return client_obj


def get_client_or_404(db: Session, client_id: str) -> ClientModel:
    """Fetches a client by client_id or raises 404."""
    cid = validate_client_id_security(client_id)
    client_obj = db.query(ClientModel).filter(ClientModel.client_id == cid).first()
    if not client_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Client '{cid}' not found."
        )
    return client_obj


def list_all_clients(db: Session) -> List[ClientModel]:
    """Returns all registered clients ordered by creation date."""
    return db.query(ClientModel).order_by(ClientModel.created_at.desc()).all()


def ensure_demo_ecommerce_registered(db: Session) -> ClientModel:
    """
    Guarantees the reference client 'demo_ecommerce' is registered on startup,
    associating existing dataset and checkpoints without duplicating files.
    """
    cid = "demo_ecommerce"
    client_obj = db.query(ClientModel).filter(ClientModel.client_id == cid).first()

    if not client_obj:
        logger.info("Registering canonical reference client: demo_ecommerce...")
        client_obj = ClientModel(
            client_id=cid,
            name="Indian E-Commerce Customer Behavior & Purchase",
            description="Canonical reference dataset and research benchmark model.",
            is_system=True
        )
        db.add(client_obj)
        db.commit()
        db.refresh(client_obj)

        # Ensure directories
        workspace = Path(f"clients/{cid}")
        (workspace / "raw").mkdir(parents=True, exist_ok=True)
        (workspace / "processed").mkdir(parents=True, exist_ok=True)
        Path(f"models/{cid}").mkdir(parents=True, exist_ok=True)

        # Register canonical schema config
        config_path = workspace / "config.yaml"
        if not config_path.exists():
            demo_schema = create_demo_ecommerce_schema()
            demo_schema.to_yaml(config_path)

        # Register schema in DB
        schema_record = SchemaMappingModel(
            client_id=cid,
            mappings_json=json.dumps(create_demo_ecommerce_schema().column_mappings),
            types_json=json.dumps(create_demo_ecommerce_schema().column_types),
            capabilities_json=json.dumps(create_demo_ecommerce_schema().get_capabilities()),
            is_valid=True,
            version=1
        )
        db.add(schema_record)

        # Register reference dataset record referencing data/raw/Ecommerce.csv
        ref_raw = Path("data/raw/Ecommerce.csv")
        if ref_raw.exists():
            dataset_record = DatasetModel(
                client_id=cid,
                filename="Ecommerce.csv",
                file_path=str(ref_raw),
                file_size_bytes=ref_raw.stat().st_size,
                row_count=25000,
                column_count=29
            )
            db.add(dataset_record)

        db.commit()
        db.refresh(client_obj)

        # Register immutable model version v1 in the registry
        from backend.services.model_service import ensure_demo_ecommerce_model_registered
        ensure_demo_ecommerce_model_registered(db)
        logger.info("Canonical demo_ecommerce client registered successfully.")
    else:
        if not getattr(client_obj, "is_system", False):
            client_obj.is_system = True
            db.commit()
        # Client exists: verify model registry v1 is registered
        from backend.services.model_service import ensure_demo_ecommerce_model_registered
        ensure_demo_ecommerce_model_registered(db)

    # Clean up any legacy placeholder tags if present
    legacy_old = db.query(ModelVersionModel).filter(
        ModelVersionModel.client_id == cid,
        ModelVersionModel.version_tag == "v1.0.0-canonical"
    ).all()
    if legacy_old:
        for old in legacy_old:
            db.delete(old)
        db.commit()

    return client_obj


def delete_client(db: Session, client_id: str, delete_physical_files: bool = False) -> dict:
    """
    Safely deletes a client registration and its dependent records from the platform.
    Guarantees atomic deletion and blocks deletion of protected reference clients
    or clients with active training runs.
    """
    cid = validate_client_id_security(client_id)

    client_obj = db.query(ClientModel).filter(ClientModel.client_id == cid).first()
    if not client_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Client '{cid}' not found."
        )

    # 1. System protection rule: block deletion of reference clients
    if getattr(client_obj, "is_system", False) or cid == "demo_ecommerce":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Client '{cid}' is a protected system reference client and cannot be deleted."
        )

    # 2. Concurrency/Job protection rule: block deletion if training run is currently active
    active_statuses = ["NOT_STARTED", "PREPROCESSING", "TRAINING", "EVALUATING"]
    active_run = (
        db.query(TrainingRunModel)
        .filter(TrainingRunModel.client_id == cid, TrainingRunModel.status.in_(active_statuses))
        .first()
    )
    if active_run:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Cannot delete client '{cid}' while a training run is active "
                f"(status: {active_run.status}, run_id: {active_run.id}). "
                f"Please wait for training to complete or terminate before deleting."
            )
        )

    # 3. Invalidate in-memory model cache
    from backend.services.model_cache import invalidate_client_cache
    invalidate_client_cache(cid)

    # 4. Atomic database cleanup of client and all dependent records
    try:
        feedback_count = db.query(FeedbackEventModel).filter(FeedbackEventModel.client_id == cid).delete(synchronize_session=False)
        rec_count = db.query(RecommendationEventModel).filter(RecommendationEventModel.client_id == cid).delete(synchronize_session=False)
        model_count = db.query(ModelVersionModel).filter(ModelVersionModel.client_id == cid).delete(synchronize_session=False)
        training_count = db.query(TrainingRunModel).filter(TrainingRunModel.client_id == cid).delete(synchronize_session=False)
        schema_count = db.query(SchemaMappingModel).filter(SchemaMappingModel.client_id == cid).delete(synchronize_session=False)
        dataset_count = db.query(DatasetModel).filter(DatasetModel.client_id == cid).delete(synchronize_session=False)

        db.delete(client_obj)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to delete client '{cid}' from database: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while deleting client '{cid}': {str(e)}"
        )

    # 5. Handle physical file removal if explicitly requested
    physical_deleted = False
    if delete_physical_files:
        try:
            # Client data directory validation and cleanup
            client_dir = (Path("clients") / cid).resolve()
            clients_root = Path("clients").resolve()
            if str(client_dir).startswith(str(clients_root)) and client_dir != clients_root:
                if client_dir.exists():
                    shutil.rmtree(client_dir)
                    physical_deleted = True

            # Model checkpoints directory validation and cleanup
            model_dir = (Path("models") / cid).resolve()
            models_root = Path("models").resolve()
            if str(model_dir).startswith(str(models_root)) and model_dir != models_root:
                if model_dir.exists():
                    shutil.rmtree(model_dir)
                    physical_deleted = True
        except Exception as e:
            logger.warning(f"Error removing physical files for client '{cid}': {e}")

    logger.info(
        f"Client '{cid}' deleted successfully. DB records removed: "
        f"datasets={dataset_count}, schemas={schema_count}, runs={training_count}, "
        f"models={model_count}, recs={rec_count}, feedback={feedback_count}. "
        f"Physical files deleted: {physical_deleted}"
    )

    return {
        "client_id": cid,
        "message": f"Client '{cid}' successfully deleted.",
        "deleted_records": {
            "datasets": dataset_count,
            "schema_mappings": schema_count,
            "training_runs": training_count,
            "model_versions": model_count,
            "recommendation_events": rec_count,
            "feedback_events": feedback_count
        },
        "physical_files_deleted": physical_deleted
    }
