"""
Service for client management, isolation enforcement, and demo_ecommerce bootstrapping.
"""

import re
import json
import logging
from pathlib import Path
from typing import List, Optional
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from backend.database.models import (
    ClientModel,
    DatasetModel,
    SchemaMappingModel,
    ModelVersionModel,
    TrainingRunModel
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
            description="Canonical reference dataset and research benchmark model."
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
