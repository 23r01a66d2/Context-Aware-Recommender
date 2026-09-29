"""
Model Registry API Endpoints.
Provides sanitized access to versioned model artifacts, metadata, and activation.
Does NOT expose filesystem paths to clients.
"""

from typing import List
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from backend.database.database import get_db
from backend.schemas.model import (
    ModelVersionSummaryResponse,
    ModelVersionDetailResponse,
    ModelActivationResponse
)
from backend.services.model_service import (
    list_client_models,
    get_client_model_version,
    activate_client_model
)

router = APIRouter(prefix="/clients/{client_id}/models", tags=["Model Registry"])


@router.get("", response_model=List[ModelVersionSummaryResponse])
def api_list_models(
    client_id: str,
    db: Session = Depends(get_db)
):
    """
    List all registered model versions for a client.
    Does not expose server filesystem paths.
    """
    return list_client_models(db, client_id)


@router.get("/{version}", response_model=ModelVersionDetailResponse)
def api_get_model_version(
    client_id: str,
    version: str,
    db: Session = Depends(get_db)
):
    """
    Retrieve metadata, feature dimensions, and evaluation metrics for a specific model version.
    Returns 404 if version does not exist or does not belong to this client.
    """
    return get_client_model_version(db, client_id, version)


@router.post("/{version}/activate", response_model=ModelActivationResponse)
def api_activate_model(
    client_id: str,
    version: str,
    db: Session = Depends(get_db)
):
    """
    Activate a completed model version for this client.
    Deactivates any previously active version for this client only.
    Other clients remain completely unaffected.
    """
    return activate_client_model(db, client_id, version)
