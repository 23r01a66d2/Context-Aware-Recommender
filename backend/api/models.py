"""
Model Registry API Endpoints.
Provides sanitized access to versioned model artifacts, metadata, and activation.
Does NOT expose filesystem paths to clients.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, status, Query
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
    activate_client_model,
    list_all_models_global,
    get_model_analytics_summary
)

router = APIRouter(tags=["Model Registry"])


@router.get("/models", response_model=List[Dict[str, Any]])
def api_list_all_models(db: Session = Depends(get_db)):
    """
    List all registered models across all clients with architecture,
    modalities, metrics, artifact existence, and activation status.
    """
    return list_all_models_global(db)


@router.get("/models/analytics", response_model=Dict[str, Any])
@router.get("/analytics", response_model=Dict[str, Any])
def api_get_model_analytics(
    client_id: Optional[str] = Query(None, description="Client/tenant identifier to scope analytics"),
    db: Session = Depends(get_db)
):
    """
    Retrieve comprehensive dynamic comparative ranking metrics,
    modality ablation contributions, warm vs cold cohorts, and loss curves.
    """
    return get_model_analytics_summary(client_id=client_id, db=db)


@router.get("/clients/{client_id}/analytics", response_model=Dict[str, Any])
def api_get_client_analytics(
    client_id: str,
    db: Session = Depends(get_db)
):
    """
    Retrieve model benchmark analytics scoped strictly to the specified client/tenant.
    """
    return get_model_analytics_summary(client_id=client_id, db=db)


@router.get("/clients/{client_id}/models", response_model=List[ModelVersionSummaryResponse])
def api_list_models(
    client_id: str,
    db: Session = Depends(get_db)
):
    """
    List all registered model versions for a client.
    Does not expose server filesystem paths.
    """
    return list_client_models(db, client_id)


@router.get("/clients/{client_id}/models/{version}", response_model=ModelVersionDetailResponse)
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


@router.post("/clients/{client_id}/models/{version}/activate", response_model=ModelActivationResponse)
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
