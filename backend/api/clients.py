"""
Client Management API Endpoints.
"""

from pathlib import Path
from typing import List
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from backend.database.database import get_db
from backend.database.models import ClientModel, DatasetModel, SchemaMappingModel, ModelVersionModel
from backend.schemas.client import (
    ClientCreate,
    ClientResponse,
    ClientListResponse,
    ClientDeleteResponse,
    ClientOverviewListResponse
)
from backend.services.client_service import (
    create_client,
    get_client_or_404,
    list_all_clients,
    delete_client,
    get_clients_overview,
    evaluate_client_inference_readiness
)

router = APIRouter(prefix="/clients", tags=["Clients"])


def _to_client_response(client: ClientModel, db: Session) -> ClientResponse:
    dataset_count = db.query(DatasetModel).filter(DatasetModel.client_id == client.client_id).count()
    has_schema = (
        db.query(SchemaMappingModel).filter(SchemaMappingModel.client_id == client.client_id).first() is not None
        or Path(f"clients/{client.client_id}/config.yaml").exists()
    )
    active_model = (
        db.query(ModelVersionModel)
        .filter(ModelVersionModel.client_id == client.client_id, ModelVersionModel.is_active == True)
        .first()
    )
    active_version = active_model.version_tag if active_model else None
    has_model = (
        active_version is not None
        or db.query(ModelVersionModel).filter(ModelVersionModel.client_id == client.client_id).first() is not None
        or (client.client_id == "demo_ecommerce" and Path("models/best_multimodal_model.pt").exists())
    )
    if not active_version and client.client_id == "demo_ecommerce" and has_model:
        active_version = "v1"

    is_system = bool(getattr(client, "is_system", False) or client.client_id == "demo_ecommerce")

    readiness = evaluate_client_inference_readiness(client.client_id, db, active_version)

    return ClientResponse(
        client_id=client.client_id,
        name=client.name,
        description=client.description or "",
        is_system=is_system,
        created_at=client.created_at,
        dataset_count=dataset_count,
        has_schema=has_schema,
        has_model=has_model,
        active_model_version=active_version,
        dataset_available=readiness["dataset_available"],
        model_available=readiness["model_available"],
        candidate_data_available=readiness["candidate_data_available"],
        features_contract_valid=readiness["features_contract_valid"],
        is_inference_ready=readiness["is_inference_ready"],
        status=readiness["status"],
        not_ready_reason=readiness["not_ready_reason"],
        readiness_category=readiness.get("readiness_category", "Ready for Inference"),
        action_required=readiness.get("action_required")
    )


@router.post("", response_model=ClientResponse, status_code=status.HTTP_201_CREATED)
def api_create_client(client_in: ClientCreate, db: Session = Depends(get_db)):
    """Create a new client and provision its isolated workspace."""
    client_obj = create_client(db, client_in)
    return _to_client_response(client_obj, db)


@router.get("/overview", response_model=ClientOverviewListResponse)
def api_get_clients_overview(
    include_diagnostic: bool = Query(False, description="Include diagnostic test clients"),
    db: Session = Depends(get_db)
):
    """
    Returns available datasets, recommendation domains, user/item statistics,
    available models, contextual capabilities, and implicit feedback support.
    """
    overviews = get_clients_overview(db, include_diagnostic=include_diagnostic)
    return ClientOverviewListResponse(domains=overviews, total=len(overviews))


@router.get("", response_model=ClientListResponse)
def api_list_clients(db: Session = Depends(get_db)):
    """List all registered clients with metadata summary."""
    clients = list_all_clients(db)
    responses = [_to_client_response(c, db) for c in clients]
    return ClientListResponse(clients=responses, total=len(responses))


@router.get("/{client_id}", response_model=ClientResponse)
def api_get_client(client_id: str, db: Session = Depends(get_db)):
    """Get metadata for a specific client."""
    client_obj = get_client_or_404(db, client_id)
    return _to_client_response(client_obj, db)


@router.delete("/{client_id}", response_model=ClientDeleteResponse)
def api_delete_client(
    client_id: str,
    delete_physical_files: bool = Query(
        False,
        description="Whether to also permanently delete client folder and model checkpoints from disk"
    ),
    db: Session = Depends(get_db)
):
    """Safely delete a client registration and its dependent platform/database records."""
    return delete_client(db=db, client_id=client_id, delete_physical_files=delete_physical_files)
