"""
Schema Mapping & Validation API Endpoints.
"""

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from backend.database.database import get_db
from backend.schemas.schema_map import (
    SchemaResponse,
    SchemaUpdateRequest,
    SchemaValidationRequest,
    SchemaValidationResponse
)
from backend.services.schema_service import (
    get_or_suggest_schema_for_client,
    save_schema_for_client,
    validate_client_schema
)

router = APIRouter(prefix="/clients/{client_id}", tags=["Schema Mapping"])


@router.get("/schema", response_model=SchemaResponse)
def api_get_schema(client_id: str, db: Session = Depends(get_db)):
    """
    Get client's schema mapping.
    If none is confirmed, automatically suggests role mappings based on dataset columns.
    """
    return get_or_suggest_schema_for_client(db, client_id)


@router.put("/schema", response_model=SchemaResponse)
@router.post("/schema", response_model=SchemaResponse)
def api_save_schema(
    client_id: str,
    schema_in: SchemaUpdateRequest,
    db: Session = Depends(get_db)
):
    """
    Validate, save, and persist schema mappings for the client.
    Supports both PUT and POST HTTP verbs for frontend and API compatibility.
    """
    return save_schema_for_client(db, client_id, schema_in)


@router.post("/validate", response_model=SchemaValidationResponse)
def api_validate_schema(
    client_id: str,
    validation_in: SchemaValidationRequest
):
    """
    Validate schema mappings against platform policies without persisting changes.
    Returns validity, errors, warnings, capabilities, and detected roles.
    """
    res = validate_client_schema(client_id, validation_in.column_mappings)
    return SchemaValidationResponse(
        valid=res["valid"],
        errors=res["errors"],
        warnings=res["warnings"],
        capabilities=res["capabilities"],
        detected_roles=res["detected_roles"]
    )
