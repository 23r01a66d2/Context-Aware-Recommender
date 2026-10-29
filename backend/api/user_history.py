"""
API Endpoints for User History, Cold Start vs Warm Start Detection,
and Interaction Tracking.
"""

from typing import Optional
from fastapi import APIRouter, Query, status, HTTPException
from backend.schemas.user_history import (
    UserStatusResponse,
    UserHistoryResponse,
    RecordUserHistoryRequest,
    RecordUserHistoryResponse
)
from backend.services.user_history_service import (
    detect_user_status,
    get_user_history,
    record_user_history_entry,
    validate_user_id
)

router = APIRouter(prefix="/user-history", tags=["User History"])


@router.get("/status", response_model=UserStatusResponse, status_code=status.HTTP_200_OK)
def api_get_user_status(
    client_id: str = Query(..., description="Client/tenant identifier"),
    user_id: str = Query(..., description="User identifier")
):
    """
    Evaluates whether a user is Cold Start (0 interactions in tenant)
    or Warm Start (>= 1 interactions in tenant).
    Validates user ID format and isolates storage by tenant.
    """
    is_valid, msg = validate_user_id(user_id)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=msg
        )
    try:
        return detect_user_status(client_id, user_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("", response_model=UserHistoryResponse, status_code=status.HTTP_200_OK)
def api_get_user_history(
    client_id: str = Query(..., description="Client/tenant identifier"),
    user_id: str = Query(..., description="User identifier"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0)
):
    """
    Retrieves the persistent interaction and recommendation history
    for the specified user in the active tenant.
    """
    is_valid, msg = validate_user_id(user_id)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=msg
        )
    try:
        return get_user_history(client_id, user_id, limit=limit, offset=offset)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/record", response_model=RecordUserHistoryResponse, status_code=status.HTTP_201_CREATED)
def api_record_user_history(req: RecordUserHistoryRequest):
    """
    Appends an interaction or recommendation event to persistent user storage.
    """
    is_valid, msg = validate_user_id(req.user_id)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=msg
        )
    try:
        return record_user_history_entry(
            client_id=req.client_id,
            user_id=req.user_id,
            entry_type=req.entry_type,
            dataset_id=req.dataset_id,
            items_or_charts=req.items_or_charts,
            metadata=req.metadata,
            feedback=req.feedback,
            session_id=req.session_id
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
