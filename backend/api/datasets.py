"""
Dataset Upload & Preview API Endpoints.
"""

from fastapi import APIRouter, Depends, UploadFile, File, Query, status
from sqlalchemy.orm import Session

from backend.database.database import get_db
from backend.schemas.dataset import DatasetUploadResponse, DatasetPreviewResponse
from backend.services.dataset_service import save_uploaded_dataset, get_dataset_preview_for_client

router = APIRouter(prefix="/clients/{client_id}/dataset", tags=["Datasets"])


@router.post("", response_model=DatasetUploadResponse, status_code=status.HTTP_201_CREATED)
def api_upload_dataset(
    client_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload a CSV dataset for the specified client.
    Enforces format validation, size checks, path traversal protection, and metadata inspection.
    """
    dataset_obj = save_uploaded_dataset(db, client_id, file)
    return DatasetUploadResponse(
        client_id=client_id,
        filename=dataset_obj.filename,
        file_size_bytes=dataset_obj.file_size_bytes,
        rows=dataset_obj.row_count,
        columns=dataset_obj.column_count,
        message="Dataset uploaded and validated successfully."
    )


@router.get("/preview", response_model=DatasetPreviewResponse)
def api_get_dataset_preview(
    client_id: str,
    limit: int = Query(10, ge=1, le=50, description="Number of sample rows (max 50)"),
    db: Session = Depends(get_db)
):
    """
    Preview the client's dataset: returns dimensions, column data types, null counts, and first N rows.
    """
    return get_dataset_preview_for_client(db, client_id, limit=limit)
