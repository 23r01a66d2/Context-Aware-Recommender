"""
Service for secure dataset upload, inspection, and preview generation.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from fastapi import UploadFile, HTTPException, status
from sqlalchemy.orm import Session

from backend.database.models import DatasetModel
from backend.schemas.dataset import DatasetPreviewResponse, ColumnMetadata
from backend.services.client_service import get_client_or_404, validate_client_id_security
from src.schema_mapper import infer_column_type

logger = logging.getLogger("DatasetService")

# Maximum upload size: 50MB
MAX_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024
SAFE_FILENAME_REGEX = re.compile(r"^[a-zA-Z0-9_.-]+$")


def sanitize_filename(filename: str) -> str:
    """Sanitizes filename against path traversal and dangerous characters."""
    clean_name = os.path.basename(filename).strip()
    if not clean_name.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Only CSV files (.csv) are accepted."
        )
    # Strip any characters outside alphanumeric, dot, hyphen, underscore
    sanitized = re.sub(r"[^a-zA-Z0-9_.-]", "_", clean_name)
    if ".." in sanitized:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Path traversal sequences are forbidden in filename."
        )
    return sanitized


def save_uploaded_dataset(db: Session, client_id: str, file: UploadFile) -> DatasetModel:
    """
    Validates, securely writes, and inspects an uploaded CSV dataset for a client.
    """
    client = get_client_or_404(db, client_id)
    cid = client.client_id

    # 1. Filename & format validation
    orig_name = file.filename or "dataset.csv"
    safe_name = sanitize_filename(orig_name)

    # 2. Destination path inside isolated client raw directory
    client_raw_dir = Path(f"clients/{cid}/raw")
    client_raw_dir.mkdir(parents=True, exist_ok=True)
    target_path = client_raw_dir / safe_name

    # 3. Read content in chunks to check size and write
    file_size = 0
    with open(target_path, "wb") as f_out:
        while True:
            chunk = file.file.read(1024 * 1024) # 1MB chunks
            if not chunk:
                break
            file_size += len(chunk)
            if file_size > MAX_UPLOAD_SIZE_BYTES:
                # Cleanup partial file
                target_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File exceeds maximum allowed size of {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)}MB."
                )
            f_out.write(chunk)

    if file_size == 0:
        target_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty."
        )

    # 4. Inspect dataset
    try:
        df = pd.read_csv(target_path, nrows=5000) # Read sample or full for metadata
    except Exception as e:
        target_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse CSV file: {str(e)}"
        )

    row_count = len(df)
    col_count = len(df.columns)

    if col_count == 0 or row_count == 0:
        target_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset contains no usable rows or columns."
        )

    # Inspect columns
    col_metadata = []
    for col in df.columns:
        s = df[col]
        col_type = infer_column_type(s)
        samples = [str(x) for x in s.dropna().head(3).tolist()]
        col_metadata.append({
            "name": str(col),
            "dtype": col_type,
            "null_count": int(s.isnull().sum()),
            "unique_count": int(s.nunique()),
            "sample_values": samples
        })

    # Sample rows for fast preview
    sample_records = df.head(10).fillna("").to_dict(orient="records")

    # 5. Persist record in database
    dataset_record = DatasetModel(
        client_id=cid,
        filename=safe_name,
        file_path=str(target_path),
        file_size_bytes=file_size,
        row_count=row_count,
        column_count=col_count,
        columns_json=json.dumps(col_metadata),
        sample_json=json.dumps(sample_records)
    )
    db.add(dataset_record)
    db.commit()
    db.refresh(dataset_record)

    logger.info(f"Dataset '{safe_name}' uploaded for client '{cid}': {row_count} rows, {col_count} columns.")
    return dataset_record


def get_dataset_preview_for_client(db: Session, client_id: str, limit: int = 10) -> DatasetPreviewResponse:
    """
    Returns column metadata, dimensions, and first N rows of client's dataset.
    """
    client = get_client_or_404(db, client_id)
    cid = client.client_id
    limit = min(max(1, limit), 50)  # Safe cap between 1 and 50 rows

    # Find dataset record
    dataset_record = db.query(DatasetModel).filter(DatasetModel.client_id == cid).order_by(DatasetModel.created_at.desc()).first()

    if dataset_record and Path(dataset_record.file_path).exists():
        file_path = Path(dataset_record.file_path)
        filename = dataset_record.filename
    elif cid == "demo_ecommerce" and Path("data/raw/Ecommerce.csv").exists():
        file_path = Path("data/raw/Ecommerce.csv")
        filename = "Ecommerce.csv"
    else:
        # Check client raw directory
        raw_files = list(Path(f"clients/{cid}/raw").glob("*.csv"))
        if raw_files:
            file_path = raw_files[0]
            filename = file_path.name
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No dataset found for client '{cid}'. Please upload a dataset first."
            )

    try:
        df_preview = pd.read_csv(file_path, nrows=limit)
        # Determine total rows safely without loading huge file
        total_rows = dataset_record.row_count if dataset_record else len(df_preview)
        total_cols = len(df_preview.columns)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reading dataset preview: {str(e)}"
        )

    # Column metadata
    columns: List[ColumnMetadata] = []
    for col in df_preview.columns:
        s = df_preview[col]
        dtype = infer_column_type(s)
        samples = [str(x) for x in s.dropna().head(3).tolist()]
        columns.append(ColumnMetadata(
            name=str(col),
            dtype=dtype,
            null_count=int(s.isnull().sum()),
            unique_count=int(s.nunique()),
            sample_values=samples
        ))

    sample_data = df_preview.fillna("").to_dict(orient="records")

    return DatasetPreviewResponse(
        client_id=cid,
        filename=filename,
        total_rows=total_rows,
        total_columns=total_cols,
        preview_limit=len(sample_data),
        columns=columns,
        sample_data=sample_data
    )
