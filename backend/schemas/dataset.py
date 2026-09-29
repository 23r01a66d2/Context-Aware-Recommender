"""
Pydantic schemas for dataset operations and inspection.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class ColumnMetadata(BaseModel):
    name: str
    dtype: str
    null_count: int
    unique_count: int
    sample_values: List[Any] = []


class DatasetUploadResponse(BaseModel):
    client_id: str
    filename: str
    file_size_bytes: int
    rows: int
    columns: int
    message: str


class DatasetPreviewResponse(BaseModel):
    client_id: str
    filename: str
    total_rows: int
    total_columns: int
    preview_limit: int
    columns: List[ColumnMetadata]
    sample_data: List[Dict[str, Any]]
