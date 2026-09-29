"""
Pydantic schemas for schema mapping and validation requests/responses.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class SchemaUpdateRequest(BaseModel):
    column_mappings: Dict[str, str] = Field(..., description="Map of column name to semantic SchemaRole")
    column_types: Optional[Dict[str, str]] = Field(default_factory=dict, description="Map of column name to DataType")
    timestamp_format: Optional[str] = Field("%d-%m-%Y", description="Date format string if timestamp is present")
    outlier_columns: Optional[List[str]] = Field(default_factory=list, description="Columns to apply IQR outlier capping")
    cold_start_threshold: Optional[int] = Field(3, ge=1, le=20, description="Primary cold-start threshold K")


class SchemaValidationRequest(BaseModel):
    column_mappings: Dict[str, str] = Field(..., description="Map of column name to semantic SchemaRole")


class SchemaValidationResponse(BaseModel):
    valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    capabilities: Dict[str, bool] = {}
    detected_roles: Dict[str, str] = {}


class SchemaResponse(BaseModel):
    client_id: str
    column_mappings: Dict[str, str]
    column_types: Dict[str, str]
    capabilities: Dict[str, bool]
    is_valid: bool
    timestamp_format: Optional[str] = "%d-%m-%Y"
    outlier_columns: List[str] = []
    cold_start_threshold: int = 3
    version: int = 1
