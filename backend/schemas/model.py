"""
Pydantic contracts for Model Registry API requests and responses.
Strictly sanitized: internal filesystem paths are never exposed to clients.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class ModelVersionSummaryResponse(BaseModel):
    client_id: str
    version: str = Field(..., description="Semantic version tag, e.g. v1, v2")
    is_active: bool = Field(..., description="Whether this version is currently the serving model")
    legacy_imported: bool = Field(default=False, description="Flag indicating model was imported from baseline")
    feature_dimensions: Optional[Dict[str, int]] = Field(None, description="Modality feature vector dimensions")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Overall and cohort validation/test metrics")
    training_run_id: Optional[str] = Field(None, description="Associated training run identifier")
    created_at: datetime


class ModelVersionDetailResponse(BaseModel):
    client_id: str
    version: str
    is_active: bool
    legacy_imported: bool = False
    feature_dimensions: Optional[Dict[str, int]] = None
    metrics: Optional[Dict[str, Any]] = None
    training_run_id: Optional[str] = None
    cold_start_threshold: int = 3
    random_seed: int = 42
    dataset_reference: Optional[str] = None
    schema_snapshot: Optional[Dict[str, Any]] = None
    created_at: datetime


class ModelActivationResponse(BaseModel):
    client_id: str
    version: str
    is_active: bool
    message: str
