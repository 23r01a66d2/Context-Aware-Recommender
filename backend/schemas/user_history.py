"""
Pydantic Schemas for Persistent User History & Cold/Warm Start Detection.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class UserStatusResponse(BaseModel):
    client_id: str
    user_id: str
    status: str = Field(..., description="'cold_start' if 0 interactions, otherwise 'warm_start'")
    is_cold_start: bool
    history_count: int
    last_active: Optional[str] = None
    recent_datasets: List[str] = Field(default_factory=list)
    created_at: Optional[str] = None


class UserHistoryRecord(BaseModel):
    event_id: str
    entry_type: str
    timestamp: str
    dataset_id: Optional[str] = None
    session_id: Optional[str] = None
    items_or_charts: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    feedback: Optional[str] = None


class UserHistoryResponse(BaseModel):
    client_id: str
    user_id: str
    status: str
    total_records: int
    limit: int
    offset: int
    history: List[Dict[str, Any]] = Field(default_factory=list)


class RecordUserHistoryRequest(BaseModel):
    client_id: str = Field(..., description="Tenant / Client identifier")
    user_id: str = Field(..., description="User identifier")
    entry_type: str = Field(..., description="Type of event (recommendation, chart_selection, product_view, feedback, upload, etc.)")
    dataset_id: Optional[str] = Field(None, description="Dataset identifier or filename")
    session_id: Optional[str] = Field(None, description="Client session ID")
    items_or_charts: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Metadata of items or charts recommended / selected")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context metadata")
    feedback: Optional[str] = Field(None, description="Optional user feedback string or rating")


class RecordUserHistoryResponse(BaseModel):
    status: str
    event_id: str
    client_id: str
    user_id: str
    total_history_count: int
    start_status: str
