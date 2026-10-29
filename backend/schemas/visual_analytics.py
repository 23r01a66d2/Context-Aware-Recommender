"""
Pydantic Schemas & DTOs for Visual Analytics.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class VisualDatasetSummary(BaseModel):
    id: str
    name: str
    filename: str
    client_id: Optional[str] = None
    is_canonical: bool = False
    size_bytes: int = 0
    size_mb: float = 0.0
    rows: Optional[int] = None
    columns: Optional[int] = None
    description: Optional[str] = None


class ChartDataRequest(BaseModel):
    dataset_id: str = Field(..., description="Dataset ID or filename")
    client_id: Optional[str] = Field(None, description="Optional associated client")
    chart_type: str = Field(..., description="bar, line, histogram, scatter, heatmap")
    x_column: str = Field(..., description="Column for X axis / primary dimension")
    y_column: Optional[str] = Field(None, description="Optional column for Y axis / metric")
    aggregation: Optional[str] = Field("count", description="Aggregation: count, sum, mean, min, max")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional column filters")


class VisualRecommendRequest(BaseModel):
    dataset_id: str = Field(..., description="Dataset ID or filename")
    client_id: Optional[str] = Field(None, description="Client ID if model-assisted personalization is desired")
    user_id: Optional[str] = Field(None, description="Optional customer/user ID for behavioral profile lookup")
    model_type: Optional[str] = Field(None, description="Recommendation model: 'mlvis', 'context_aware', 'statistical', or 'custom'")
    session_context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Session context: current_chart_type, selected_columns, etc.")
    top_k: Optional[int] = Field(6, ge=1, le=20, description="Top K recommendations to return")


class VisualInteractionRequest(BaseModel):
    session_id: str = Field(..., description="Client session identifier")
    event_type: str = Field(..., description="chart_impression, recommendation_click, chart_selection, filter_change, chart_view, helpful_feedback, unhelpful_feedback")
    client_id: Optional[str] = Field(None, description="Associated client ID")
    user_id: Optional[str] = Field(None, description="Optional user ID")
    chart_id: Optional[str] = Field(None, description="Identifier of candidate or chart")
    chart_type: Optional[str] = Field(None, description="bar, scatter, histogram, heatmap, line")
    recommendation_id: Optional[str] = Field(None, description="Optional recommendation session ID")
    details: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Arbitrary event metadata")


class VisualInteractionResponse(BaseModel):
    id: int
    session_id: str
    event_type: str
    status: str = "recorded"
