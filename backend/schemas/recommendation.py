"""
Pydantic contracts for Recommendation requests and responses.
Grounded in real-time inference engine requirements (Phase 5).
"""

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, field_validator


class RecommendationFilter(BaseModel):
    min_price: Optional[float] = Field(None, ge=0.0)
    max_price: Optional[float] = Field(None, ge=0.0)
    category: Optional[str] = None


class RecommendationRequest(BaseModel):
    client_id: str = Field(..., min_length=1, max_length=64)
    user_id: str = Field(..., min_length=1, max_length=64)
    context: Dict[str, Any] = Field(default_factory=dict, description="Session and environmental context")
    filters: Optional[Union[Dict[str, Any], RecommendationFilter]] = None
    top_k: int = Field(5, ge=1, le=50, description="Number of candidate items to return")

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, v: int) -> int:
        if v < 1 or v > 50:
            raise ValueError("top_k must be between 1 and 50")
        return v


class RecommendedItem(BaseModel):
    rank: int
    item_id: str
    score: float
    score_type: str = "sigmoid_score"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    attributes: Optional[Dict[str, Any]] = None
    explanation: str = ""


class RecommendationResponse(BaseModel):
    recommendation_id: str
    client_id: str
    model_version: str
    user_id: str
    status: str  # cold_start, warm_start
    history_count: int
    cold_start_threshold: int
    modality_weights: Dict[str, float]
    recommendations: List[RecommendedItem]
    candidate_count: int
    unsupported_candidate_count: int = 0
    latency_ms: float
