"""
Pydantic contracts for Feedback logging requests and responses.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class FeedbackActionEnum(str, Enum):
    CLICK = "CLICK"
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    PURCHASE = "PURCHASE"


class FeedbackCreateRequest(BaseModel):
    client_id: str = Field(..., min_length=1, max_length=64)
    user_id: str = Field(..., min_length=1, max_length=64)
    item_id: str = Field(..., min_length=1, max_length=64)
    action: FeedbackActionEnum = Field(..., description="Interaction outcome action")
    position: Optional[int] = Field(None, ge=1, le=50, description="Rank position of the recommendation")
    recommendation_id: Optional[str] = Field(None, description="Optional ID of the recommendation event")


class FeedbackResponse(BaseModel):
    id: int
    client_id: str
    user_id: str
    item_id: str
    action: str
    recommendation_id: Optional[str] = None
    created_at: datetime
    status: str = "recorded"
