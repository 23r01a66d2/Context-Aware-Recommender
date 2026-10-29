"""
Pydantic contracts for Feedback logging requests and responses.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class FeedbackActionEnum(str, Enum):
    IMPRESSION = "IMPRESSION"
    VIEW = "VIEW"
    LONG_VIEW = "LONG_VIEW"
    CLICK = "CLICK"
    IGNORE = "IGNORE"
    ADD_TO_CART = "ADD_TO_CART"
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    PURCHASE = "PURCHASE"


DEFAULT_FEEDBACK_WEIGHTS = {
    "PURCHASE": 5.0,
    "ADD_TO_CART": 4.0,
    "CLICK": 2.0,
    "ACCEPT": 1.5,
    "LONG_VIEW": 1.0,
    "VIEW": 0.5,
    "IMPRESSION": 0.0,
    "IGNORE": -0.1,
    "REJECT": -3.0,
}


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
