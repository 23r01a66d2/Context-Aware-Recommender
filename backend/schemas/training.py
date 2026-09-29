"""
Pydantic schemas for training trigger and status contracts.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class TrainingStatusEnum(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    PREPROCESSING = "PREPROCESSING"
    TRAINING = "TRAINING"
    EVALUATING = "EVALUATING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class TrainingTriggerRequest(BaseModel):
    epochs: Optional[int] = Field(15, ge=1, le=100)
    batch_size: Optional[int] = Field(256, ge=16, le=2048)
    learning_rate: Optional[float] = Field(0.001, gt=0.0)


class TrainingStatusResponse(BaseModel):
    client_id: str
    run_id: str
    status: TrainingStatusEnum
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
