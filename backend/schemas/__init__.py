"""
Pydantic data validation schemas for API request and response contracts.
"""
from backend.schemas.client import ClientCreate, ClientResponse, ClientListResponse
from backend.schemas.dataset import DatasetPreviewResponse, DatasetUploadResponse
from backend.schemas.schema_map import (
    SchemaUpdateRequest,
    SchemaValidationRequest,
    SchemaValidationResponse,
    SchemaResponse
)
from backend.schemas.training import TrainingTriggerRequest, TrainingStatusResponse
from backend.schemas.recommendation import (
    RecommendationRequest,
    RecommendationResponse,
    RecommendedItem,
    RecommendationFilter
)
from backend.schemas.feedback import FeedbackCreateRequest, FeedbackResponse, FeedbackActionEnum

from backend.schemas.model import (
    ModelVersionSummaryResponse,
    ModelVersionDetailResponse,
    ModelActivationResponse
)

__all__ = [
    "ClientCreate",
    "ClientResponse",
    "ClientListResponse",
    "DatasetPreviewResponse",
    "DatasetUploadResponse",
    "SchemaUpdateRequest",
    "SchemaValidationRequest",
    "SchemaValidationResponse",
    "SchemaResponse",
    "TrainingTriggerRequest",
    "TrainingStatusResponse",
    "RecommendationRequest",
    "RecommendationResponse",
    "RecommendedItem",
    "RecommendationFilter",
    "FeedbackCreateRequest",
    "FeedbackResponse",
    "FeedbackActionEnum",
    "ModelVersionSummaryResponse",
    "ModelVersionDetailResponse",
    "ModelActivationResponse"
]
