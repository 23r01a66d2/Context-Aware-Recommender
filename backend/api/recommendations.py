"""
Real-Time Recommendation API Endpoints.
Performs live online candidate scoring using the active client model.
"""

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from backend.database.database import get_db
from backend.schemas.recommendation import (
    RecommendationRequest,
    RecommendationResponse
)
from backend.services.inference_service import generate_recommendations

router = APIRouter(prefix="/recommend", tags=["Recommendations"])


@router.post("", response_model=RecommendationResponse, status_code=status.HTTP_200_OK)
def api_get_recommendations(
    request: RecommendationRequest,
    db: Session = Depends(get_db)
):
    """
    Real-time multi-modal recommendation inference endpoint.
    Retrieves active model, scores candidate items from client catalog,
    applies filters, adapts for cold/warm users, generates grounded explanations,
    and returns ranked Top-K items.
    """
    return generate_recommendations(request, db)
