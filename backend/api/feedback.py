"""
Feedback Logging API Endpoints.
Validates client existence, feedback actions, and optional recommendation_id linkage.
"""

import json
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.database.database import get_db
from backend.database.models import FeedbackEventModel, RecommendationEventModel
from backend.schemas.feedback import FeedbackCreateRequest, FeedbackResponse
from backend.services.client_service import get_client_or_404

router = APIRouter(prefix="/feedback", tags=["Feedback"])


@router.get("", response_model=List[FeedbackResponse])
def api_get_feedback(
    client_id: Optional[str] = Query(None, description="Filter by client ID"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    limit: int = Query(500, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """
    Returns recorded feedback events for audit logging and verification.
    """
    query = db.query(FeedbackEventModel).order_by(FeedbackEventModel.created_at.desc())
    if client_id:
        query = query.filter(FeedbackEventModel.client_id == client_id.strip())
    if user_id:
        query = query.filter(FeedbackEventModel.user_id == user_id.strip())

    events = query.limit(limit).all()
    return [
        FeedbackResponse(
            id=e.id,
            client_id=e.client_id,
            user_id=e.user_id,
            item_id=e.item_id,
            action=e.action,
            recommendation_id=e.recommendation_id,
            created_at=e.created_at,
            status="recorded"
        )
        for e in events
    ]



@router.post("", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED)
def api_record_feedback(
    feedback_in: FeedbackCreateRequest,
    db: Session = Depends(get_db)
):
    """
    Records a user interaction feedback event (CLICK, ACCEPT, REJECT, PURCHASE).
    Validates client existence and optional recommendation_id linkage.
    Persists feedback metadata without triggering retraining.
    """
    client = get_client_or_404(db, feedback_in.client_id)

    rec_id = None
    if feedback_in.recommendation_id:
        rec_id_clean = feedback_in.recommendation_id.strip()
        rec_event = (
            db.query(RecommendationEventModel)
            .filter(RecommendationEventModel.id == rec_id_clean)
            .first()
        )
        if not rec_event:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Recommendation event '{rec_id_clean}' not found."
            )
        if rec_event.client_id != client.client_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Recommendation event belongs to a different client."
            )

        # Validate that item_id appeared in that recommendation
        try:
            items_list = json.loads(rec_event.recommended_items_json)
            if feedback_in.item_id.strip() not in [str(i) for i in items_list]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Item '{feedback_in.item_id}' was not in the recommended items for recommendation event '{rec_id_clean}'."
                )
        except json.JSONDecodeError:
            pass

        rec_id = rec_id_clean

    event = FeedbackEventModel(
        client_id=client.client_id,
        user_id=feedback_in.user_id.strip(),
        item_id=feedback_in.item_id.strip(),
        action=feedback_in.action.value,
        position=feedback_in.position,
        recommendation_id=rec_id
    )
    db.add(event)
    db.commit()
    db.refresh(event)

    # Automatically persist into user history
    try:
        from backend.services.user_history_service import record_user_history_entry
        record_user_history_entry(
            client_id=client.client_id,
            user_id=feedback_in.user_id.strip(),
            entry_type=f"product_feedback_{feedback_in.action.value.lower()}",
            dataset_id=f"catalog_{client.client_id}",
            items_or_charts=[{"item_id": feedback_in.item_id.strip(), "action": feedback_in.action.value}],
            metadata={"position": feedback_in.position, "recommendation_id": rec_id},
            feedback=feedback_in.action.value
        )
    except Exception as e_hist:
        import logging
        logging.getLogger("FeedbackRouter").warning(f"Failed to record feedback in user history: {e_hist}")

    return FeedbackResponse(
        id=event.id,
        client_id=event.client_id,
        user_id=event.user_id,
        item_id=event.item_id,
        action=event.action,
        recommendation_id=event.recommendation_id,
        created_at=event.created_at,
        status="recorded"
    )

