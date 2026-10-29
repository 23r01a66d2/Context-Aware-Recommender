"""
API Endpoints for Visual Analytics, Interactive Charts,
Visual Recommendations, and Interaction Logging.
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, UploadFile, File, Query, status, HTTPException
from sqlalchemy.orm import Session

from backend.database.database import get_db
from backend.schemas.visual_analytics import (
    VisualDatasetSummary,
    ChartDataRequest,
    VisualRecommendRequest,
    VisualInteractionRequest,
    VisualInteractionResponse
)
from backend.services.visual_analytics_service import (
    list_available_visual_datasets,
    get_dataset_dataframe,
    save_visual_analytics_upload,
    record_visual_interaction,
    get_visual_interactions_history,
    compute_visual_analytics_metrics
)
from src.visual_analytics import (
    profile_dataset,
    generate_chart_data,
    build_visualization_catalog,
    rank_visualization_recommendations
)

router = APIRouter(prefix="/visual-analytics", tags=["Visual Analytics"])


@router.get("/datasets", response_model=List[VisualDatasetSummary])
def api_list_visual_datasets(db: Session = Depends(get_db)):
    """
    List available datasets for visual analytics exploration,
    including the canonical Indian E-Commerce benchmark and registered client catalogs.
    """
    return list_available_visual_datasets(db)


@router.post("/upload", status_code=status.HTTP_201_CREATED)
def api_upload_visual_dataset(
    file: UploadFile = File(...),
    client_id: Optional[str] = Query(None, description="Optional associated client ID"),
    user_id: Optional[str] = Query(None, description="Optional associated user ID"),
    db: Session = Depends(get_db)
):
    """
    Upload a CSV dataset for visual exploration and recommendation.
    Enforces format validation, size constraints, and path traversal protection.
    """
    res = save_visual_analytics_upload(db, file, client_id=client_id)
    if user_id:
        try:
            from backend.services.user_history_service import record_user_history_entry
            cid = client_id or "demo_ecommerce"
            record_user_history_entry(
                client_id=cid,
                user_id=user_id.strip(),
                entry_type="dataset_upload",
                dataset_id=res.get("filename") or res.get("id"),
                metadata={"filename": res.get("filename"), "dataset_id": res.get("id"), "rows": res.get("rows")}
            )
        except Exception as e_hist:
            import logging
            logging.getLogger("VisualAnalyticsRouter").warning(f"Failed to record upload in user history: {e_hist}")
    return res



@router.get("/profile")
def api_get_dataset_profile(
    dataset_id: str = Query("demo_ecommerce", description="Dataset identifier or filename"),
    client_id: Optional[str] = Query(None, description="Optional associated client"),
    db: Session = Depends(get_db)
):
    """
    Returns dataset dimensions, column types, missingness, summary statistics,
    and correlation matrix for numerical attributes.
    """
    df, filename, path = get_dataset_dataframe(db, dataset_id, client_id)
    profile = profile_dataset(df)
    profile["dataset_id"] = dataset_id
    profile["filename"] = filename
    return profile


@router.post("/chart-data")
def api_get_chart_data(
    req: ChartDataRequest,
    db: Session = Depends(get_db)
):
    """
    Renders structured chart data (Bar, Line, Histogram, Scatter, Heatmap)
    computed directly from real dataset values with optional column filters.
    """
    df, _, _ = get_dataset_dataframe(db, req.dataset_id, req.client_id)
    try:
        data = generate_chart_data(
            df=df,
            chart_type=req.chart_type,
            x_col=req.x_column,
            y_col=req.y_column,
            aggregation=req.aggregation or "count",
            filters=req.filters
        )
        return data
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/recommend")
def api_get_visual_recommendations(
    req: VisualRecommendRequest,
    db: Session = Depends(get_db)
):
    """
    Generates ranked visualization recommendations:
    - Provides sensible cold-start recommendations using dataset characteristics and chart suitability.
    - Integrates the active Two-Tower recommendation model & multimodal fusion pipeline where supported.
    - Transparently tags recommendation scores as neural model confidence or heuristic suitability.
    """
    df, _, _ = get_dataset_dataframe(db, req.dataset_id, req.client_id)
    catalog = build_visualization_catalog(df, req.client_id)

    # Fetch recent interaction history for this user/session if available
    recent_events = []
    if req.session_context and "session_id" in req.session_context:
        recent_events = get_visual_interactions_history(
            db, session_id=req.session_context["session_id"], limit=10
        )

    recommendations = rank_visualization_recommendations(
        candidates=catalog,
        df=df,
        session_context=req.session_context,
        interaction_history=recent_events,
        user_id=req.user_id,
        client_id=req.client_id,
        db_session=db,
        top_k=req.top_k or 6,
        model_type=req.model_type
    )


    # Persist into user history if user_id is provided
    if req.user_id:
        try:
            from backend.services.user_history_service import record_user_history_entry
            cid = req.client_id or "demo_ecommerce"
            recs_data = recommendations.get("recommendations", [])
            record_user_history_entry(
                client_id=cid,
                user_id=req.user_id.strip(),
                entry_type="visual_recommendation",
                dataset_id=req.dataset_id,
                session_id=req.session_context.get("session_id") if req.session_context else None,
                items_or_charts=[
                    {
                        "candidate_id": r.get("candidate_id"),
                        "chart_type": r.get("chart_type"),
                        "title": r.get("title"),
                        "x_column": r.get("x_column"),
                        "y_column": r.get("y_column"),
                        "score": r.get("recommendation_score")
                    }
                    for r in recs_data[:req.top_k or 6]
                ],
                metadata={
                    "model_type": recommendations.get("model_type"),
                    "model_version": recommendations.get("model_version"),
                    "dataset_id": req.dataset_id
                }
            )
        except Exception as e_hist:
            import logging
            logging.getLogger("VisualAnalyticsRouter").warning(f"Failed to record visual rec in user history: {e_hist}")

    return recommendations



@router.post("/interactions", response_model=VisualInteractionResponse, status_code=status.HTTP_201_CREATED)
def api_record_interaction(
    req: VisualInteractionRequest,
    db: Session = Depends(get_db)
):
    """
    Records a user visual interaction event: chart impression, click, selection, filter change, or feedback.
    Associates the event with session and user identifiers.
    """
    ev = record_visual_interaction(
        db=db,
        session_id=req.session_id,
        event_type=req.event_type,
        client_id=req.client_id,
        user_id=req.user_id,
        chart_id=req.chart_id,
        chart_type=req.chart_type,
        recommendation_id=req.recommendation_id,
        details=req.details
    )
    # Also persist into user history if user_id is present
    if req.user_id:
        try:
            from backend.services.user_history_service import record_user_history_entry
            cid = req.client_id or "demo_ecommerce"
            record_user_history_entry(
                client_id=cid,
                user_id=req.user_id.strip(),
                entry_type=req.event_type,
                dataset_id=req.details.get("dataset_id") if req.details else None,
                session_id=req.session_id,
                items_or_charts=[{
                    "chart_id": req.chart_id,
                    "chart_type": req.chart_type,
                    "recommendation_id": req.recommendation_id
                }] if req.chart_id else [],
                metadata=req.details or {},
                feedback="helpful" if req.event_type == "helpful_feedback" else ("unhelpful" if req.event_type == "unhelpful_feedback" else None)
            )
        except Exception as e_hist:
            import logging
            logging.getLogger("VisualAnalyticsRouter").warning(f"Failed to record visual interaction in user history: {e_hist}")

    return VisualInteractionResponse(
        id=ev.id,
        session_id=ev.session_id,
        event_type=ev.event_type,
        status="recorded"
    )



@router.get("/interactions")
def api_get_interactions(
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
    client_id: Optional[str] = Query(None, description="Filter by client ID"),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db)
):
    """
    Retrieves recent visual analytics interactions for audit logging and verification.
    """
    return get_visual_interactions_history(db, session_id=session_id, client_id=client_id, limit=limit)


@router.get("/analytics")
def api_get_visual_analytics(
    client_id: Optional[str] = Query(None, description="Filter by client ID"),
    db: Session = Depends(get_db)
):
    """
    Returns empirical analytics for recommendation impressions, click-through rate (CTR),
    helpful/unhelpful feedback, chart-type distribution, and cold-start vs returning performance.
    Clearly distinguishes live measured data from offline benchmark references.
    """
    return compute_visual_analytics_metrics(db, client_id=client_id)
