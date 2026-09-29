import json
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, status, BackgroundTasks
from sqlalchemy.orm import Session

from backend.database.database import get_db
from backend.database.models import TrainingRunModel
from backend.schemas.training import (
    TrainingTriggerRequest,
    TrainingStatusResponse,
    TrainingStatusEnum
)
from backend.services.client_service import get_client_or_404
from backend.services.training_service import trigger_training_run

router = APIRouter(prefix="/clients/{client_id}", tags=["Training"])


@router.post("/train", response_model=TrainingStatusResponse, status_code=status.HTTP_202_ACCEPTED)
def api_trigger_training(
    client_id: str,
    train_in: TrainingTriggerRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Trigger a new background training run for the client.
    Validates prerequisites and returns 202 Accepted without blocking HTTP connection.
    """
    return trigger_training_run(
        db=db,
        background_tasks=background_tasks,
        client_id=client_id,
        request=train_in
    )


@router.get("/training-status", response_model=TrainingStatusResponse)
def api_get_training_status(
    client_id: str,
    db: Session = Depends(get_db)
):
    """
    Get the latest real training status for a client.
    Returns real status (NOT_STARTED, PREPROCESSING, TRAINING, EVALUATING, COMPLETED, FAILED),
    current epoch, total epochs, and actual persisted loss/metrics.
    """
    client = get_client_or_404(db, client_id)
    cid = client.client_id

    latest_run = (
        db.query(TrainingRunModel)
        .filter(TrainingRunModel.client_id == cid)
        .order_by(TrainingRunModel.created_at.desc())
        .first()
    )

    if not latest_run:
        now = datetime.now(timezone.utc)
        return TrainingStatusResponse(
            client_id=cid,
            run_id="none",
            status=TrainingStatusEnum.NOT_STARTED,
            current_epoch=0,
            total_epochs=0,
            error_message=None,
            metrics=None,
            created_at=now,
            completed_at=None
        )

    metrics = None
    if latest_run.metrics_json:
        try:
            metrics = json.loads(latest_run.metrics_json)
        except Exception:
            metrics = None

    return TrainingStatusResponse(
        client_id=cid,
        run_id=latest_run.id,
        status=TrainingStatusEnum(latest_run.status),
        current_epoch=latest_run.current_epoch,
        total_epochs=latest_run.total_epochs,
        error_message=latest_run.error_message,
        metrics=metrics,
        created_at=latest_run.created_at,
        completed_at=latest_run.completed_at
    )
