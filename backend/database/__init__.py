"""
Database layer with SQLAlchemy ORM abstraction.
"""
from backend.database.database import Base, engine, SessionLocal, get_db, init_db
from backend.database.models import (
    ClientModel,
    DatasetModel,
    SchemaMappingModel,
    TrainingRunModel,
    ModelVersionModel,
    RecommendationEventModel,
    FeedbackEventModel
)

__all__ = [
    "Base",
    "engine",
    "SessionLocal",
    "get_db",
    "init_db",
    "ClientModel",
    "DatasetModel",
    "SchemaMappingModel",
    "TrainingRunModel",
    "ModelVersionModel",
    "RecommendationEventModel",
    "FeedbackEventModel"
]
