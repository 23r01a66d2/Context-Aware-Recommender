"""
SQLAlchemy ORM models for platform metadata.
Stores metadata, references, and configurations only.
Large files (CSVs, PyTorch checkpoints, tensors) reside in the filesystem.
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    Integer,
    BigInteger,
    Float,
    String,
    Text,
    Boolean,
    DateTime,
    ForeignKey
)
from sqlalchemy.orm import relationship
from backend.database.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ClientModel(Base):
    __tablename__ = "clients"

    id = Column(Integer, primary_key=True, autoincrement=True)
    client_id = Column(String(64), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, default="", nullable=True)
    created_at = Column(DateTime, default=utcnow, nullable=False)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)

    # Relationships
    datasets = relationship("DatasetModel", back_populates="client", cascade="all, delete-orphan")
    schema_mappings = relationship("SchemaMappingModel", back_populates="client", cascade="all, delete-orphan")
    training_runs = relationship("TrainingRunModel", back_populates="client", cascade="all, delete-orphan")
    model_versions = relationship("ModelVersionModel", back_populates="client", cascade="all, delete-orphan")
    recommendation_events = relationship("RecommendationEventModel", back_populates="client", cascade="all, delete-orphan")
    feedback_events = relationship("FeedbackEventModel", back_populates="client", cascade="all, delete-orphan")


class DatasetModel(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    client_id = Column(String(64), ForeignKey("clients.client_id", ondelete="CASCADE"), index=True, nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_size_bytes = Column(BigInteger, default=0, nullable=False)
    row_count = Column(Integer, default=0, nullable=False)
    column_count = Column(Integer, default=0, nullable=False)
    columns_json = Column(Text, nullable=True)  # JSON-encoded column names, inferred types, null counts
    sample_json = Column(Text, nullable=True)   # JSON-encoded sample rows for fast preview
    created_at = Column(DateTime, default=utcnow, nullable=False)

    client = relationship("ClientModel", back_populates="datasets")


class SchemaMappingModel(Base):
    __tablename__ = "schema_mappings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    client_id = Column(String(64), ForeignKey("clients.client_id", ondelete="CASCADE"), index=True, nullable=False)
    mappings_json = Column(Text, nullable=False)     # JSON: column_name -> role
    types_json = Column(Text, nullable=True)         # JSON: column_name -> dtype
    capabilities_json = Column(Text, nullable=True)  # JSON: platform capabilities
    is_valid = Column(Boolean, default=False, nullable=False)
    version = Column(Integer, default=1, nullable=False)
    created_at = Column(DateTime, default=utcnow, nullable=False)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)

    client = relationship("ClientModel", back_populates="schema_mappings")


class TrainingRunModel(Base):
    __tablename__ = "training_runs"

    id = Column(String(64), primary_key=True)  # run_id
    client_id = Column(String(64), ForeignKey("clients.client_id", ondelete="CASCADE"), index=True, nullable=False)
    status = Column(String(32), default="NOT_STARTED", nullable=False)  # NOT_STARTED, PREPROCESSING, TRAINING, EVALUATING, COMPLETED, FAILED
    current_epoch = Column(Integer, default=0, nullable=False)
    total_epochs = Column(Integer, default=0, nullable=False)
    error_message = Column(Text, nullable=True)
    metrics_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    client = relationship("ClientModel", back_populates="training_runs")


class ModelVersionModel(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    client_id = Column(String(64), ForeignKey("clients.client_id", ondelete="CASCADE"), index=True, nullable=False)
    version_tag = Column(String(64), index=True, nullable=False)
    training_run_id = Column(String(64), nullable=True)
    checkpoint_path = Column(String(512), nullable=False)
    encoders_path = Column(String(512), nullable=False)
    metrics_json = Column(Text, nullable=True)
    feature_dimensions_json = Column(Text, nullable=True)
    schema_snapshot_json = Column(Text, nullable=True)
    cold_start_threshold = Column(Integer, default=3, nullable=False)
    random_seed = Column(Integer, default=42, nullable=False)
    dataset_reference = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=False, nullable=False)
    legacy_imported = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=utcnow, nullable=False)

    client = relationship("ClientModel", back_populates="model_versions")


class RecommendationEventModel(Base):
    __tablename__ = "recommendation_events"

    id = Column(String(64), primary_key=True)  # recommendation_id
    client_id = Column(String(64), ForeignKey("clients.client_id", ondelete="CASCADE"), index=True, nullable=False)
    user_id = Column(String(64), nullable=False)
    model_version = Column(String(64), nullable=True)
    status = Column(String(32), nullable=True)  # cold_start, warm_start
    history_count = Column(Integer, default=0, nullable=True)
    candidate_count = Column(Integer, default=0, nullable=True)
    top_k = Column(Integer, default=5, nullable=True)
    context_json = Column(Text, nullable=True)
    filters_json = Column(Text, nullable=True)
    recommended_items_json = Column(Text, nullable=False)
    scores_json = Column(Text, nullable=True)
    latency_ms = Column(Float, default=0.0, nullable=False)
    created_at = Column(DateTime, default=utcnow, nullable=False)

    client = relationship("ClientModel", back_populates="recommendation_events")


class FeedbackEventModel(Base):
    __tablename__ = "feedback_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    client_id = Column(String(64), ForeignKey("clients.client_id", ondelete="CASCADE"), index=True, nullable=False)
    user_id = Column(String(64), nullable=False)
    item_id = Column(String(64), nullable=False)
    action = Column(String(32), nullable=False)  # CLICK, ACCEPT, REJECT, PURCHASE
    position = Column(Integer, nullable=True)
    recommendation_id = Column(String(64), ForeignKey("recommendation_events.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime, default=utcnow, nullable=False)

    client = relationship("ClientModel", back_populates="feedback_events")
