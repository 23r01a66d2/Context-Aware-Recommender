"""
Database connection, session management, and table initialization.
Uses SQLAlchemy abstraction allowing SQLite for local prototype and PostgreSQL for production.
"""

import os
from pathlib import Path
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# Database URL: default to local SQLite, but can be overridden by environment variable for PostgreSQL
DB_DIR = Path("data")
DB_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_SQLITE_URL = f"sqlite:///{DB_DIR / 'platform.db'}"

DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_SQLITE_URL)

# SQLite requires check_same_thread=False for multithreaded FastAPI requests
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args, echo=False)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """Dependency that provides a database session and safely closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Initializes all database tables registered with Base and handles dynamic column additions for SQLite."""
    import backend.database.models  # Ensure models are imported before creating tables
    Base.metadata.create_all(bind=engine)

    from sqlalchemy import inspect, text
    inspector = inspect(engine)
    if "clients" in inspector.get_table_names():
        cols = {c["name"] for c in inspector.get_columns("clients")}
        with engine.connect() as conn:
            if "is_system" not in cols:
                conn.execute(text("ALTER TABLE clients ADD COLUMN is_system BOOLEAN DEFAULT 0 NOT NULL"))
            conn.execute(text("UPDATE clients SET is_system = 1 WHERE client_id = 'demo_ecommerce'"))
            conn.commit()

    if "training_runs" in inspector.get_table_names():
        cols = {c["name"] for c in inspector.get_columns("training_runs")}
        with engine.connect() as conn:
            if "current_epoch" not in cols:
                conn.execute(text("ALTER TABLE training_runs ADD COLUMN current_epoch INTEGER DEFAULT 0 NOT NULL"))
            if "total_epochs" not in cols:
                conn.execute(text("ALTER TABLE training_runs ADD COLUMN total_epochs INTEGER DEFAULT 0 NOT NULL"))
            conn.commit()

    if "model_versions" in inspector.get_table_names():
        cols = {c["name"] for c in inspector.get_columns("model_versions")}
        with engine.connect() as conn:
            if "training_run_id" not in cols:
                conn.execute(text("ALTER TABLE model_versions ADD COLUMN training_run_id VARCHAR(64)"))
            if "metrics_json" not in cols:
                conn.execute(text("ALTER TABLE model_versions ADD COLUMN metrics_json TEXT"))
            if "feature_dimensions_json" not in cols:
                conn.execute(text("ALTER TABLE model_versions ADD COLUMN feature_dimensions_json TEXT"))
            if "schema_snapshot_json" not in cols:
                conn.execute(text("ALTER TABLE model_versions ADD COLUMN schema_snapshot_json TEXT"))
            if "cold_start_threshold" not in cols:
                conn.execute(text("ALTER TABLE model_versions ADD COLUMN cold_start_threshold INTEGER DEFAULT 3 NOT NULL"))
            if "random_seed" not in cols:
                conn.execute(text("ALTER TABLE model_versions ADD COLUMN random_seed INTEGER DEFAULT 42 NOT NULL"))
            if "dataset_reference" not in cols:
                conn.execute(text("ALTER TABLE model_versions ADD COLUMN dataset_reference VARCHAR(255)"))
            if "legacy_imported" not in cols:
                conn.execute(text("ALTER TABLE model_versions ADD COLUMN legacy_imported BOOLEAN DEFAULT 0 NOT NULL"))
            conn.commit()

    if "recommendation_events" in inspector.get_table_names():
        cols = {c["name"] for c in inspector.get_columns("recommendation_events")}
        with engine.connect() as conn:
            if "model_version" not in cols:
                conn.execute(text("ALTER TABLE recommendation_events ADD COLUMN model_version VARCHAR(64)"))
            if "status" not in cols:
                conn.execute(text("ALTER TABLE recommendation_events ADD COLUMN status VARCHAR(32)"))
            if "history_count" not in cols:
                conn.execute(text("ALTER TABLE recommendation_events ADD COLUMN history_count INTEGER DEFAULT 0"))
            if "candidate_count" not in cols:
                conn.execute(text("ALTER TABLE recommendation_events ADD COLUMN candidate_count INTEGER DEFAULT 0"))
            if "top_k" not in cols:
                conn.execute(text("ALTER TABLE recommendation_events ADD COLUMN top_k INTEGER DEFAULT 5"))
            if "filters_json" not in cols:
                conn.execute(text("ALTER TABLE recommendation_events ADD COLUMN filters_json TEXT"))
            if "scores_json" not in cols:
                conn.execute(text("ALTER TABLE recommendation_events ADD COLUMN scores_json TEXT"))
            conn.commit()

    if "feedback_events" in inspector.get_table_names():
        cols = {c["name"] for c in inspector.get_columns("feedback_events")}
        with engine.connect() as conn:
            if "recommendation_id" not in cols:
                conn.execute(text("ALTER TABLE feedback_events ADD COLUMN recommendation_id VARCHAR(64)"))
            conn.commit()
