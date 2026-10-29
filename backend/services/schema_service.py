"""
Service for schema mapping discovery, validation, and persistence.
Reuses src/schema_mapper.py without duplicating logic.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

import yaml
from backend.database.models import SchemaMappingModel, DatasetModel
from backend.schemas.schema_map import SchemaResponse, SchemaUpdateRequest
from backend.services.client_service import get_client_or_404
from src.schema_mapper import (
    ClientSchemaConfig,
    auto_suggest_mappings,
    validate_schema,
    compute_dataset_capabilities,
    inspect_and_validate_schema,
    create_demo_ecommerce_schema,
    get_or_create_client_config,
    repair_and_augment_dataset_dataframe
)

logger = logging.getLogger("SchemaService")


def get_or_suggest_schema_for_client(db: Session, client_id: str) -> SchemaResponse:
    """
    Retrieves confirmed schema mapping for client.
    If none is confirmed, automatically suggests mappings based on uploaded dataset columns.
    """
    client = get_client_or_404(db, client_id)
    cid = client.client_id

    # 1. Check existing saved config.yaml
    cfg_path = Path(f"clients/{cid}/config.yaml")
    if cfg_path.exists():
        cfg = ClientSchemaConfig.from_yaml(cfg_path)
        is_valid, errors, warnings = validate_schema(cfg.column_mappings)
        generated_fields = {}
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                ydata = yaml.safe_load(f) or {}
                generated_fields = ydata.get("generated_fields", {})
        except Exception:
            pass

        return SchemaResponse(
            client_id=cid,
            column_mappings=cfg.column_mappings,
            column_types=cfg.column_types or {},
            capabilities=cfg.get_capabilities(),
            is_valid=is_valid,
            timestamp_format=cfg.timestamp_format,
            outlier_columns=cfg.outlier_columns or [],
            cold_start_threshold=cfg.cold_start_threshold,
            version=1,
            generated_fields=generated_fields
        )

    # 2. If demo_ecommerce, bootstrap canonical schema
    if cid == "demo_ecommerce":
        cfg = create_demo_ecommerce_schema()
        cfg.to_yaml(cfg_path)
        return SchemaResponse(
            client_id=cid,
            column_mappings=cfg.column_mappings,
            column_types=cfg.column_types,
            capabilities=cfg.get_capabilities(),
            is_valid=True,
            timestamp_format=cfg.timestamp_format,
            outlier_columns=cfg.outlier_columns,
            cold_start_threshold=cfg.cold_start_threshold,
            version=1
        )

    # 3. Check for uploaded dataset to run auto-suggestion
    raw_files = list(Path(f"clients/{cid}/raw").glob("*.csv"))
    if not raw_files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No dataset found for client '{cid}'. Upload a dataset before configuring schema."
        )

    df_sample = pd.read_csv(raw_files[0], nrows=100)
    suggested_mappings, inferred_types = auto_suggest_mappings(df_sample)
    is_valid, errors, warnings = validate_schema(suggested_mappings, list(df_sample.columns))
    capabilities = compute_dataset_capabilities(suggested_mappings)

    return SchemaResponse(
        client_id=cid,
        column_mappings=suggested_mappings,
        column_types=inferred_types,
        capabilities=capabilities,
        is_valid=is_valid,
        timestamp_format="%Y-%m-%d %H:%M:%S",
        outlier_columns=[],
        cold_start_threshold=3,
        version=1
    )


def save_schema_for_client(db: Session, client_id: str, schema_in: SchemaUpdateRequest) -> SchemaResponse:
    """
    Validates, saves to client config.yaml, and persists in database.
    """
    client = get_client_or_404(db, client_id)
    cid = client.client_id

    # 1. Validation check
    is_valid, errors, warnings = validate_schema(schema_in.column_mappings)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid schema mapping: {'; '.join(errors)}"
        )

    capabilities = compute_dataset_capabilities(schema_in.column_mappings)

    # 2. Build ClientSchemaConfig and write to filesystem
    cfg = ClientSchemaConfig(
        client_id=cid,
        client_name=client.name,
        column_mappings=schema_in.column_mappings,
        column_types=schema_in.column_types or {},
        timestamp_format=schema_in.timestamp_format,
        outlier_columns=schema_in.outlier_columns or [],
        cold_start_threshold=schema_in.cold_start_threshold or 3
    )
    cfg_path = Path(f"clients/{cid}/config.yaml")
    cfg.to_yaml(cfg_path)

    # 3. Upsert into database
    mapping_record = db.query(SchemaMappingModel).filter(SchemaMappingModel.client_id == cid).first()
    if mapping_record:
        mapping_record.mappings_json = json.dumps(schema_in.column_mappings)
        mapping_record.types_json = json.dumps(schema_in.column_types or {})
        mapping_record.capabilities_json = json.dumps(capabilities)
        mapping_record.is_valid = True
        mapping_record.version += 1
    else:
        mapping_record = SchemaMappingModel(
            client_id=cid,
            mappings_json=json.dumps(schema_in.column_mappings),
            types_json=json.dumps(schema_in.column_types or {}) if schema_in.column_types else "{}",
            capabilities_json=json.dumps(capabilities),
            is_valid=True,
            version=1
        )
        db.add(mapping_record)

    db.commit()
    db.refresh(mapping_record)

    logger.info(f"Schema mapping v{mapping_record.version} saved for client '{cid}'.")
    return SchemaResponse(
        client_id=cid,
        column_mappings=schema_in.column_mappings,
        column_types=schema_in.column_types or {},
        capabilities=capabilities,
        is_valid=True,
        timestamp_format=schema_in.timestamp_format,
        outlier_columns=schema_in.outlier_columns or [],
        cold_start_threshold=schema_in.cold_start_threshold or 3,
        version=mapping_record.version
    )


def validate_client_schema(client_id: str, mappings: Dict[str, str]) -> Dict[str, Any]:
    """
    Runs schema validation and capability assessment using src/schema_mapper.py.
    """
    safe_cid = client_id.strip()
    return inspect_and_validate_schema(mappings)


def auto_repair_schema_for_client(db: Session, client_id: str) -> SchemaResponse:
    """
    Safely auto-repairs missing IDs, derives display name, detects images,
    and updates dataset & schema mappings without fabricating behavioral history.
    """
    client = get_client_or_404(db, client_id)
    cid = client.client_id

    raw_files = list(Path(f"clients/{cid}/raw").glob("*.csv"))
    if not raw_files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No dataset found for client '{cid}'. Upload a dataset before auto-repairing schema."
        )

    target_csv = raw_files[0]
    df = pd.read_csv(target_csv)
    repaired_df, generated_fields = repair_and_augment_dataset_dataframe(df, client_id=cid)

    # Save repaired dataset
    repaired_df.to_csv(target_csv, index=False)

    suggested_mappings, inferred_types = auto_suggest_mappings(repaired_df)
    is_valid, errors, warnings = validate_schema(suggested_mappings, list(repaired_df.columns))
    capabilities = compute_dataset_capabilities(suggested_mappings, inferred_types)

    # Save config.yaml
    cfg_path = Path(f"clients/{cid}/config.yaml")
    cfg = ClientSchemaConfig(
        client_id=cid,
        client_name=client.name,
        column_mappings=suggested_mappings,
        column_types=inferred_types,
        timestamp_format="%Y-%m-%d %H:%M:%S",
        outlier_columns=[],
        cold_start_threshold=3
    )
    cfg_dict = cfg.to_dict()
    cfg_dict["generated_fields"] = generated_fields
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False, sort_keys=False)

    # Upsert into database
    mapping_record = db.query(SchemaMappingModel).filter(SchemaMappingModel.client_id == cid).first()
    caps_data = {"capabilities": capabilities, "generated_fields": generated_fields}
    if mapping_record:
        mapping_record.mappings_json = json.dumps(suggested_mappings)
        mapping_record.types_json = json.dumps(inferred_types)
        mapping_record.capabilities_json = json.dumps(caps_data)
        mapping_record.is_valid = is_valid
        mapping_record.version += 1
    else:
        mapping_record = SchemaMappingModel(
            client_id=cid,
            mappings_json=json.dumps(suggested_mappings),
            types_json=json.dumps(inferred_types),
            capabilities_json=json.dumps(caps_data),
            is_valid=is_valid,
            version=1
        )
        db.add(mapping_record)
    db.commit()
    db.refresh(mapping_record)

    return SchemaResponse(
        client_id=cid,
        column_mappings=suggested_mappings,
        column_types=inferred_types,
        capabilities=capabilities,
        is_valid=is_valid,
        timestamp_format="%Y-%m-%d %H:%M:%S",
        outlier_columns=[],
        cold_start_threshold=3,
        version=mapping_record.version,
        generated_fields=generated_fields
    )
