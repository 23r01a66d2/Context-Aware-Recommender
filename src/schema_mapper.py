"""
Module: Schema Mapping Engine & Dynamic Client Configuration
Provides semantic role definitions, automated schema suggestion heuristics,
schema validation, leakage prevention checks, and client configuration serialization.
"""

import os
import re
import json
import logging
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import yaml
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SchemaMapper")


class SchemaRole(str, Enum):
    """Semantic roles assigned to dataset columns in the recommendation pipeline."""
    USER_ID = "USER_ID"                  # Primary user/customer identifier
    ITEM_ID = "ITEM_ID"                  # Candidate product/item/content identifier
    TIMESTAMP = "TIMESTAMP"              # Interaction timestamp or date for chronological order
    TARGET = "TARGET"                    # Recommendation objective (e.g., purchased, clicked, converted)
    BEHAVIOR_FEATURE = "BEHAVIOR_FEATURE"# User behavioral & engagement metrics (point-in-time)
    CONTENT_FEATURE = "CONTENT_FEATURE"  # Candidate item metadata (category, price, brand)
    CONTEXT_FEATURE = "CONTEXT_FEATURE"  # Recommendation-time session context (device, channel, time)
    IGNORE = "IGNORE"                    # Post-decision outcomes, leakage-prone fields, or unneeded cols
    DROP = "IGNORE"                      # Alias for IGNORE


class ColumnType(str, Enum):
    """Underlying data types for feature engineering and transformation."""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    ID = "id"
    TEXT = "text"


# Regex patterns for auto-suggestion heuristics
ROLE_PATTERNS = {
    SchemaRole.USER_ID: [
        r"^user(_id|_code|_uuid)?$", r"^cust(omer)?(_id|_code)?$", r"^buyer(_id|_code|_uuid)?$",
        r"^visitor(_id)?$", r"^client(_id)?$", r"^account(_id)?$",
        r".*_user_id$", r".*_customer_id$"
    ],
    SchemaRole.ITEM_ID: [
        r"^item(_id|_code)?$", r"^product(_id|_code)?$", r"^prod(_id|_code)?$",
        r"^sku$", r"^article(_id|_code)?$", r"^asin$", r"^goods(_id|_code)?$",
        r".*_product_id$", r".*_item_id$", r".*_item_code$"
    ],
    SchemaRole.TIMESTAMP: [
        r"^timestamp$", r"^visit_date$", r"^event_time$", r"^date$",
        r"^created_at$", r"^interaction_time$", r"^session_time$",
        r"^order_date$", r"^time$"
    ],
    SchemaRole.TARGET: [
        r"^purchased?$", r"^is_purchased?$", r"^bought$", r"^is_bought$",
        r"^is_ordered?$", r"^converted?$", r"^conversion$", r"^clicked?$",
        r"^is_clicked?$", r"^target$", r"^label$", r".*_bought$"
    ],
    SchemaRole.IGNORE: [
        r"^revenue(_normalized)?$", r"^rating$", r"^review(_text|_helpful_votes)?$",
        r"^cart_abandoned?$", r"^payment_method$", r"^refunded?$",
        r"^returned?$", r"^order_status$", r"^feedback$", r"^added_to_cart$"
    ],
    SchemaRole.BEHAVIOR_FEATURE: [
        r"^pages?_viewed$", r"^time_on_site(_sec)?$", r"^session_duration$",
        r"^clicks?(_count)?$", r"^dwell_time$", r"^scroll_depth$",
        r"^interaction_count$", r"^view_count$"
    ],
    SchemaRole.CONTENT_FEATURE: [
        r"^product_category$", r"^category(_id|_code)?$", r"^unit_price$",
        r"^price(_usd)?$", r"^discount_percent$", r"^discount_amount$",
        r"^brand(_id)?$", r"^department$", r"^cost$", r"^tag_.*$"
    ],
    SchemaRole.CONTEXT_FEATURE: [
        r"^device(_type)?$", r"^user_type$", r"^marketing_channel$",
        r"^channel$", r"^location$", r"^city$", r"^country(_code)?$",
        r"^visit_season$", r"^season$", r"^visit_day$", r"^visit_month$",
        r"^visit_weekday$", r"^browser$", r"^os$", r"^platform$",
        r"^ip_country$", r"^hour(_of_day)?$"
    ]
}

# Known leakage outcome keywords that must not be used as input features
KNOWN_OUTCOME_KEYWORDS = [
    "revenue", "rating", "review", "abandon", "payment",
    "refund", "return", "status", "feedback"
]


class ClientSchemaConfig:
    """
    Configuration specification for a client's dataset schema, mappings,
    and preprocessing parameters.
    """

    def __init__(
        self,
        client_id: str,
        client_name: str,
        column_mappings: Dict[str, str],
        column_types: Optional[Dict[str, str]] = None,
        timestamp_format: Optional[str] = "%d-%m-%Y",
        split_config: Optional[Dict[str, Any]] = None,
        outlier_columns: Optional[List[str]] = None,
        cold_start_threshold: int = 3,
        evaluated_thresholds: Optional[List[int]] = None,
        target_positive_value: Any = 1,
        description: str = ""
    ):
        self.client_id = client_id
        self.client_name = client_name
        self.column_mappings = column_mappings
        self.column_types = column_types or {}
        self.timestamp_format = timestamp_format
        self.split_config = split_config or {
            "strategy": "chronological",
            "train_start": "2024-01-01",
            "train_end": "2024-07-31",
            "val_start": "2024-08-01",
            "val_end": "2024-09-30",
            "test_start": "2024-10-01",
            "test_end": "2024-12-30"
        }
        self.outlier_columns = outlier_columns or []
        self.cold_start_threshold = cold_start_threshold
        self.evaluated_thresholds = evaluated_thresholds or [1, 2, 3, 5]
        self.target_positive_value = target_positive_value
        self.description = description

    def get_columns_by_role(self, role: Union[SchemaRole, str]) -> List[str]:
        """Returns all column names matching a given SchemaRole."""
        target_role = role.value if isinstance(role, SchemaRole) else str(role)
        return [col for col, r in self.column_mappings.items() if r == target_role]

    def get_user_id_col(self) -> str:
        cols = self.get_columns_by_role(SchemaRole.USER_ID)
        if not cols:
            raise ValueError(f"No USER_ID column defined in schema for client '{self.client_id}'.")
        return cols[0]

    def get_item_id_col(self) -> str:
        cols = self.get_columns_by_role(SchemaRole.ITEM_ID)
        if not cols:
            raise ValueError(f"No ITEM_ID column defined in schema for client '{self.client_id}'.")
        return cols[0]

    def get_timestamp_col(self) -> Optional[str]:
        cols = self.get_columns_by_role(SchemaRole.TIMESTAMP)
        return cols[0] if cols else None

    def get_capabilities(self) -> Dict[str, bool]:
        return compute_dataset_capabilities(self.column_mappings)

    def get_target_col(self) -> Optional[str]:
        cols = self.get_columns_by_role(SchemaRole.TARGET)
        return cols[0] if cols else None

    def get_behavior_cols(self) -> List[str]:
        return self.get_columns_by_role(SchemaRole.BEHAVIOR_FEATURE)

    def get_content_cols(self) -> List[str]:
        return self.get_columns_by_role(SchemaRole.CONTENT_FEATURE)

    def get_context_cols(self) -> List[str]:
        return self.get_columns_by_role(SchemaRole.CONTEXT_FEATURE)

    def get_ignored_cols(self) -> List[str]:
        return self.get_columns_by_role(SchemaRole.IGNORE)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "client_id": self.client_id,
            "client_name": self.client_name,
            "column_mappings": self.column_mappings,
            "column_types": self.column_types,
            "timestamp_format": self.timestamp_format,
            "split_config": self.split_config,
            "outlier_columns": self.outlier_columns,
            "cold_start_threshold": self.cold_start_threshold,
            "evaluated_thresholds": self.evaluated_thresholds,
            "target_positive_value": self.target_positive_value,
            "description": self.description
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClientSchemaConfig":
        return cls(
            client_id=data.get("client_id", "default"),
            client_name=data.get("client_name", "Default Client"),
            column_mappings=data.get("column_mappings", {}),
            column_types=data.get("column_types", {}),
            timestamp_format=data.get("timestamp_format", "%d-%m-%Y"),
            split_config=data.get("split_config"),
            outlier_columns=data.get("outlier_columns"),
            cold_start_threshold=data.get("cold_start_threshold", 3),
            evaluated_thresholds=data.get("evaluated_thresholds", [1, 2, 3, 5]),
            target_positive_value=data.get("target_positive_value", 1),
            description=data.get("description", "")
        )

    def to_yaml(self, filepath: Union[str, Path]) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved client schema configuration to {filepath}")

    @classmethod
    def from_yaml(cls, filepath: Union[str, Path]) -> "ClientSchemaConfig":
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


def infer_column_type(series: pd.Series) -> str:
    """Infers the general data type for a pandas Series."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return ColumnType.DATETIME.value
    if pd.api.types.is_numeric_dtype(series):
        # If binary 0/1 or boolean, could be target or binary flag
        unique_vals = set(series.dropna().unique())
        if unique_vals.issubset({0, 1, 0.0, 1.0, True, False}):
            return ColumnType.NUMERIC.value
        # If high-cardinality integer with ID-like pattern
        if series.nunique() > 1000 and "id" in series.name.lower():
            return ColumnType.ID.value
        return ColumnType.NUMERIC.value
    # Object or string
    # Check if convertible to datetime
    if "date" in series.name.lower() or "time" in series.name.lower():
        return ColumnType.DATETIME.value
    # Cardinality check
    if series.nunique() <= 50:
        return ColumnType.CATEGORICAL.value
    if "id" in series.name.lower() or "code" in series.name.lower():
        return ColumnType.ID.value
    return ColumnType.CATEGORICAL.value


def auto_suggest_mappings(df: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Analyzes DataFrame columns and suggests semantic SchemaRoles and data types.
    Returns:
        (mappings, inferred_types)
    """
    mappings: Dict[str, str] = {}
    inferred_types: Dict[str, str] = {}

    assigned_roles = set()

    for col in df.columns:
        col_lower = col.lower().strip()
        dtype = infer_column_type(df[col])
        inferred_types[col] = dtype

        suggested_role = None

        # 1. Check exact pattern matches for single-instance roles first
        if SchemaRole.USER_ID not in assigned_roles:
            for pat in ROLE_PATTERNS[SchemaRole.USER_ID]:
                if re.match(pat, col_lower):
                    suggested_role = SchemaRole.USER_ID.value
                    assigned_roles.add(SchemaRole.USER_ID)
                    break

        if suggested_role is None and SchemaRole.ITEM_ID not in assigned_roles:
            for pat in ROLE_PATTERNS[SchemaRole.ITEM_ID]:
                if re.match(pat, col_lower):
                    suggested_role = SchemaRole.ITEM_ID.value
                    assigned_roles.add(SchemaRole.ITEM_ID)
                    break

        if suggested_role is None and SchemaRole.TIMESTAMP not in assigned_roles:
            for pat in ROLE_PATTERNS[SchemaRole.TIMESTAMP]:
                if re.match(pat, col_lower):
                    suggested_role = SchemaRole.TIMESTAMP.value
                    assigned_roles.add(SchemaRole.TIMESTAMP)
                    break

        if suggested_role is None and SchemaRole.TARGET not in assigned_roles:
            for pat in ROLE_PATTERNS[SchemaRole.TARGET]:
                if re.match(pat, col_lower):
                    suggested_role = SchemaRole.TARGET.value
                    assigned_roles.add(SchemaRole.TARGET)
                    break

        # 2. Check for outcome / leakage fields -> IGNORE
        if suggested_role is None:
            for pat in ROLE_PATTERNS[SchemaRole.IGNORE]:
                if re.match(pat, col_lower) or any(kw in col_lower for kw in KNOWN_OUTCOME_KEYWORDS):
                    suggested_role = SchemaRole.IGNORE.value
                    break

        # 3. Check feature roles
        if suggested_role is None:
            for pat in ROLE_PATTERNS[SchemaRole.BEHAVIOR_FEATURE]:
                if re.match(pat, col_lower):
                    suggested_role = SchemaRole.BEHAVIOR_FEATURE.value
                    break

        if suggested_role is None:
            for pat in ROLE_PATTERNS[SchemaRole.CONTENT_FEATURE]:
                if re.match(pat, col_lower):
                    suggested_role = SchemaRole.CONTENT_FEATURE.value
                    break

        if suggested_role is None:
            for pat in ROLE_PATTERNS[SchemaRole.CONTEXT_FEATURE]:
                if re.match(pat, col_lower):
                    suggested_role = SchemaRole.CONTEXT_FEATURE.value
                    break

        # 4. Fallback based on column name & data type
        if suggested_role is None:
            if "id" in col_lower:
                suggested_role = SchemaRole.IGNORE.value  # Secondary ID like session_id
            elif dtype == ColumnType.DATETIME.value and SchemaRole.TIMESTAMP not in assigned_roles:
                suggested_role = SchemaRole.TIMESTAMP.value
                assigned_roles.add(SchemaRole.TIMESTAMP)
            elif "view" in col_lower or "click" in col_lower or "time" in col_lower:
                suggested_role = SchemaRole.BEHAVIOR_FEATURE.value
            elif "cat" in col_lower or "price" in col_lower or "cost" in col_lower:
                suggested_role = SchemaRole.CONTENT_FEATURE.value
            else:
                suggested_role = SchemaRole.CONTEXT_FEATURE.value

        mappings[col] = suggested_role

    return mappings, inferred_types


def compute_dataset_capabilities(mappings: Dict[str, str]) -> Dict[str, bool]:
    """
    Computes explicit platform capabilities supported by the current schema mappings:
    - recommendation_compatible: USER_ID and ITEM_ID present, plus at least one feature
    - supervised_training: recommendation_compatible AND TARGET present
    - temporal_training: recommendation_compatible AND TIMESTAMP present
    - cold_start_history: recommendation_compatible AND TIMESTAMP AND BEHAVIOR_FEATURE present
    """
    user_cols = [c for c, r in mappings.items() if r == SchemaRole.USER_ID.value]
    item_cols = [c for c, r in mappings.items() if r == SchemaRole.ITEM_ID.value]
    timestamp_cols = [c for c, r in mappings.items() if r == SchemaRole.TIMESTAMP.value]
    target_cols = [c for c, r in mappings.items() if r == SchemaRole.TARGET.value]
    feat_cols = [c for c, r in mappings.items() if r in [
        SchemaRole.BEHAVIOR_FEATURE.value,
        SchemaRole.CONTENT_FEATURE.value,
        SchemaRole.CONTEXT_FEATURE.value
    ]]
    behavior_cols = [c for c, r in mappings.items() if r == SchemaRole.BEHAVIOR_FEATURE.value]

    has_identity = (len(user_cols) == 1) and (len(item_cols) == 1)
    has_timestamp = len(timestamp_cols) >= 1
    has_target = len(target_cols) == 1
    has_features = len(feat_cols) >= 1

    rec_compat = has_identity and has_features
    return {
        "recommendation_compatible": bool(rec_compat),
        "supervised_training": bool(rec_compat and has_target),
        "temporal_training": bool(rec_compat and has_timestamp),
        "cold_start_history": bool(rec_compat and has_timestamp and len(behavior_cols) > 0)
    }


def validate_schema(
    mappings: Dict[str, str],
    df_columns: Optional[List[str]] = None,
    require_target: bool = False
) -> Tuple[bool, List[str], List[str]]:
    """
    Validates semantic schema mappings according to platform policy:
    - USER_ID and ITEM_ID are mandatory.
    - TIMESTAMP is strongly recommended: absence produces a warning and disables temporal training,
      but does not block registration/inspection.
    - TARGET is required for supervised training: absence produces a warning, but does not block mapping.
    Returns:
        (is_valid, error_messages, warning_messages)
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Count occurrences of single-instance roles
    user_id_cols = [col for col, role in mappings.items() if role == SchemaRole.USER_ID.value]
    item_id_cols = [col for col, role in mappings.items() if role == SchemaRole.ITEM_ID.value]
    timestamp_cols = [col for col, role in mappings.items() if role == SchemaRole.TIMESTAMP.value]
    target_cols = [col for col, role in mappings.items() if role == SchemaRole.TARGET.value]

    # Required: USER_ID (Mandatory)
    if len(user_id_cols) == 0:
        errors.append("Schema Error: Exactly one column must be mapped to role 'USER_ID'. None found.")
    elif len(user_id_cols) > 1:
        errors.append(f"Schema Error: Multiple columns mapped to 'USER_ID': {user_id_cols}. Only 1 permitted.")

    # Required: ITEM_ID (Mandatory)
    if len(item_id_cols) == 0:
        errors.append("Schema Error: Exactly one column must be mapped to role 'ITEM_ID'. None found.")
    elif len(item_id_cols) > 1:
        errors.append(f"Schema Error: Multiple columns mapped to 'ITEM_ID': {item_id_cols}. Only 1 permitted.")

    # Strongly Recommended: TIMESTAMP (Policy: Warning only, non-blocking)
    if len(timestamp_cols) == 0:
        warnings.append(
            "Schema Warning: No column mapped to 'TIMESTAMP'. Strict chronological splitting, "
            "point-in-time temporal claims, and zero-leakage chronological guarantees are disabled for this dataset."
        )
    elif len(timestamp_cols) > 1:
        warnings.append(f"Multiple columns mapped to 'TIMESTAMP': {timestamp_cols}. Using first: {timestamp_cols[0]}.")

    # TARGET: Warning if absent (or error only if strictly required for immediate supervised training)
    if len(target_cols) == 0:
        if require_target:
            errors.append("Schema Error: Exactly one column must be mapped to role 'TARGET' for supervised training.")
        else:
            warnings.append(
                "Schema Warning: No column mapped to 'TARGET'. Supervised training capability is disabled "
                "until an outcome/target column is specified."
            )
    elif len(target_cols) > 1:
        errors.append(f"Schema Error: Multiple columns mapped to 'TARGET': {target_cols}. Only 1 permitted.")

    # Verify at least some features are specified
    feat_cols = [col for col, role in mappings.items() if role in [
        SchemaRole.BEHAVIOR_FEATURE.value,
        SchemaRole.CONTENT_FEATURE.value,
        SchemaRole.CONTEXT_FEATURE.value
    ]]
    if len(feat_cols) == 0:
        errors.append("Schema Error: At least one feature column (BEHAVIOR, CONTENT, or CONTEXT) must be mapped.")

    # Verify columns exist in DataFrame if provided
    if df_columns is not None:
        missing_in_df = [c for c in mappings.keys() if c not in df_columns]
        if missing_in_df:
            errors.append(f"Schema Error: Mapped columns not present in dataset: {missing_in_df}")

        unmapped = [c for c in df_columns if c not in mappings]
        if unmapped:
            warnings.append(f"Dataset columns unmapped in schema (defaulting to IGNORE): {unmapped}")

    # Leakage check: warn if known outcome keywords are mapped as predictive features
    for col, role in mappings.items():
        if role in [SchemaRole.BEHAVIOR_FEATURE.value, SchemaRole.CONTENT_FEATURE.value, SchemaRole.CONTEXT_FEATURE.value]:
            col_lower = col.lower()
            if any(kw in col_lower for kw in KNOWN_OUTCOME_KEYWORDS):
                warnings.append(
                    f"POTENTIAL DATA LEAKAGE WARNING: Column '{col}' resembles a post-decision outcome "
                    f"but is assigned to predictive role '{role}'. Consider mapping to 'IGNORE'."
                )

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def inspect_and_validate_schema(
    mappings: Dict[str, str],
    df_columns: Optional[List[str]] = None,
    require_target: bool = False
) -> Dict[str, Any]:
    """
    Unified validation & capability assessment response for API services:
    returns valid, errors, warnings, capabilities, and detected_roles.
    """
    is_valid, errors, warnings = validate_schema(mappings, df_columns, require_target)
    capabilities = compute_dataset_capabilities(mappings)
    return {
        "valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "capabilities": capabilities,
        "detected_roles": mappings
    }


def create_demo_ecommerce_schema() -> ClientSchemaConfig:
    """
    Constructs the canonical, verified schema configuration for 'demo_ecommerce'
    reproducing exact feature dimensions: Behavior=16, Content=12, Context=21.
    """
    column_mappings = {
        "customer_id": SchemaRole.USER_ID.value,
        "session_id": SchemaRole.IGNORE.value,
        "visit_date": SchemaRole.TIMESTAMP.value,
        "device_type": SchemaRole.CONTEXT_FEATURE.value,
        "user_type": SchemaRole.CONTEXT_FEATURE.value,
        "marketing_channel": SchemaRole.CONTEXT_FEATURE.value,
        "product_id": SchemaRole.ITEM_ID.value,
        "product_category": SchemaRole.CONTENT_FEATURE.value,
        "unit_price": SchemaRole.CONTENT_FEATURE.value,
        "quantity": SchemaRole.IGNORE.value,
        "discount_percent": SchemaRole.CONTENT_FEATURE.value,
        "discount_amount": SchemaRole.CONTENT_FEATURE.value,
        "revenue": SchemaRole.IGNORE.value,
        "pages_viewed": SchemaRole.BEHAVIOR_FEATURE.value,
        "time_on_site_sec": SchemaRole.BEHAVIOR_FEATURE.value,
        "added_to_cart": SchemaRole.IGNORE.value,
        "purchased": SchemaRole.TARGET.value,
        "cart_abandoned": SchemaRole.IGNORE.value,
        "rating": SchemaRole.IGNORE.value,
        "review_text": SchemaRole.IGNORE.value,
        "review_helpful_votes": SchemaRole.IGNORE.value,
        "payment_method": SchemaRole.IGNORE.value,
        "visit_day": SchemaRole.CONTEXT_FEATURE.value,
        "visit_month": SchemaRole.CONTEXT_FEATURE.value,
        "visit_weekday": SchemaRole.CONTEXT_FEATURE.value,
        "visit_season": SchemaRole.CONTEXT_FEATURE.value,
        "session_duration_bucket": SchemaRole.IGNORE.value,
        "revenue_normalized": SchemaRole.IGNORE.value,
        "location": SchemaRole.CONTEXT_FEATURE.value
    }

    column_types = {
        "customer_id": ColumnType.ID.value,
        "session_id": ColumnType.ID.value,
        "visit_date": ColumnType.DATETIME.value,
        "device_type": ColumnType.CATEGORICAL.value,
        "user_type": ColumnType.CATEGORICAL.value,
        "marketing_channel": ColumnType.CATEGORICAL.value,
        "product_id": ColumnType.ID.value,
        "product_category": ColumnType.CATEGORICAL.value,
        "unit_price": ColumnType.NUMERIC.value,
        "quantity": ColumnType.NUMERIC.value,
        "discount_percent": ColumnType.NUMERIC.value,
        "discount_amount": ColumnType.NUMERIC.value,
        "revenue": ColumnType.NUMERIC.value,
        "pages_viewed": ColumnType.NUMERIC.value,
        "time_on_site_sec": ColumnType.NUMERIC.value,
        "added_to_cart": ColumnType.NUMERIC.value,
        "purchased": ColumnType.NUMERIC.value,
        "cart_abandoned": ColumnType.NUMERIC.value,
        "rating": ColumnType.NUMERIC.value,
        "review_text": ColumnType.CATEGORICAL.value,
        "review_helpful_votes": ColumnType.NUMERIC.value,
        "payment_method": ColumnType.CATEGORICAL.value,
        "visit_day": ColumnType.NUMERIC.value,
        "visit_month": ColumnType.NUMERIC.value,
        "visit_weekday": ColumnType.NUMERIC.value,
        "visit_season": ColumnType.CATEGORICAL.value,
        "session_duration_bucket": ColumnType.CATEGORICAL.value,
        "revenue_normalized": ColumnType.NUMERIC.value,
        "location": ColumnType.NUMERIC.value
    }

    split_config = {
        "strategy": "chronological",
        "date_column": "visit_date",
        "date_format": "%d-%m-%Y",
        "train_start": "2024-01-01",
        "train_end": "2024-07-31",
        "val_start": "2024-08-01",
        "val_end": "2024-09-30",
        "test_start": "2024-10-01",
        "test_end": "2024-12-30"
    }

    outlier_columns = ["unit_price", "discount_amount", "time_on_site_sec", "pages_viewed"]

    return ClientSchemaConfig(
        client_id="demo_ecommerce",
        client_name="Indian E-Commerce Customer Behavior & Purchase",
        column_mappings=column_mappings,
        column_types=column_types,
        timestamp_format="%d-%m-%Y",
        split_config=split_config,
        outlier_columns=outlier_columns,
        cold_start_threshold=3,
        evaluated_thresholds=[1, 2, 3, 5],
        target_positive_value=1,
        description="Canonical reference dataset configuration matching Phase 1 model (16 Beh, 12 Cont, 21 Ctx)."
    )


def get_or_create_client_config(client_id: str, clients_dir: Union[str, Path] = "clients") -> ClientSchemaConfig:
    """
    Loads an existing ClientSchemaConfig or creates default config if demo_ecommerce.
    """
    clients_dir = Path(clients_dir)
    client_config_path = clients_dir / client_id / "config.yaml"

    if client_config_path.exists():
        return ClientSchemaConfig.from_yaml(client_config_path)

    if client_id == "demo_ecommerce":
        config = create_demo_ecommerce_schema()
        config.to_yaml(client_config_path)
        return config

    raise FileNotFoundError(f"Client configuration not found for '{client_id}' at {client_config_path}")


if __name__ == "__main__":
    demo_cfg = create_demo_ecommerce_schema()
    is_valid, errors, warnings = validate_schema(demo_cfg.column_mappings)
    print("Demo E-commerce Schema Validation:")
    print("  Valid:", is_valid)
    print("  Errors:", errors)
    print("  Warnings:", warnings)
    demo_cfg.to_yaml("clients/demo_ecommerce/config.yaml")
    print(f"Saved canonical config to clients/demo_ecommerce/config.yaml")
