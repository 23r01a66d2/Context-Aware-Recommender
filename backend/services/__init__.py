"""
Application service layer for clients, datasets, schemas, and recommendations.
"""
from backend.services.client_service import (
    validate_client_id_security,
    create_client,
    get_client_or_404,
    list_all_clients,
    ensure_demo_ecommerce_registered
)
from backend.services.dataset_service import (
    save_uploaded_dataset,
    get_dataset_preview_for_client
)
from backend.services.schema_service import (
    get_or_suggest_schema_for_client,
    save_schema_for_client,
    validate_client_schema
)

__all__ = [
    "validate_client_id_security",
    "create_client",
    "get_client_or_404",
    "list_all_clients",
    "ensure_demo_ecommerce_registered",
    "save_uploaded_dataset",
    "get_dataset_preview_for_client",
    "get_or_suggest_schema_for_client",
    "save_schema_for_client",
    "validate_client_schema"
]
