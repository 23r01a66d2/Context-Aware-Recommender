"""
Pydantic schemas for Client entity and API contracts.
"""

import re
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator


CLIENT_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


class ClientCreate(BaseModel):
    client_id: str = Field(..., description="Unique client identifier (alphanumeric, hyphen, underscore)")
    name: str = Field(..., min_length=1, max_length=255, description="Client human-readable name")
    description: Optional[str] = Field("", max_length=1000, description="Optional client description")

    @field_validator("client_id")
    @classmethod
    def validate_client_id_format(cls, v: str) -> str:
        v_clean = v.strip()
        if not v_clean:
            raise ValueError("client_id cannot be empty")
        if not CLIENT_ID_REGEX.match(v_clean):
            raise ValueError(
                "Invalid client_id format. Must contain only alphanumeric characters, "
                "underscores, or hyphens (1-64 characters). Path traversals ('..', '/') are strictly forbidden."
            )
        if ".." in v_clean or "/" in v_clean or "\\" in v_clean:
            raise ValueError("Path traversal sequences are strictly forbidden in client_id.")
        return v_clean


class ClientResponse(BaseModel):
    client_id: str
    name: str
    description: Optional[str] = ""
    is_system: bool = False
    created_at: datetime
    dataset_count: int = 0
    has_schema: bool = False
    has_model: bool = False
    active_model_version: Optional[str] = None

    class Config:
        from_attributes = True


class ClientListResponse(BaseModel):
    clients: List[ClientResponse]
    total: int


class ClientDeleteResponse(BaseModel):
    client_id: str
    message: str
    deleted_records: dict
    physical_files_deleted: bool
