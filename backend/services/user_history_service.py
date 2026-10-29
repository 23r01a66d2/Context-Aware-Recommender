"""
Persistent JSON-based User History Service.
Provides strict tenant-isolated storage, atomic writes, thread locking,
corrupted file recovery, cold/warm start classification, and event tracking.
"""

import os
import re
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("UserHistoryService")

# Base directory for user histories
USER_HISTORY_BASE_DIR = Path("data/user_history")
_FILE_LOCKS: Dict[str, threading.Lock] = {}
_META_LOCK = threading.Lock()


def _get_lock(key: str) -> threading.Lock:
    """Gets or creates a threading Lock for a specific tenant/user file."""
    with _META_LOCK:
        if key not in _FILE_LOCKS:
            _FILE_LOCKS[key] = threading.Lock()
        return _FILE_LOCKS[key]


def sanitize_id(identifier: Optional[str], default_name: str = "default") -> str:
    """
    Sanitizes tenant / user ID to prevent path traversal or unsafe characters.
    Allows alphanumeric, underscores, hyphens, and periods.
    """
    if not identifier:
        return default_name
    s = str(identifier).strip()
    clean = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', s)
    clean = clean.strip('._')
    return clean if clean else default_name


def validate_user_id(user_id: Optional[str]) -> Tuple[bool, str]:
    """
    Validates a user ID string.
    Returns (is_valid, sanitized_or_error_message).
    User ID must be non-empty, max 128 characters, and contain valid chars.
    """
    if user_id is None:
        return False, "User ID cannot be empty."
    s = str(user_id).strip()
    if not s:
        return False, "User ID cannot be empty or blank."
    if len(s) > 128:
        return False, "User ID exceeds maximum length of 128 characters."
    if ".." in s or "/" in s or "\\" in s:
        return False, "User ID contains invalid path characters."
    
    clean = sanitize_id(s)
    if not clean:
        return False, "User ID contains only invalid characters."
    return True, clean


def _get_user_file_path(client_id: str, clean_user_id: str) -> Path:
    """Returns the dedicated path for client/user JSON file."""
    clean_tenant = sanitize_id(client_id, "default_tenant")
    tenant_dir = USER_HISTORY_BASE_DIR / clean_tenant
    tenant_dir.mkdir(parents=True, exist_ok=True)
    return tenant_dir / f"{clean_user_id}.json"


def _read_user_file(file_path: Path, client_id: str, user_id: str) -> Dict[str, Any]:
    """
    Safely reads and deserializes the JSON history file.
    If missing or corrupted, handles gracefully with backup/fallback.
    """
    default_structure = {
        "tenant_id": client_id,
        "user_id": user_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "history_count": 0,
        "interactions": []
    }

    if not file_path.exists():
        return default_structure

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("JSON root is not an object")
            if "interactions" not in data or not isinstance(data["interactions"], list):
                data["interactions"] = []
            data["history_count"] = len(data["interactions"])
            data["tenant_id"] = client_id
            data["user_id"] = user_id
            return data
    except Exception as e:
        logger.warning(f"Corrupted or unreadable history file {file_path}: {e}. Creating fallback recovery.")
        # Attempt to rename corrupted file for post-mortem analysis
        try:
            corrupted_backup = file_path.with_suffix(f".corrupted_{int(datetime.now().timestamp())}")
            file_path.rename(corrupted_backup)
            logger.info(f"Backed up corrupted history file to {corrupted_backup}")
        except Exception:
            pass
        return default_structure


def _write_user_file(file_path: Path, data: Dict[str, Any]) -> None:
    """
    Safely writes data to JSON file atomically using a temporary file.
    """
    tmp_path = file_path.with_suffix(".tmp")
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    data["history_count"] = len(data.get("interactions", []))

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())

    # Atomic replace
    tmp_path.replace(file_path)


def detect_user_status(client_id: str, user_id: str) -> Dict[str, Any]:
    """
    Detects whether a user is Cold Start or Warm Start in the given tenant.
    - If 0 interactions -> status: 'cold_start'
    - If >= 1 interactions -> status: 'warm_start'
    Also returns history_count, last_active, recent_datasets, and interaction summary.
    """
    is_valid, result = validate_user_id(user_id)
    if not is_valid:
        raise ValueError(result)
    clean_uid = result
    clean_cid = sanitize_id(client_id, "default_tenant")

    lock_key = f"{clean_cid}::{clean_uid}"
    lock = _get_lock(lock_key)

    with lock:
        path = _get_user_file_path(clean_cid, clean_uid)
        user_data = _read_user_file(path, clean_cid, clean_uid)

    interactions = user_data.get("interactions", [])
    history_count = len(interactions)
    is_cold = (history_count == 0)

    # Collect unique datasets interacted with
    recent_datasets = []
    seen_ds = set()
    for item in reversed(interactions):
        ds_id = item.get("dataset_id")
        if ds_id and ds_id not in seen_ds:
            seen_ds.add(ds_id)
            recent_datasets.append(ds_id)

    last_active = interactions[-1]["timestamp"] if interactions else None

    return {
        "client_id": clean_cid,
        "user_id": clean_uid,
        "status": "cold_start" if is_cold else "warm_start",
        "is_cold_start": is_cold,
        "history_count": history_count,
        "last_active": last_active,
        "recent_datasets": recent_datasets[:5],
        "created_at": user_data.get("created_at")
    }


def get_user_chart_preferences(
    client_id: str,
    user_id: str
) -> Dict[str, Any]:
    """
    Analyzes persistent user interactions to compute bounded preference weights
    for visual chart types, column affinity, and positive/negative feedback.
    
    Returns:
    {
        "has_history": bool,
        "chart_type_weights": Dict[str, float], # relative preference deltas [-0.15, +0.15]
        "positive_charts": List[str],
        "negative_charts": List[str],
        "selected_charts_count": Dict[str, int],
        "total_interactions": int
    }
    """
    is_valid, result = validate_user_id(user_id)
    if not is_valid:
        return {
            "has_history": False,
            "has_relevant_preferences": False,
            "chart_type_weights": {},
            "positive_charts": [],
            "negative_charts": [],
            "selected_charts_count": {},
            "total_interactions": 0,
            "total_genuine_selections": 0
        }
    clean_uid = result
    clean_cid = sanitize_id(client_id, "default_tenant")

    lock_key = f"{clean_cid}::{clean_uid}"
    lock = _get_lock(lock_key)

    with lock:
        path = _get_user_file_path(clean_cid, clean_uid)
        user_data = _read_user_file(path, clean_cid, clean_uid)

    interactions = user_data.get("interactions", [])
    if not interactions:
        return {
            "has_history": False,
            "has_relevant_preferences": False,
            "chart_type_weights": {},
            "positive_charts": [],
            "negative_charts": [],
            "selected_charts_count": {},
            "total_interactions": 0,
            "total_genuine_selections": 0
        }

    chart_scores: Dict[str, float] = {}
    chart_counts: Dict[str, int] = {}
    positive_charts = set()
    negative_charts = set()

    # Canonical chart synonyms / normalization
    TYPE_SYNONYMS = {
        "box_plot": "box",
        "boxplot": "box",
        "stackedbar": "stacked_bar",
        "groupedbar": "grouped_bar",
        "piechart": "pie",
        "donutchart": "donut",
        "areachart": "area",
        "bubblechart": "bubble"
    }

    # Explicit event separation to prevent displayed impressions from polluting preferences
    GENUINE_SELECTION_TYPES = {"recommendation_click", "chart_selection", "chart_apply", "chart_render"}
    POSITIVE_FEEDBACK_TYPES = {"helpful_feedback"}
    NEGATIVE_FEEDBACK_TYPES = {"unhelpful_feedback"}
    IGNORE_ENTRY_TYPES = {"visual_recommendation", "chart_impression", "impression", "dataset_upload"}

    total_genuine_selections = 0

    for ev in interactions:
        e_type = ev.get("entry_type", "")
        # Strictly ignore mere recommendation displays and impressions
        if e_type in IGNORE_ENTRY_TYPES:
            continue

        fb = (ev.get("feedback") or "").lower()

        # Extract chart types from items_or_charts or metadata
        c_types = []
        for it in ev.get("items_or_charts", []):
            ct = it.get("chart_type")
            if ct:
                c_types.append(TYPE_SYNONYMS.get(ct.lower(), ct.lower()))
        
        # Check metadata details
        meta_ct = ev.get("metadata", {}).get("chart_type")
        if meta_ct:
            c_types.append(TYPE_SYNONYMS.get(meta_ct.lower(), meta_ct.lower()))

        distinct_cts = set(c_types)
        is_positive = (e_type in POSITIVE_FEEDBACK_TYPES or fb in ("helpful", "like", "positive", "accept", "upvote"))
        is_negative = (e_type in NEGATIVE_FEEDBACK_TYPES or fb in ("unhelpful", "dislike", "negative", "reject", "downvote"))
        is_selection = (e_type in GENUINE_SELECTION_TYPES)

        if not (is_positive or is_negative or is_selection):
            continue

        for ct in distinct_cts:
            current = chart_scores.get(ct, 0.0)

            # Weight by deliberate user action
            if is_selection:
                chart_counts[ct] = chart_counts.get(ct, 0) + 1
                total_genuine_selections += 1
                current += 0.05
            
            if is_positive:
                current += 0.08
                positive_charts.add(ct)
            elif is_negative:
                current -= 0.10
                negative_charts.add(ct)
            
            chart_scores[ct] = current

    # Strictly bound the preference weights to [-0.12, +0.12]
    # so user history can never overpower dataset suitability or MLVis confidence
    bounded_weights = {}
    for ct, score in chart_scores.items():
        bounded_weights[ct] = round(max(min(score, 0.12), -0.12), 3)

    has_relevant_preferences = (
        total_genuine_selections > 0 or len(positive_charts) > 0 or len(negative_charts) > 0
    )

    return {
        "has_history": len(interactions) > 0,
        "has_relevant_preferences": has_relevant_preferences,
        "chart_type_weights": bounded_weights,
        "positive_charts": list(positive_charts),
        "negative_charts": list(negative_charts),
        "selected_charts_count": chart_counts,
        "total_interactions": len(interactions),
        "total_genuine_selections": total_genuine_selections
    }



def get_user_history(
    client_id: str,
    user_id: str,
    limit: int = 50,
    offset: int = 0
) -> Dict[str, Any]:
    """
    Retrieves full or paginated interaction history for a user in a tenant.
    """
    is_valid, result = validate_user_id(user_id)
    if not is_valid:
        raise ValueError(result)
    clean_uid = result
    clean_cid = sanitize_id(client_id, "default_tenant")

    lock_key = f"{clean_cid}::{clean_uid}"
    lock = _get_lock(lock_key)

    with lock:
        path = _get_user_file_path(clean_cid, clean_uid)
        user_data = _read_user_file(path, clean_cid, clean_uid)

    interactions = user_data.get("interactions", [])
    # Return reversed order (newest first)
    sorted_interactions = list(reversed(interactions))
    paginated = sorted_interactions[offset:offset + limit]

    return {
        "client_id": clean_cid,
        "user_id": clean_uid,
        "status": "cold_start" if len(interactions) == 0 else "warm_start",
        "total_records": len(interactions),
        "limit": limit,
        "offset": offset,
        "history": paginated
    }


def record_user_history_entry(
    client_id: str,
    user_id: str,
    entry_type: str,
    dataset_id: Optional[str] = None,
    items_or_charts: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    feedback: Optional[str] = None,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Appends a new interaction event to the user's persistent history file.
    Does NOT store full CSVs or oversized data - only structured metadata.
    """
    is_valid, result = validate_user_id(user_id)
    if not is_valid:
        raise ValueError(result)
    clean_uid = result
    clean_cid = sanitize_id(client_id, "default_tenant")

    lock_key = f"{clean_cid}::{clean_uid}"
    lock = _get_lock(lock_key)

    entry = {
        "event_id": f"evt_{int(datetime.now().timestamp() * 1000)}",
        "entry_type": str(entry_type).strip(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_id": str(dataset_id).strip() if dataset_id else None,
        "session_id": str(session_id).strip() if session_id else None,
        "items_or_charts": items_or_charts or [],
        "metadata": metadata or {},
        "feedback": str(feedback).strip() if feedback else None
    }

    with lock:
        path = _get_user_file_path(clean_cid, clean_uid)
        user_data = _read_user_file(path, clean_cid, clean_uid)
        user_data.setdefault("interactions", []).append(entry)
        _write_user_file(path, user_data)

    logger.info(f"Recorded user history entry '{entry_type}' for user '{clean_uid}' in tenant '{clean_cid}'.")
    return {
        "status": "recorded",
        "event_id": entry["event_id"],
        "client_id": clean_cid,
        "user_id": clean_uid,
        "total_history_count": len(user_data["interactions"]),
        "start_status": "warm_start"
    }


def clear_user_history(client_id: str, user_id: str) -> bool:
    """
    Clears interaction history for a user in a tenant (useful for testing or reset).
    """
    is_valid, result = validate_user_id(user_id)
    if not is_valid:
        raise ValueError(result)
    clean_uid = result
    clean_cid = sanitize_id(client_id, "default_tenant")

    lock_key = f"{clean_cid}::{clean_uid}"
    lock = _get_lock(lock_key)

    with lock:
        path = _get_user_file_path(clean_cid, clean_uid)
        if path.exists():
            path.unlink()
            return True
    return False
