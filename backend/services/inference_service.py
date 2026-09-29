"""
Real-Time Inference Engine & Live Candidate Scoring Service.
Handles online candidate generation, transform-only feature preparation,
batched PyTorch neural scoring, cold-start adaptation, grounded explainability,
and recommendation event logging.
"""

import time
import json
import uuid
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from backend.database.models import ModelVersionModel, RecommendationEventModel
from backend.schemas.recommendation import (
    RecommendationRequest,
    RecommendationResponse,
    RecommendedItem,
    RecommendationFilter
)
from backend.services.client_service import get_client_or_404
from backend.services.model_cache import get_model_entry, ModelCacheEntry
from src.explainability import generate_recommendation_explanation

logger = logging.getLogger("InferenceService")

# Outcome and target fields strictly forbidden as recommendation-time predictors
FORBIDDEN_FIELDS = {
    "revenue", "rating", "review_text", "review_helpful_votes",
    "cart_abandoned", "payment_method", "revenue_normalized",
    "added_to_cart", "purchased"
}


def _resolve_and_validate_filters(
    filters_in: Optional[Any],
    client_id: str,
    entry: ModelCacheEntry
) -> Dict[str, Any]:
    """
    Validates request filter keys against the client schema's content features.
    Raises 422 Unprocessable Entity if unknown filter keys are provided.
    """
    if not filters_in:
        return {}

    if isinstance(filters_in, RecommendationFilter):
        raw_filters = filters_in.model_dump(exclude_none=True)
    elif isinstance(filters_in, dict):
        raw_filters = {k: v for k, v in filters_in.items() if v is not None}
    else:
        raw_filters = {}

    if not raw_filters:
        return {}

    # Determine supported filter fields from schema or catalog
    if client_id == "demo_ecommerce":
        allowed_exact = {"product_category", "unit_price", "discount_percent", "discount_amount", "product_id"}
        allowed_prefixes = {"price": "unit_price", "unit_price": "unit_price", "discount": "discount_percent"}
        allowed_direct_aliases = {"category": "product_category", "max_price": ("unit_price", "max"), "min_price": ("unit_price", "min")}
    else:
        cfg = entry.schema_config
        content_cols = cfg.get_content_cols() if cfg else []
        item_col = cfg.get_item_id_col() if cfg else "item_id"
        allowed_exact = set(content_cols) | {item_col}
        allowed_prefixes = {c: c for c in content_cols}
        allowed_direct_aliases = {}

    validated_filters = {}
    for k, v in raw_filters.items():
        k_clean = k.strip()

        # Direct match
        if k_clean in allowed_exact:
            validated_filters[k_clean] = ("eq", v)
            continue

        # Direct alias match
        if k_clean in allowed_direct_aliases:
            alias_val = allowed_direct_aliases[k_clean]
            if isinstance(alias_val, tuple):
                col, op = alias_val
                validated_filters[f"{op}_{col}"] = (op, v)
            else:
                validated_filters[alias_val] = ("eq", v)
            continue

        # min_ / max_ prefix match
        if k_clean.startswith("min_"):
            field = k_clean[4:]
            target_col = allowed_prefixes.get(field, field if field in allowed_exact else None)
            if target_col:
                validated_filters[f"min_{target_col}"] = ("min", v)
                continue

        if k_clean.startswith("max_"):
            field = k_clean[4:]
            target_col = allowed_prefixes.get(field, field if field in allowed_exact else None)
            if target_col:
                validated_filters[f"max_{target_col}"] = ("max", v)
                continue

        # Unknown filter key
        supported_str = ", ".join(sorted(list(allowed_exact) + ["min_price", "max_price", "category"]))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid filter field '{k_clean}'. Supported filter fields for client '{client_id}' are: {supported_str}"
        )

    return validated_filters


def _apply_filters_to_catalog(
    catalog: List[Dict[str, Any]],
    filters: Dict[str, Any],
    client_id: str,
    entry: ModelCacheEntry
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Applies validated filters and excludes unsupported items.
    Returns (eligible_items, unsupported_count).
    """
    unsupported_count = 0
    eligible = []

    for item in catalog:
        # Check item ID validity
        if client_id == "demo_ecommerce":
            try:
                pid = int(item["product_id"])
                if pid < 0 or pid >= 1000:
                    unsupported_count += 1
                    continue
            except (ValueError, TypeError):
                unsupported_count += 1
                continue
        else:
            item_col = entry.schema_config.get_item_id_col() if entry.schema_config else "item_id"
            raw_id = item.get(item_col)
            if entry.item_id_map is not None and raw_id not in entry.item_id_map:
                unsupported_count += 1
                continue

        # Apply content filters
        match = True
        for key, (op, val) in filters.items():
            if op == "eq":
                col_name = key
                if col_name in item and str(item[col_name]) != str(val):
                    match = False
                    break
            elif op == "min":
                col_name = key[4:]  # strip 'min_'
                if col_name in item:
                    try:
                        if float(item[col_name]) < float(val):
                            match = False
                            break
                    except (ValueError, TypeError):
                        match = False
                        break
            elif op == "max":
                col_name = key[4:]  # strip 'max_'
                if col_name in item:
                    try:
                        if float(item[col_name]) > float(val):
                            match = False
                            break
                    except (ValueError, TypeError):
                        match = False
                        break

        if match:
            eligible.append(item)

    return eligible, unsupported_count


def _build_context_vector(
    context_dict: Dict[str, Any],
    client_id: str,
    entry: ModelCacheEntry
) -> np.ndarray:
    """
    Constructs context feature vector using fitted encoders without fitting anything.
    """
    if client_id == "demo_ecommerce":
        ctx_cat_encoder = entry.encoders["ctx_cat_encoder"]
        cat_cols = ["device_type", "user_type", "marketing_channel", "visit_season"]
        cat_vals = [context_dict.get(c, 0) for c in cat_cols]
        cat_df = pd.DataFrame([cat_vals], columns=cat_cols)
        cat_enc = ctx_cat_encoder.transform(cat_df)

        day = float(context_dict.get("visit_day", 15))
        month = float(context_dict.get("visit_month", 6))
        weekday = float(context_dict.get("visit_weekday", 2))
        loc = float(context_dict.get("location", 50))

        day_sin = np.sin(2 * np.pi * day / 31.0)
        day_cos = np.cos(2 * np.pi * day / 31.0)
        month_sin = np.sin(2 * np.pi * month / 12.0)
        month_cos = np.cos(2 * np.pi * month / 12.0)
        weekday_sin = np.sin(2 * np.pi * weekday / 7.0)
        weekday_cos = np.cos(2 * np.pi * weekday / 7.0)
        loc_norm = loc / 225.0

        ctx_num = np.array([[day_sin, day_cos, month_sin, month_cos, weekday_sin, weekday_cos, loc_norm]], dtype=np.float32)
        return np.hstack([cat_enc, ctx_num]).astype(np.float32)
    else:
        cfg = entry.schema_config
        ctx_cols = cfg.get_context_cols() if cfg else []
        ctx_cat_encoder = entry.encoders.get("ctx_cat_encoder")
        ctx_num_scaler = entry.encoders.get("ctx_num_scaler")

        ctx_cat_cols = list(ctx_cat_encoder.feature_names_in_) if (ctx_cat_encoder and hasattr(ctx_cat_encoder, "feature_names_in_")) else [c for c in ctx_cols if (cfg.column_types.get(c) == "categorical")]
        ctx_num_cols = list(ctx_num_scaler.feature_names_in_) if (ctx_num_scaler and hasattr(ctx_num_scaler, "feature_names_in_")) else [c for c in ctx_cols if c not in ctx_cat_cols]

        parts = []
        if ctx_cat_encoder and ctx_cat_cols:
            vals = []
            for i, c in enumerate(ctx_cat_cols):
                cat_dtype = ctx_cat_encoder.categories_[i].dtype if hasattr(ctx_cat_encoder, "categories_") and i < len(ctx_cat_encoder.categories_) else None
                if c in context_dict and context_dict[c] is not None and str(context_dict[c]).strip() != "":
                    raw_val = context_dict[c]
                    if cat_dtype is not None and cat_dtype.kind in ('i', 'u', 'f'):
                        try:
                            vals.append(int(float(raw_val)))
                        except (ValueError, TypeError):
                            vals.append(0)
                    else:
                        vals.append(str(raw_val))
                else:
                    if cat_dtype is not None and cat_dtype.kind in ('i', 'u', 'f'):
                        vals.append(0)
                    else:
                        vals.append("Unknown")

            df_cat = pd.DataFrame([vals], columns=ctx_cat_cols)
            parts.append(ctx_cat_encoder.transform(df_cat))

        if ctx_num_scaler and ctx_num_cols:
            vals = []
            for c in ctx_num_cols:
                raw_val = context_dict.get(c)
                if raw_val is not None and str(raw_val).strip() != "":
                    try:
                        vals.append(float(raw_val))
                    except (ValueError, TypeError):
                        vals.append(0.0)
                else:
                    vals.append(0.0)
            df_num = pd.DataFrame([vals], columns=ctx_num_cols)
            parts.append(ctx_num_scaler.transform(df_num))

        if not parts:
            if entry.context_dim != 1:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Feature contract violation: model expects context dimension {entry.context_dim}, but client schema has no context features configured."
                )
            return np.ones((1, entry.context_dim), dtype=np.float32)
        res = np.hstack(parts).astype(np.float32)
        if res.shape[1] != entry.context_dim:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Feature contract violation: transformed context dimension ({res.shape[1]}) does not match model expected dimension ({entry.context_dim})."
            )
        return res


def _build_candidate_content_matrix(
    candidates: List[Dict[str, Any]],
    client_id: str,
    entry: ModelCacheEntry
) -> np.ndarray:
    """
    Constructs candidate content feature matrix for all eligible items.
    Enforces strict feature contract matching the active model's training-time preprocessors.
    """
    if client_id == "demo_ecommerce":
        prod_cat_encoder = entry.encoders["prod_cat_encoder"]
        content_scaler = entry.encoders["content_scaler"]

        required_cols = ["product_id", "product_category", "unit_price", "discount_percent", "discount_amount"]
        for col in required_cols:
            if any(col not in item for item in candidates):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Inference feature contract violation: candidate items are missing required feature column '{col}' for demo_ecommerce model."
                )

        pids = np.array([[int(item["product_id"])] for item in candidates], dtype=np.float32)
        cats = [item["product_category"] for item in candidates]
        cat_df = pd.DataFrame({"product_category": cats})
        cat_enc = prod_cat_encoder.transform(cat_df)

        num_cols = ["unit_price", "discount_percent", "discount_amount"]
        nums = [[item["unit_price"], item["discount_percent"], item["discount_amount"]] for item in candidates]
        num_df = pd.DataFrame(nums, columns=num_cols)
        num_scaled = content_scaler.transform(num_df)

        mat = np.hstack([pids, cat_enc, num_scaled]).astype(np.float32)
        if mat.shape[1] != entry.content_dim:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Feature contract violation: transformed candidate content dimension ({mat.shape[1]}) does not match model expected dimension ({entry.content_dim})."
            )
        return mat
    else:
        cfg = entry.schema_config
        item_col = cfg.get_item_id_col() if cfg else "item_id"
        content_cols = [c for c in cfg.get_content_cols() if c != item_col] if cfg else []
        cont_cat_encoder = entry.encoders.get("cont_cat_encoder")
        content_scaler = entry.encoders.get("content_scaler")

        cont_cat_cols = list(cont_cat_encoder.feature_names_in_) if (cont_cat_encoder and hasattr(cont_cat_encoder, "feature_names_in_")) else [c for c in content_cols if cfg.column_types.get(c) == "categorical"]
        cont_num_cols = list(content_scaler.feature_names_in_) if (content_scaler and hasattr(content_scaler, "feature_names_in_")) else [c for c in content_cols if c not in cont_cat_cols]

        # Verify candidate items contain all required content columns
        missing_cat = [c for c in cont_cat_cols if any(c not in item for item in candidates)]
        missing_num = [c for c in cont_num_cols if any(c not in item for item in candidates)]
        if missing_cat or missing_num:
            missing_cols = sorted(list(set(missing_cat + missing_num)))
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Inference feature contract violation: candidate items are missing required feature columns: {missing_cols}"
            )

        pids = np.array([[entry.item_id_map.get(item.get(item_col), 0)] for item in candidates], dtype=np.float32)
        parts = [pids]

        if cont_cat_encoder and cont_cat_cols:
            cats = []
            for item in candidates:
                row_vals = {}
                for i, c in enumerate(cont_cat_cols):
                    cat_dtype = cont_cat_encoder.categories_[i].dtype if hasattr(cont_cat_encoder, "categories_") and i < len(cont_cat_encoder.categories_) else None
                    val = item.get(c)
                    if val is not None and str(val).strip() != "":
                        if cat_dtype is not None and cat_dtype.kind in ('i', 'u', 'f'):
                            try:
                                row_vals[c] = int(float(val))
                            except (ValueError, TypeError):
                                row_vals[c] = 0
                        else:
                            row_vals[c] = str(val)
                    else:
                        if cat_dtype is not None and cat_dtype.kind in ('i', 'u', 'f'):
                            row_vals[c] = 0
                        else:
                            row_vals[c] = "Unknown"
                cats.append(row_vals)
            df_cat = pd.DataFrame(cats)
            parts.append(cont_cat_encoder.transform(df_cat))

        if content_scaler and cont_num_cols:
            nums = [{c: float(item.get(c, 0.0)) for c in cont_num_cols} for item in candidates]
            df_num = pd.DataFrame(nums)
            parts.append(content_scaler.transform(df_num))

        mat = np.hstack(parts).astype(np.float32)
        if mat.shape[1] != entry.content_dim:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Feature contract violation: transformed candidate content dimension ({mat.shape[1]}) does not match model expected dimension ({entry.content_dim})."
            )
        return mat


def generate_recommendations(
    request: RecommendationRequest,
    db: Session
) -> RecommendationResponse:
    """
    Real-time recommendation generation pipeline.
    Validates client, loads cached model, processes context and history,
    scores eligible candidate items, ranks them, generates grounded explanations,
    measures latency, and persists the recommendation event.
    """
    t_start = time.perf_counter()

    # 1. Validate client
    client = get_client_or_404(db, request.client_id)
    cid = client.client_id

    # 2. Retrieve active model version
    active_version = (
        db.query(ModelVersionModel)
        .filter(ModelVersionModel.client_id == cid, ModelVersionModel.is_active == True)
        .first()
    )
    if not active_version:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No active model is available for this client."
        )

    # 3. Retrieve model cache entry (thread-safe, verified architecture)
    entry = get_model_entry(cid, active_version.version_tag, db)

    # 4. Filters validation and candidate selection
    validated_filters = _resolve_and_validate_filters(request.filters, cid, entry)
    eligible_candidates, unsupported_count = _apply_filters_to_catalog(
        catalog=entry.catalog,
        filters=validated_filters,
        client_id=cid,
        entry=entry
    )
    candidate_count = len(eligible_candidates)

    # 5. User historical profile retrieval & cold-start determination
    # User ID can be int or str in customer_profiles
    uid_key = request.user_id.strip()
    profile = entry.customer_profiles.get(uid_key)
    if profile is None:
        try:
            profile = entry.customer_profiles.get(int(uid_key))
        except (ValueError, TypeError):
            profile = None

    if profile is not None:
        history_count = int(profile["count"])
        is_cold = (history_count < entry.cold_start_threshold)
        status_str = "cold_start" if is_cold else "warm_start"
        top_past_category = profile.get("top_past_category")
        last_product_id = profile.get("last_product_id")

        if cid == "demo_ecommerce":
            beh_scaler = entry.encoders["beh_scaler"]
            beh_num = [
                profile["count"], profile["purchases"], profile["purchase_rate"],
                profile["revenue"], profile["cart_add_rate"], profile["avg_pages_viewed"],
                profile["avg_time_on_site"], profile["recency_score"]
            ]
            df_beh = pd.DataFrame([beh_num], columns=entry.encoders["beh_num_cols"])
            scaled_num = beh_scaler.transform(df_beh)
            cat_prefs = np.array(profile["cat_prefs"]).reshape(1, -1)
            x_beh = np.hstack([scaled_num, cat_prefs]).astype(np.float32)
        else:
            beh_scaler = entry.encoders["beh_scaler"]
            beh_num = [profile["count"], profile["purchases"], profile["purchase_rate"], profile["recency_score"]]
            for beh_col in entry.schema_config.get_behavior_cols():
                avg_val = profile.get(f"hist_avg_{beh_col}", 0.0)
                beh_num.append(avg_val)
            df_beh = pd.DataFrame([beh_num], columns=entry.encoders["beh_num_cols"])
            scaled_num = beh_scaler.transform(df_beh)
            beh_cat_pref_cols = entry.encoders.get("beh_cat_pref_cols", [])
            if beh_cat_pref_cols:
                cat_prefs = profile.get("cat_prefs")
                if cat_prefs is None or len(cat_prefs) != len(beh_cat_pref_cols):
                    cat_prefs = np.zeros((1, len(beh_cat_pref_cols)), dtype=np.float32)
                else:
                    cat_prefs = np.array(cat_prefs, dtype=np.float32).reshape(1, -1)
                x_beh = np.hstack([scaled_num, cat_prefs]).astype(np.float32)
            else:
                x_beh = scaled_num.astype(np.float32)
    else:
        # Completely unknown user: history_count = 0, zero behavioral representation
        history_count = 0
        is_cold = True
        status_str = "cold_start"
        top_past_category = None
        last_product_id = None
        x_beh = np.zeros((1, entry.behavior_dim), dtype=np.float32)

    if x_beh.shape[1] != entry.behavior_dim:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Feature contract violation: transformed behavior dimension ({x_beh.shape[1]}) does not match model expected dimension ({entry.behavior_dim})."
        )

    # 6. Context vector
    x_ctx = _build_context_vector(request.context, cid, entry)

    # 7. Candidate scoring
    recommendations_list: List[RecommendedItem] = []
    w_beh, w_cont, w_ctx = 0.0, 0.0, 0.0

    if candidate_count > 0:
        # Build candidate matrix
        X_cands = _build_candidate_content_matrix(eligible_candidates, cid, entry)

        # Session anchor content representation
        # If user has an anchor item in history, use it; otherwise use candidate centroid
        anchor_idx = None
        if last_product_id is not None:
            id_col = "product_id" if cid == "demo_ecommerce" else (entry.schema_config.get_item_id_col() if entry.schema_config else "item_id")
            for i, c in enumerate(eligible_candidates):
                if str(c.get(id_col)) == str(last_product_id):
                    anchor_idx = i
                    break

        if anchor_idx is not None:
            x_cont_anchor = X_cands[anchor_idx:anchor_idx + 1]
        else:
            x_cont_anchor = X_cands.mean(axis=0, keepdims=True)

        # PyTorch batched forward pass in eval mode without gradients
        with torch.no_grad():
            t_beh = torch.from_numpy(x_beh)
            t_anchor = torch.from_numpy(x_cont_anchor)
            t_ctx = torch.from_numpy(x_ctx)
            t_hist = torch.tensor([history_count], dtype=torch.long)
            t_cands = torch.from_numpy(X_cands)

            e_user, weights, cold_mask = entry.model.encode_user_context(
                x_beh=t_beh,
                x_cont_anchor=t_anchor,
                x_ctx=t_ctx,
                hist_counts=t_hist
            )
            scores = entry.model.score_candidates(e_user, t_cands)
            scores_np = scores.cpu().numpy()

            w_beh = round(float(weights[0, 0].item()), 4)
            w_cont = round(float(weights[0, 1].item()), 4)
            w_ctx = round(float(weights[0, 2].item()), 4)

        # 8. Candidate ranking and Top-K selection
        k = min(request.top_k, candidate_count)
        ranked_indices = np.argsort(-scores_np)[:k]

        id_key = "product_id" if cid == "demo_ecommerce" else (entry.schema_config.get_item_id_col() if entry.schema_config else "item_id")
        cat_key = "product_category" if cid == "demo_ecommerce" else "category"

        device_name = request.context.get("device_type", "Desktop")
        if isinstance(device_name, int):
            device_names = {0: "Desktop", 1: "Mobile", 2: "Tablet"}
            device_name = device_names.get(device_name, f"Device {device_name}")

        for rank_idx, idx in enumerate(ranked_indices, start=1):
            cand_item = eligible_candidates[idx]
            raw_item_id = str(cand_item.get(id_key))
            score_val = round(float(scores_np[idx]), 4)
            item_cat = cand_item.get(cat_key, 0)
            try:
                item_cat_code = int(item_cat)
            except (ValueError, TypeError):
                item_cat_code = 0

            # Grounded truthful explanation
            explanation_data = generate_recommendation_explanation(
                customer_id=request.user_id,
                product_id=raw_item_id,
                product_category=item_cat_code,
                hist_count=history_count,
                modality_weights=[w_beh, w_cont, w_ctx],
                top_past_category=top_past_category,
                device_name=str(device_name)
            )

            # Metadata without target or outcome fields
            clean_meta = {
                k_m: v_m for k_m, v_m in cand_item.items()
                if k_m not in FORBIDDEN_FIELDS
            }

            recommendations_list.append(RecommendedItem(
                rank=rank_idx,
                item_id=raw_item_id,
                score=score_val,
                score_type="sigmoid_score",
                metadata=clean_meta,
                attributes=clean_meta,
                explanation=explanation_data["explanation"]
            ))

    # 9. Timing & Event Persistence
    t_end = time.perf_counter()
    latency_ms = round((t_end - t_start) * 1000.0, 2)
    rec_id = f"rec_{uuid.uuid4().hex[:12]}"

    recommended_ids = [r.item_id for r in recommendations_list]
    scores_list = [r.score for r in recommendations_list]

    event = RecommendationEventModel(
        id=rec_id,
        client_id=cid,
        user_id=request.user_id.strip(),
        model_version=active_version.version_tag,
        status=status_str,
        history_count=history_count,
        candidate_count=candidate_count,
        top_k=request.top_k,
        context_json=json.dumps(request.context),
        filters_json=json.dumps(validated_filters),
        recommended_items_json=json.dumps(recommended_ids),
        scores_json=json.dumps(scores_list),
        latency_ms=latency_ms
    )
    db.add(event)
    db.commit()

    return RecommendationResponse(
        recommendation_id=rec_id,
        client_id=cid,
        model_version=active_version.version_tag,
        user_id=request.user_id.strip(),
        status=status_str,
        history_count=history_count,
        cold_start_threshold=entry.cold_start_threshold,
        modality_weights={
            "behavior": w_beh,
            "content": w_cont,
            "context": w_ctx
        },
        recommendations=recommendations_list,
        candidate_count=candidate_count,
        unsupported_candidate_count=unsupported_count,
        latency_ms=latency_ms
    )
