"""
Model Cache Service for Real-Time Inference.
Manages in-memory cached PyTorch models, fitted preprocessors, client item catalogs,
and customer historical profiles keyed by (client_id, model_version).
Provides thread-safe loading, validation against checkpoint architecture,
per-client cache invalidation, and global cache clearing.
"""

import os
import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional
import numpy as np
import pandas as pd
import joblib
import torch

from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from backend.database.models import ModelVersionModel, DatasetModel
from src.models.recommender import TwoTowerMultiModalRecommender
from src.schema_mapper import ClientSchemaConfig

logger = logging.getLogger("ModelCache")


@dataclass
class ModelCacheEntry:
    client_id: str
    model_version: str
    model: TwoTowerMultiModalRecommender
    encoders: Dict[str, Any]
    metadata: Dict[str, Any]
    schema_config: Optional[ClientSchemaConfig]
    catalog: List[Dict[str, Any]]
    customer_profiles: Dict[Any, Dict[str, Any]]
    item_id_map: Optional[Dict[Any, int]]
    cold_start_threshold: int
    behavior_dim: int
    content_dim: int
    context_dim: int


_cache: Dict[Tuple[str, str], ModelCacheEntry] = {}
_lock = threading.Lock()


def _build_demo_ecommerce_catalog_and_profiles() -> Tuple[List[Dict[str, Any]], Dict[Any, Dict[str, Any]]]:
    """
    Constructs the item catalog and point-in-time customer historical profiles
    for demo_ecommerce strictly from the validated historical training split (train.csv).
    Guarantees zero future or test leakage.
    """
    train_path = Path("data/processed/train.csv")
    if not train_path.exists():
        raise FileNotFoundError(f"Historical training data for demo_ecommerce not found at {train_path}")

    df_train = pd.read_csv(train_path)

    # 1. Item Catalog (latest valid content attributes per product_id)
    content_cols = ["product_category", "unit_price", "discount_percent", "discount_amount"]
    catalog_df = (
        df_train[["product_id"] + content_cols]
        .drop_duplicates(subset=["product_id"], keep="last")
        .sort_values("product_id")
        .reset_index(drop=True)
    )
    catalog = catalog_df.to_dict(orient="records")

    # 2. Point-in-time Customer Profiles
    if "parsed_visit_date" not in df_train.columns:
        df_train["parsed_visit_date"] = pd.to_datetime(df_train["visit_date"], format="%d-%m-%Y")
    else:
        df_train["parsed_visit_date"] = pd.to_datetime(df_train["parsed_visit_date"])

    df_sorted = df_train.sort_values(by=["parsed_visit_date", "session_id"]).reset_index(drop=True)
    max_train_date = df_sorted["parsed_visit_date"].max()

    customer_profiles: Dict[Any, Dict[str, Any]] = {}
    for _, row in df_sorted.iterrows():
        cust_id = int(row["customer_id"])
        pid = int(row["product_id"])
        cat = int(row["product_category"])
        purch = int(row.get("purchased", 0))
        rev = float(row.get("revenue", 0.0))
        cart = int(row.get("added_to_cart", 0))
        pages = float(row.get("pages_viewed", 0.0))
        tos = float(row.get("time_on_site_sec", 0.0))
        dt = row["parsed_visit_date"]

        if cust_id not in customer_profiles:
            cat_counts = np.zeros(8, dtype=np.float32)
            cat_counts[min(cat, 7)] += 1.0
            customer_profiles[cust_id] = {
                "count": 1,
                "purchases": purch,
                "revenue": rev,
                "cart_adds": cart,
                "total_pages": pages,
                "total_time": tos,
                "last_date": dt,
                "cat_counts": cat_counts,
                "last_product_id": pid
            }
        else:
            p = customer_profiles[cust_id]
            p["count"] += 1
            p["purchases"] += purch
            p["revenue"] += rev
            p["cart_adds"] += cart
            p["total_pages"] += pages
            p["total_time"] += tos
            p["last_date"] = dt
            p["cat_counts"][min(cat, 7)] += 1.0
            p["last_product_id"] = pid

    # Finalize derived historical metrics per customer
    for cust_id, p in customer_profiles.items():
        cnt = p["count"]
        p["purchase_rate"] = p["purchases"] / cnt if cnt > 0 else 0.0
        p["cart_add_rate"] = p["cart_adds"] / cnt if cnt > 0 else 0.0
        p["avg_pages_viewed"] = p["total_pages"] / cnt if cnt > 0 else 0.0
        p["avg_time_on_site"] = p["total_time"] / cnt if cnt > 0 else 0.0
        delta_days = (max_train_date - p["last_date"]).days
        p["recency_days"] = max(0.0, float(delta_days))
        p["recency_score"] = float(np.exp(-0.01 * p["recency_days"]))
        p["cat_prefs"] = (p["cat_counts"] / cnt).tolist()
        p["top_past_category"] = int(np.argmax(p["cat_counts"]))

    logger.info(f"Loaded demo_ecommerce catalog ({len(catalog)} products) and profiles ({len(customer_profiles)} customers).")
    return catalog, customer_profiles


def _build_generic_catalog_and_profiles(
    client_id: str,
    schema_config: ClientSchemaConfig,
    encoders: Dict[str, Any],
    db: Session
) -> Tuple[List[Dict[str, Any]], Dict[Any, Dict[str, Any]]]:
    """
    Constructs isolated item catalog and customer historical profiles for a generic client
    using its configured schema and training interaction files.
    """
    item_col = schema_config.get_item_id_col()
    user_col = schema_config.get_user_id_col()
    target_col = schema_config.get_target_col()
    content_cols = [c for c in schema_config.get_content_cols() if c != item_col]

    # Locate training or processed dataset
    data_candidates = [
        Path(f"clients/{client_id}/processed/train.csv"),
        Path(f"clients/{client_id}/processed/cleaned_dataset.csv")
    ]
    train_file = None
    for p in data_candidates:
        if p.exists():
            train_file = p
            break

    if train_file is None:
        ds_record = db.query(DatasetModel).filter(DatasetModel.client_id == client_id).first()
        if ds_record and Path(ds_record.file_path).exists():
            train_file = Path(ds_record.file_path)

    if train_file is None:
        logger.warning(f"No training data found for client '{client_id}'. Empty catalog initialized.")
        return [], {}

    df = pd.read_csv(train_file)

    # 1. Item Catalog
    keep_cols = [item_col] + content_cols
    existing_cols = [c for c in keep_cols if c in df.columns]
    catalog_df = (
        df[existing_cols]
        .drop_duplicates(subset=[item_col], keep="last")
        .reset_index(drop=True)
    )
    catalog = catalog_df.to_dict(orient="records")

    # 2. Customer Profiles
    customer_profiles: Dict[Any, Dict[str, Any]] = {}
    beh_cat_pref_cols = encoders.get("beh_cat_pref_cols", [])
    num_cat_prefs = len(beh_cat_pref_cols)
    cat_col = None
    cat_map: Dict[Any, int] = {}
    if num_cat_prefs > 0:
        for c in content_cols:
            if c in df.columns and (schema_config.column_types.get(c) == "categorical" or df[c].nunique() <= 30):
                cat_col = c
                unique_cats = sorted(df[c].dropna().unique())
                cat_map = {val: idx for idx, val in enumerate(unique_cats)}
                break

    if user_col in df.columns:
        for cust_id, group in df.groupby(user_col):
            cnt = len(group)
            purch = int(group[target_col].sum()) if target_col and target_col in group.columns else 0
            last_item = group.iloc[-1][item_col] if item_col in group.columns else None

            profile = {
                "count": cnt,
                "purchases": purch,
                "purchase_rate": purch / cnt if cnt > 0 else 0.0,
                "recency_score": 1.0,
                "last_product_id": last_item
            }
            # Add averages for any configured behavioral columns
            for beh_col in schema_config.get_behavior_cols():
                if beh_col in group.columns and pd.api.types.is_numeric_dtype(group[beh_col]):
                    profile[f"hist_avg_{beh_col}"] = float(group[beh_col].mean())

            if num_cat_prefs > 0 and cat_col and cat_col in group.columns:
                counts = np.zeros(num_cat_prefs, dtype=np.float32)
                for val in group[cat_col].dropna():
                    idx = cat_map.get(val)
                    if idx is not None and idx < num_cat_prefs:
                        counts[idx] += 1
                total_c = counts.sum()
                profile["cat_prefs"] = (counts / total_c).tolist() if total_c > 0 else counts.tolist()
                profile["top_past_category"] = int(np.argmax(counts))
            elif num_cat_prefs > 0:
                profile["cat_prefs"] = [0.0] * num_cat_prefs

            customer_profiles[str(cust_id)] = profile
            customer_profiles[cust_id] = profile

    logger.info(f"Loaded generic client '{client_id}' catalog ({len(catalog)} items) and {len(customer_profiles)} profiles.")
    return catalog, customer_profiles


def get_model_entry(client_id: str, model_version: str, db: Session) -> ModelCacheEntry:
    """
    Retrieves or loads the ModelCacheEntry for (client_id, model_version).
    Verifies architectural agreement between metadata and checkpoint weights.
    """
    cache_key = (client_id, model_version)
    if cache_key in _cache:
        return _cache[cache_key]

    with _lock:
        if cache_key in _cache:
            return _cache[cache_key]

        # 1. Query database record for model version
        version_record = (
            db.query(ModelVersionModel)
            .filter(
                ModelVersionModel.client_id == client_id,
                ModelVersionModel.version_tag == model_version
            )
            .first()
        )
        if not version_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model version '{model_version}' not found for client '{client_id}'."
            )

        ckpt_path = Path(version_record.checkpoint_path)
        enc_path = Path(version_record.encoders_path)

        if not ckpt_path.exists():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Model checkpoint artifact is missing on server storage."
            )
        if not enc_path.exists():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Model feature encoders artifact is missing on server storage."
            )

        # 2. Read training metadata
        meta_path = ckpt_path.parent / "training_metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            metadata = {
                "client_id": client_id,
                "model_version": model_version,
                "feature_dimensions": json.loads(version_record.feature_dimensions_json or "{}"),
                "cold_start_threshold": version_record.cold_start_threshold,
                "random_seed": version_record.random_seed
            }

        dims = metadata.get("feature_dimensions", {})
        beh_dim = int(dims.get("behavior", 16))
        cont_dim = int(dims.get("content", 12))
        ctx_dim = int(dims.get("context", 21))
        cold_threshold = int(metadata.get("cold_start_threshold", version_record.cold_start_threshold or 3))

        # 3. Load state dict & verify architecture agreement
        try:
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except Exception as e:
            logger.error(f"Failed to load checkpoint at {ckpt_path}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to load model state dictionary."
            )

        # Verify state dict dimensions against metadata
        if "content_encoder.id_embedding.weight" not in state_dict:
            raise ValueError("Invalid checkpoint: missing content_encoder.id_embedding.weight")
        num_products = state_dict["content_encoder.id_embedding.weight"].shape[0]

        sd_beh_dim = state_dict["behavior_encoder.net.0.weight"].shape[1]
        sd_ctx_dim = state_dict["context_encoder.net.0.weight"].shape[1]
        sd_cont_meta_dim = state_dict["content_encoder.meta_net.0.weight"].shape[1]

        if sd_beh_dim != beh_dim:
            raise ValueError(
                f"Architecture mismatch for client '{client_id}': metadata behavior dim ({beh_dim}) != checkpoint ({sd_beh_dim})"
            )
        if sd_ctx_dim != ctx_dim:
            raise ValueError(
                f"Architecture mismatch for client '{client_id}': metadata context dim ({ctx_dim}) != checkpoint ({sd_ctx_dim})"
            )
        if (sd_cont_meta_dim + 1) != cont_dim:
            raise ValueError(
                f"Architecture mismatch for client '{client_id}': metadata content dim ({cont_dim}) != checkpoint ({sd_cont_meta_dim + 1})"
            )

        # Detect embedding_dim from checkpoint if possible
        embedding_dim = 64
        if "fusion_network.layer_norm.weight" in state_dict:
            embedding_dim = int(state_dict["fusion_network.layer_norm.weight"].shape[0])

        # 4. Instantiate TwoTowerMultiModalRecommender
        model = TwoTowerMultiModalRecommender(
            num_products=num_products,
            beh_dim=beh_dim,
            ctx_dim=ctx_dim,
            cont_meta_dim=sd_cont_meta_dim,
            embedding_dim=embedding_dim,
            cold_threshold=cold_threshold,
            apply_cold_start_adaptation=True
        )
        model.load_state_dict(state_dict)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        # 5. Load preprocessors & schema config
        encoders = joblib.load(enc_path)

        schema_config = None
        cfg_path = ckpt_path.parent / "config.yaml"
        if cfg_path.exists():
            schema_config = ClientSchemaConfig.from_yaml(cfg_path)
        else:
            client_cfg_file = Path(f"clients/{client_id}/config.yaml")
            if client_cfg_file.exists():
                schema_config = ClientSchemaConfig.from_yaml(client_cfg_file)

        # 6. Build item catalog and customer profiles
        if client_id == "demo_ecommerce":
            catalog, profiles = _build_demo_ecommerce_catalog_and_profiles()
            item_id_map = None
        else:
            catalog, profiles = _build_generic_catalog_and_profiles(
                client_id=client_id,
                schema_config=schema_config,
                encoders=encoders,
                db=db
            )
            item_id_map = encoders.get("item_id_map")

        entry = ModelCacheEntry(
            client_id=client_id,
            model_version=model_version,
            model=model,
            encoders=encoders,
            metadata=metadata,
            schema_config=schema_config,
            catalog=catalog,
            customer_profiles=profiles,
            item_id_map=item_id_map,
            cold_start_threshold=cold_threshold,
            behavior_dim=beh_dim,
            content_dim=cont_dim,
            context_dim=ctx_dim
        )
        _cache[cache_key] = entry
        logger.info(f"Successfully cached model entry for ({client_id}, {model_version}).")
        return entry


def invalidate_client_cache(client_id: str) -> None:
    """Invalidates cached entries for client_id without disturbing other clients."""
    with _lock:
        keys_to_delete = [k for k in _cache.keys() if k[0] == client_id]
        for k in keys_to_delete:
            del _cache[k]
        if keys_to_delete:
            logger.info(f"Invalidated model cache for client '{client_id}': {keys_to_delete}")


def clear_all_cache() -> None:
    """Clears all cached entries across all clients (useful for testing)."""
    with _lock:
        _cache.clear()
        logger.info("Cleared all model cache entries.")
