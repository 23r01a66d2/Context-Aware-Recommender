"""
Main FastAPI Application Entrypoint for Real-Time Multi-Client Recommendation Platform.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.database.database import init_db, SessionLocal
from backend.services.client_service import (
    ensure_demo_ecommerce_registered,
    ensure_all_canonical_clients_registered
)
from backend.api import (
    clients_router,
    datasets_router,
    schemas_router,
    training_router,
    recommendations_router,
    feedback_router,
    models_router,
    visual_analytics_router,
    user_history_router
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BackendMain")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    logger.info("Initializing database tables...")
    init_db()

    logger.info("Verifying canonical reference clients registration (demo_ecommerce, movielens_dataset)...")
    db = SessionLocal()
    try:
        ensure_all_canonical_clients_registered(db)
        from backend.services.model_service import ensure_demo_ecommerce_model_registered
        ensure_demo_ecommerce_model_registered(db)
    finally:
        db.close()

    yield
    logger.info("Application shutting down.")


app = FastAPI(
    title="Real-Time Context-Aware Recommendation Platform",
    description="Multi-client API platform for multi-modal behavioral recommendation with cold-start gating.",
    version="1.0.0",
    lifespan=lifespan
)

# Development CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://context-aware-recommender.local:8000",
        "http://context-aware-recommender.local:3000",
        "http://context-aware-recommender.local",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Security: Safe Error Handlers (never leak stack traces or internal filesystem paths)
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for err in exc.errors():
        field = ".".join(str(loc) for loc in err.get("loc", []))
        errors.append(f"{field}: {err.get('msg', 'Invalid value')}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Input validation error", "errors": errors}
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled server error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )


# Register API routers with /api prefix
app.include_router(clients_router, prefix="/api")
app.include_router(datasets_router, prefix="/api")
app.include_router(schemas_router, prefix="/api")
app.include_router(training_router, prefix="/api")
app.include_router(recommendations_router, prefix="/api")
app.include_router(feedback_router, prefix="/api")
app.include_router(models_router, prefix="/api")
app.include_router(visual_analytics_router, prefix="/api")
app.include_router(user_history_router, prefix="/api")


from pathlib import Path
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import os
import mimetypes

@app.get("/api/health")
def health_check():
    """
    Health check endpoint returning exact service status.
    Does not fabricate model status.
    """
    return {
        "status": "ok",
        "service": "context-aware-recommender"
    }


@app.get("/api/images/{client_id}/{image_name}")
def get_client_image(client_id: str, image_name: str):
    """
    Securely serves dataset/client images with path traversal protection.
    Searches isolated client directories, data/datasets, and archive/data.
    """
    clean_cid = os.path.basename(client_id.strip())
    clean_img = os.path.basename(image_name.strip())
    if not clean_img or ".." in clean_img:
        raise StarletteHTTPException(status_code=400, detail="Invalid image request.")

    potential_paths = [
        Path(f"clients/{clean_cid}/images/{clean_img}"),
        Path(f"clients/{clean_cid}/data/{clean_img}"),
        Path(f"clients/{clean_cid}/data/images/{clean_img}"),
        Path(f"clients/{clean_cid}/raw/{clean_img}"),
        Path(f"data/datasets/fashion/images/{clean_img}"),
        Path(f"archive/data/{clean_img}"),
        Path(f"archive/{clean_img}"),
        Path(f"data/raw/{clean_img}"),
    ]
    for p in potential_paths:
        if p.exists() and p.is_file():
            mime, _ = mimetypes.guess_type(str(p))
            return FileResponse(p, media_type=mime or "image/jpeg")

    raise StarletteHTTPException(status_code=404, detail="Image not found.")


# Mount production React build if dist directory exists
dist_dir = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if dist_dir.exists():
    assets_dir = dist_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="spa_assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # Exclude reserved paths
        if full_path.startswith("api/") or full_path in ("api", "docs", "redoc", "openapi.json"):
            raise StarletteHTTPException(status_code=404, detail="Not Found")
        target_file = dist_dir / full_path
        if target_file.is_file():
            return FileResponse(target_file)
        return FileResponse(dist_dir / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
