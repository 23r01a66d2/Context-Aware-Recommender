"""
API Routers for client endpoints.
"""
from backend.api.clients import router as clients_router
from backend.api.datasets import router as datasets_router
from backend.api.schemas import router as schemas_router
from backend.api.training import router as training_router
from backend.api.recommendations import router as recommendations_router
from backend.api.feedback import router as feedback_router

from backend.api.models import router as models_router

__all__ = [
    "clients_router",
    "datasets_router",
    "schemas_router",
    "training_router",
    "recommendations_router",
    "feedback_router",
    "models_router"
]
