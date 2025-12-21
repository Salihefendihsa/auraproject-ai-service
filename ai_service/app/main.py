"""
AuraProject AI Service v1.4.4
Frontend Demo + MongoDB + Cache + Hybrid LLM + Try-On.
"""
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from ai_service.app.routes import router
from ai_service.core.storage import storage
from ai_service.config import get_settings, get_provider_status, validate_provider_config
from ai_service.cache import cache_manager
from ai_service.db import mongo

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Frontend path
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 50)
    logger.info("AuraProject AI Service v1.4.4 Starting...")
    logger.info("Features: Frontend + MongoDB + Cache + Hybrid LLM + Try-On")
    logger.info("=" * 50)
    
    storage.ensure_directories()
    
    mongo_connected = mongo.connect()
    if mongo_connected:
        logger.info("✓ MongoDB connected")
    else:
        logger.warning("⚠ MongoDB not connected")
    
    provider_status = get_provider_status()
    logger.info(f"Active LLM: {provider_status.get('active_provider', 'none')}")
    
    cache_status = cache_manager.get_status()
    logger.info(f"Cache: {'enabled' if cache_status['enabled'] else 'disabled'}")
    
    logger.info(f"Frontend: {FRONTEND_DIR}")
    logger.info("✓ Service ready! Open http://localhost:8000")
    logger.info("=" * 50)
    
    yield
    
    logger.info("Service shutting down...")


app = FastAPI(
    title="AuraProject AI Service",
    description="Frontend Demo + MongoDB + Cache + Hybrid LLM + Try-On",
    version="1.4.4",
    lifespan=lifespan
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Mount static files for frontend
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# API routes
app.include_router(router)


@app.get("/")
async def serve_frontend():
    """Serve the frontend demo."""
    return FileResponse(FRONTEND_DIR / "index.html")
