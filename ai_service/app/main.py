"""
AuraProject AI Service v2.7.0
Single Best Outfit Mode + ControlNet + Gemini 3 Pro Ready.
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
from ai_service.config import get_settings, get_provider_status
from ai_service.cache import cache_manager
from ai_service.db import mongo
from ai_service.observability import is_logging_enabled

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 50)
    logger.info("AuraProject AI Service v2.7.0 Starting...")
    logger.info("Features: Single-Mode + ControlNet + Gemini-3 + Self-Critique")
    logger.info("=" * 50)
    
    storage.ensure_directories()
    
    mongo_connected = mongo.connect()
    logger.info(f"MongoDB: {'connected' if mongo_connected else 'disconnected'}")
    
    provider_status = get_provider_status()
    logger.info(f"Active LLM: {provider_status.get('active_provider', 'none')}")
    
    cache_status = cache_manager.get_status()
    logger.info(f"Cache: {'enabled' if cache_status['enabled'] else 'disabled'}")
    
    logger.info(f"Logging: {'enabled' if is_logging_enabled() else 'disabled'}")
    
    logger.info("✓ Service ready! http://localhost:8000")
    logger.info("✓ Metrics available at /metrics")
    logger.info("=" * 50)
    
    yield
    
    logger.info("Service shutting down...")


app = FastAPI(
    title="AuraProject AI Service",
    description="Single Best Outfit Mode + ControlNet + Gemini 3 Pro Ready",
    version="2.7.0",
    lifespan=lifespan
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
app.include_router(router)


@app.get("/")
async def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")
