"""
AuraProject AI Service v1.4.3
MongoDB + Cache + Hybrid LLM + Try-On.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ai_service.app.routes import router
from ai_service.core.storage import storage
from ai_service.config import get_settings, get_provider_status, validate_provider_config
from ai_service.cache import cache_manager
from ai_service.db import mongo

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 50)
    logger.info("AuraProject AI Service v1.4.3 Starting...")
    logger.info("Features: MongoDB + Cache + Hybrid LLM + Try-On")
    logger.info("=" * 50)
    
    storage.ensure_directories()
    
    # Connect to MongoDB
    mongo_connected = mongo.connect()
    if mongo_connected:
        logger.info("✓ MongoDB connected")
    else:
        logger.warning("⚠ MongoDB not connected - jobs will not persist")
    
    # LLM status
    provider_status = get_provider_status()
    logger.info(f"Active LLM: {provider_status.get('active_provider', 'none')}")
    
    # Cache status
    cache_status = cache_manager.get_status()
    logger.info(f"Cache: {'enabled' if cache_status['enabled'] else 'disabled'}")
    
    logger.info("✓ Service ready!")
    logger.info("=" * 50)
    
    yield
    
    logger.info("Service shutting down...")


app = FastAPI(
    title="AuraProject AI Service",
    description="MongoDB + Cache + Hybrid LLM + Try-On",
    version="1.4.3",
    lifespan=lifespan
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(router)


@app.get("/")
async def root():
    return {
        "service": "AuraProject AI Service",
        "version": "1.4.3",
        "features": ["mongodb", "cache", "segmentation", "attributes", "hybrid_llm", "virtual_tryon"],
        "docs": "/docs"
    }
