"""
AuraProject AI Service v1.4.2
Caching + Config + Hybrid LLM + Try-On.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ai_service.app.routes import router
from ai_service.core.storage import storage
from ai_service.config import get_settings, get_provider_status, validate_provider_config
from ai_service.cache import cache_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("=" * 50)
    logger.info("AuraProject AI Service v1.4.2 Starting...")
    logger.info("Features: Cache + Config + Hybrid LLM + Try-On")
    logger.info("=" * 50)
    
    # Initialize storage
    storage.ensure_directories()
    
    # Load settings
    settings = get_settings()
    provider_status = get_provider_status()
    cache_status = cache_manager.get_status()
    
    # Log LLM status
    logger.info(f"LLM Enabled: {settings.llm_enabled}")
    logger.info(f"Active Provider: {provider_status.get('active_provider', 'none')}")
    
    # Log cache status
    logger.info(f"Cache Enabled: {cache_status['enabled']}")
    logger.info(f"Cache TTL: {cache_status['ttl_minutes']} minutes")
    logger.info(f"Cache Entries: {cache_status['entries']}")
    
    # Log warnings
    warnings = validate_provider_config()
    for warning in warnings:
        logger.warning(f"⚠ {warning}")
    
    logger.info("✓ Service ready!")
    logger.info("=" * 50)
    
    yield
    
    logger.info("Service shutting down...")


# Create app
app = FastAPI(
    title="AuraProject AI Service",
    description="Caching + Config + Hybrid LLM + Try-On",
    version="1.4.2",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "AuraProject AI Service",
        "version": "1.4.2",
        "features": ["cache", "config", "segmentation", "attributes", "hybrid_llm", "virtual_tryon"],
        "docs": "/docs"
    }
