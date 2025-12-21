"""
AuraProject AI Service v1.4.1
Config-based provider management + Hybrid LLM + Try-On.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ai_service.app.routes import router
from ai_service.core.storage import storage
from ai_service.config import get_settings, get_provider_status, validate_provider_config

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
    logger.info("AuraProject AI Service v1.4.1 Starting...")
    logger.info("Features: Config + Hybrid LLM + Segmentation + Try-On")
    logger.info("=" * 50)
    
    # Initialize storage
    storage.ensure_directories()
    
    # Load and display settings
    settings = get_settings()
    provider_status = get_provider_status()
    
    logger.info(f"LLM Enabled: {settings.llm_enabled}")
    logger.info(f"Primary Provider: {settings.llm_primary}")
    logger.info(f"Secondary Provider: {settings.llm_secondary}")
    logger.info(f"Daily Limit: {settings.llm_daily_limit}")
    
    # Check provider availability
    if provider_status["availability"].get("openai"):
        logger.info("✓ OpenAI configured")
    else:
        logger.warning("⚠ OpenAI not configured")
    
    if provider_status["availability"].get("gemini"):
        logger.info("✓ Gemini configured")
    else:
        logger.info("ℹ Gemini not configured (optional)")
    
    # Log active provider
    active = provider_status.get("active_provider")
    if active:
        logger.info(f"✓ Active provider: {active}")
    else:
        logger.warning("⚠ No active provider - outfit planning will fail")
    
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
    description="Config-based LLM Management + Hybrid LLM + Try-On",
    version="1.4.1",
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
        "version": "1.4.1",
        "features": ["config", "segmentation", "attributes", "hybrid_llm", "virtual_tryon"],
        "docs": "/docs"
    }
