"""
AuraProject AI Service v1.2.0
Segmentation + Attributes + LLM + Virtual Try-On.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ai_service.app.routes import router
from ai_service.core.storage import storage

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
    logger.info("AuraProject AI Service v1.2.0 Starting...")
    logger.info("Features: Segmentation + Attributes + LLM + Try-On")
    logger.info("=" * 50)
    
    # Initialize storage
    storage.ensure_directories()
    
    import os
    if os.getenv("OPENAI_API_KEY"):
        logger.info("✓ OpenAI API key configured")
    else:
        logger.warning("⚠ OPENAI_API_KEY not set")
    
    logger.info("✓ Service ready!")
    logger.info("Note: First request downloads models (~3GB total)")
    logger.info("=" * 50)
    
    yield
    
    logger.info("Service shutting down...")


# Create app
app = FastAPI(
    title="AuraProject AI Service",
    description="Segmentation + Attributes + LLM + Virtual Try-On",
    version="1.2.0",
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
        "version": "1.2.0",
        "features": ["segmentation", "attribute_extraction", "llm_planning", "virtual_tryon"],
        "docs": "/docs"
    }
