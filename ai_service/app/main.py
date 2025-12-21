"""
AuraProject AI Service v1.3.0
Hybrid LLM (OpenAI + Gemini) + Segmentation + Try-On.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ai_service.app.routes import router
from ai_service.core.storage import storage
from ai_service.llm import router as llm_router

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
    logger.info("AuraProject AI Service v1.3.0 Starting...")
    logger.info("Features: Hybrid LLM + Segmentation + Try-On")
    logger.info("=" * 50)
    
    # Initialize storage
    storage.ensure_directories()
    
    # Check LLM providers
    llm_status = llm_router.get_provider_status()
    
    if llm_status["openai"]:
        logger.info("✓ OpenAI configured (primary)")
    else:
        logger.warning("⚠ OPENAI_API_KEY not set")
    
    if llm_status["gemini"]:
        logger.info("✓ Gemini configured (advisor)")
    else:
        logger.info("ℹ Gemini not configured (optional)")
    
    logger.info("✓ Service ready!")
    logger.info("=" * 50)
    
    yield
    
    logger.info("Service shutting down...")


# Create app
app = FastAPI(
    title="AuraProject AI Service",
    description="Hybrid LLM (OpenAI + Gemini) + Segmentation + Try-On",
    version="1.3.0",
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
        "version": "1.3.0",
        "features": ["segmentation", "attributes", "hybrid_llm", "virtual_tryon"],
        "docs": "/docs"
    }
