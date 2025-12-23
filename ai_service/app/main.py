"""
AuraProject AI Service v2.7.0
Single Best Outfit Mode + ControlNet + Gemini 3 Pro Ready.

============================================================================
CDN READINESS DOCUMENTATION
============================================================================

This service is structured for CDN integration. All static assets are served 
from predictable, absolute paths that can be fronted by a CDN.

STATIC ASSET ROUTES (CDN CANDIDATES):
-------------------------------------
1. /ai/assets/jobs/{job_id}/renders/*.png   - Rendered outfit images
2. /ai/assets/jobs/{job_id}/masks/*.png     - Segmentation masks
3. /ai/assets/jobs/{job_id}/input.jpg       - Input images
4. /ai/assets/wardrobe/{user_id}/*          - User wardrobe items
5. /static/*                                 - Frontend static files

CDN INTEGRATION NOTES:
----------------------
- All render_url values are absolute paths (e.g., /ai/assets/jobs/{job_id}/renders/outfit_1.png)
- Paths map directly to storage.base_data_dir structure
- Cache-Control headers can be added at CDN level
- No CDN integration implemented yet - this is structural readiness only

API ROUTES (NOT CDN CANDIDATES):
--------------------------------
- /ai/outfit-seed      - Main outfit generation endpoint
- /ai/outfit           - Legacy outfit endpoint
- /ai/wardrobe/*       - Wardrobe management
- /health, /metrics    - Health and monitoring

STORAGE STRUCTURE:
------------------
ai_service/data/
├── jobs/{job_id}/
│   ├── renders/       <- CDN CANDIDATE: outfit renders
│   │   └── outfit_*.png
│   ├── masks/         <- CDN CANDIDATE: segmentation masks
│   │   └── *.png
│   └── input.jpg      <- CDN CANDIDATE: input image
└── wardrobe/{user_id}/ <- CDN CANDIDATE: user wardrobe
    └── *.png

============================================================================
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

# ============================================================================
# STATIC ASSET PATHS (CDN CANDIDATES)
# ============================================================================
# These paths serve static files that can be fronted by a CDN.
# All paths are absolute and map to predictable storage locations.
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
    
    # Log CDN-ready paths
    logger.info(f"Static assets path: {storage.base_data_dir}")
    logger.info(f"CDN-ready route: /ai/assets/*")
    
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

# ============================================================================
# MIDDLEWARE
# ============================================================================
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ============================================================================
# STATIC ASSET ROUTES (CDN CANDIDATES)
# ============================================================================
# /static/* - Frontend static files (HTML, CSS, JS)
# These are suitable for CDN caching with long TTLs.
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# /ai/assets/* is handled by routes.py with path traversal protection
# This serves rendered images, masks, and user wardrobe items.
# See: routes.py -> serve_asset() function

# ============================================================================
# API ROUTES (NOT CDN CANDIDATES)
# ============================================================================
# All API routes are dynamic and should NOT be cached by CDN.
# These include /ai/outfit-seed, /ai/outfit, /ai/wardrobe/*, /health, /metrics
app.include_router(router)


@app.get("/")
async def serve_frontend():
    """Serve the main frontend index.html."""
    return FileResponse(FRONTEND_DIR / "index.html")
