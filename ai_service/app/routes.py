"""
API Routes for AuraProject AI Service v1.4.2
Caching + Config + Hybrid LLM + Try-On.
"""
import logging
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from ai_service.core.storage import storage
from ai_service.core.orchestrator import run_pipeline
from ai_service.config import get_provider_status, get_settings
from ai_service.cache import cache_manager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint with provider and cache status."""
    settings = get_settings()
    provider_status = get_provider_status()
    cache_status = cache_manager.get_status()
    
    return {
        "status": "ok",
        "version": "1.4.2",
        "llm": {
            "enabled": provider_status["enabled"],
            "primary": provider_status["primary"],
            "secondary": provider_status["secondary"],
            "availability": provider_status["availability"],
            "active_provider": provider_status["active_provider"],
            "daily_limit": provider_status["daily_limit"],
        },
        "cache": cache_status,
        "features": ["segmentation", "attributes", "hybrid_llm", "tryon", "config", "cache"]
    }


@router.post("/ai/outfit")
async def create_outfit(
    image: UploadFile = File(..., description="User photo"),
    user_note: Optional[str] = Form(None, description="Optional notes")
):
    """
    Generate 5 outfit recommendations with try-on renders.
    
    v1.4.2 Features:
    - Request-level caching to reduce LLM costs
    - Cache based on image hash + attributes + provider
    - 24-hour TTL by default
    """
    try:
        # Create job
        job_id = storage.create_job()
        job_path = storage.get_job_path(job_id)
        
        logger.info(f"New job: {job_id}")
        
        # Save image
        image_path = storage.save_input_image(job_id, image.file)
        
        # Run pipeline (with caching)
        result = await run_pipeline(
            job_id=job_id,
            image_path=str(image_path),
            user_note=user_note
        )
        
        # Build response
        response = {
            "job_id": job_id,
            "seed": {
                "input_image": f"/ai/assets/jobs/{job_id}/input.jpg"
            },
            "detected_clothing": result["detected_clothing"],
            "detected_items": result.get("detected_items", {}),
            "masks": {
                k: f"/ai/assets/jobs/{job_id}/masks/{v}"
                for k, v in result.get("masks", {}).items()
            },
            "raw_labels": result.get("raw_labels", []),
            "outfits": result.get("outfits", []),
            "cache_hit": result.get("cache_hit", False),
            "status": result.get("status", "completed"),
            "note": "v1.4.2 - caching enabled for cost reduction"
        }
        
        if result.get("error"):
            response["error"] = result["error"]
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Outfit generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ai/assets/{file_path:path}")
async def serve_asset(file_path: str):
    """Serve static files from job directories."""
    try:
        full_path = storage.get_file_path(file_path)
        
        if not full_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        if not full_path.is_file():
            raise HTTPException(status_code=400, detail="Not a file")
        
        return FileResponse(full_path)
        
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
