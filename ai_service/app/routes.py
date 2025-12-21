"""
API Routes for AuraProject AI Service v1.4.3
MongoDB persistence + Caching + Hybrid LLM + Try-On.
"""
import logging
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from ai_service.core.storage import storage
from ai_service.core.orchestrator import run_pipeline
from ai_service.config import get_provider_status, get_settings
from ai_service.cache import cache_manager
from ai_service.db import mongo

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint with all service statuses."""
    provider_status = get_provider_status()
    cache_status = cache_manager.get_status()
    mongo_status = mongo.health_check()
    
    return {
        "status": "ok",
        "version": "1.4.3",
        "llm": {
            "enabled": provider_status["enabled"],
            "primary": provider_status["primary"],
            "secondary": provider_status["secondary"],
            "availability": provider_status["availability"],
            "active_provider": provider_status["active_provider"],
        },
        "cache": cache_status,
        "mongo": mongo_status,
        "features": ["segmentation", "attributes", "hybrid_llm", "tryon", "cache", "mongodb"]
    }


@router.post("/ai/outfit")
async def create_outfit(
    image: UploadFile = File(..., description="User photo"),
    user_note: Optional[str] = Form(None, description="Optional notes")
):
    """
    Generate 5 outfit recommendations with MongoDB persistence.
    
    v1.4.3: Jobs are persisted to MongoDB and survive server restarts.
    """
    try:
        job_id = storage.create_job()
        logger.info(f"New job: {job_id}")
        
        image_path = storage.save_input_image(job_id, image.file)
        
        result = await run_pipeline(
            job_id=job_id,
            image_path=str(image_path),
            user_note=user_note
        )
        
        response = {
            "job_id": job_id,
            "seed": {"input_image": f"/ai/assets/jobs/{job_id}/input.jpg"},
            "detected_clothing": result["detected_clothing"],
            "detected_items": result.get("detected_items", {}),
            "masks": {k: f"/ai/assets/jobs/{job_id}/masks/{v}" for k, v in result.get("masks", {}).items()},
            "raw_labels": result.get("raw_labels", []),
            "outfits": result.get("outfits", []),
            "cache_hit": result.get("cache_hit", False),
            "status": result.get("status", "completed"),
            "note": "v1.4.3 - job persisted to MongoDB"
        }
        
        if result.get("error"):
            response["error"] = result["error"]
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Outfit generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ai/jobs/{job_id}")
async def get_job(job_id: str):
    """
    Get a persisted job by ID from MongoDB.
    
    Returns the full job document including outfits, renders, and status.
    """
    job = mongo.get_job(job_id)
    
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return JSONResponse(content=job)


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
