"""
API Routes for AuraProject AI Service v1.5.0
With /metrics endpoint and enhanced /health.
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
from ai_service.observability import get_metrics, is_logging_enabled

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check with observability info."""
    provider_status = get_provider_status()
    cache_status = cache_manager.get_status()
    mongo_status = mongo.health_check()
    metrics = get_metrics()
    
    return {
        "status": "ok",
        "version": "1.5.0",
        "llm": {
            "enabled": provider_status["enabled"],
            "primary": provider_status["primary"],
            "secondary": provider_status["secondary"],
            "availability": provider_status["availability"],
            "active_provider": provider_status["active_provider"],
        },
        "cache": cache_status,
        "mongo": mongo_status,
        "observability": {
            "logging_enabled": is_logging_enabled(),
            "total_requests": metrics["total_requests"],
            "cache_hit_ratio": metrics["cache_hit_ratio"],
            "total_cost_usd": metrics["total_cost_usd"]
        },
        "features": ["segmentation", "attributes", "hybrid_llm", "tryon", "cache", "mongodb", "observability"]
    }


@router.get("/metrics")
async def get_metrics_endpoint():
    """Get detailed metrics for monitoring."""
    metrics = get_metrics()
    return JSONResponse(content=metrics)


@router.post("/ai/outfit")
async def create_outfit(
    image: UploadFile = File(..., description="User photo"),
    user_note: Optional[str] = Form(None, description="Optional notes")
):
    """
    Generate 5 outfit recommendations.
    
    v1.5.0: Includes cost tracking and request logging.
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
            "cost": result.get("cost"),
            "status": result.get("status", "completed"),
            "note": "v1.5.0 - observability & production readiness"
        }
        
        if result.get("error"):
            response["error"] = result["error"]
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Outfit generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ai/jobs/{job_id}")
async def get_job(job_id: str):
    """Get a persisted job by ID."""
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
