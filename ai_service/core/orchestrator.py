"""
Pipeline Orchestrator (v1.4.3)
Coordinates segmentation + LLM + try-on + caching + MongoDB persistence.
"""
import time
import logging
from typing import Dict, Any, Optional

from ai_service.core.storage import storage
from ai_service.vision.segmenter import segmenter
from ai_service.llm import router as llm_router
from ai_service.cache import cache_manager, get_cache_key
from ai_service.config import get_active_provider
from ai_service.db import mongo

logger = logging.getLogger(__name__)


async def run_pipeline(
    job_id: str,
    image_path: str,
    user_note: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run the full analysis and rendering pipeline with MongoDB persistence.
    
    Steps (v1.4.3):
    1. Create job in MongoDB (status=pending)
    2. Run segmentation → update DB
    3. Check cache
    4. Call LLM → update DB
    5. Render try-on → update DB
    6. Save to cache
    7. Mark job complete in DB
    """
    start_time = time.time()
    
    result = {
        "job_id": job_id,
        "detected_clothing": {
            "top": False,
            "bottom": False,
            "outerwear": False,
            "shoes": False
        },
        "detected_items": {},
        "masks": {},
        "raw_labels": [],
        "outfits": [],
        "renders": {},
        "status": "completed",
        "cache_hit": False,
        "error": None
    }
    
    job_path = storage.get_job_path(job_id)
    masks_dir = str(job_path / "masks")
    renders_dir = str(job_path / "renders")
    active_provider = get_active_provider()
    
    # ================================================
    # STEP 0: Create job in MongoDB
    # ================================================
    job_doc = mongo.create_job_document(job_id)
    job_doc["assets"]["input_image"] = f"/ai/assets/jobs/{job_id}/input.jpg"
    mongo.insert_job(job_doc)
    
    # ================================================
    # STEP 1: Segmentation with Attributes
    # ================================================
    logger.info(f"[{job_id}] Running segmentation...")
    
    try:
        seg_result = segmenter.segment(image_path, extract_attributes=True)
        
        result["detected_clothing"]["top"] = seg_result.get("top", False)
        result["detected_clothing"]["bottom"] = seg_result.get("bottom", False)
        result["detected_clothing"]["outerwear"] = seg_result.get("outerwear", False)
        result["detected_clothing"]["shoes"] = seg_result.get("shoes", False)
        result["raw_labels"] = seg_result.get("raw_labels", [])
        result["detected_items"] = seg_result.get("detected_items", {})
        
        # Save masks
        masks = seg_result.get("masks", {})
        if masks:
            saved_masks = segmenter.save_masks(masks, masks_dir)
            result["masks"] = saved_masks
        
        # Update MongoDB
        mongo.update_job(job_id, {
            "detected_clothing": result["detected_clothing"],
            "attributes": result["detected_items"],
            "assets.masks": {k: f"/ai/assets/jobs/{job_id}/masks/{v}" for k, v in result["masks"].items()}
        })
            
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        for cat in ["top", "bottom", "outerwear", "shoes"]:
            result["detected_items"][cat] = {"present": False}
    
    # ================================================
    # STEP 2: Check Cache
    # ================================================
    cache_key = None
    
    if cache_manager.enabled:
        cache_key = get_cache_key(
            image_path=image_path,
            detected_clothing=result["detected_clothing"],
            detected_items=result["detected_items"],
            user_note=user_note,
            active_provider=active_provider or "none"
        )
        
        cached_response = cache_manager.get(cache_key)
        
        if cached_response:
            logger.info(f"[{job_id}] Cache hit!")
            result["outfits"] = cached_response.get("outfits", [])
            result["cache_hit"] = True
            
            for i, outfit in enumerate(result["outfits"], start=1):
                outfit["render_url"] = f"/ai/assets/jobs/{job_id}/renders/outfit_{i}.png"
            
            _copy_cached_renders(cached_response.get("_render_dir"), renders_dir, len(result["outfits"]))
            
            # Update MongoDB with cached data
            mongo.update_job(job_id, {
                "outfits": result["outfits"],
                "cached": True,
                "provider_used": active_provider or "cached",
                "status": "completed",
                "assets.renders": [f"/ai/assets/jobs/{job_id}/renders/outfit_{i}.png" for i in range(1, len(result["outfits"]) + 1)]
            })
            
            result["processing_time_ms"] = int((time.time() - start_time) * 1000)
            return result
    
    # ================================================
    # STEP 3: LLM Outfit Generation
    # ================================================
    logger.info(f"[{job_id}] Generating outfits with LLM...")
    
    try:
        outfits = await llm_router.plan_outfits(
            detected_items=result["detected_items"],
            user_note=user_note,
            season_hint=None
        )
        result["outfits"] = outfits
        
        # Update MongoDB
        mongo.update_job(job_id, {
            "outfits": outfits,
            "provider_used": active_provider or "unknown"
        })
        
    except Exception as e:
        logger.error(f"LLM failed: {e}")
        result["error"] = str(e)
        result["status"] = "failed"
        mongo.update_job(job_id, {"status": "failed", "error": str(e)})
        return result
    
    # ================================================
    # STEP 4: Try-On Rendering
    # ================================================
    if result["outfits"]:
        logger.info(f"[{job_id}] Rendering try-on images...")
        
        try:
            from ai_service.renderer.tryon import render_all_outfits
            
            render_result = render_all_outfits(
                input_image_path=image_path,
                masks_dir=masks_dir,
                outfits=result["outfits"],
                output_dir=renders_dir
            )
            
            result["renders"] = render_result.get("renders", {})
            render_urls = []
            
            for i, outfit in enumerate(result["outfits"], start=1):
                if i in result["renders"]:
                    url = f"/ai/assets/jobs/{job_id}/renders/{result['renders'][i]}"
                    outfit["render_url"] = url
                    outfit["tryon_method"] = "inpainting"
                    render_urls.append(url)
            
            # Update MongoDB
            mongo.update_job(job_id, {"assets.renders": render_urls})
            
        except Exception as e:
            logger.error(f"Rendering failed: {e}")
            _create_fallback_renders(image_path, renders_dir, result)
    
    # ================================================
    # STEP 5: Save to Cache & Finalize
    # ================================================
    if cache_manager.enabled and cache_key and result["outfits"]:
        cache_manager.set(cache_key, {"outfits": result["outfits"], "_render_dir": renders_dir})
    
    # Mark job complete in MongoDB
    mongo.update_job(job_id, {"status": "completed", "cached": False})
    
    result["processing_time_ms"] = int((time.time() - start_time) * 1000)
    logger.info(f"[{job_id}] Pipeline complete in {result['processing_time_ms']}ms")
    
    return result


def _create_fallback_renders(image_path: str, renders_dir: str, result: Dict):
    import shutil
    from pathlib import Path
    
    try:
        output_path = Path(renders_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i, outfit in enumerate(result["outfits"], start=1):
            render_file = f"outfit_{i}.png"
            shutil.copy(image_path, output_path / render_file)
            result["renders"][i] = render_file
            outfit["render_url"] = f"/ai/assets/jobs/{result['job_id']}/renders/{render_file}"
            outfit["tryon_method"] = "fallback"
    except Exception as e:
        logger.error(f"Fallback render failed: {e}")


def _copy_cached_renders(source_dir: Optional[str], target_dir: str, count: int):
    import shutil
    from pathlib import Path
    
    if not source_dir:
        return
    
    try:
        source_path = Path(source_dir)
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        
        for i in range(1, count + 1):
            src = source_path / f"outfit_{i}.png"
            dst = target_path / f"outfit_{i}.png"
            if src.exists():
                shutil.copy(src, dst)
    except Exception as e:
        logger.warning(f"Failed to copy cached renders: {e}")
