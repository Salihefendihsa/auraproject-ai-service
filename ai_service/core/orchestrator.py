"""
Pipeline Orchestrator (v1.4.2)
Coordinates segmentation + attributes + hybrid LLM + try-on + caching.
"""
import time
import logging
from typing import Dict, Any, Optional

from ai_service.core.storage import storage
from ai_service.vision.segmenter import segmenter
from ai_service.llm import router as llm_router
from ai_service.cache import cache_manager, get_cache_key
from ai_service.config import get_active_provider

logger = logging.getLogger(__name__)


async def run_pipeline(
    job_id: str,
    image_path: str,
    user_note: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run the full analysis and rendering pipeline with caching.
    
    Steps (v1.4.2):
    1. Run segmentation with attribute extraction
    2. Check cache
    3. If cache hit → return cached response
    4. Call hybrid LLM
    5. Render try-on images
    6. Save to cache
    7. Return result
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
    
    # ================================================
    # STEP 1: Segmentation with Attributes
    # ================================================
    logger.info(f"[{job_id}] Running segmentation with attribute extraction...")
    
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
        
        if seg_result.get("error"):
            logger.warning(f"Segmentation warning: {seg_result['error']}")
            
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        for cat in ["top", "bottom", "outerwear", "shoes"]:
            result["detected_items"][cat] = {"present": False}
    
    # ================================================
    # STEP 2: Check Cache
    # ================================================
    active_provider = get_active_provider()
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
            logger.info(f"[{job_id}] Cache hit! Returning cached response.")
            
            # Merge cached data with current job
            result["outfits"] = cached_response.get("outfits", [])
            result["cache_hit"] = True
            
            # Update render URLs for this job
            for i, outfit in enumerate(result["outfits"], start=1):
                if "render_url" in outfit:
                    # Copy cached renders to new job (if needed)
                    outfit["render_url"] = f"/ai/assets/jobs/{job_id}/renders/outfit_{i}.png"
            
            # Copy render files from cache info
            _copy_cached_renders(
                cached_response.get("_render_dir"),
                renders_dir,
                len(result["outfits"])
            )
            
            processing_time = int((time.time() - start_time) * 1000)
            result["processing_time_ms"] = processing_time
            
            return result
    
    # ================================================
    # STEP 3: Hybrid LLM Outfit Generation
    # ================================================
    logger.info(f"[{job_id}] Generating outfits with hybrid LLM...")
    
    try:
        outfits = await llm_router.plan_outfits(
            detected_items=result["detected_items"],
            user_note=user_note,
            season_hint=None
        )
        result["outfits"] = outfits
        logger.info(f"[{job_id}] Generated {len(outfits)} outfits")
        
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        result["error"] = str(e)
        result["status"] = "partial"
    
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
            
            for i, outfit in enumerate(result["outfits"], start=1):
                if i in result["renders"]:
                    outfit["render_url"] = f"/ai/assets/jobs/{job_id}/renders/{result['renders'][i]}"
                    outfit["tryon_method"] = "inpainting"
            
        except ImportError as e:
            logger.warning(f"Try-on renderer not available: {e}")
            _create_fallback_renders(image_path, renders_dir, result)
        except Exception as e:
            logger.error(f"Try-on rendering failed: {e}")
            _create_fallback_renders(image_path, renders_dir, result)
    
    # ================================================
    # STEP 5: Save to Cache
    # ================================================
    if cache_manager.enabled and cache_key and result["outfits"]:
        cache_data = {
            "outfits": result["outfits"],
            "_render_dir": renders_dir,
        }
        cache_manager.set(cache_key, cache_data)
        logger.info(f"[{job_id}] Saved to cache")
    
    processing_time = int((time.time() - start_time) * 1000)
    result["processing_time_ms"] = processing_time
    
    logger.info(f"[{job_id}] Pipeline complete in {processing_time}ms")
    
    return result


def _create_fallback_renders(image_path: str, renders_dir: str, result: Dict):
    """Create fallback renders by copying input image."""
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
        
        logger.info(f"Created {len(result['outfits'])} fallback renders")
    except Exception as e:
        logger.error(f"Fallback render creation failed: {e}")


def _copy_cached_renders(source_dir: Optional[str], target_dir: str, count: int):
    """Copy cached render files to new job directory."""
    import shutil
    from pathlib import Path
    
    if not source_dir:
        return
    
    try:
        source_path = Path(source_dir)
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        
        for i in range(1, count + 1):
            src_file = source_path / f"outfit_{i}.png"
            dst_file = target_path / f"outfit_{i}.png"
            if src_file.exists():
                shutil.copy(src_file, dst_file)
                
    except Exception as e:
        logger.warning(f"Failed to copy cached renders: {e}")
