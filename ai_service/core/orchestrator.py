"""
Pipeline Orchestrator (v2.7.0)
Single Best Outfit Mode + self-critique pipeline.
"""
import time
import logging
import asyncio
from typing import Dict, Any, Optional

from ai_service.core.storage import storage
from ai_service.vision.segmenter import segmenter
from ai_service.llm import router as llm_router
from ai_service.cache import cache_manager, get_cache_key
from ai_service.config import get_active_provider
from ai_service.db import mongo
from ai_service.db.wardrobe import build_wardrobe_context
from ai_service.observability import log_request, increment_request, estimate_cost, is_logging_enabled
from ai_service.llm.tryon_judge import judge_all_outfits, identify_worst_outfits
from ai_service.llm.outfit_regenerator import regenerate_worst_outfits

logger = logging.getLogger(__name__)


# ==================== OUTFIT SEED PIPELINE ====================
# ============================================================================
# DO NOT MODIFY BASELINE LOGIC BELOW - DEMO BASELINE FREEZE v1.0
# Seed lock behavior and full → partial fallback logic is FROZEN.
# ============================================================================

from ai_service.core.outfit_recommender import (
    load_catalog,
    build_seed_object,
    generate_outfits,
    plan_slots,
    validate_catalog_slots,
)


class OutfitSeedError(Exception):
    """Error during outfit seed job."""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


def run_outfit_seed_job(
    job_id: str,
    seed_image_path: Optional[str],
    person_image_path: Optional[str],
    gender: str,
    event: Optional[str],
    season: Optional[str],
    mode: str,
    seed_category: Optional[str] = None
) -> Dict[str, Any]:
    """
    Initialize a seed job and generate outfits.
    
    Args:
        job_id: UUID for the job
        seed_image_path: Path to seed garment image (optional)
        person_image_path: Path to person image (optional)
        gender: male or female
        event: Optional event type
        season: Optional season
        mode: mock, partial_tryon, or full_tryon
        seed_category: Optional explicit seed category (top/bottom/outerwear/shoes/accessory)
    
    Returns:
        Dict with subject_type, seed, and outfits
    
    Raises:
        OutfitSeedError: If catalog missing items or detection confidence too low
    """
    logger.info(f"[{job_id}] Initializing seed job (mode={mode})")
    
    # Step 1: Create job folder
    storage.create_seed_job(job_id)
    
    # Step 2: Determine subject_type
    subject_type = "person" if person_image_path else "mannequin"
    
    # Step 3: Build seed object with detection
    seed, confidence = build_seed_object(
        seed_image_path=seed_image_path,
        seed_category=seed_category
    )
    
    # Step 3b: Check detection confidence if no explicit category
    if not seed_category and seed_image_path and confidence < 0.4:
        logger.warning(f"[{job_id}] Low detection confidence: {confidence:.2f}")
        raise OutfitSeedError(
            f"Unable to detect garment category with confidence. "
            f"Please provide seed_category explicitly (top/bottom/outerwear/shoes/accessory).",
            status_code=400
        )
    
    # Step 4: Load catalog
    catalog = load_catalog()
    if not catalog.get("items"):
        raise OutfitSeedError("Catalog is empty or failed to load", status_code=500)
    
    # Step 5: Validate catalog has items for all required slots
    slots_to_fill = plan_slots(seed["category"])
    missing_slots = validate_catalog_slots(catalog, slots_to_fill, gender)
    if missing_slots:
        raise OutfitSeedError(
            f"Catalog missing items for slots: {', '.join(missing_slots)} (gender={gender})",
            status_code=400
        )
    
    # Step 6: Generate 5 outfits
    outfits = generate_outfits(
        seed=seed,
        catalog=catalog,
        gender=gender,
        event=event,
        season=season
    )
    
    # Step 6: Try-On Rendering based on mode
    # ========================================================================
    # EXISTING MODES (FROZEN - DO NOT MODIFY):
    #   - full_tryon: Full body try-on with partial fallback
    #   - partial_tryon: Upper body try-on only
    #   - mock: No rendering, preview only
    #
    # NEW MODE (ISOLATED EXTENSION):
    #   - user_photo_tryon: Render onto user's uploaded photo
    # ========================================================================
    
    if mode == "user_photo_tryon":
        # ISOLATED EXTENSION: User photo try-on
        # Fallback to partial_tryon on any failure
        outfits = _render_user_photo_tryon_outfits(
            job_id=job_id,
            outfits=outfits,
            person_image_path=person_image_path,
            gender=gender
        )
    elif mode == "full_tryon":
        # Attempt full try-on, fallback to partial on failure
        outfits = _render_full_tryon_outfits(
            job_id=job_id,
            outfits=outfits,
            person_image_path=person_image_path,
            gender=gender
        )
    elif mode == "partial_tryon":
        outfits = _render_partial_tryon_outfits(
            job_id=job_id,
            outfits=outfits,
            person_image_path=person_image_path,
            gender=gender
        )
    else:
        # For mock mode, just mark tryon_mode
        for outfit in outfits:
            outfit["tryon_mode"] = "mock"
            outfit["render_url"] = None
    
    # Step 7: Save job.json with outfits
    storage.save_seed_job_json(
        job_id=job_id,
        seed_image_path=seed_image_path,
        person_image_path=person_image_path,
        gender=gender,
        event=event,
        season=season,
        mode=mode,
        subject_type=subject_type
    )
    
    # Step 8: Update job.json with outfits and stage
    _update_job_with_outfits(job_id, seed, outfits)
    
    logger.info(f"[{job_id}] Seed job complete. subject_type={subject_type}, outfits={len(outfits)}")
    
    return {
        "subject_type": subject_type,
        "seed": seed,
        "outfits": outfits
    }


def _render_partial_tryon_outfits(
    job_id: str,
    outfits: list,
    person_image_path: Optional[str],
    gender: str
) -> list:
    """
    Render partial try-on for all outfits.
    
    Only renders upper body (top + outerwear).
    Lower body and shoes remain mock.
    
    ========================================================================
    DO NOT MODIFY BASELINE LOGIC BELOW - DEMO BASELINE FREEZE v1.0
    Partial try-on pipeline is FROZEN as of 2024-12-23.
    ========================================================================
    """
    from ai_service.renderer.partial_tryon import render_outfit_partial
    
    job_path = storage.get_job_path(job_id)
    renders_dir = job_path / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"[{job_id}] Rendering partial try-on for {len(outfits)} outfits")
    
    for outfit in outfits:
        rank = outfit.get("rank", 1)
        
        # Attempt partial try-on render
        render_filename = render_outfit_partial(
            outfit=outfit,
            job_id=job_id,
            person_image_path=person_image_path,
            gender=gender,
            output_dir=renders_dir
        )
        
        if render_filename:
            outfit["render_url"] = f"/ai/assets/jobs/{job_id}/renders/{render_filename}"
            outfit["tryon_mode"] = "partial"
            logger.info(f"[{job_id}] Outfit {rank}: partial render OK")
        else:
            outfit["render_url"] = None
            outfit["tryon_mode"] = "mock"
            logger.warning(f"[{job_id}] Outfit {rank}: partial render failed, using mock")
    
    return outfits


def _render_full_tryon_outfits(
    job_id: str,
    outfits: list,
    person_image_path: Optional[str],
    gender: str
) -> list:
    """
    Render full try-on for all outfits.
    
    Attempts full body try-on with warp-based fitting.
    Falls back to partial try-on if full fails.
    
    ========================================================================
    DO NOT MODIFY BASELINE LOGIC BELOW - DEMO BASELINE FREEZE v1.0
    Full try-on with partial fallback is FROZEN as of 2024-12-23.
    ========================================================================
    """
    from ai_service.renderer.full_tryon import render_outfit_full
    from ai_service.renderer.partial_tryon import render_outfit_partial
    
    job_path = storage.get_job_path(job_id)
    renders_dir = job_path / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"[{job_id}] Rendering full try-on for {len(outfits)} outfits")
    
    for outfit in outfits:
        rank = outfit.get("rank", 1)
        
        # Attempt full try-on render
        render_filename = render_outfit_full(
            outfit=outfit,
            job_id=job_id,
            person_image_path=person_image_path,
            gender=gender,
            output_dir=renders_dir
        )
        
        if render_filename:
            outfit["render_url"] = f"/ai/assets/jobs/{job_id}/renders/{render_filename}"
            outfit["tryon_mode"] = "full"
            logger.info(f"[{job_id}] Outfit {rank}: full render OK")
        else:
            # Fallback to partial try-on
            logger.warning(f"[{job_id}] Outfit {rank}: full render failed, falling back to partial")
            
            render_filename = render_outfit_partial(
                outfit=outfit,
                job_id=job_id,
                person_image_path=person_image_path,
                gender=gender,
                output_dir=renders_dir
            )
            
            if render_filename:
                outfit["render_url"] = f"/ai/assets/jobs/{job_id}/renders/{render_filename}"
                outfit["tryon_mode"] = "partial_fallback"
                logger.info(f"[{job_id}] Outfit {rank}: partial fallback OK")
            else:
                outfit["render_url"] = None
                outfit["tryon_mode"] = "mock"
                logger.warning(f"[{job_id}] Outfit {rank}: all renders failed, using mock")
    
    return outfits


# ============================================================================
# ISOLATED EXTENSION: User Photo Try-On
# ============================================================================
# This function is ONLY called when mode == "user_photo_tryon".
# It does NOT modify any existing mannequin try-on logic.
# On any failure, it falls back to partial_tryon on mannequin.
# ============================================================================

def _render_user_photo_tryon_outfits(
    job_id: str,
    outfits: list,
    person_image_path: Optional[str],
    gender: str
) -> list:
    """
    Render outfits onto user's uploaded photo.
    
    ISOLATED EXTENSION: This function is completely separate from
    mannequin try-on. It uses the user_photo_tryon module.
    
    Fallback Guarantee:
    - If person_image_path is missing → fallback to partial_tryon
    - If person detection fails → fallback to partial_tryon
    - If any render error occurs → mark as user_photo_fallback
    """
    logger.info(f"[{job_id}] Starting user photo try-on render")
    
    # Validation: person_image is required for user photo try-on
    if not person_image_path:
        logger.warning(f"[{job_id}] No person image provided, falling back to partial_tryon")
        return _render_partial_tryon_outfits(
            job_id=job_id,
            outfits=outfits,
            person_image_path=None,
            gender=gender
        )
    
    try:
        # Step 1: Validate that image is a valid human photo
        from ai_service.renderer.user_photo_detection import detect_user_photo
        detection_result = detect_user_photo(person_image_path)
        
        if not detection_result.get("is_valid_for_tryon", False):
            logger.warning(
                f"[{job_id}] User photo not valid for try-on: "
                f"{detection_result.get('reason', 'Unknown')}. "
                f"Falling back to mannequin partial_tryon."
            )
            fallback_outfits = _render_partial_tryon_outfits(
                job_id=job_id,
                outfits=outfits,
                person_image_path=person_image_path,
                gender=gender
            )
            for outfit in fallback_outfits:
                if outfit.get("tryon_mode") in ("partial", "partial_fallback"):
                    outfit["tryon_mode"] = "mannequin_fallback"
            return fallback_outfits
        
        logger.info(
            f"[{job_id}] User photo validated: {detection_result.get('reason')} "
            f"(confidence={detection_result.get('confidence', 0):.2f})"
        )
        
        # Step 2: Render outfits onto user photo
        from ai_service.renderer.user_photo_tryon import render_all_outfits_user_photo
        
        job_path = storage.get_job_path(job_id)
        
        rendered_outfits = render_all_outfits_user_photo(
            job_id=job_id,
            person_image_path=person_image_path,
            outfits=outfits,
            output_dir=job_path
        )
        
        # Step 3: Check if any renders succeeded
        success_count = sum(
            1 for o in rendered_outfits 
            if o.get("tryon_mode") == "user_photo"
        )
        
        if success_count == 0:
            logger.warning(f"[{job_id}] All user photo renders failed, falling back to mannequin")
            fallback_outfits = _render_partial_tryon_outfits(
                job_id=job_id,
                outfits=outfits,
                person_image_path=person_image_path,
                gender=gender
            )
            for outfit in fallback_outfits:
                outfit["tryon_mode"] = "user_photo_fallback"
            return fallback_outfits
        
        logger.info(f"[{job_id}] User photo try-on complete: {success_count}/{len(rendered_outfits)} succeeded")
        return rendered_outfits
        
    except ImportError as e:
        logger.error(f"[{job_id}] User photo try-on module not available: {e}")
        fallback_outfits = _render_partial_tryon_outfits(
            job_id=job_id,
            outfits=outfits,
            person_image_path=person_image_path,
            gender=gender
        )
        for outfit in fallback_outfits:
            outfit["tryon_mode"] = "user_photo_fallback"
        return fallback_outfits
        
    except Exception as e:
        logger.error(f"[{job_id}] User photo try-on failed: {e}")
        fallback_outfits = _render_partial_tryon_outfits(
            job_id=job_id,
            outfits=outfits,
            person_image_path=person_image_path,
            gender=gender
        )
        for outfit in fallback_outfits:
            outfit["tryon_mode"] = "user_photo_fallback"
        return fallback_outfits


def _update_job_with_outfits(job_id: str, seed: Dict[str, Any], outfits: list) -> None:
    """Update job.json with generated outfits."""
    import json
    from datetime import datetime, timezone
    
    job_path = storage.get_job_path(job_id)
    job_json_path = job_path / "job.json"
    
    try:
        with open(job_json_path, "r", encoding="utf-8") as f:
            job_data = json.load(f)
    except Exception:
        job_data = {}
    
    job_data["seed"] = {
        "status": "detected",
        "category": seed.get("category"),
        "color": seed.get("color"),
        "style": [],
        "locked": True
    }
    job_data["outfits"] = outfits
    job_data["current_stage"] = "outfits_generated"
    job_data["updated_at"] = datetime.now(timezone.utc).isoformat()
    
    with open(job_json_path, "w", encoding="utf-8") as f:
        json.dump(job_data, f, indent=2)
    
    logger.info(f"[{job_id}] Updated job.json with {len(outfits)} outfits")


# ==================== MAIN PIPELINE ====================


async def run_pipeline(
    job_id: str,
    image_path: str,
    user_note: Optional[str] = None,
    owner_user_id: Optional[str] = None,
    event: Optional[str] = None,
    weather_context: Optional[str] = None,
    mode: str = "full",  # v2.7.0: full | single
    turbo: bool = False  # v2.9.0: Turbo mode
) -> Dict[str, Any]:
    """
    Run the full pipeline with observability tracking.
    
    Args:
        job_id: Unique job identifier
        image_path: Path to input image
        user_note: Optional user notes
        owner_user_id: User ID for job ownership (v2.0)
        event: Event type for context (v2.1)
        weather_context: Weather context string (v2.1)
        mode: Product mode - 'full' (5 outfits) or 'single' (best only, v2.7.0)
    """
    logger.info(f"Pipeline starting: job={job_id}, mode={mode}")
    start_time = time.time()
    
    result = {
        "job_id": job_id,
        "detected_clothing": {"top": False, "bottom": False, "outerwear": False, "shoes": False},
        "detected_items": {},
        "masks": {},
        "raw_labels": [],
        "outfits": [],
        "renders": {},
        "status": "completed",
        "cache_hit": False,
        "cost": None,
        "error": None
    }
    
    job_path = storage.get_job_path(job_id)
    masks_dir = str(job_path / "masks")
    renders_dir = str(job_path / "renders")
    active_provider = get_active_provider()
    
    # Create job in MongoDB with owner
    job_doc = mongo.create_job_document(job_id)
    job_doc["assets"]["input_image"] = f"/ai/assets/jobs/{job_id}/input.jpg"
    if owner_user_id:
        job_doc["owner_user_id"] = owner_user_id
    if event:
        job_doc["event"] = event
    if weather_context:
        job_doc["weather_context"] = weather_context
    mongo.insert_job(job_doc)
    
    # Step 1: Segmentation & Wardrobe Context (Parallel)
    logger.info(f"[{job_id}] Starting parallel step: Segmentation + Wardrobe")
    
    async def get_seg():
        try:
            return segmenter.segment(image_path, extract_attributes=True)
        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            return {"detected_items": {cat: {"present": False} for cat in ["top", "bottom", "outerwear", "shoes"]}}

    async def get_wardrobe():
        if owner_user_id:
            return build_wardrobe_context(owner_user_id)
        return None

    seg_task = asyncio.create_task(asyncio.to_thread(segmenter.segment, image_path, extract_attributes=True))
    wardrobe_task = asyncio.create_task(asyncio.to_thread(build_wardrobe_context, owner_user_id)) if owner_user_id else asyncio.Future()
    if not owner_user_id: wardrobe_task.set_result(None)

    seg_result, wardrobe_ctx = await asyncio.gather(seg_task, wardrobe_task)
    
    # Process Segmentation Results
    result["detected_clothing"]["top"] = seg_result.get("top", False)
    result["detected_clothing"]["bottom"] = seg_result.get("bottom", False)
    result["detected_clothing"]["outerwear"] = seg_result.get("outerwear", False)
    result["detected_clothing"]["shoes"] = seg_result.get("shoes", False)
    result["raw_labels"] = seg_result.get("raw_labels", [])
    result["detected_items"] = seg_result.get("detected_items", {})
    
    masks = seg_result.get("masks", {})
    if masks:
        saved_masks = segmenter.save_masks(masks, masks_dir)
        result["masks"] = saved_masks
    
    mongo.update_job(job_id, {
        "detected_clothing": result["detected_clothing"],
        "attributes": result["detected_items"],
        "assets.masks": {k: f"/ai/assets/jobs/{job_id}/masks/{v}" for k, v in result["masks"].items()}
    })

    # Check Cache
    cache_key = None
    tokens, cost_usd = 0, 0.0
    
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
            
            mongo.update_job(job_id, {
                "outfits": result["outfits"],
                "cached": True,
                "provider_used": "cached",
                "status": "completed",
                "assets.renders": [f"/ai/assets/jobs/{job_id}/renders/outfit_{i}.png" for i in range(1, len(result["outfits"]) + 1)]
            })
            
            latency_ms = int((time.time() - start_time) * 1000)
            result["processing_time_ms"] = latency_ms
            _track_request(job_id, "cached", True, latency_ms, "success", 0, 0.0)
            return result

    # Step 2: LLM Generation
    try:
        outfits = await llm_router.plan_outfits(
            detected_items=result["detected_items"],
            user_note=user_note,
            season_hint=None,
            event=event,
            weather_context=weather_context,
            wardrobe_context=wardrobe_ctx
        )
        result["outfits"] = outfits
        
        # Estimate cost
        tokens, cost_usd = estimate_cost(active_provider or "openai")
        result["cost"] = {
            "provider": active_provider,
            "tokens": tokens,
            "estimated_usd": cost_usd
        }
        
        mongo.update_job(job_id, {
            "outfits": outfits,
            "provider_used": active_provider or "unknown",
            "cost": result["cost"]
        })
        
    except Exception as e:
        logger.error(f"LLM failed: {e}")
        result["error"] = str(e)
        result["status"] = "failed"
        
        latency_ms = int((time.time() - start_time) * 1000)
        _track_request(job_id, active_provider, False, latency_ms, "fail", 0, 0.0, str(e))
        
        mongo.update_job(job_id, {"status": "failed", "error": str(e)})
        return result
    
    # Step 3: Rendering
    if result["outfits"]:
        try:
            from ai_service.renderer.tryon import render_all_outfits
            
            # v2.9.0: Skip self-critique if turbo=True or mode='single'
            skip_critique = turbo or mode == "single"
            
            render_result = await render_all_outfits(
                input_image_path=image_path,
                masks_dir=masks_dir,
                outfits=result["outfits"],
                output_dir=renders_dir,
                turbo=turbo
            )
            
            result["renders"] = render_result.get("renders", {})
            render_urls = []
            
            for i, outfit in enumerate(result["outfits"], start=1):
                if i in result["renders"]:
                    url = f"/ai/assets/jobs/{job_id}/renders/{result['renders'][i]}"
                    outfit["render_url"] = url
                    outfit["tryon_method"] = "inpainting"
                    render_urls.append(url)
            
            mongo.update_job(job_id, {"assets.renders": render_urls})
            
            # v2.4.0: Self-Critique Pipeline (SKIP IF TURBO/SINGLE)
            if not skip_critique:
                try:
                    result = await _run_self_critique(
                        result=result,
                        job_id=job_id,
                        image_path=image_path,
                        renders_dir=renders_dir,
                        masks_dir=masks_dir,
                        detected_items=result["detected_items"],
                        event=event,
                        weather_context=weather_context,
                        wardrobe_ctx=wardrobe_ctx
                    )
                except Exception as critique_error:
                    logger.warning(f"Self-critique failed (continuing): {critique_error}")
            else:
                logger.info(f"[{job_id}] Skipping self-critique (mode={mode}, turbo={turbo})")
            
        except Exception as e:
            logger.error(f"Rendering failed: {e}")
            _create_fallback_renders(image_path, renders_dir, result)
    
    # Save to Cache & Finalize
    if cache_manager.enabled and cache_key and result["outfits"]:
        cache_manager.set(cache_key, {"outfits": result["outfits"], "_render_dir": renders_dir})
    
    mongo.update_job(job_id, {"status": "completed", "cached": False})
    
    latency_ms = int((time.time() - start_time) * 1000)
    result["processing_time_ms"] = latency_ms
    
    # Log and track metrics
    _track_request(job_id, active_provider, False, latency_ms, "success", tokens, cost_usd)
    
    logger.info(f"[{job_id}] Pipeline complete in {latency_ms}ms (cost: ${cost_usd:.4f})")
    
    return result


def _track_request(job_id: str, provider: str, cache_hit: bool, latency_ms: int, status: str, tokens: int, cost_usd: float, error: str = None):
    """Log and track request metrics."""
    if is_logging_enabled():
        log_request(job_id, provider or "unknown", cache_hit, latency_ms, status, error, tokens, cost_usd)
    
    increment_request(provider or "unknown", cache_hit, tokens, cost_usd, error is not None)


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


async def _run_self_critique(
    result: Dict[str, Any],
    job_id: str,
    image_path: str,
    renders_dir: str,
    masks_dir: str,
    detected_items: Dict[str, Any],
    event: Optional[str],
    weather_context: Optional[str],
    wardrobe_ctx: Optional[str]
) -> Dict[str, Any]:
    """
    v2.4.0: Self-critique pipeline.
    
    1. Gemini judges all try-on renders
    2. Identify worst outfits (score < 0.6)
    3. OpenAI regenerates replacement outfits
    4. SD re-renders only the replaced outfits
    5. Ensures exactly 5 outfits returned
    """
    from ai_service.renderer.tryon import TryOnRenderer
    
    logger.info(f"[{job_id}] Running self-critique pipeline...")
    
    # Track stats
    stats = {
        "replaced_count": 0,
        "avg_score_before": 0.0,
        "avg_score_after": 0.0
    }
    
    # Step 1: Gemini judges all outfits
    judge_results = await judge_all_outfits(
        original_image_path=image_path,
        renders_dir=renders_dir,
        outfit_count=len(result["outfits"])
    )
    
    # Calculate average score before
    if judge_results:
        stats["avg_score_before"] = sum(r.quality_score for r in judge_results) / len(judge_results)
    
    # Add quality scores to outfits
    for jr in judge_results:
        idx = jr.outfit_index - 1
        if 0 <= idx < len(result["outfits"]):
            result["outfits"][idx]["quality_score"] = jr.quality_score
            result["outfits"][idx]["quality_verdict"] = jr.verdict
            if jr.issues:
                result["outfits"][idx]["quality_issues"] = jr.issues
    
    logger.info(f"[{job_id}] Gemini judged {len(judge_results)} outfits, avg score: {stats['avg_score_before']:.2f}")
    
    # Step 2: Identify worst outfits
    worst = identify_worst_outfits(judge_results, threshold=0.6, max_replace=2)
    
    if not worst:
        logger.info(f"[{job_id}] All outfits pass quality threshold")
        result["self_critique"] = {
            "replaced_count": 0,
            "avg_score": stats["avg_score_before"]
        }
        return result
    
    logger.info(f"[{job_id}] Found {len(worst)} low-quality outfits to replace")
    
    # Step 3: OpenAI regenerates replacements
    worst_dicts = [w.to_dict() for w in worst]
    
    updated_outfits = await regenerate_worst_outfits(
        worst_results=worst_dicts,
        detected_items=detected_items,
        outfits=result["outfits"],
        event=event,
        weather_context=weather_context,
        wardrobe_context=wardrobe_ctx
    )
    
    result["outfits"] = updated_outfits
    stats["replaced_count"] = len(worst)
    
    # Step 4: Re-render only replaced outfits in parallel
    renderer = TryOnRenderer()
    
    async def render_task(w_obj):
        idx = w_obj.outfit_index
        if 0 < idx <= len(result["outfits"]):
            outfit = result["outfits"][idx - 1]
            render_path = f"{renders_dir}/outfit_{idx}.png"
            
            success = await asyncio.to_thread(
                renderer.render_outfit,
                input_image_path=image_path,
                masks_dir=masks_dir,
                outfit=outfit,
                output_path=render_path,
                resolution=512
            )
            
            if success:
                outfit["render_url"] = f"/ai/assets/jobs/{job_id}/renders/outfit_{idx}.png"
                outfit["regenerated"] = True
                logger.info(f"[{job_id}] Re-rendered outfit {idx}")

    if worst:
        await asyncio.gather(*[render_task(w) for w in worst])
    
    # Step 5: Re-judge regenerated outfits to get new scores
    new_judge_results = await judge_all_outfits(
        original_image_path=image_path,
        renders_dir=renders_dir,
        outfit_count=len(result["outfits"])
    )
    
    if new_judge_results:
        stats["avg_score_after"] = sum(r.quality_score for r in new_judge_results) / len(new_judge_results)
        
        # Update final scores
        for jr in new_judge_results:
            idx = jr.outfit_index - 1
            if 0 <= idx < len(result["outfits"]):
                result["outfits"][idx]["quality_score"] = jr.quality_score
                result["outfits"][idx]["quality_verdict"] = jr.verdict
    
    # Ensure exactly 5 outfits
    while len(result["outfits"]) < 5:
        result["outfits"].append({
            "rank": len(result["outfits"]) + 1,
            "style_tag": "fallback",
            "items": {},
            "quality_score": 0.5,
            "quality_verdict": "fallback"
        })
    
    result["self_critique"] = {
        "replaced_count": stats["replaced_count"],
        "avg_score_before": round(stats["avg_score_before"], 3),
        "avg_score_after": round(stats["avg_score_after"], 3),
        "improvement": round(stats["avg_score_after"] - stats["avg_score_before"], 3)
    }
    
    logger.info(
        f"[{job_id}] Self-critique complete: "
        f"replaced {stats['replaced_count']}, "
        f"score {stats['avg_score_before']:.2f} → {stats['avg_score_after']:.2f}"
    )
    
    return result
