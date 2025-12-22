"""
API Routes for AuraProject AI Service v2.7.0
Single Best Outfit Mode + ControlNet pose lock.
"""
import logging
from pathlib import Path
from typing import Optional, Literal
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Request, Query
from fastapi.responses import FileResponse, JSONResponse

from ai_service.core.storage import storage
from ai_service.core.orchestrator import run_pipeline, run_outfit_seed_job
from ai_service.core.validation import (
    ValidationError,
    validate_image_upload,
    sanitize_asset_path,
    validate_outfit_seed_input,
)
from ai_service.core.auth import (
    User,
    get_current_user,
    get_optional_user,
    create_user,
    check_job_ownership,
    increment_concurrent_jobs,
    decrement_concurrent_jobs,
)
from ai_service.core.rate_limit import (
    check_rate_limit,
    check_concurrent_jobs,
    get_rate_limit_headers,
    get_client_ip,
)
from ai_service.config import get_provider_status, get_settings
from ai_service.cache import cache_manager
from ai_service.db import mongo
from ai_service.db.history import (
    get_user_history,
    get_history_count,
    add_favorite,
    get_favorites,
    remove_favorite,
    add_feedback,
    get_user_feedback_stats,
)
from ai_service.services.weather import get_weather, is_configured as weather_configured
from ai_service.observability import get_metrics, is_logging_enabled

logger = logging.getLogger(__name__)

router = APIRouter()

# Allowed event types
EVENT_TYPES = Literal["business", "casual", "sport", "wedding", "party", "date"]


def _generate_winner_explanation(winner: dict, event: Optional[str]) -> str:
    """Generate a human-readable explanation for the winning outfit."""
    score = winner.get("quality_score", 0.7)
    items = winner.get("items", {})
    
    item_count = len([v for v in items.values() if v])
    
    if score >= 0.85:
        quality = "excellent visual quality"
    elif score >= 0.7:
        quality = "good overall appearance"
    else:
        quality = "balanced styling"
    
    event_text = f"for {event}" if event else ""
    
    return f"Selected as the best outfit {event_text} with {quality} and {item_count} coordinated pieces."


# ==================== PUBLIC ENDPOINTS ====================

@router.get("/health")
async def health_check():
    """Health check with observability info."""
    provider_status = get_provider_status()
    cache_status = cache_manager.get_status()
    mongo_status = mongo.health_check()
    metrics = get_metrics()
    
    # Get LLM config status (v2.5.0)
    from ai_service.config.llm_config import get_all_configs_dict, get_controlnet_config
    llm_configs = get_all_configs_dict()
    controlnet_config = get_controlnet_config()
    
    return {
        "status": "ok",
        "version": "2.7.0",
        "llm": {
            "enabled": provider_status["enabled"],
            "primary": provider_status["primary"],
            "secondary": provider_status["secondary"],
            "availability": provider_status["availability"],
            "active_provider": provider_status["active_provider"],
        },
        "llm_config": llm_configs,
        "controlnet": controlnet_config.to_dict(),
        "product_mode_support": ["full", "single"],  # v2.7.0
        "cache": cache_status,
        "mongo": mongo_status,
        "weather": {"enabled": weather_configured()},
        "observability": {
            "logging_enabled": is_logging_enabled(),
            "total_requests": metrics["total_requests"],
            "cache_hit_ratio": metrics["cache_hit_ratio"],
            "total_cost_usd": metrics["total_cost_usd"]
        },
        "features": [
            "segmentation", "attributes", "hybrid_llm", "tryon", 
            "cache", "mongodb", "observability",
            "input_validation", "api_key_auth", "rate_limiting",
            "event_context", "weather", "history", "favorites",
            "wardrobe", "duplicate_detection",
            "self_critique", "gemini_judge", "outfit_regeneration",
            "gemini_3_pro_ready", "gpt_5_ready",
            "controlnet_pose_lock",
            "single_best_outfit_mode"  # v2.7 feature
        ]
    }


@router.get("/metrics")
async def get_metrics_endpoint():
    """Get detailed metrics for monitoring."""
    metrics = get_metrics()
    return JSONResponse(content=metrics)


# ==================== USER MANAGEMENT ====================

@router.post("/ai/users")
async def create_new_user(
    name: str = Form(..., description="User display name")
):
    """
    Create a new user and get API key.
    
    WARNING: The API key is only shown once!
    """
    user = create_user(name)
    
    if user is None:
        raise HTTPException(
            status_code=500,
            detail="Failed to create user. Is MongoDB running?"
        )
    
    return JSONResponse(content=user, status_code=201)


# ==================== OUTFIT SEED (v3.0.0) ====================

from ai_service.core.orchestrator import OutfitSeedError

ALLOWED_SEED_CATEGORIES = {"top", "bottom", "outerwear", "shoes", "accessory"}


def _get_file_extension(upload_file: UploadFile) -> str:
    """Get appropriate file extension from upload file."""
    # Try content type first
    content_type_map = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
    }
    ext = content_type_map.get(upload_file.content_type, None)
    if ext:
        return ext
    
    # Fallback to filename
    if upload_file.filename and "." in upload_file.filename:
        return "." + upload_file.filename.rsplit(".", 1)[-1].lower()
    
    # Default to jpg
    return ".jpg"


def _compute_aggregate_tryon_mode(outfits: list) -> str:
    """Compute aggregate tryon_mode from all outfits."""
    modes = [o.get("tryon_mode", "mock") for o in outfits]
    
    if all(m == "full" for m in modes):
        return "full"
    elif all(m == "mock" for m in modes):
        return "mock"
    elif all(m == "partial" for m in modes):
        return "partial"
    elif all(m == "partial_fallback" for m in modes):
        return "partial_fallback"
    else:
        return "mixed"


@router.post("/ai/outfit-seed")
async def create_outfit_seed(
    request: Request,
    seed_image: Optional[UploadFile] = File(None, description="Seed garment image"),
    person_image: Optional[UploadFile] = File(None, description="Full-body person image"),
    gender: str = Form(..., description="Gender: male or female"),
    seed_category: Optional[str] = Form(None, description="Seed category: top, bottom, outerwear, shoes, accessory"),
    event: Optional[str] = Form(None, description="Event type: work, date, party, casual"),
    season: Optional[str] = Form(None, description="Season: summer, winter"),
    mode: Optional[str] = Form("mock", description="Mode: mock, partial_tryon, full_tryon"),
    user: User = Depends(get_current_user)
):
    """
    POST /ai/outfit-seed (v3.0.0)
    
    Create a seed-locked outfit job with 5 outfits. The seed item
    is locked and appears in all outfits.
    
    Headers:
        X-API-Key: Your API key (required)
    
    Form Parameters:
        - seed_image: Seed garment image (optional)
        - person_image: Full-body person image (optional)
        - gender: male | female (required)
        - seed_category: top | bottom | outerwear | shoes | accessory (optional, auto-detected if not provided)
        - event: work | date | party | casual (optional)
        - season: summer | winter (optional)
        - mode: mock | partial_tryon | full_tryon (default: mock)
    
    At least one of seed_image or person_image must be provided.
    """
    import uuid
    
    # Rate limiting
    await check_rate_limit(request, user)
    
    try:
        # Validate input
        seed_image_provided = seed_image is not None and seed_image.filename
        person_image_provided = person_image is not None and person_image.filename
        
        try:
            validated = validate_outfit_seed_input(
                gender=gender,
                seed_image_provided=seed_image_provided,
                person_image_provided=person_image_provided,
                event=event,
                season=season,
                mode=mode
            )
        except ValidationError as ve:
            raise HTTPException(status_code=ve.status_code, detail=ve.message)
        
        # Validate seed_category if provided
        validated_seed_category = None
        if seed_category:
            if seed_category.lower() not in ALLOWED_SEED_CATEGORIES:
                raise HTTPException(
                    status_code=400,
                    detail=f"seed_category must be one of: {', '.join(ALLOWED_SEED_CATEGORIES)}"
                )
            validated_seed_category = seed_category.lower()
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        logger.info(f"Outfit-seed request: job_id={job_id}, gender={validated['gender']}, mode={validated['mode']}, seed_category={validated_seed_category}")
        
        # Save images with proper extensions
        seed_image_path = None
        person_image_path = None
        
        job_path = storage.get_job_path(job_id)
        job_path.mkdir(parents=True, exist_ok=True)
        
        if seed_image_provided:
            ext = _get_file_extension(seed_image)
            seed_path = job_path / f"seed_image{ext}"
            content = await seed_image.read()
            with open(seed_path, "wb") as f:
                f.write(content)
            seed_image_path = str(seed_path)
        
        if person_image_provided:
            ext = _get_file_extension(person_image)
            person_path = job_path / f"person_image{ext}"
            content = await person_image.read()
            with open(person_path, "wb") as f:
                f.write(content)
            person_image_path = str(person_path)
        
        # Run seed job pipeline
        try:
            result = run_outfit_seed_job(
                job_id=job_id,
                seed_image_path=seed_image_path,
                person_image_path=person_image_path,
                gender=validated["gender"],
                event=validated["event"],
                season=validated["season"],
                mode=validated["mode"],
                seed_category=validated_seed_category
            )
        except OutfitSeedError as ose:
            raise HTTPException(status_code=ose.status_code, detail=ose.message)
        
        # Extract results
        subject_type = result["subject_type"]
        seed = result["seed"]
        outfits = result["outfits"]
        
        # Compute aggregate tryon_mode from actual render results
        actual_tryon_mode = _compute_aggregate_tryon_mode(outfits)
        
        # Return response contract
        response = {
            "version": "3.0.0",
            "job_id": job_id,
            "seed_locked": True,
            "seed": {
                "status": "detected",
                "category": seed.get("category"),
                "color": seed.get("color"),
                "confidence": seed.get("detection_confidence", 1.0),
                "style": seed.get("style", [])
            },
            "outfits": outfits,
            "tryon_mode": actual_tryon_mode,
            "subject_type": subject_type,
            "message": "Job created. Pipeline completed."
        }
        
        headers = get_rate_limit_headers(request, user)
        return JSONResponse(content=response, headers=headers)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Outfit-seed failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== OUTFIT GENERATION ====================

@router.post("/ai/outfit")
async def create_outfit(
    request: Request,
    image: UploadFile = File(..., description="User photo"),
    user_note: Optional[str] = Form(None, description="Optional notes"),
    event: Optional[str] = Form(None, description="Event type: business, casual, sport, wedding, party, date"),
    city: Optional[str] = Form(None, description="City for weather context"),
    mode: Optional[str] = Form("full", description="Mode: full (5 outfits) | single (best outfit only)"),
    turbo: Optional[bool] = Form(False, description="Turbo mode for faster processing (v2.9.0)"),
    user: User = Depends(get_current_user)
):
    """
    Generate outfit recommendations.
    
    v2.7.0: Added mode parameter for single best outfit selection.
    
    Headers:
        X-API-Key: Your API key (required)
    
    Form Parameters:
        - image: User photo (required)
        - user_note: Optional styling notes
        - event: business | casual | sport | wedding | party | date
        - city: City name for weather (e.g., Istanbul, London)
        - mode: full (default, 5 outfits) | single (best outfit only)
    """
    # Validate event type
    valid_events = ["business", "casual", "sport", "wedding", "party", "date"]
    if event and event not in valid_events:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid event type. Allowed: {', '.join(valid_events)}"
        )
    
    # Validate mode
    valid_modes = ["full", "single"]
    if mode not in valid_modes:
        mode = "full"
    
    logger.info(f"Outfit request: mode={mode}")
    
    # Rate limiting check
    await check_rate_limit(request, user)
    
    # Concurrent job limit check
    await check_concurrent_jobs(user)
    
    try:
        # Input validation
        try:
            content, validated_image = await validate_image_upload(
                image,
                image.content_type
            )
        except ValidationError as ve:
            raise HTTPException(status_code=ve.status_code, detail=ve.message)
        
        # Fetch weather if city provided
        weather_info = None
        weather_context = None
        if city:
            weather_info = await get_weather(city)
            if weather_info:
                weather_context = weather_info.to_prompt_context()
                logger.info(f"Weather context: {weather_info.city} - {weather_info.layer_hint}")
        
        # Increment concurrent jobs
        increment_concurrent_jobs(user.user_id)
        
        try:
            job_id = storage.create_job()
            logger.info(f"New job: {job_id} (user: {user.user_id}, event: {event}, mode: {mode})")
            
            # Save with validated image
            image_path = storage.save_input_image(job_id, image.file)
            
            result = await run_pipeline(
                job_id=job_id,
                image_path=str(image_path),
                user_note=user_note,
                owner_user_id=user.user_id,
                event=event,
                weather_context=weather_context,
                mode=mode,  # v2.7.0: Pass mode to orchestrator
                turbo=turbo  # v2.9.0: Pass turbo flag
            )
            
            # v2.7.0: Handle single mode response
            if mode == "single":
                outfits = result.get("outfits", [])
                if outfits:
                    # Sort by quality score and pick best
                    sorted_outfits = sorted(
                        outfits, 
                        key=lambda x: x.get("quality_score", 0), 
                        reverse=True
                    )
                    winner = sorted_outfits[0]
                    
                    logger.info(f"Single mode winner: index={winner.get('rank', 1)}, confidence={winner.get('quality_score', 0):.2f}")
                    
                    response = {
                        "job_id": job_id,
                        "title": f"Best {event or 'casual'} outfit",
                        "outfit": winner.get("items", {}),
                        "render_url": winner.get("render_url"),
                        "confidence": winner.get("quality_score", 0.7),
                        "why": _generate_winner_explanation(winner, event),
                        "version": "single-mode",
                        "context": {
                            "event": event,
                            "city": city,
                            "weather": weather_info.to_dict() if weather_info else None
                        }
                    }
                else:
                    response = {
                        "job_id": job_id,
                        "title": "No outfit found",
                        "outfit": {},
                        "render_url": None,
                        "confidence": 0.0,
                        "why": "Could not generate outfit recommendations.",
                        "version": "single-mode"
                    }
                
                headers = get_rate_limit_headers(request, user)
                return JSONResponse(content=response, headers=headers)
            
            # Full mode response (unchanged)
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
                "context": {
                    "event": event,
                    "city": city,
                    "weather": weather_info.to_dict() if weather_info else None
                },
                "note": "v2.7.0 - mode=full (5 outfits)"
            }
            
            if result.get("error"):
                response["error"] = result["error"]
            
            # Add rate limit headers
            headers = get_rate_limit_headers(request, user)
            return JSONResponse(content=response, headers=headers)
            
        finally:
            # Always decrement concurrent jobs
            decrement_concurrent_jobs(user.user_id)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Outfit generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== JOBS ====================

@router.get("/ai/jobs/{job_id}")
async def get_job(
    job_id: str,
    user: User = Depends(get_current_user)
):
    """
    Get a persisted job by ID.
    
    Only returns jobs owned by the authenticated user.
    """
    job = mongo.get_job(job_id)
    
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    # Check ownership
    if not check_job_ownership(job, user):
        raise HTTPException(status_code=403, detail="Access denied: not your job")
    
    return JSONResponse(content=job)


# ==================== HISTORY ====================

@router.get("/ai/history")
async def get_history(
    limit: int = Query(20, ge=1, le=100, description="Max results"),
    offset: int = Query(0, ge=0, description="Skip for pagination"),
    user: User = Depends(get_current_user)
):
    """
    Get user's outfit generation history.
    
    Returns a list of past jobs with summaries.
    """
    history = get_user_history(user.user_id, limit=limit, offset=offset)
    total = get_history_count(user.user_id)
    
    return JSONResponse(content={
        "history": history,
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": offset + len(history) < total
    })


# ==================== FAVORITES ====================

@router.post("/ai/favorites")
async def add_favorite_outfit(
    job_id: str = Form(..., description="Job ID containing the outfit"),
    outfit_index: int = Form(..., ge=1, le=10, description="Outfit index (1-based)"),
    note: Optional[str] = Form(None, description="Optional note"),
    user: User = Depends(get_current_user)
):
    """
    Save an outfit as favorite.
    
    The outfit is saved with its full snapshot for future reference.
    """
    favorite = add_favorite(
        user_id=user.user_id,
        job_id=job_id,
        outfit_index=outfit_index,
        note=note
    )
    
    if favorite is None:
        raise HTTPException(
            status_code=400,
            detail="Failed to add favorite. Check job_id and outfit_index."
        )
    
    return JSONResponse(content=favorite, status_code=201)


@router.get("/ai/favorites")
async def get_favorites_list(
    limit: int = Query(50, ge=1, le=100, description="Max results"),
    offset: int = Query(0, ge=0, description="Skip for pagination"),
    user: User = Depends(get_current_user)
):
    """
    Get user's favorite outfits.
    """
    favorites = get_favorites(user.user_id, limit=limit, offset=offset)
    
    return JSONResponse(content={
        "favorites": favorites,
        "count": len(favorites),
        "limit": limit,
        "offset": offset
    })


@router.delete("/ai/favorites/{job_id}/{outfit_index}")
async def delete_favorite(
    job_id: str,
    outfit_index: int,
    user: User = Depends(get_current_user)
):
    """
    Remove a favorite.
    """
    success = remove_favorite(user.user_id, job_id, outfit_index)
    
    if not success:
        raise HTTPException(status_code=404, detail="Favorite not found")
    
    return JSONResponse(content={"message": "Favorite removed"})


# ==================== FEEDBACK ====================

@router.post("/ai/feedback")
async def add_outfit_feedback(
    job_id: str = Form(..., description="Job ID"),
    outfit_index: int = Form(..., ge=1, le=10, description="Outfit index (1-based)"),
    feedback_type: str = Form(..., description="like or dislike"),
    reason: Optional[str] = Form(None, description="Reason for dislike"),
    user: User = Depends(get_current_user)
):
    """
    Add feedback (like/dislike) for an outfit.
    
    Dislike reasons help improve future recommendations:
    - too_formal, too_casual, wrong_colors, wrong_style, not_my_taste, etc.
    """
    if feedback_type not in ["like", "dislike"]:
        raise HTTPException(
            status_code=400,
            detail="feedback_type must be 'like' or 'dislike'"
        )
    
    feedback = add_feedback(
        user_id=user.user_id,
        job_id=job_id,
        outfit_index=outfit_index,
        feedback_type=feedback_type,
        reason=reason
    )
    
    if feedback is None:
        raise HTTPException(
            status_code=400,
            detail="Failed to add feedback. Check job ownership."
        )
    
    return JSONResponse(content=feedback, status_code=201)


@router.get("/ai/feedback/stats")
async def get_feedback_stats(
    user: User = Depends(get_current_user)
):
    """
    Get feedback statistics for the user.
    
    Shows like/dislike counts and top dislike reasons.
    """
    stats = get_user_feedback_stats(user.user_id)
    return JSONResponse(content=stats)


# ==================== WARDROBE ====================

from ai_service.db.wardrobe import (
    create_wardrobe_item,
    get_wardrobe_items as db_get_wardrobe_items,
    get_wardrobe_item,
    delete_wardrobe_item,
    find_duplicate_in_wardrobe,
    get_wardrobe_count,
    compute_phash,
)
from ai_service.vision.segmenter import segmenter


@router.post("/ai/wardrobe/items")
async def upload_wardrobe_item(
    request: Request,
    image: UploadFile = File(..., description="Clothing item photo"),
    category: str = Form(..., description="Category: top, bottom, outerwear, shoes"),
    season: Optional[str] = Form("all", description="Season: spring, summer, fall, winter, all"),
    style_tags: Optional[str] = Form(None, description="Comma-separated style tags"),
    user: User = Depends(get_current_user)
):
    """
    Upload a clothing item to user's wardrobe.
    
    v2.3.0: Auto-segments clothing and detects duplicates.
    
    Flow:
    1. Validate image
    2. Compute pHash for duplicate detection
    3. Check for duplicates in wardrobe
    4. Segment clothing to extract mask
    5. Extract attributes (color, style)
    6. Save item to wardrobe
    """
    # Validate category
    valid_categories = ["top", "bottom", "outerwear", "shoes"]
    if category not in valid_categories:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category. Allowed: {', '.join(valid_categories)}"
        )
    
    # Rate limiting
    await check_rate_limit(request, user)
    
    try:
        # Input validation
        try:
            content, validated_image = await validate_image_upload(
                image.file,
                image.content_type
            )
        except ValidationError as ve:
            raise HTTPException(status_code=ve.status_code, detail=ve.message)
        
        # Save image temporarily for processing
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.file.seek(0)
            tmp.write(image.file.read())
            tmp_path = tmp.name
        
        try:
            # Compute pHash for duplicate detection
            phash = compute_phash(tmp_path)
            
            # Check for duplicates
            if phash:
                duplicate = find_duplicate_in_wardrobe(user.user_id, phash)
                if duplicate:
                    return JSONResponse(
                        status_code=409,
                        content={
                            "error": "duplicate_detected",
                            "message": "This item appears to already exist in your wardrobe",
                            "existing_item": duplicate
                        }
                    )
            
            # Segment clothing
            seg_result = segmenter.segment(tmp_path, extract_attributes=True)
            
            # Extract attributes for the specified category
            detected = seg_result.get("detected_items", {}).get(category, {})
            color = detected.get("color", "unknown")
            style = detected.get("style", "casual")
            
            # Parse style tags
            tags = []
            if style_tags:
                tags = [t.strip() for t in style_tags.split(",") if t.strip()]
            if style and style not in tags:
                tags.append(style)
            
            # Create wardrobe storage directory
            wardrobe_dir = Path("ai_service/data/wardrobe") / user.user_id
            wardrobe_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate unique filename
            import secrets
            item_filename = f"{secrets.token_hex(8)}.jpg"
            mask_filename = f"{secrets.token_hex(8)}_mask.png"
            
            # Save image
            image_path = wardrobe_dir / item_filename
            validated_image.save(str(image_path), quality=95)
            
            # Save mask if available
            mask_url = None
            if category in seg_result.get("masks", {}):
                mask = seg_result["masks"][category]
                mask_path = wardrobe_dir / mask_filename
                mask.save(str(mask_path))
                mask_url = f"/ai/assets/wardrobe/{user.user_id}/{mask_filename}"
            
            image_url = f"/ai/assets/wardrobe/{user.user_id}/{item_filename}"
            
            # Create wardrobe item
            item = create_wardrobe_item(
                owner_user_id=user.user_id,
                category=category,
                image_url=image_url,
                mask_url=mask_url,
                color_palette=[color] if color != "unknown" else [],
                style_tags=tags,
                season=season,
                phash=phash
            )
            
            if item is None:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to save wardrobe item"
                )
            
            return JSONResponse(content=item, status_code=201)
            
        finally:
            # Cleanup temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Wardrobe upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ai/wardrobe/items")
async def list_wardrobe_items(
    category: Optional[str] = Query(None, description="Filter by category"),
    season: Optional[str] = Query(None, description="Filter by season"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user: User = Depends(get_current_user)
):
    """
    List user's wardrobe items.
    """
    items = db_get_wardrobe_items(
        user_id=user.user_id,
        category=category,
        season=season,
        limit=limit,
        offset=offset
    )
    
    total = get_wardrobe_count(user.user_id)
    
    return JSONResponse(content={
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": offset + len(items) < total
    })


@router.get("/ai/wardrobe/items/{item_id}")
async def get_single_wardrobe_item(
    item_id: str,
    user: User = Depends(get_current_user)
):
    """
    Get a specific wardrobe item.
    """
    item = get_wardrobe_item(user.user_id, item_id)
    
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    
    return JSONResponse(content=item)


@router.delete("/ai/wardrobe/items/{item_id}")
async def remove_wardrobe_item(
    item_id: str,
    user: User = Depends(get_current_user)
):
    """
    Delete a wardrobe item.
    """
    success = delete_wardrobe_item(user.user_id, item_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Item not found")
    
    return JSONResponse(content={"message": "Item deleted"})


# ==================== CATALOG-BASED OUTFIT ====================

from ai_service.outfit_recommender import generate_outfit_combos, generate_single_item_combos
from ai_service.outfit_renderer import render_all_outfit_cards


@router.post("/ai/outfit-catalog")
async def catalog_outfit(
    request: Request,
    image: UploadFile = File(..., description="User photo or clothing item"),
    item_category: Optional[str] = Form(None, description="Category of uploaded item: top, bottom, outerwear, shoes"),
    item_color: Optional[str] = Form(None, description="Color of uploaded item"),
    gender: Optional[str] = Form("unisex", description="Gender for outfit recommendations: male, female, unisex"),
    user: User = Depends(get_current_user)
):
    """
    Generate catalog-based outfit recommendations.
    
    v3.0.0: New catalog-based outfit system.
    
    Flow:
    1. User uploads a photo of themselves OR a single clothing item
    2. If item_category is provided, treat as single item upload
    3. Generate 5 matching outfit combinations from catalog
    4. Render outfit cards on model images
    5. Return outfit cards with garment details
    """
    import uuid
    
    # Rate limiting
    await check_rate_limit(request, user)
    
    try:
        # Validate image
        try:
            content, validated_image = await validate_image_upload(
                image,
                image.content_type
            )
        except ValidationError as ve:
            raise HTTPException(status_code=ve.status_code, detail=ve.message)
        
        job_id = str(uuid.uuid4())
        logger.info(f"Catalog outfit job: {job_id}")
        
        # Determine if this is a single item upload or full photo
        if item_category:
            # User uploaded a single clothing item
            if item_category not in ["top", "bottom", "outerwear", "shoes"]:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid item_category. Use: top, bottom, outerwear, shoes"
                )
            
            # Use provided color or detect
            color = item_color or "black"
            
            # Generate outfits that match this item
            outfits = generate_single_item_combos(
                item_category=item_category,
                item_color=color,
                item_name=f"Your {item_category}",
                num_outfits=5,
                gender=gender or "unisex"
            )
            
        else:
            # Full photo - detect what user is wearing, suggest rest
            # For now, default to bottom (user has pants, need top/outerwear/shoes)
            fixed_item = {"name": "Your Current Outfit", "color": "neutral"}
            outfits = generate_outfit_combos(
                fixed_item=fixed_item,
                fixed_category="bottom",  # Assume user's bottom is kept
                num_outfits=5,
                gender=gender or "unisex"
            )
        
        if not outfits:
            raise HTTPException(status_code=500, detail="Failed to generate outfits")
        
        # Create renders directory
        renders_dir = storage.get_job_path(job_id) / "renders"
        renders_dir.mkdir(parents=True, exist_ok=True)
        
        # Render outfit cards
        model_gender = "male" if gender == "male" else ("female" if gender == "female" else "male")
        renders = render_all_outfit_cards(
            outfits=outfits,
            output_dir=str(renders_dir),
            model_gender=model_gender
        )
        
        # Add render URLs to outfits
        for outfit in outfits:
            rank = outfit.get("rank", 1)
            if rank in renders:
                outfit["render_url"] = f"/ai/assets/jobs/{job_id}/renders/{renders[rank]}"
        
        return JSONResponse(content={
            "job_id": job_id,
            "outfits": outfits,
            "count": len(outfits),
            "version": "catalog-v3.0.0"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Catalog outfit failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ai/assets/{file_path:path}")
async def serve_asset(
    file_path: str,
    user: Optional[User] = Depends(get_optional_user)
):
    """
    Serve static files from job directories.
    
    With path traversal protection.
    """
    try:
        # Get base directory for assets
        settings = get_settings()
        base_dir = Path(settings.data_dir) if hasattr(settings, 'data_dir') else storage.base_data_dir
        
        # Sanitize path to prevent traversal
        try:
            full_path = storage.get_file_path(file_path)
            
            # Additional security check
            sanitize_asset_path(str(full_path), base_dir)
            
        except ValidationError as ve:
            raise HTTPException(status_code=ve.status_code, detail=ve.message)
        
        if not full_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        if not full_path.is_file():
            raise HTTPException(status_code=400, detail="Not a file")
        
        return FileResponse(full_path)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Asset serve error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
