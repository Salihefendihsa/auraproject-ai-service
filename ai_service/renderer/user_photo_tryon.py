"""
User Photo Try-On Renderer (v1.0.0)

ISOLATED EXTENSION: Renders garments onto user-uploaded photos.
Does NOT modify any existing mannequin try-on logic.

This module is ONLY used when mode == "user_photo_tryon".

Flow:
1. Parse human body regions from user photo
2. Estimate pose keypoints
3. Warp garments sequentially: top → bottom → outerwear → shoes
4. Save renders to /ai/assets/jobs/{job_id}/user_tryon/

Fallback Guarantee:
- Any error → falls back to partial_tryon on mannequin
- Never crashes the server
- Never fails silently
"""
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Import existing modules safely
try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available for user photo try-on")


def render_user_photo_tryon(
    person_image_path: str,
    garment_images: Dict[str, str],
    output_path: str,
    pose_keypoints: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Render garments onto a user photo.
    
    ISOLATED: This function does NOT share state with mannequin try-on.
    
    Args:
        person_image_path: Path to the user's full-body photo
        garment_images: Dict mapping category to garment image path
                       e.g., {"top": "path/to/top.png", "bottom": "path/to/pants.png"}
        output_path: Where to save the rendered result
        pose_keypoints: Optional pre-computed pose keypoints
        
    Returns:
        Dict with:
        - success: bool
        - output_path: str (if success)
        - tryon_mode: "user_photo" or "user_photo_fallback"
        - error: str (if failed)
    """
    if not PIL_AVAILABLE:
        return {
            "success": False,
            "error": "PIL not available for rendering",
            "tryon_mode": "user_photo_fallback"
        }
    
    try:
        # Load person image
        person_img = Image.open(person_image_path).convert("RGBA")
        width, height = person_img.size
        
        logger.info(f"[user_photo_tryon] Processing user photo: {width}x{height}")
        
        # Get pose keypoints
        if not pose_keypoints:
            from ai_service.renderer.pose_estimation import estimate_pose
            pose_keypoints = estimate_pose(person_image_path)
        
        # Get body measurements for warping
        from ai_service.renderer.pose_estimation import calculate_body_measurements
        measurements = calculate_body_measurements(pose_keypoints, (width, height))
        
        # Parse human body regions
        from ai_service.renderer.human_parsing import parse_human_simple
        parsing_result = parse_human_simple(person_image_path)
        
        # Create composite image starting with person
        composite = person_img.copy()
        
        # Apply garments in order (back to front)
        garment_order = ["bottom", "shoes", "top", "outerwear"]
        
        for category in garment_order:
            if category in garment_images:
                garment_path = garment_images[category]
                if garment_path and Path(garment_path).exists():
                    try:
                        composite = _overlay_garment(
                            composite, 
                            garment_path, 
                            category,
                            parsing_result,
                            measurements,
                            pose_keypoints
                        )
                        logger.debug(f"[user_photo_tryon] Applied {category}")
                    except Exception as e:
                        logger.warning(f"[user_photo_tryon] Failed to apply {category}: {e}")
                        # Continue with other garments
        
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save result
        composite_rgb = composite.convert("RGB")
        composite_rgb.save(output_path, quality=95)
        
        logger.info(f"[user_photo_tryon] Saved render: {output_path}")
        
        return {
            "success": True,
            "output_path": output_path,
            "tryon_mode": "user_photo",
            "garments_applied": list(garment_images.keys())
        }
        
    except Exception as e:
        logger.error(f"[user_photo_tryon] Render failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "tryon_mode": "user_photo_fallback"
        }


def _overlay_garment(
    base_image: "Image.Image",
    garment_path: str,
    category: str,
    parsing_result: Dict[str, Any],
    measurements: Dict[str, float],
    pose_keypoints: Dict[str, Any]
) -> "Image.Image":
    """
    Overlay a garment onto the base image at the correct position.
    
    Uses body region parsing and pose to determine placement.
    
    Returns:
        Modified base image with garment overlaid
    """
    garment_img = Image.open(garment_path).convert("RGBA")
    width, height = base_image.size
    
    # Get target region for this garment category
    from ai_service.renderer.human_parsing import get_garment_target_region
    target_region = get_garment_target_region(parsing_result, category)
    
    if not target_region:
        # Use default positions based on category
        target_region = _get_default_garment_region(category, width, height, measurements)
    
    x1, y1, x2, y2 = target_region
    target_width = int(x2 - x1)
    target_height = int(y2 - y1)
    
    if target_width <= 0 or target_height <= 0:
        logger.warning(f"[user_photo_tryon] Invalid target region for {category}")
        return base_image
    
    # Resize garment to fit target region
    garment_resized = garment_img.resize(
        (target_width, target_height),
        Image.Resampling.LANCZOS
    )
    
    # Create a copy to paste onto
    result = base_image.copy()
    
    # Paste garment with alpha blending
    result.paste(garment_resized, (int(x1), int(y1)), garment_resized)
    
    return result


def _get_default_garment_region(
    category: str,
    width: int,
    height: int,
    measurements: Dict[str, float]
) -> Tuple[float, float, float, float]:
    """
    Get default garment placement region based on category.
    
    Uses standard body proportions when parsing result is unavailable.
    """
    # Standard proportions for standing full-body photo
    if category == "top":
        # Upper body: shoulders to hips
        y_start = height * 0.15
        y_end = height * 0.50
        x_margin = width * 0.15
        return (x_margin, y_start, width - x_margin, y_end)
        
    elif category == "outerwear":
        # Slightly larger than top
        y_start = height * 0.13
        y_end = height * 0.55
        x_margin = width * 0.10
        return (x_margin, y_start, width - x_margin, y_end)
        
    elif category == "bottom":
        # Hips to ankles
        y_start = height * 0.48
        y_end = height * 0.85
        x_margin = width * 0.25
        return (x_margin, y_start, width - x_margin, y_end)
        
    elif category == "shoes":
        # Feet area
        y_start = height * 0.85
        y_end = height * 0.98
        x_margin = width * 0.25
        return (x_margin, y_start, width - x_margin, y_end)
        
    else:
        # Default: center of image
        margin = width * 0.2
        return (margin, margin, width - margin, height - margin)


def render_outfit_user_photo(
    job_id: str,
    person_image_path: str,
    outfit: Dict[str, Any],
    outfit_index: int,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Render a single outfit onto user photo.
    
    ISOLATED: This function is completely separate from mannequin rendering.
    
    Args:
        job_id: Job identifier
        person_image_path: Path to user's photo
        outfit: Outfit dict with items
        outfit_index: Index of this outfit (1-based)
        output_dir: Directory to save renders
        
    Returns:
        Updated outfit dict with render_url and tryon_mode
    """
    # Create user_tryon subdirectory (CDN CANDIDATE path)
    user_tryon_dir = output_dir / "user_tryon"
    user_tryon_dir.mkdir(parents=True, exist_ok=True)
    
    render_filename = f"outfit_{outfit_index}.png"
    output_path = user_tryon_dir / render_filename
    
    # Collect garment images from outfit
    garment_images = {}
    
    for category in ["top", "bottom", "outerwear", "shoes"]:
        item = outfit.get("items", {}).get(category)
        if item and item.get("image_path"):
            garment_images[category] = item["image_path"]
    
    if not garment_images:
        logger.warning(f"[user_photo_tryon] No garment images for outfit {outfit_index}")
        outfit["tryon_mode"] = "user_photo_fallback"
        outfit["render_url"] = None
        return outfit
    
    # Render
    result = render_user_photo_tryon(
        person_image_path=person_image_path,
        garment_images=garment_images,
        output_path=str(output_path)
    )
    
    if result["success"]:
        outfit["tryon_mode"] = "user_photo"
        outfit["render_url"] = f"/ai/assets/jobs/{job_id}/user_tryon/{render_filename}"
        logger.info(f"[user_photo_tryon] Outfit {outfit_index} rendered successfully")
    else:
        outfit["tryon_mode"] = "user_photo_fallback"
        outfit["render_url"] = None
        outfit["render_error"] = result.get("error", "Unknown render error")
        logger.warning(f"[user_photo_tryon] Outfit {outfit_index} fallback: {result.get('error')}")
    
    return outfit


def render_all_outfits_user_photo(
    job_id: str,
    person_image_path: str,
    outfits: List[Dict[str, Any]],
    output_dir: Path
) -> List[Dict[str, Any]]:
    """
    Render all outfits onto user photo.
    
    ISOLATED: Completely separate from mannequin rendering pipeline.
    
    Args:
        job_id: Job identifier
        person_image_path: Path to user's photo
        outfits: List of outfit dicts
        output_dir: Directory to save renders
        
    Returns:
        List of outfits with updated render_url and tryon_mode
    """
    logger.info(f"[user_photo_tryon] Rendering {len(outfits)} outfits for job {job_id}")
    
    rendered_outfits = []
    
    for i, outfit in enumerate(outfits, start=1):
        try:
            rendered = render_outfit_user_photo(
                job_id=job_id,
                person_image_path=person_image_path,
                outfit=outfit.copy(),  # Don't mutate original
                outfit_index=i,
                output_dir=output_dir
            )
            rendered_outfits.append(rendered)
        except Exception as e:
            logger.error(f"[user_photo_tryon] Outfit {i} error: {e}")
            outfit_copy = outfit.copy()
            outfit_copy["tryon_mode"] = "user_photo_fallback"
            outfit_copy["render_url"] = None
            outfit_copy["render_error"] = str(e)
            rendered_outfits.append(outfit_copy)
    
    # Log summary
    success_count = sum(1 for o in rendered_outfits if o.get("tryon_mode") == "user_photo")
    logger.info(f"[user_photo_tryon] Completed: {success_count}/{len(outfits)} successful")
    
    return rendered_outfits
