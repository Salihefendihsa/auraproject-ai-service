"""
Human Parsing Module (v3.0.0)

Body part segmentation for virtual try-on.
Segments human body into regions for garment placement.
"""
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import hashlib

try:
    from PIL import Image, ImageDraw
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

logger = logging.getLogger(__name__)

# Cache directory for parsed results
CACHE_DIR = Path(__file__).parent.parent / "data" / "parsing_cache"

# Body part labels (simplified CIHP-style)
BODY_PARTS = {
    "background": 0,
    "head": 1,
    "torso": 2,
    "upper_arm_left": 3,
    "upper_arm_right": 4,
    "lower_arm_left": 5,
    "lower_arm_right": 6,
    "upper_leg_left": 7,
    "upper_leg_right": 8,
    "lower_leg_left": 9,
    "lower_leg_right": 10,
    "feet": 11,
}

# Garment-to-body-part mapping
GARMENT_REGIONS = {
    "top": ["torso", "upper_arm_left", "upper_arm_right"],
    "outerwear": ["torso", "upper_arm_left", "upper_arm_right"],
    "bottom": ["upper_leg_left", "upper_leg_right", "lower_leg_left", "lower_leg_right"],
    "shoes": ["feet"],
}


def get_cache_key(image_path: str) -> str:
    """Generate cache key for an image."""
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def get_cached_parsing(image_path: str) -> Optional[Dict[str, Any]]:
    """Load cached parsing result if available."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = get_cache_key(image_path)
    cache_file = CACHE_DIR / f"{cache_key}.npz"
    
    if cache_file.exists():
        try:
            import numpy as np
            data = np.load(cache_file, allow_pickle=True)
            return {
                "segmentation": data["segmentation"],
                "regions": data["regions"].item(),
            }
        except Exception as e:
            logger.warning(f"Failed to load cached parsing: {e}")
    
    return None


def save_cached_parsing(image_path: str, result: Dict[str, Any]) -> None:
    """Save parsing result to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = get_cache_key(image_path)
    cache_file = CACHE_DIR / f"{cache_key}.npz"
    
    try:
        import numpy as np
        np.savez(
            cache_file,
            segmentation=result["segmentation"],
            regions=result["regions"]
        )
    except Exception as e:
        logger.warning(f"Failed to save parsing cache: {e}")


def parse_human_simple(image_path: str) -> Dict[str, Any]:
    """
    Simple rule-based human parsing (fallback).
    
    Uses fixed proportions based on typical human body ratios.
    Good for mannequins and standardized poses.
    
    Returns:
        Dict with segmentation mask and region bounding boxes
    """
    if not PIL_AVAILABLE:
        logger.error("PIL not available for human parsing")
        return {"segmentation": None, "regions": {}}
    
    # Check cache first
    cached = get_cached_parsing(image_path)
    if cached:
        logger.debug(f"Using cached parsing for {image_path}")
        return cached
    
    try:
        img = Image.open(image_path)
        width, height = img.size
        
        # Create segmentation mask using typical body proportions
        segmentation = np.zeros((height, width), dtype=np.uint8)
        regions = {}
        
        # Head region (top 15% centered)
        head_y_start = 0
        head_y_end = int(height * 0.15)
        head_x_start = int(width * 0.35)
        head_x_end = int(width * 0.65)
        segmentation[head_y_start:head_y_end, head_x_start:head_x_end] = BODY_PARTS["head"]
        regions["head"] = (head_x_start, head_y_start, head_x_end, head_y_end)
        
        # Torso region (15% to 55% vertically)
        torso_y_start = int(height * 0.15)
        torso_y_end = int(height * 0.55)
        torso_x_start = int(width * 0.25)
        torso_x_end = int(width * 0.75)
        segmentation[torso_y_start:torso_y_end, torso_x_start:torso_x_end] = BODY_PARTS["torso"]
        regions["torso"] = (torso_x_start, torso_y_start, torso_x_end, torso_y_end)
        
        # Upper arms (sides of torso region, top half)
        arm_y_start = int(height * 0.18)
        arm_y_end = int(height * 0.40)
        
        # Left arm
        left_arm_x_start = int(width * 0.10)
        left_arm_x_end = int(width * 0.25)
        segmentation[arm_y_start:arm_y_end, left_arm_x_start:left_arm_x_end] = BODY_PARTS["upper_arm_left"]
        regions["upper_arm_left"] = (left_arm_x_start, arm_y_start, left_arm_x_end, arm_y_end)
        
        # Right arm
        right_arm_x_start = int(width * 0.75)
        right_arm_x_end = int(width * 0.90)
        segmentation[arm_y_start:arm_y_end, right_arm_x_start:right_arm_x_end] = BODY_PARTS["upper_arm_right"]
        regions["upper_arm_right"] = (right_arm_x_start, arm_y_start, right_arm_x_end, arm_y_end)
        
        # Lower arms
        lower_arm_y_start = int(height * 0.40)
        lower_arm_y_end = int(height * 0.55)
        segmentation[lower_arm_y_start:lower_arm_y_end, left_arm_x_start:left_arm_x_end] = BODY_PARTS["lower_arm_left"]
        segmentation[lower_arm_y_start:lower_arm_y_end, right_arm_x_start:right_arm_x_end] = BODY_PARTS["lower_arm_right"]
        regions["lower_arm_left"] = (left_arm_x_start, lower_arm_y_start, left_arm_x_end, lower_arm_y_end)
        regions["lower_arm_right"] = (right_arm_x_start, lower_arm_y_start, right_arm_x_end, lower_arm_y_end)
        
        # Upper legs (55% to 75%)
        leg_y_start = int(height * 0.55)
        leg_y_end = int(height * 0.75)
        leg_left_x_start = int(width * 0.30)
        leg_left_x_end = int(width * 0.48)
        leg_right_x_start = int(width * 0.52)
        leg_right_x_end = int(width * 0.70)
        
        segmentation[leg_y_start:leg_y_end, leg_left_x_start:leg_left_x_end] = BODY_PARTS["upper_leg_left"]
        segmentation[leg_y_start:leg_y_end, leg_right_x_start:leg_right_x_end] = BODY_PARTS["upper_leg_right"]
        regions["upper_leg_left"] = (leg_left_x_start, leg_y_start, leg_left_x_end, leg_y_end)
        regions["upper_leg_right"] = (leg_right_x_start, leg_y_start, leg_right_x_end, leg_y_end)
        
        # Lower legs (75% to 92%)
        lower_leg_y_start = int(height * 0.75)
        lower_leg_y_end = int(height * 0.92)
        segmentation[lower_leg_y_start:lower_leg_y_end, leg_left_x_start:leg_left_x_end] = BODY_PARTS["lower_leg_left"]
        segmentation[lower_leg_y_start:lower_leg_y_end, leg_right_x_start:leg_right_x_end] = BODY_PARTS["lower_leg_right"]
        regions["lower_leg_left"] = (leg_left_x_start, lower_leg_y_start, leg_left_x_end, lower_leg_y_end)
        regions["lower_leg_right"] = (leg_right_x_start, lower_leg_y_start, leg_right_x_end, lower_leg_y_end)
        
        # Feet (92% to 100%)
        feet_y_start = int(height * 0.92)
        feet_y_end = height
        feet_x_start = int(width * 0.25)
        feet_x_end = int(width * 0.75)
        segmentation[feet_y_start:feet_y_end, feet_x_start:feet_x_end] = BODY_PARTS["feet"]
        regions["feet"] = (feet_x_start, feet_y_start, feet_x_end, feet_y_end)
        
        result = {
            "segmentation": segmentation,
            "regions": regions,
            "image_size": (width, height),
        }
        
        # Cache result
        save_cached_parsing(image_path, result)
        
        logger.info(f"Parsed human body: {len(regions)} regions")
        return result
        
    except Exception as e:
        logger.error(f"Human parsing failed: {e}")
        return {"segmentation": None, "regions": {}}


def get_garment_target_region(
    parsing_result: Dict[str, Any],
    garment_category: str
) -> Optional[Tuple[int, int, int, int]]:
    """
    Get the target bounding box for a garment category.
    
    Returns:
        (x1, y1, x2, y2) bounding box or None
    """
    regions = parsing_result.get("regions", {})
    target_parts = GARMENT_REGIONS.get(garment_category, [])
    
    if not target_parts:
        return None
    
    # Combine all relevant body part regions
    x1, y1, x2, y2 = float('inf'), float('inf'), 0, 0
    
    for part in target_parts:
        if part in regions:
            px1, py1, px2, py2 = regions[part]
            x1 = min(x1, px1)
            y1 = min(y1, py1)
            x2 = max(x2, px2)
            y2 = max(y2, py2)
    
    if x1 == float('inf'):
        return None
    
    return (int(x1), int(y1), int(x2), int(y2))


def create_parsing_mask(
    parsing_result: Dict[str, Any],
    garment_category: str
) -> Optional[Image.Image]:
    """
    Create a binary mask for the garment target region.
    
    Returns:
        PIL Image mask (white = target region)
    """
    if not PIL_AVAILABLE:
        return None
    
    segmentation = parsing_result.get("segmentation")
    if segmentation is None:
        return None
    
    target_parts = GARMENT_REGIONS.get(garment_category, [])
    target_labels = [BODY_PARTS.get(p, -1) for p in target_parts]
    
    # Create binary mask
    mask = np.zeros_like(segmentation, dtype=np.uint8)
    for label in target_labels:
        if label >= 0:
            mask[segmentation == label] = 255
    
    return Image.fromarray(mask, mode="L")
