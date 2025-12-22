"""
Partial Try-On Renderer (v3.0.0)

Deterministic upper-body try-on using alpha blending.
NO diffusion, NO generative models.

Supports:
- top garments
- outerwear garments

Lower body and shoes remain MOCK (not rendered).
"""
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

try:
    from PIL import Image, ImageOps
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

logger = logging.getLogger(__name__)

# Base paths
STATIC_DIR = Path(__file__).parent.parent / "static"
MANNEQUIN_DIR = STATIC_DIR / "mannequins"

# Upper body positioning (relative to image dimensions)
# These are hardcoded ratios for MVP - can be refined later
UPPER_BODY_REGION = {
    "top": {
        "y_start": 0.15,   # Start 15% from top
        "y_end": 0.55,     # End 55% from top  
        "x_start": 0.15,   # Start 15% from left
        "x_end": 0.85,     # End 85% from left
        "scale": 0.70,     # Scale garment to 70% of region
    },
    "outerwear": {
        "y_start": 0.12,   # Slightly higher for outerwear
        "y_end": 0.60,     # Slightly lower
        "x_start": 0.10,   # Wider for outerwear
        "x_end": 0.90,
        "scale": 0.75,
    },
}


def ensure_mannequin_images() -> None:
    """Create placeholder mannequin images if they don't exist."""
    MANNEQUIN_DIR.mkdir(parents=True, exist_ok=True)
    
    for gender in ["male", "female"]:
        mannequin_path = MANNEQUIN_DIR / f"mannequin_{gender}.jpg"
        if not mannequin_path.exists():
            # Create a simple placeholder mannequin silhouette
            if PIL_AVAILABLE:
                _create_placeholder_mannequin(mannequin_path, gender)
            else:
                logger.warning(f"PIL not available, cannot create mannequin placeholder")


def _create_placeholder_mannequin(path: Path, gender: str) -> None:
    """Create a simple placeholder mannequin image."""
    # Create a neutral gray background with silhouette shape
    width, height = 512, 768
    
    # Background color
    bg_color = (240, 240, 240)  # Light gray
    silhouette_color = (200, 200, 200)  # Darker gray for body area
    
    img = Image.new("RGB", (width, height), bg_color)
    
    # Draw a simple body silhouette (oval for torso, rectangle hint)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    
    # Head (circle at top)
    head_center = (width // 2, int(height * 0.08))
    head_radius = int(width * 0.08)
    draw.ellipse([
        head_center[0] - head_radius, head_center[1] - head_radius,
        head_center[0] + head_radius, head_center[1] + head_radius
    ], fill=silhouette_color)
    
    # Neck
    neck_width = int(width * 0.06)
    draw.rectangle([
        width // 2 - neck_width, int(height * 0.12),
        width // 2 + neck_width, int(height * 0.18)
    ], fill=silhouette_color)
    
    # Torso (trapezoid-ish)
    shoulder_width = int(width * 0.35)
    waist_width = int(width * 0.25)
    torso_top = int(height * 0.18)
    torso_bottom = int(height * 0.55)
    
    draw.polygon([
        (width // 2 - shoulder_width, torso_top),
        (width // 2 + shoulder_width, torso_top),
        (width // 2 + waist_width, torso_bottom),
        (width // 2 - waist_width, torso_bottom),
    ], fill=silhouette_color)
    
    # Legs (two rectangles)
    leg_width = int(width * 0.12)
    leg_gap = int(width * 0.04)
    
    # Left leg
    draw.rectangle([
        width // 2 - waist_width, int(height * 0.55),
        width // 2 - leg_gap, int(height * 0.95)
    ], fill=silhouette_color)
    
    # Right leg
    draw.rectangle([
        width // 2 + leg_gap, int(height * 0.55),
        width // 2 + waist_width, int(height * 0.95)
    ], fill=silhouette_color)
    
    img.save(path, quality=90)
    logger.info(f"Created placeholder mannequin: {path}")


def load_garment_image(image_path: str) -> Optional[Image.Image]:
    """
    Load garment image and ensure it has alpha channel.
    
    If no alpha channel, attempts simple background removal.
    """
    if not PIL_AVAILABLE:
        logger.error("PIL not available")
        return None
    
    try:
        img = Image.open(image_path)
        
        # Convert to RGBA if needed
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        
        return img
    except Exception as e:
        logger.error(f"Failed to load garment image {image_path}: {e}")
        return None


def simple_background_removal(img: Image.Image, bg_color: Tuple[int, int, int] = (255, 255, 255), threshold: int = 30) -> Image.Image:
    """
    Simple background removal by making near-white pixels transparent.
    
    This is a basic approach - works best with white/light backgrounds.
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    
    pixels = img.load()
    width, height = img.size
    
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            
            # Check if pixel is close to background color
            diff = abs(r - bg_color[0]) + abs(g - bg_color[1]) + abs(b - bg_color[2])
            
            if diff < threshold * 3:
                # Make transparent
                pixels[x, y] = (r, g, b, 0)
    
    return img


def overlay_garment(
    base_image: Image.Image,
    garment_image: Image.Image,
    category: str
) -> Image.Image:
    """
    Overlay a garment onto the base image at the appropriate position.
    
    Args:
        base_image: Background image (person or mannequin)
        garment_image: Garment with alpha channel
        category: "top" or "outerwear"
    
    Returns:
        Composited image
    """
    base_width, base_height = base_image.size
    
    # Get positioning config
    config = UPPER_BODY_REGION.get(category, UPPER_BODY_REGION["top"])
    
    # Calculate target region
    region_x_start = int(base_width * config["x_start"])
    region_x_end = int(base_width * config["x_end"])
    region_y_start = int(base_height * config["y_start"])
    region_y_end = int(base_height * config["y_end"])
    
    region_width = region_x_end - region_x_start
    region_height = region_y_end - region_y_start
    
    # Scale garment to fit region
    garment_aspect = garment_image.width / garment_image.height
    region_aspect = region_width / region_height
    
    if garment_aspect > region_aspect:
        # Garment is wider, fit to width
        new_width = int(region_width * config["scale"])
        new_height = int(new_width / garment_aspect)
    else:
        # Garment is taller, fit to height
        new_height = int(region_height * config["scale"])
        new_width = int(new_height * garment_aspect)
    
    # Resize garment
    garment_resized = garment_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Calculate centered position within region
    x_offset = region_x_start + (region_width - new_width) // 2
    y_offset = region_y_start + (region_height - new_height) // 2
    
    # Create result by compositing
    result = base_image.copy()
    if result.mode != "RGBA":
        result = result.convert("RGBA")
    
    # Paste garment using its alpha channel as mask
    result.paste(garment_resized, (x_offset, y_offset), garment_resized)
    
    return result


def get_base_image(
    person_image_path: Optional[str],
    gender: str
) -> Optional[Image.Image]:
    """
    Get the base image for try-on rendering.
    
    Uses person image if available, otherwise falls back to mannequin.
    """
    if not PIL_AVAILABLE:
        logger.error("PIL not available")
        return None
    
    # Try person image first
    if person_image_path and Path(person_image_path).exists():
        try:
            return Image.open(person_image_path).convert("RGBA")
        except Exception as e:
            logger.warning(f"Failed to load person image: {e}")
    
    # Fall back to mannequin
    ensure_mannequin_images()
    mannequin_path = MANNEQUIN_DIR / f"mannequin_{gender}.jpg"
    
    if mannequin_path.exists():
        try:
            return Image.open(mannequin_path).convert("RGBA")
        except Exception as e:
            logger.error(f"Failed to load mannequin: {e}")
    
    # Last resort: create blank canvas
    logger.warning("No base image available, using blank canvas")
    return Image.new("RGBA", (512, 768), (240, 240, 240, 255))


def get_garment_image_path(item: Dict[str, Any], catalog_images_dir: Path) -> Optional[str]:
    """
    Resolve the full path to a garment image.
    
    Checks multiple possible locations.
    """
    image_filename = item.get("image")
    if not image_filename:
        return None
    
    # Check catalog images directory
    possible_paths = [
        catalog_images_dir / image_filename,
        catalog_images_dir / "garments" / image_filename,
        Path(__file__).parent.parent / "static" / "garments" / image_filename,
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    return None


def render_partial_tryon(
    base_image_path: Optional[str],
    top_image_path: Optional[str],
    outerwear_image_path: Optional[str],
    output_path: str,
    person_image_path: Optional[str] = None,
    gender: str = "male"
) -> bool:
    """
    Render partial try-on for upper body garments.
    
    Args:
        base_image_path: Path to base image (deprecated, use person_image_path)
        top_image_path: Path to top garment image
        outerwear_image_path: Path to outerwear garment image
        output_path: Where to save the result
        person_image_path: Path to person image (if available)
        gender: Gender for mannequin fallback
    
    Returns:
        True if rendering succeeded, False otherwise
    """
    if not PIL_AVAILABLE:
        logger.error("PIL not available for partial try-on rendering")
        return False
    
    try:
        # Get base image (person or mannequin)
        effective_base = person_image_path or base_image_path
        base_image = get_base_image(effective_base, gender)
        
        if base_image is None:
            logger.error("Failed to get base image")
            return False
        
        result = base_image
        layers_applied = 0
        
        # Layer 1: Top garment (if provided)
        if top_image_path and Path(top_image_path).exists():
            top_img = load_garment_image(top_image_path)
            if top_img:
                # Apply simple background removal if needed
                top_img = simple_background_removal(top_img)
                result = overlay_garment(result, top_img, "top")
                layers_applied += 1
                logger.debug(f"Applied top layer: {top_image_path}")
        
        # Layer 2: Outerwear (on top of the top)
        if outerwear_image_path and Path(outerwear_image_path).exists():
            outerwear_img = load_garment_image(outerwear_image_path)
            if outerwear_img:
                outerwear_img = simple_background_removal(outerwear_img)
                result = overlay_garment(result, outerwear_img, "outerwear")
                layers_applied += 1
                logger.debug(f"Applied outerwear layer: {outerwear_image_path}")
        
        # Convert to RGB for JPEG output
        if output_path.lower().endswith(".jpg") or output_path.lower().endswith(".jpeg"):
            result = result.convert("RGB")
        
        # Save result
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result.save(output_path, quality=90)
        logger.info(f"Partial try-on render saved: {output_path} ({layers_applied} layers)")
        
        return True
    
    except Exception as e:
        logger.error(f"Partial try-on rendering failed: {e}")
        return False


def render_outfit_partial(
    outfit: Dict[str, Any],
    job_id: str,
    person_image_path: Optional[str],
    gender: str,
    output_dir: Path,
    catalog_images_dir: Optional[Path] = None
) -> Optional[str]:
    """
    Render a single outfit using partial try-on.
    
    Args:
        outfit: Outfit dict with items
        job_id: Job ID for logging
        person_image_path: Person image path
        gender: Gender for mannequin fallback
        output_dir: Directory to save renders
        catalog_images_dir: Directory containing garment images
    
    Returns:
        Render filename if successful, None otherwise
    """
    rank = outfit.get("rank", 1)
    items = outfit.get("items", {})
    
    # Get top and outerwear image paths
    top_item = items.get("top", {})
    outerwear_item = items.get("outerwear", {})
    
    # Resolve image paths from catalog
    if catalog_images_dir is None:
        catalog_images_dir = Path(__file__).parent.parent / "static" / "garments"
    
    top_image_path = None
    outerwear_image_path = None
    
    if top_item.get("source") == "catalog" and top_item.get("image"):
        top_image_path = get_garment_image_path(top_item, catalog_images_dir)
    
    if outerwear_item.get("source") == "catalog" and outerwear_item.get("image"):
        outerwear_image_path = get_garment_image_path(outerwear_item, catalog_images_dir)
    
    # Output path
    output_filename = f"outfit_{rank}_partial.png"
    output_path = output_dir / output_filename
    
    # Render
    success = render_partial_tryon(
        base_image_path=None,
        top_image_path=top_image_path,
        outerwear_image_path=outerwear_image_path,
        output_path=str(output_path),
        person_image_path=person_image_path,
        gender=gender
    )
    
    if success:
        return output_filename
    
    return None
