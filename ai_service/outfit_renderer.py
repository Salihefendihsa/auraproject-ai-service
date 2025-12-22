"""
Outfit Display Renderer (v2.0.0)

Creates visual cards showing outfit combinations with REAL garment images.
Uses PIL/Pillow for composite creation, no GPU required.
"""
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Paths
MODELS_DIR = Path(__file__).parent / "data" / "models"
GARMENTS_DIR = Path(__file__).parent / "static" / "garments"
DEFAULT_MALE_MODEL = MODELS_DIR / "model_male.jpg"
DEFAULT_FEMALE_MODEL = MODELS_DIR / "model_female.jpg"

# Garment positioning on card (relative to 400x600 card)
GARMENT_POSITIONS = {
    "top": {"x": 270, "y": 50, "width": 120, "height": 150},
    "bottom": {"x": 270, "y": 210, "width": 120, "height": 150},
    "outerwear": {"x": 270, "y": 370, "width": 120, "height": 100},
    "shoes": {"x": 270, "y": 480, "width": 120, "height": 100}
}


def load_garment_image(image_filename: str, size: tuple) -> Optional[Image.Image]:
    """Load and resize a garment image."""
    if not image_filename:
        return None
        
    garment_path = GARMENTS_DIR / image_filename
    
    if not garment_path.exists():
        logger.warning(f"Garment image not found: {garment_path}")
        return None
    
    try:
        img = Image.open(garment_path).convert("RGBA")
        img = img.resize(size, Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        logger.error(f"Failed to load garment: {e}")
        return None


def create_outfit_card(
    outfit: Dict[str, Any],
    model_gender: str = "male",
    output_path: Optional[str] = None,
    card_width: int = 400,
    card_height: int = 600
) -> Image.Image:
    """
    Create a visual card with REAL garment images.
    
    v2.0.0: Now composites actual garment photos onto the card.
    
    Args:
        outfit: Outfit dictionary with items (must include 'image' field)
        model_gender: male or female
        output_path: Where to save (None = don't save)
        card_width: Output width
        card_height: Output height
        
    Returns:
        PIL Image of the outfit card
    """
    # Load base model image
    model_path = DEFAULT_MALE_MODEL if model_gender == "male" else DEFAULT_FEMALE_MODEL
    
    try:
        if model_path.exists():
            base_image = Image.open(model_path).convert("RGBA")
            base_image = base_image.resize((card_width, card_height), Image.Resampling.LANCZOS)
        else:
            base_image = Image.new("RGBA", (card_width, card_height), color=(240, 240, 240, 255))
    except Exception as e:
        logger.error(f"Failed to load model image: {e}")
        base_image = Image.new("RGBA", (card_width, card_height), color=(240, 240, 240, 255))
    
    # Get items
    items = outfit.get("items", {})
    
    # Composite garment images on the right side
    for category in ["top", "bottom", "outerwear", "shoes"]:
        item = items.get(category, {})
        if not item:
            continue
            
        image_filename = item.get("image", "")
        if not image_filename:
            continue
        
        pos = GARMENT_POSITIONS.get(category, {})
        if not pos:
            continue
        
        garment_img = load_garment_image(
            image_filename, 
            (pos["width"], pos["height"])
        )
        
        if garment_img:
            # Create rounded rectangle background
            bg = Image.new("RGBA", (pos["width"] + 10, pos["height"] + 10), (255, 255, 255, 230))
            base_image.paste(bg, (pos["x"] - 5, pos["y"] - 5), bg)
            
            # Paste garment
            base_image.paste(garment_img, (pos["x"], pos["y"]), garment_img)
    
    # Convert to RGB for drawing
    base_image = base_image.convert("RGB")
    draw = ImageDraw.Draw(base_image)
    
    # Load fonts
    try:
        font_title = ImageFont.truetype("arial.ttf", 20)
        font_item = ImageFont.truetype("arial.ttf", 12)
    except:
        font_title = ImageFont.load_default()
        font_item = ImageFont.load_default()
    
    # Draw style tag header
    style_tag = outfit.get("style_tag", "Outfit")
    rank = outfit.get("rank", 1)
    
    # Semi-transparent header
    header = Image.new("RGBA", (card_width, 40), (0, 0, 0, 200))
    base_image.paste(header, (0, 0), header)
    draw = ImageDraw.Draw(base_image)
    draw.text((15, 10), f"#{rank} {style_tag}", fill="white", font=font_title)
    
    # Draw item labels under each garment
    for category in ["top", "bottom", "outerwear", "shoes"]:
        item = items.get(category, {})
        if not item:
            continue
            
        pos = GARMENT_POSITIONS.get(category, {})
        if not pos:
            continue
        
        name = item.get("name", category)[:20]  # Truncate long names
        color = item.get("color", "")
        source = item.get("source", "suggested")
        icon = "👤" if source == "user" else "✨"
        
        label_y = pos["y"] + pos["height"] + 2
        label = f"{icon}{color}"
        draw.text((pos["x"], label_y), label, fill="black", font=font_item)
    
    # Save if path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        base_image.save(output_path, quality=95)
        logger.info(f"Saved outfit card: {output_path}")
    
    return base_image


def render_all_outfit_cards(
    outfits: List[Dict[str, Any]],
    output_dir: str,
    model_gender: str = "male"
) -> Dict[int, str]:
    """
    Render cards for all outfits.
    
    Args:
        outfits: List of outfit dictionaries
        output_dir: Directory to save cards
        model_gender: male or female
        
    Returns:
        Dict mapping outfit rank to output path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for outfit in outfits:
        rank = outfit.get("rank", len(results) + 1)
        card_path = output_path / f"outfit_{rank}.png"
        
        try:
            create_outfit_card(
                outfit=outfit,
                model_gender=model_gender,
                output_path=str(card_path)
            )
            results[rank] = f"outfit_{rank}.png"
        except Exception as e:
            logger.error(f"Failed to render outfit {rank}: {e}")
    
    logger.info(f"Rendered {len(results)} outfit cards")
    return results


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test outfit with image fields
    test_outfit = {
        "rank": 1,
        "style_tag": "Casual Chic",
        "items": {
            "bottom": {"name": "Beige Chinos", "color": "beige", "source": "user", "image": "bottom_002_beige_chinos.jpg"},
            "top": {"name": "Navy Oxford", "color": "navy", "source": "suggested", "image": "top_002_navy_oxford.jpg"},
            "outerwear": {"name": "Navy Trench", "color": "navy", "source": "suggested", "image": "outerwear_003_navy_trench.jpg"},
            "shoes": {"name": "White Sneakers", "color": "white", "source": "suggested", "image": "shoes_001_white_sneakers.jpg"}
        }
    }
    
    img = create_outfit_card(test_outfit, output_path="test_outfit_card_v2.png")
    print(f"Created card: {img.size}")
