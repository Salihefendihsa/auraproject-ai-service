"""
Catalog-Based Outfit Recommender (v1.0.0)

This module generates outfit recommendations based on:
1. User's detected/uploaded item (kept FIXED)
2. Matching items from our catalog (trend combos)
3. Color and style compatibility rules
"""
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Load catalog
CATALOG_PATH = Path(__file__).parent / "catalog.json"


def load_catalog() -> Dict:
    """Load the garment catalog."""
    try:
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load catalog: {e}")
        return {"items": [], "color_compatibility": {}}


def get_compatible_colors(color: str, catalog: Dict) -> List[str]:
    """Get colors that match the given color."""
    compat = catalog.get("color_compatibility", {})
    return compat.get(color, ["white", "black", "gray", "beige"])


def filter_items_by_category(items: List[Dict], category: str) -> List[Dict]:
    """Filter catalog items by category."""
    return [item for item in items if item.get("category") == category]


def filter_items_by_color_match(items: List[Dict], compatible_colors: List[str]) -> List[Dict]:
    """Filter items by color compatibility."""
    return [item for item in items if item.get("color") in compatible_colors or item.get("color") == "black" or item.get("color") == "white"]


def filter_items_by_gender(items: List[Dict], gender: str) -> List[Dict]:
    """Filter items by gender (unisex always included)."""
    return [item for item in items if item.get("gender") in ["unisex", gender]]


def generate_outfit_combos(
    fixed_item: Dict[str, Any],
    fixed_category: str,
    num_outfits: int = 5,
    gender: str = "unisex"
) -> List[Dict[str, Any]]:
    """
    Generate outfit combinations based on a fixed item.
    
    Args:
        fixed_item: The item user wants to keep (e.g., their pants)
        fixed_category: Category of fixed item (top, bottom, outerwear, shoes)
        num_outfits: Number of combinations to generate
        gender: Filter by gender
        
    Returns:
        List of outfit dictionaries with items for each category
    """
    catalog = load_catalog()
    all_items = catalog.get("items", [])
    
    if not all_items:
        logger.error("Catalog is empty")
        return []
    
    # Get color compatibility
    fixed_color = fixed_item.get("color", "black")
    compatible_colors = get_compatible_colors(fixed_color, catalog)
    
    logger.info(f"Fixed item: {fixed_item.get('name')} ({fixed_color})")
    logger.info(f"Compatible colors: {compatible_colors}")
    
    # Get items for each missing category
    categories_needed = [c for c in ["top", "bottom", "outerwear", "shoes"] if c != fixed_category]
    
    category_options = {}
    for cat in categories_needed:
        cat_items = filter_items_by_category(all_items, cat)
        cat_items = filter_items_by_gender(cat_items, gender)
        cat_items = filter_items_by_color_match(cat_items, compatible_colors)
        
        if not cat_items:
            # Fallback: just filter by category and gender
            cat_items = filter_items_by_category(all_items, cat)
            cat_items = filter_items_by_gender(cat_items, gender)
        
        category_options[cat] = cat_items
        logger.debug(f"Category {cat}: {len(cat_items)} options")
    
    # Generate unique outfits
    outfits = []
    used_combos = set()
    
    style_tags = ["Casual Chic", "Smart Casual", "Streetwear", "Athleisure", "Classic Elegant"]
    
    attempts = 0
    max_attempts = num_outfits * 10
    
    while len(outfits) < num_outfits and attempts < max_attempts:
        attempts += 1
        
        # Random selection for each category
        outfit_items = {fixed_category: fixed_item}
        combo_key = []
        
        for cat in categories_needed:
            options = category_options.get(cat, [])
            if options:
                selected = random.choice(options)
                outfit_items[cat] = {
                    "id": selected.get("id"),
                    "name": selected.get("name"),
                    "color": selected.get("color"),
                    "image": selected.get("image", ""),
                    "source": "suggested"
                }
                combo_key.append(selected.get("id"))
        
        # Check if combo is unique
        combo_str = "-".join(sorted(combo_key))
        if combo_str in used_combos:
            continue
        used_combos.add(combo_str)
        
        # Mark fixed item
        outfit_items[fixed_category]["source"] = "user"
        
        # Create outfit
        rank = len(outfits) + 1
        outfit = {
            "rank": rank,
            "style_tag": style_tags[rank - 1] if rank <= len(style_tags) else f"Style {rank}",
            "items": outfit_items,
            "explanation": f"A {style_tags[rank-1].lower() if rank <= len(style_tags) else 'stylish'} look that complements your {fixed_color} {fixed_item.get('name', fixed_category)}."
        }
        
        outfits.append(outfit)
    
    logger.info(f"Generated {len(outfits)} unique outfit combinations")
    return outfits


def generate_single_item_combos(
    item_category: str,
    item_color: str,
    item_name: str,
    num_outfits: int = 5,
    gender: str = "unisex"
) -> List[Dict[str, Any]]:
    """
    Generate outfits when user uploads a SINGLE clothing item (not wearing it).
    
    Args:
        item_category: What type of item (top, bottom, etc)
        item_color: Detected or specified color
        item_name: Name/description of the item
        num_outfits: How many combos to generate
        gender: male/female/unisex
        
    Returns:
        List of outfit dictionaries
    """
    fixed_item = {
        "name": item_name,
        "color": item_color,
        "source": "user"
    }
    
    return generate_outfit_combos(
        fixed_item=fixed_item,
        fixed_category=item_category,
        num_outfits=num_outfits,
        gender=gender
    )


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test: User has beige pants, wants outfit recommendations
    fixed = {"name": "Beige Chinos", "color": "beige"}
    outfits = generate_outfit_combos(fixed, "bottom", num_outfits=5)
    
    print("\n=== GENERATED OUTFITS ===")
    for outfit in outfits:
        print(f"\n{outfit['rank']}. {outfit['style_tag']}")
        for cat, item in outfit["items"].items():
            source = item.get("source", "?")
            print(f"   {cat}: {item.get('color', '')} {item.get('name', '')} [{source}]")
