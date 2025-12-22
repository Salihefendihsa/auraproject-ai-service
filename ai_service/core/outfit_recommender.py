"""
Outfit Recommender (v3.1.0)
Seed-locked outfit generation with EVENT and TREND AWARE scoring.

The seed item is LOCKED and appears in ALL outfits.
Scoring uses: color compatibility (40%) + style compatibility (30%) + event (20%) + trend (10%)
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)

# Catalog path
CATALOG_PATH = Path(__file__).parent.parent / "catalog.json"
TRENDS_PATH = Path(__file__).parent.parent / "trends.json"

# Slot rules based on seed category
SLOT_RULES = {
    "top": ["bottom", "outerwear", "shoes"],
    "bottom": ["top", "outerwear", "shoes"],
    "outerwear": ["top", "bottom", "shoes"],
    "shoes": ["top", "bottom", "outerwear"],
    "accessory": ["top", "bottom", "outerwear", "shoes"],
}

# ==================== EVENT SCORING CONFIG ====================
# Maps event types to style preferences with bonuses and penalties

EVENT_STYLE_WEIGHTS = {
    "work": {
        # Positive styles for work
        "bonus": ["smart_casual", "professional", "classic", "formal", "minimal", "elegant"],
        # Negative styles for work
        "penalty": ["streetwear", "edgy", "sporty", "bohemian"],
        # Bonus item types for work
        "bonus_items": ["blazer", "oxford", "trousers", "loafers"],
    },
    "date": {
        "bonus": ["smart_casual", "elegant", "classic", "romantic", "sophisticated"],
        "penalty": ["sporty", "utility", "rugged"],
        # Soft/neutral colors work better for dates
        "bonus_colors": ["navy", "burgundy", "beige", "white", "black"],
    },
    "party": {
        "bonus": ["edgy", "streetwear", "elegant", "bold", "statement"],
        "penalty": ["minimal", "rugged", "utility"],
        # Bold colors for party
        "bonus_colors": ["black", "burgundy", "pink"],
    },
    "casual": {
        "bonus": ["casual", "minimal", "athleisure", "comfortable", "relaxed", "cozy"],
        "penalty": ["formal", "professional"],
        # Casual-friendly items
        "bonus_items": ["sneakers", "hoodie", "t-shirt", "jeans", "joggers"],
    },
}


def load_catalog() -> Dict[str, Any]:
    """Load the clothing catalog."""
    try:
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load catalog: {e}")
        return {"items": [], "color_compatibility": {}}


def load_trends() -> Dict[str, Any]:
    """Load trends data (optional)."""
    try:
        with open(TRENDS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load trends: {e}")
        return {}


# ==================== BRAND LOOKBOOK SYSTEM ====================
# Lookbook rules influence outfit ranking without overriding seed preferences.
# This is a permanent system designed for extensibility to future brands.
# ZARA is the PRIMARY backbone (weight=1.0), other brands provide diversity.

LOOKBOOK_DIR = Path(__file__).parent.parent / "lookbook"

# Brand weights for scoring influence
# Higher weight = more dominant in rankings
BRAND_WEIGHTS = {
    "zara": 1.0,    # PRIMARY backbone - must dominate
    "hm": 0.75,     # SECONDARY diversity layer
}

# Brand file mappings
BRAND_FILES = {
    "zara": "zara_fw_2024.json",
    "hm": "hm_fw_2024.json",
}


def load_lookbook_rules(brands: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Load lookbook rules for one or more brands.
    
    Brands are loaded in order, with earlier brands taking priority.
    Zara should always be first to ensure dominance.
    
    Args:
        brands: List of brand names. Defaults to ["zara", "hm"]
    
    Returns:
        List of rule dicts with brand metadata attached.
        Each rule includes: rule data + _brand + _brand_weight
        System continues normally if any lookbook is missing.
    """
    if brands is None:
        brands = ["zara", "hm"]  # Default: Zara first, H&M second
    
    all_rules = []
    
    for brand in brands:
        brand_lower = brand.lower()
        filename = BRAND_FILES.get(brand_lower)
        
        if not filename:
            logger.debug(f"No lookbook file configured for brand: {brand}")
            continue
        
        lookbook_path = LOOKBOOK_DIR / filename
        
        try:
            with open(lookbook_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                # Get brand weight from file or global config
                brand_weight = data.get("brand_weight", BRAND_WEIGHTS.get(brand_lower, 0.5))
                
                # Attach brand metadata to each rule
                for rule in data.get("rules", []):
                    rule["_brand"] = data.get("brand", brand_lower)
                    rule["_brand_weight"] = brand_weight
                    all_rules.append(rule)
                
                logger.info(f"Loaded lookbook: {brand} ({len(data.get('rules', []))} rules, weight={brand_weight})")
                
        except FileNotFoundError:
            logger.debug(f"Lookbook file not found: {lookbook_path}")
            continue
        except Exception as e:
            logger.warning(f"Failed to load lookbook {brand}: {e}")
            continue
    
    logger.info(f"Total lookbook rules loaded: {len(all_rules)}")
    return all_rules


def lookbook_rule_match_score(
    outfit_items: Dict[str, Any],
    rules: List[Dict[str, Any]],
    event: Optional[str] = None
) -> Tuple[float, Optional[str], float, Optional[str]]:
    """
    Calculate bonus score based on lookbook rule matching.
    
    Compares outfit composition with brand rules.
    NEVER overrides seed lock or forces exact colors.
    Only influences ranking through bonus points.
    
    WHY CONFIDENCE MULTIPLIER EXISTS:
    - Real-world trends have varying dominance (e.g., "Power Dressing" is more 
      prevalent in Zara FW24 than "Edgy Urban")
    - High-confidence rules (0.9) should produce higher bonuses than low-confidence (0.6)
    - This ensures frequent Zara styles dominate rankings while niche styles 
      appear less often but correctly
    
    WHY BRAND WEIGHT EXISTS:
    - Zara is PRIMARY backbone (weight=1.0) and MUST dominate rankings
    - H&M is SECONDARY diversity layer (weight=0.75) for variation
    - Formula: final_bonus = min(raw_score × confidence × brand_weight, 0.25)
    
    Args:
        outfit_items: Dict of category -> item info
        rules: List of rule dicts from lookbook (with _brand and _brand_weight attached)
        event: Optional event type for style matching
    
    Returns:
        Tuple of (bonus_score, matched_rule_id, rule_confidence, matched_brand)
        Bonus range: 0.0 (no match) to 0.25 (strong match with high confidence)
    """
    if not rules:
        return (0.0, None, 1.0, None)
    
    best_raw_score = 0.0
    best_final_score = 0.0
    best_rule_id = None
    best_confidence = 1.0
    best_brand = None
    best_brand_weight = 1.0
    
    # Extract outfit characteristics
    outfit_colors = set()
    outfit_styles = set()
    outfit_keywords = []
    
    for category, item in outfit_items.items():
        if isinstance(item, dict):
            if item.get("color"):
                outfit_colors.add(item["color"].lower())
            if item.get("style"):
                outfit_styles.update(s.lower() for s in item.get("style", []))
            if item.get("name"):
                outfit_keywords.extend(item["name"].lower().split())
    
    for rule in rules:
        raw_score = 0.0
        # Default confidence to 1.0 if missing (full weight)
        rule_confidence = rule.get("confidence", 1.0)
        # Get brand weight from rule metadata (attached by load_lookbook_rules)
        brand_weight = rule.get("_brand_weight", 1.0)
        brand_name = rule.get("_brand", "unknown")
        
        # 1. Style tag overlap (max 0.10)
        rule_styles = set(s.lower() for s in rule.get("style_tags", []))
        if rule_styles:
            style_overlap = len(outfit_styles & rule_styles) / len(rule_styles)
            raw_score += 0.10 * style_overlap
        
        # 2. Preferred color match (max 0.08)
        preferred_colors = set(c.lower() for c in rule.get("preferred_colors", []))
        if preferred_colors and outfit_colors:
            color_overlap = len(outfit_colors & preferred_colors) / len(outfit_colors)
            raw_score += 0.08 * color_overlap
        
        # 3. Composition keyword match (max 0.07)
        rule_keywords = rule.get("keywords", {})
        keyword_matches = 0
        keyword_total = 0
        
        for category, keywords in rule_keywords.items():
            if category in outfit_items:
                keyword_total += 1
                item_name = outfit_items[category].get("name", "").lower() if isinstance(outfit_items[category], dict) else ""
                for kw in keywords:
                    if kw.lower() in item_name:
                        keyword_matches += 1
                        break
        
        if keyword_total > 0:
            raw_score += 0.07 * (keyword_matches / keyword_total)
        
        # CRITICAL: Apply confidence AND brand weight AFTER calculating raw match score
        # This ensures Zara (weight=1.0) dominates over H&M (weight=0.75)
        # Formula: final_bonus = min(raw_score * confidence * brand_weight, 0.25)
        final_score = raw_score * rule_confidence * brand_weight
        
        if final_score > best_final_score:
            best_raw_score = raw_score
            best_final_score = final_score
            best_rule_id = rule.get("rule_id")
            best_confidence = rule_confidence
            best_brand = brand_name
            best_brand_weight = brand_weight
    
    # Cap at 0.25 max bonus (as per scoring guarantees)
    best_final_score = min(best_final_score, 0.25)
    
    if best_final_score > 0.05:
        logger.debug(
            f"Lookbook match: {best_brand}/{best_rule_id} | "
            f"raw={best_raw_score:.3f} × conf={best_confidence:.2f} × brand={best_brand_weight:.2f} = bonus={best_final_score:.3f}"
        )
    
    return (best_final_score, best_rule_id, best_confidence, best_brand)


# ==================== SEED DETECTION ====================

# Category keywords for simple detection
CATEGORY_KEYWORDS = {
    "top": ["shirt", "blouse", "t-shirt", "tee", "sweater", "hoodie", "polo", "henley", "tank", "crop"],
    "bottom": ["pants", "jeans", "trousers", "shorts", "skirt", "jogger", "chino", "legging"],
    "outerwear": ["jacket", "coat", "blazer", "cardigan", "vest", "bomber", "trench", "parka"],
    "shoes": ["shoe", "sneaker", "boot", "loafer", "sandal", "heel", "flat", "oxford"],
    "accessory": ["bag", "belt", "hat", "scarf", "watch", "jewelry", "tie", "glasses"],
}

# Catalog colors for mapping
CATALOG_COLORS = [
    "black", "white", "gray", "navy", "blue", "red", "green", "beige", 
    "brown", "burgundy", "olive", "pink", "camel", "charcoal", "tan", "light_blue"
]

# Color RGB mappings for matching
COLOR_RGB_MAP = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "gray": (128, 128, 128),
    "navy": (0, 0, 128),
    "blue": (0, 0, 255),
    "red": (255, 0, 0),
    "green": (0, 128, 0),
    "beige": (245, 245, 220),
    "brown": (139, 69, 19),
    "burgundy": (128, 0, 32),
    "olive": (128, 128, 0),
    "pink": (255, 192, 203),
    "camel": (193, 154, 107),
    "charcoal": (54, 69, 79),
    "tan": (210, 180, 140),
    "light_blue": (173, 216, 230),
}


def detect_seed_category(
    seed_image_path: Optional[str],
    explicit_category: Optional[str] = None
) -> Tuple[str, float]:
    """
    Detect the category of the seed garment.
    
    Priority:
    1. If explicit_category provided, use it (confidence=1.0)
    2. Try CLIP-based zero-shot classification
    3. Fallback to aspect ratio heuristics
    
    Args:
        seed_image_path: Path to seed image
        explicit_category: User-provided category override
    
    Returns:
        Tuple of (category, confidence)
    """
    VALID_CATEGORIES = {"top", "bottom", "outerwear", "shoes", "accessory"}
    
    # Priority 1: Explicit category
    if explicit_category and explicit_category.lower() in VALID_CATEGORIES:
        logger.info(f"Using explicit seed_category: {explicit_category}")
        return (explicit_category.lower(), 1.0)
    
    if not seed_image_path:
        logger.warning("No seed image path provided, defaulting to 'top'")
        return ("top", 0.3)
    
    try:
        from PIL import Image
        img = Image.open(seed_image_path)
        width, height = img.size
        aspect_ratio = width / height
        
        # Try CLIP-based detection first
        category, confidence = _detect_with_clip(seed_image_path)
        if confidence >= 0.5:
            logger.info(f"CLIP detected category: {category} (confidence={confidence:.2f})")
            return (category, confidence)
        
        # Fallback: Aspect ratio heuristics
        if aspect_ratio > 1.5:
            # Very wide - likely shoes or accessory
            category = "shoes"
            confidence = 0.4
        elif aspect_ratio < 0.6:
            # Very tall - likely bottom (pants)
            category = "bottom"
            confidence = 0.4
        elif 0.7 <= aspect_ratio <= 1.3:
            # Square-ish - likely top or outerwear
            category = "top"
            confidence = 0.4
        else:
            category = "top"
            confidence = 0.3
        
        logger.info(f"Aspect ratio heuristic: {category} (confidence={confidence:.2f})")
        return (category, confidence)
        
    except Exception as e:
        logger.error(f"Seed category detection failed: {e}")
        return ("top", 0.2)


def _detect_with_clip(image_path: str) -> Tuple[str, float]:
    """
    Use CLIP zero-shot classification for category detection.
    Falls back gracefully if CLIP not available.
    """
    try:
        # Try to use transformers CLIP
        from transformers import CLIPProcessor, CLIPModel
        import torch
        from PIL import Image
        
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        image = Image.open(image_path).convert("RGB")
        
        # Category prompts
        categories = ["top", "bottom", "outerwear", "shoes", "accessory"]
        prompts = [
            "a photo of a shirt, blouse, or upper body garment",
            "a photo of pants, jeans, or lower body garment",
            "a photo of a jacket, coat, or outerwear",
            "a photo of shoes, sneakers, or footwear",
            "a photo of a fashion accessory like a bag or belt",
        ]
        
        inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits_per_image[0]
            probs = torch.softmax(logits, dim=0)
        
        best_idx = probs.argmax().item()
        confidence = probs[best_idx].item()
        
        return (categories[best_idx], confidence)
        
    except ImportError:
        logger.debug("CLIP not available, using fallback detection")
        return ("top", 0.0)
    except Exception as e:
        logger.debug(f"CLIP detection failed: {e}")
        return ("top", 0.0)


def extract_seed_color(seed_image_path: Optional[str]) -> Tuple[str, List[Tuple[int, int, int]]]:
    """
    Extract dominant color from seed garment image.
    
    Uses k-means clustering on non-background pixels.
    
    Args:
        seed_image_path: Path to seed image
    
    Returns:
        Tuple of (color_name, palette as list of RGB tuples)
    """
    if not seed_image_path:
        return ("neutral", [])
    
    try:
        from PIL import Image
        import numpy as np
        
        img = Image.open(seed_image_path).convert("RGB")
        img_array = np.array(img)
        
        # Simple background removal: ignore near-white pixels
        # Reshape to (N, 3)
        pixels = img_array.reshape(-1, 3)
        
        # Filter out background (near-white or near-black)
        brightness = pixels.mean(axis=1)
        mask = (brightness > 20) & (brightness < 240)
        foreground_pixels = pixels[mask]
        
        if len(foreground_pixels) < 100:
            # Not enough foreground pixels
            logger.warning("Not enough foreground pixels for color extraction")
            return ("neutral", [])
        
        # K-means clustering to find dominant colors
        palette = _kmeans_colors(foreground_pixels, k=3)
        
        # Get dominant color (largest cluster)
        dominant_rgb = palette[0]
        
        # Map to catalog color
        color_name = _map_to_catalog_color(dominant_rgb)
        
        logger.info(f"Extracted seed color: {color_name} from RGB{dominant_rgb}")
        return (color_name, palette)
        
    except Exception as e:
        logger.error(f"Color extraction failed: {e}")
        return ("neutral", [])


def _kmeans_colors(pixels: 'np.ndarray', k: int = 3) -> List[Tuple[int, int, int]]:
    """Simple k-means clustering for color extraction."""
    import numpy as np
    
    # Random initialization
    np.random.seed(42)  # Deterministic
    indices = np.random.choice(len(pixels), k, replace=False)
    centroids = pixels[indices].astype(float)
    
    for _ in range(10):  # 10 iterations
        # Assign pixels to nearest centroid
        distances = np.sqrt(((pixels[:, np.newaxis] - centroids) ** 2).sum(axis=2))
        labels = distances.argmin(axis=1)
        
        # Update centroids
        new_centroids = np.array([
            pixels[labels == i].mean(axis=0) if (labels == i).sum() > 0 else centroids[i]
            for i in range(k)
        ])
        
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    
    # Sort by cluster size (most frequent first)
    sizes = [(labels == i).sum() for i in range(k)]
    order = np.argsort(sizes)[::-1]
    
    palette = [tuple(int(c) for c in centroids[i]) for i in order]
    return palette


def _map_to_catalog_color(rgb: Tuple[int, int, int]) -> str:
    """Map RGB value to nearest catalog color name."""
    import numpy as np
    
    best_color = "neutral"
    best_distance = float('inf')
    
    for color_name, color_rgb in COLOR_RGB_MAP.items():
        distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(rgb, color_rgb)))
        if distance < best_distance:
            best_distance = distance
            best_color = color_name
    
    return best_color


def plan_slots(seed_category: str) -> List[str]:
    """
    Plan which slots need to be filled based on seed category.
    
    Args:
        seed_category: Category of the locked seed item
    
    Returns:
        List of slot categories to fill
    """
    slots = SLOT_RULES.get(seed_category, ["top", "bottom", "outerwear", "shoes"])
    slots = [s for s in slots if s != seed_category]
    return slots


def filter_catalog_by_gender(items: List[Dict], gender: str) -> List[Dict]:
    """Filter catalog items by gender compatibility."""
    return [
        item for item in items
        if item.get("gender") in ["unisex", gender]
    ]


def filter_catalog_by_category(items: List[Dict], category: str) -> List[Dict]:
    """Filter catalog items by category."""
    return [item for item in items if item.get("category") == category]


def get_compatible_colors(seed_color: str, catalog: Dict) -> List[str]:
    """Get colors that are compatible with the seed color."""
    compatibility = catalog.get("color_compatibility", {})
    return compatibility.get(seed_color, [])


# ==================== SCORING FUNCTIONS ====================

def color_compatibility_score(item_color: str, seed_color: str, compatible_colors: List[str]) -> float:
    """
    Calculate color compatibility score.
    
    Range: 0.0 to 1.0
    - 1.0: Item color is in the compatible colors list
    - 0.5: Item color matches seed color (monochrome - acceptable but not ideal)
    - 0.3: Neutral colors (black, white, gray, beige) - always safe
    - 0.0: No compatibility found
    """
    # Best: color is explicitly compatible with seed
    if item_color in compatible_colors:
        return 1.0
    
    # Monochrome matching (same color family)
    if item_color == seed_color:
        return 0.5
    
    # Neutral colors are always safe fallbacks
    neutral_colors = {"black", "white", "gray", "beige", "navy", "charcoal"}
    if item_color in neutral_colors:
        return 0.3
    
    return 0.0


def style_compatibility_score(item_styles: List[str], seed_styles: List[str]) -> float:
    """
    Calculate style compatibility score.
    
    Range: 0.0 to 1.0
    - Checks for overlapping styles between item and seed
    - More overlap = higher score
    """
    if not item_styles:
        return 0.3  # No style info, assume neutral
    
    if not seed_styles:
        # If seed has no style info, favor versatile/casual items
        versatile_styles = {"casual", "minimal", "classic", "versatile"}
        overlap = len(set(item_styles) & versatile_styles)
        return min(0.5 + (overlap * 0.25), 1.0)
    
    # Calculate Jaccard-like similarity
    item_set = set(item_styles)
    seed_set = set(seed_styles)
    overlap = len(item_set & seed_set)
    
    if overlap > 0:
        return min(0.5 + (overlap * 0.25), 1.0)
    
    return 0.2  # No overlap but not incompatible


def event_score(item: Dict, event: Optional[str]) -> float:
    """
    Calculate event-based score.
    
    Range: -1.0 to +1.0
    - Positive for styles/items that match the event
    - Negative for styles/items that clash with the event
    """
    if not event or event not in EVENT_STYLE_WEIGHTS:
        return 0.0  # No event context, neutral score
    
    config = EVENT_STYLE_WEIGHTS[event]
    item_styles = set(item.get("style", []))
    item_color = item.get("color", "")
    item_name = item.get("name", "").lower()
    
    score = 0.0
    
    # Style bonuses (+0.3 each, max +0.6)
    bonus_styles = set(config.get("bonus", []))
    style_bonus_count = len(item_styles & bonus_styles)
    score += min(style_bonus_count * 0.3, 0.6)
    
    # Style penalties (-0.3 each, max -0.6)
    penalty_styles = set(config.get("penalty", []))
    style_penalty_count = len(item_styles & penalty_styles)
    score -= min(style_penalty_count * 0.3, 0.6)
    
    # Color bonuses for events that specify them (+0.2)
    bonus_colors = config.get("bonus_colors", [])
    if bonus_colors and item_color in bonus_colors:
        score += 0.2
    
    # Item name bonuses (+0.2)
    bonus_items = config.get("bonus_items", [])
    for bonus_item in bonus_items:
        if bonus_item in item_name:
            score += 0.2
            break
    
    # Clamp to range [-1.0, +1.0]
    return max(-1.0, min(1.0, score))


def trend_score(item: Dict, season: Optional[str], trends: Dict) -> float:
    """
    Calculate trend-based score.
    
    Range: 0.0 to 1.0
    - Based on seasonal colors and trending styles from trends.json
    """
    if not trends:
        return 0.0
    
    item_color = item.get("color", "")
    item_styles = set(item.get("style", []))
    score = 0.0
    
    # Seasonal color bonus (up to +0.3)
    if season:
        # Map our seasons to trends.json seasons
        season_map = {"summer": "summer", "winter": "winter"}
        mapped_season = season_map.get(season, season)
        seasonal_colors = trends.get("seasonal_colors", {}).get(mapped_season, [])
        if item_color in seasonal_colors:
            score += 0.3
    
    # Trend style/color matching (up to +0.7 based on popularity)
    best_trend_score = 0.0
    for trend in trends.get("trends", []):
        trend_styles = set(trend.get("styles", []))
        trend_colors = set(trend.get("colors", []))
        popularity = trend.get("popularity", 0.5)
        
        # Check style match
        style_match = len(item_styles & trend_styles) > 0
        # Check color match
        color_match = item_color in trend_colors
        
        if style_match or color_match:
            # Weight by trend popularity (0.0 to 1.0)
            trend_contribution = popularity * 0.7
            if style_match and color_match:
                trend_contribution *= 1.2  # Bonus for matching both
            best_trend_score = max(best_trend_score, trend_contribution)
    
    score += best_trend_score
    
    # Clamp to range [0.0, 1.0]
    return min(1.0, score)


def calculate_final_score(
    item: Dict,
    seed_color: str,
    seed_styles: List[str],
    compatible_colors: List[str],
    event: Optional[str],
    season: Optional[str],
    trends: Dict
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate final weighted score for an item.
    
    Weights:
    - 40% color compatibility
    - 30% style compatibility
    - 20% event score
    - 10% trend score
    
    Returns:
        Tuple of (final_score, score_breakdown)
    """
    item_color = item.get("color", "")
    item_styles = item.get("style", [])
    
    # Calculate component scores
    color_score = color_compatibility_score(item_color, seed_color, compatible_colors)
    style_score = style_compatibility_score(item_styles, seed_styles)
    evt_score = event_score(item, event)
    trd_score = trend_score(item, season, trends)
    
    # Event score is [-1, 1], normalize to [0, 1] for weighted sum
    evt_score_normalized = (evt_score + 1.0) / 2.0
    
    # Weighted final score
    final = (
        0.4 * color_score +
        0.3 * style_score +
        0.2 * evt_score_normalized +
        0.1 * trd_score
    )
    
    breakdown = {
        "color": round(color_score, 3),
        "style": round(style_score, 3),
        "event": round(evt_score, 3),
        "trend": round(trd_score, 3),
        "final": round(final, 3),
    }
    
    return final, breakdown


# ==================== MAIN GENERATION ====================

def generate_outfits(
    seed: Dict[str, Any],
    catalog: Dict[str, Any],
    gender: str,
    event: Optional[str] = None,
    season: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Generate 5 complete outfits with the seed item locked.
    
    Uses weighted scoring:
    - 40% color compatibility
    - 30% style compatibility  
    - 20% event score
    - 10% trend score
    
    Args:
        seed: Seed item dict with source, category, locked, color
        catalog: Full catalog with items and color_compatibility
        gender: male or female
        event: Optional event type (work/date/party/casual)
        season: Optional season (summer/winter)
    
    Returns:
        List of 5 outfit dicts, each containing the seed + filled slots
    """
    seed_category = seed.get("category", "top")
    seed_color = seed.get("color", "neutral")
    seed_styles = seed.get("style", [])
    slots_to_fill = plan_slots(seed_category)
    
    logger.info(f"Generating outfits: seed={seed_category}, event={event}, season={season}")
    
    # Load trends for trend-aware scoring
    trends = load_trends()
    
    # Load brand lookbook rules for composition influence
    # Default: loads Zara first (weight=1.0), then H&M (weight=0.75)
    # This provides bonus scoring but NEVER overrides seed lock
    lookbook_rules = load_lookbook_rules()  # Returns list of rules with brand metadata
    
    # Get compatible colors from catalog
    compatible_colors = get_compatible_colors(seed_color, catalog)
    
    # Score and rank catalog items for each slot
    slot_candidates: Dict[str, List[Tuple[float, Dict, Dict]]] = {}
    
    for slot in slots_to_fill:
        items = filter_catalog_by_category(catalog.get("items", []), slot)
        items = filter_catalog_by_gender(items, gender)
        
        # Score each item using the full scoring system
        scored_items = []
        for item in items:
            final_score, breakdown = calculate_final_score(
                item=item,
                seed_color=seed_color,
                seed_styles=seed_styles,
                compatible_colors=compatible_colors,
                event=event,
                season=season,
                trends=trends
            )
            scored_items.append((final_score, item, breakdown))
        
        # Sort by score descending (deterministic: highest score first)
        scored_items.sort(key=lambda x: (-x[0], x[1].get("id", "")))
        slot_candidates[slot] = scored_items
        
        logger.debug(f"Slot {slot}: {len(scored_items)} candidates, top score={scored_items[0][0] if scored_items else 0}")
    
    # Generate 5 unique outfits with diversity constraint
    outfits = []
    used_items: Dict[str, set] = {slot: set() for slot in slots_to_fill}
    
    for outfit_idx in range(5):
        outfit = {
            "rank": outfit_idx + 1,
            "items": {
                # Seed item is ALWAYS locked in every outfit
                seed_category: {
                    "source": "user",
                    "locked": True,
                    "category": seed_category,
                    "color": seed_color,
                    "name": seed.get("name", "User's Seed Item"),
                    "image_path": seed.get("image_path"),  # CRITICAL: propagate for try-on
                }
            },
            "seed_locked": True,
            "scores": {},  # Store score breakdowns for explainability
        }
        
        outfit_total_score = 0.0
        
        # Fill each slot with highest-scoring unused item
        for slot in slots_to_fill:
            candidates = slot_candidates.get(slot, [])
            selected_item = None
            selected_breakdown = None
            
            # DIVERSITY: Try to pick an unused item first
            for score, candidate, breakdown in candidates:
                item_id = candidate.get("id")
                if item_id not in used_items[slot]:
                    selected_item = candidate
                    selected_breakdown = breakdown
                    used_items[slot].add(item_id)
                    break
            
            # FALLBACK: If all items used, use round-robin to ensure variety
            if selected_item is None and candidates:
                # Pick item at index = outfit_idx % num_candidates
                fallback_idx = outfit_idx % len(candidates)
                _, selected_item, selected_breakdown = candidates[fallback_idx]
            
            if selected_item:
                outfit["items"][slot] = {
                    "source": "catalog",
                    "locked": False,
                    "id": selected_item.get("id"),
                    "category": slot,
                    "name": selected_item.get("name"),
                    "color": selected_item.get("color"),
                    "style": selected_item.get("style", []),
                    "image": selected_item.get("image"),
                }
                if selected_breakdown:
                    outfit["scores"][slot] = selected_breakdown
                    outfit_total_score += selected_breakdown.get("final", 0)
        
        # Average score across filled slots
        num_filled = len([s for s in slots_to_fill if s in outfit["items"]])
        base_score = outfit_total_score / max(num_filled, 1)
        
        # Apply lookbook rule bonus (0.0 to 0.25)
        # Confidence and brand weight multipliers ensure proper dominance:
        # - Zara (weight=1.0) dominates rankings
        # - H&M (weight=0.75) provides diversity
        # This influences ranking but NEVER overrides seed lock
        lookbook_bonus, matched_rule, lookbook_confidence, lookbook_brand = lookbook_rule_match_score(
            outfit_items=outfit["items"],
            rules=lookbook_rules,
            event=event
        )
        
        outfit["outfit_score"] = round(base_score + lookbook_bonus, 3)
        outfit["lookbook_bonus"] = round(lookbook_bonus, 3)
        outfit["lookbook_bonus_applied"] = lookbook_bonus > 0
        outfit["lookbook_confidence"] = round(lookbook_confidence, 2)
        outfit["lookbook_brand"] = lookbook_brand
        if matched_rule:
            outfit["matched_lookbook_rule"] = matched_rule
        
        outfits.append(outfit)
    
    # ==================== BRAND MIX CONSTRAINT ====================
    # Enforce hard constraints to maintain Zara dominance:
    #   - MIN 2 outfits from Zara (primary backbone)
    #   - MAX 2 outfits from H&M (secondary diversity)
    #   - Remaining slots filled by highest score regardless of brand
    # This affects ONLY selection, NOT scoring formulas or seed lock.
    
    # Sort all candidates by score (best first)
    outfits.sort(key=lambda x: -x.get("outfit_score", 0))
    
    # Separate by brand
    zara_outfits = [o for o in outfits if o.get("lookbook_brand") == "Zara"]
    hm_outfits = [o for o in outfits if o.get("lookbook_brand") == "H&M"]
    other_outfits = [o for o in outfits if o.get("lookbook_brand") not in ("Zara", "H&M")]
    
    # Build final selection with brand constraints
    final_outfits = []
    used_indices = set()
    
    # Step 1: Guarantee MIN 2 from Zara (if available)
    zara_count = 0
    for o in zara_outfits:
        if zara_count >= 2:
            break
        final_outfits.append(o)
        used_indices.add(id(o))
        zara_count += 1
    
    # Step 2: Add up to MAX 2 from H&M (if available and room exists)
    hm_count = 0
    for o in hm_outfits:
        if len(final_outfits) >= 5 or hm_count >= 2:
            break
        final_outfits.append(o)
        used_indices.add(id(o))
        hm_count += 1
    
    # Step 3: Fill remaining slots with highest-scoring unused outfits
    remaining = [o for o in outfits if id(o) not in used_indices]
    for o in remaining:
        if len(final_outfits) >= 5:
            break
        final_outfits.append(o)
    
    # Step 4: Fallback if we still don't have 5 (edge case: very small catalog)
    if len(final_outfits) < 5:
        logger.warning(f"Brand mix: only {len(final_outfits)} outfits available")
    
    # Re-sort final selection by score and assign ranks
    final_outfits.sort(key=lambda x: -x.get("outfit_score", 0))
    for idx, outfit in enumerate(final_outfits):
        outfit["rank"] = idx + 1
    
    # Log brand distribution
    final_brands = [o.get("lookbook_brand", "none") for o in final_outfits]
    logger.info(f"Generated {len(final_outfits)} outfits, brands={final_brands}, best score={final_outfits[0].get('outfit_score', 0) if final_outfits else 0}")
    return final_outfits


def validate_catalog_slots(
    catalog: Dict[str, Any],
    slots_to_fill: List[str],
    gender: str
) -> List[str]:
    """
    Validate that catalog has items for all required slots.
    
    Returns:
        List of missing slot names (empty if all OK)
    """
    missing_slots = []
    
    for slot in slots_to_fill:
        items = filter_catalog_by_category(catalog.get("items", []), slot)
        items = filter_catalog_by_gender(items, gender)
        
        if not items:
            missing_slots.append(slot)
    
    return missing_slots


def build_seed_object(
    seed_image_path: Optional[str],
    seed_category: Optional[str] = None,
    seed_color: Optional[str] = None,
    seed_name: Optional[str] = None
) -> Tuple[Dict[str, Any], float]:
    """
    Build a seed object for outfit generation.
    
    Uses real detection for category and color if not provided.
    
    Args:
        seed_image_path: Path to seed image
        seed_category: Override category (else detected)
        seed_color: Override color (else extracted from image)
        seed_name: Override name
    
    Returns:
        Tuple of (seed dict, detection confidence)
    """
    # Detect category with confidence
    category, confidence = detect_seed_category(seed_image_path, seed_category)
    
    # Extract or use provided color
    if seed_color:
        color = seed_color
        palette = []
    else:
        color, palette = extract_seed_color(seed_image_path)
    
    seed = {
        "source": "user",
        "category": category,
        "locked": True,
        "color": color,
        "palette": palette,
        "style": [],
        "name": seed_name or "User's Seed Garment",
        "image_path": seed_image_path,
        "detection_confidence": confidence,
    }
    
    logger.info(f"Built seed object: category={category} (conf={confidence:.2f}), color={color}")
    return (seed, confidence)
