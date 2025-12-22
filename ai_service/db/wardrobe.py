"""
Wardrobe System Module (v2.3.0)
User wardrobe item management with duplicate detection.
"""
import os
import logging
import secrets
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path

from ai_service.db import mongo

logger = logging.getLogger(__name__)

# Wardrobe storage directory
WARDROBE_DIR = os.getenv("AURA_WARDROBE_DIR", "ai_service/data/wardrobe")


# ==================== PERCEPTUAL HASH ====================

def compute_phash(image_path: str, hash_size: int = 8) -> Optional[str]:
    """
    Compute perceptual hash (pHash) for an image.
    
    Uses DCT-based perceptual hashing for robust similarity detection.
    
    Args:
        image_path: Path to image file
        hash_size: Size of hash (8x8 = 64 bits)
    
    Returns:
        Hex string of perceptual hash, or None if failed
    """
    try:
        from PIL import Image
        import numpy as np
        
        # Load and prepare image
        img = Image.open(image_path).convert("L")  # Grayscale
        
        # Resize to (hash_size+1) * 4 for DCT
        img = img.resize((hash_size * 4, hash_size * 4), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        pixels = np.array(img, dtype=np.float64)
        
        # Simple DCT-like transform (approximation without scipy.fft)
        # Use mean-based approach for CPU efficiency
        img_small = img.resize((hash_size, hash_size), Image.Resampling.LANCZOS)
        pixels = np.array(img_small, dtype=np.float64)
        
        # Compute hash based on median
        median = np.median(pixels)
        diff = pixels > median
        
        # Convert to hex string
        hash_bits = diff.flatten()
        hash_int = sum(bit << i for i, bit in enumerate(hash_bits))
        hash_hex = format(hash_int, f'0{hash_size * hash_size // 4}x')
        
        return hash_hex
        
    except Exception as e:
        logger.error(f"Failed to compute pHash: {e}")
        return None


def hamming_distance(hash1: str, hash2: str) -> int:
    """
    Compute Hamming distance between two hex hashes.
    
    Returns:
        Number of differing bits (lower = more similar)
    """
    try:
        int1 = int(hash1, 16)
        int2 = int(hash2, 16)
        xor = int1 ^ int2
        return bin(xor).count('1')
    except:
        return 64  # Max distance if comparison fails


def is_duplicate(hash1: str, hash2: str, threshold: int = 6) -> bool:
    """
    Check if two images are duplicates based on pHash.
    
    Args:
        hash1, hash2: Hex hash strings
        threshold: Max Hamming distance for duplicates (6 = ~90% similar)
    
    Returns:
        True if images are considered duplicates
    """
    return hamming_distance(hash1, hash2) <= threshold


# ==================== WARDROBE ITEM MODEL ====================

def create_wardrobe_item(
    owner_user_id: str,
    category: str,
    image_url: str,
    mask_url: Optional[str] = None,
    color_palette: Optional[List[str]] = None,
    style_tags: Optional[List[str]] = None,
    season: Optional[str] = None,
    phash: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Create a new wardrobe item.
    
    Args:
        owner_user_id: Owner's user ID
        category: top, bottom, outerwear, shoes
        image_url: URL/path to item image
        mask_url: URL/path to segmentation mask
        color_palette: List of colors (e.g., ["navy", "white"])
        style_tags: Style descriptors (e.g., ["casual", "sporty"])
        season: Season suitability (e.g., "summer", "all")
        phash: Perceptual hash for duplicate detection
    
    Returns:
        Created item document or None
    """
    try:
        collection = mongo.get_collection("wardrobe_items")
        if collection is None:
            return None
        
        item_id = secrets.token_hex(8)
        
        item = {
            "item_id": item_id,
            "owner_user_id": owner_user_id,
            "category": category,
            "image_url": image_url,
            "mask_url": mask_url,
            "color_palette": color_palette or [],
            "style_tags": style_tags or [],
            "season": season or "all",
            "phash": phash,
            "created_at": datetime.utcnow().isoformat(),
            "active": True
        }
        
        collection.insert_one(item)
        logger.info(f"Wardrobe item created: {item_id} ({category})")
        
        item.pop("_id", None)
        return item
        
    except Exception as e:
        logger.error(f"Failed to create wardrobe item: {e}")
        return None


def get_wardrobe_items(
    user_id: str,
    category: Optional[str] = None,
    season: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """
    Get user's wardrobe items with optional filters.
    
    Args:
        user_id: Owner's user ID
        category: Filter by category (top, bottom, etc.)
        season: Filter by season
        limit: Max results
        offset: Skip for pagination
    
    Returns:
        List of wardrobe items
    """
    try:
        collection = mongo.get_collection("wardrobe_items")
        if collection is None:
            return []
        
        query = {"owner_user_id": user_id, "active": True}
        
        if category:
            query["category"] = category
        if season:
            query["season"] = {"$in": [season, "all"]}
        
        cursor = collection.find(
            query,
            {"_id": 0}
        ).sort("created_at", -1).skip(offset).limit(limit)
        
        return list(cursor)
        
    except Exception as e:
        logger.error(f"Failed to get wardrobe items: {e}")
        return []


def get_wardrobe_item(user_id: str, item_id: str) -> Optional[Dict[str, Any]]:
    """Get a specific wardrobe item."""
    try:
        collection = mongo.get_collection("wardrobe_items")
        if collection is None:
            return None
        
        item = collection.find_one(
            {"item_id": item_id, "owner_user_id": user_id, "active": True},
            {"_id": 0}
        )
        
        return item
        
    except Exception as e:
        logger.error(f"Failed to get wardrobe item: {e}")
        return None


def delete_wardrobe_item(user_id: str, item_id: str) -> bool:
    """
    Soft delete a wardrobe item.
    
    Returns:
        True if deleted, False otherwise
    """
    try:
        collection = mongo.get_collection("wardrobe_items")
        if collection is None:
            return False
        
        result = collection.update_one(
            {"item_id": item_id, "owner_user_id": user_id},
            {"$set": {"active": False}}
        )
        
        return result.modified_count > 0
        
    except Exception as e:
        logger.error(f"Failed to delete wardrobe item: {e}")
        return False


def find_duplicate_in_wardrobe(
    user_id: str,
    phash: str,
    threshold: int = 6
) -> Optional[Dict[str, Any]]:
    """
    Find duplicate item in user's wardrobe.
    
    Args:
        user_id: Owner's user ID
        phash: Perceptual hash of new item
        threshold: Hamming distance threshold
    
    Returns:
        Matching duplicate item or None
    """
    try:
        items = get_wardrobe_items(user_id, limit=1000)
        
        for item in items:
            item_hash = item.get("phash")
            if item_hash and is_duplicate(phash, item_hash, threshold):
                logger.info(f"Duplicate found: {item['item_id']} (distance: {hamming_distance(phash, item_hash)})")
                return item
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to check for duplicates: {e}")
        return None


def get_wardrobe_count(user_id: str) -> int:
    """Get total wardrobe item count."""
    try:
        collection = mongo.get_collection("wardrobe_items")
        if collection is None:
            return 0
        
        return collection.count_documents({"owner_user_id": user_id, "active": True})
        
    except Exception as e:
        logger.error(f"Failed to count wardrobe items: {e}")
        return 0


def get_wardrobe_by_category(user_id: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get user's wardrobe organized by category.
    
    Returns:
        Dict with categories as keys and item lists as values
    """
    items = get_wardrobe_items(user_id, limit=500)
    
    by_category = {
        "top": [],
        "bottom": [],
        "outerwear": [],
        "shoes": []
    }
    
    for item in items:
        category = item.get("category", "")
        if category in by_category:
            by_category[category].append(item)
    
    return by_category


# ==================== WARDROBE CONTEXT FOR LLM ====================

def build_wardrobe_context(user_id: str) -> str:
    """
    Build LLM context from user's wardrobe.
    
    Returns:
        Context string describing user's wardrobe
    """
    by_category = get_wardrobe_by_category(user_id)
    
    parts = ["User's wardrobe contains:"]
    
    for category, items in by_category.items():
        if items:
            item_descs = []
            for item in items[:5]:  # Max 5 per category
                colors = ", ".join(item.get("color_palette", ["unknown"]))
                styles = ", ".join(item.get("style_tags", ["casual"]))
                item_descs.append(f"{colors} {category} ({styles})")
            
            parts.append(f"- {category.upper()}: {', '.join(item_descs)}")
    
    if len(parts) == 1:
        return ""  # Empty wardrobe
    
    return "\n".join(parts)


def get_wardrobe_items_for_outfit(user_id: str, category: str) -> Optional[Dict[str, Any]]:
    """
    Get best matching wardrobe item for a category.
    
    For outfit planning, returns the most recently added item.
    
    Returns:
        Wardrobe item or None
    """
    items = get_wardrobe_items(user_id, category=category, limit=1)
    return items[0] if items else None
