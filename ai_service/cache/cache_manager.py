"""
Cache Manager (v1.4.2)
Manages cache key generation and high-level cache operations.
"""
import os
import json
import hashlib
import logging
from typing import Dict, Any, Optional

from ai_service.cache.cache_store import CacheStore

logger = logging.getLogger(__name__)


# Cache settings from environment
CACHE_ENABLED = os.getenv("AURA_CACHE_ENABLED", "true").lower() == "true"
CACHE_TTL_MINUTES = int(os.getenv("AURA_CACHE_TTL_MINUTES", "1440"))  # 24 hours


class CacheManager:
    """High-level cache management for outfit requests."""
    
    _instance = None
    _store: Optional[CacheStore] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._store = CacheStore(ttl_minutes=CACHE_TTL_MINUTES)
        return cls._instance
    
    @property
    def enabled(self) -> bool:
        """Check if caching is enabled."""
        return CACHE_ENABLED
    
    def generate_cache_key(
        self,
        image_hash: str,
        detected_clothing: Dict[str, bool],
        detected_items: Dict[str, Any],
        user_note: Optional[str],
        active_provider: str
    ) -> str:
        """
        Generate cache key from request parameters.
        
        Args:
            image_hash: SHA256 of input image
            detected_clothing: Detection results
            detected_items: Attribute results
            user_note: User note (if any)
            active_provider: Active LLM provider
            
        Returns:
            SHA256 hash as cache key
        """
        key_data = {
            "image_hash": image_hash,
            "detected_clothing": detected_clothing,
            "detected_items": self._normalize_items(detected_items),
            "user_note": user_note or "",
            "provider": active_provider,
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def _normalize_items(self, items: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize detected_items for consistent hashing."""
        normalized = {}
        for category, item in items.items():
            if isinstance(item, dict):
                normalized[category] = {
                    "present": item.get("present", False),
                    "type": item.get("type", ""),
                    "color": item.get("color", ""),
                    "style": item.get("style", ""),
                }
            else:
                normalized[category] = {"present": False}
        return normalized
    
    def get_image_hash(self, image_path: str) -> str:
        """Calculate SHA256 hash of an image file."""
        try:
            with open(image_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.warning(f"Failed to hash image: {e}")
            return ""
    
    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached response."""
        if not self.enabled:
            return None
        return self._store.get(cache_key)
    
    def set(self, cache_key: str, response: Dict[str, Any]):
        """Cache a response."""
        if not self.enabled:
            return
        self._store.set(cache_key, response)
    
    def get_status(self) -> Dict[str, Any]:
        """Get cache status for health endpoint."""
        stats = self._store.get_stats() if self._store else {}
        return {
            "enabled": self.enabled,
            "type": "disk_json",
            "ttl_minutes": CACHE_TTL_MINUTES,
            "entries": stats.get("entries", 0),
        }


# Global instance
cache_manager = CacheManager()


def get_cache_key(
    image_path: str,
    detected_clothing: Dict[str, bool],
    detected_items: Dict[str, Any],
    user_note: Optional[str],
    active_provider: str
) -> str:
    """Convenience function to generate cache key."""
    image_hash = cache_manager.get_image_hash(image_path)
    return cache_manager.generate_cache_key(
        image_hash=image_hash,
        detected_clothing=detected_clothing,
        detected_items=detected_items,
        user_note=user_note,
        active_provider=active_provider
    )
