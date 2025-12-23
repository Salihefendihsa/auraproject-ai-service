"""
Seed Cache Helper (v1.0.0)

Non-breaking cache layer for /ai/outfit-seed endpoint.
Caches complete responses to avoid repeating expensive operations (LLM, try-on).

Cache Key Design:
-----------------
The cache key is a SHA256 hash of normalized request parameters:
- seed_image_hash: SHA256 of the seed image content
- gender: male/female
- event: work/date/party/casual or null
- season: summer/winter or null
- mode: mock/partial_tryon/full_tryon
- baseline_version: ensures cache invalidates when baseline changes

WHY these components:
- seed_image_hash: Same image should produce same outfits (deterministic behavior)
- gender: Directly affects catalog filtering
- event/season: Affects scoring weights and outfit selection
- mode: Determines try-on pipeline path
- baseline_version: Cache must respect frozen baseline; new versions = cache miss

TTL:
----
24 hours (configurable via AURA_SEED_CACHE_TTL_MINUTES)
In-memory cache is acceptable for demo; can be extended to Redis later.

Guarantees:
-----------
- Cache is transparent to frontend (response contract unchanged)
- Cache does NOT alter response data
- Cache respects baseline_version
"""
import os
import json
import hashlib
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from ai_service.config.settings import DEMO_BASELINE_VERSION

logger = logging.getLogger(__name__)

# Configuration
SEED_CACHE_ENABLED = os.getenv("AURA_SEED_CACHE_ENABLED", "true").lower() == "true"
SEED_CACHE_TTL_MINUTES = int(os.getenv("AURA_SEED_CACHE_TTL_MINUTES", "1440"))  # 24 hours


class SeedResponseCache:
    """
    In-memory cache for /ai/outfit-seed responses.
    
    This is a simple TTL-based cache that stores complete response dicts.
    For production, this can be extended to use Redis or disk storage.
    """
    
    def __init__(self, ttl_minutes: int = 1440):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._ttl = timedelta(minutes=ttl_minutes)
        self._hits = 0
        self._misses = 0
    
    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if cache entry has expired."""
        cached_at = entry.get("_cached_at")
        if not cached_at:
            return True
        try:
            cached_time = datetime.fromisoformat(cached_at)
            return datetime.utcnow() - cached_time > self._ttl
        except (ValueError, TypeError):
            return True
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached response by key.
        
        Returns None if not found or expired.
        The returned response excludes cache metadata.
        """
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None
        
        if self._is_expired(entry):
            # Clean up expired entry
            del self._cache[key]
            self._misses += 1
            logger.debug(f"Seed cache expired: {key[:16]}...")
            return None
        
        self._hits += 1
        logger.info(f"Seed cache HIT: {key[:16]}...")
        
        # Return response without cache metadata
        response = entry.get("_response")
        return response
    
    def set(self, key: str, response: Dict[str, Any]) -> None:
        """
        Store response in cache.
        
        Adds cache metadata but does NOT modify the response itself.
        """
        self._cache[key] = {
            "_response": response,
            "_cached_at": datetime.utcnow().isoformat(),
        }
        logger.info(f"Seed cache SET: {key[:16]}... (ttl={self._ttl})")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "entries": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_ratio": round(self._hits / total, 3) if total > 0 else 0.0,
            "ttl_minutes": int(self._ttl.total_seconds() / 60),
        }
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        logger.info("Seed cache cleared")


# Global singleton instance
_seed_cache = SeedResponseCache(ttl_minutes=SEED_CACHE_TTL_MINUTES)


def compute_image_hash(image_content: bytes) -> str:
    """
    Compute SHA256 hash of image content.
    
    WHY SHA256: Collision-resistant, deterministic, fast enough for demo.
    """
    return hashlib.sha256(image_content).hexdigest()


def generate_seed_cache_key(
    seed_image_hash: str,
    gender: str,
    event: Optional[str],
    season: Optional[str],
    mode: str,
    baseline_version: str = DEMO_BASELINE_VERSION
) -> str:
    """
    Generate cache key for outfit-seed request.
    
    WHY this design:
    - Deterministic: Same inputs → same key → same cached result
    - Includes baseline_version: Cache auto-invalidates when baseline changes
    - Sorted JSON: Ensures consistent key regardless of dict ordering
    
    Args:
        seed_image_hash: SHA256 of seed image content
        gender: male or female
        event: work/date/party/casual or None
        season: summer/winter or None
        mode: mock/partial_tryon/full_tryon
        baseline_version: Current baseline version (default: from config)
    
    Returns:
        SHA256 hash string as cache key
    """
    key_data = {
        "seed_image_hash": seed_image_hash,
        "gender": gender.lower(),
        "event": (event or "").lower(),
        "season": (season or "").lower(),
        "mode": mode.lower(),
        "baseline_version": baseline_version,
    }
    
    # Deterministic JSON serialization
    key_string = json.dumps(key_data, sort_keys=True)
    cache_key = hashlib.sha256(key_string.encode()).hexdigest()
    
    logger.debug(f"Seed cache key: {cache_key[:16]}... (gender={gender}, mode={mode})")
    return cache_key


def get_cached_seed_response(cache_key: str) -> Optional[Dict[str, Any]]:
    """
    Get cached response for outfit-seed request.
    
    Usage:
        cache_key = generate_seed_cache_key(...)
        cached = get_cached_seed_response(cache_key)
        if cached:
            return cached  # Skip expensive operations
    
    Returns:
        Cached response dict, or None if not found/expired/disabled.
    """
    if not SEED_CACHE_ENABLED:
        return None
    
    return _seed_cache.get(cache_key)


def set_cached_seed_response(cache_key: str, response: Dict[str, Any]) -> None:
    """
    Store response in cache.
    
    Usage:
        cache_key = generate_seed_cache_key(...)
        response = run_outfit_seed_job(...)  # Expensive operation
        set_cached_seed_response(cache_key, response)  # Cache for next time
    
    Args:
        cache_key: Key from generate_seed_cache_key()
        response: Complete response dict from outfit-seed endpoint
    """
    if not SEED_CACHE_ENABLED:
        return
    
    _seed_cache.set(cache_key, response)


def get_seed_cache_stats() -> Dict[str, Any]:
    """
    Get seed cache statistics for health/metrics endpoints.
    
    Returns:
        Dict with entries, hits, misses, hit_ratio, ttl_minutes
    """
    return {
        "enabled": SEED_CACHE_ENABLED,
        "type": "in_memory",
        **_seed_cache.get_stats(),
    }
