"""
Cache Store (v1.4.2)
Disk-based JSON cache for request responses.
"""
import os
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class CacheStore:
    """Disk-based cache using JSON files."""
    
    def __init__(self, cache_dir: str = None, ttl_minutes: int = 1440):
        """
        Initialize cache store.
        
        Args:
            cache_dir: Directory for cache files
            ttl_minutes: Time-to-live in minutes (default: 24 hours)
        """
        if cache_dir is None:
            module_dir = Path(__file__).parent.parent
            cache_dir = module_dir / "data" / "cache"
        
        self.cache_dir = Path(cache_dir)
        self.ttl_seconds = ttl_minutes * 60
        self._ensure_directory()
    
    def _ensure_directory(self):
        """Create cache directory if needed."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for a cache key."""
        return self.cache_dir / f"{cache_key}.json"
    
    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached data if exists and not expired.
        
        Args:
            cache_key: Cache key (SHA256 hash)
            
        Returns:
            Cached data dict or None
        """
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Check expiration
            cached_at = data.get("_cached_at", 0)
            if time.time() - cached_at > self.ttl_seconds:
                logger.info(f"Cache expired: {cache_key[:16]}...")
                self._delete(cache_key)
                return None
            
            logger.info(f"Cache hit: {cache_key[:16]}...")
            return data.get("response")
            
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None
    
    def set(self, cache_key: str, response: Dict[str, Any]):
        """
        Save response to cache.
        
        Args:
            cache_key: Cache key (SHA256 hash)
            response: Response data to cache
        """
        cache_path = self._get_cache_path(cache_key)
        
        try:
            data = {
                "_cached_at": time.time(),
                "_cache_key": cache_key,
                "response": response
            }
            
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Cache saved: {cache_key[:16]}...")
            
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    def _delete(self, cache_key: str):
        """Delete a cache entry."""
        cache_path = self._get_cache_path(cache_key)
        try:
            if cache_path.exists():
                cache_path.unlink()
        except Exception as e:
            logger.warning(f"Cache delete error: {e}")
    
    def clear_expired(self) -> int:
        """
        Remove all expired cache entries.
        
        Returns:
            Number of entries removed
        """
        removed = 0
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    cached_at = data.get("_cached_at", 0)
                    if time.time() - cached_at > self.ttl_seconds:
                        cache_file.unlink()
                        removed += 1
                except:
                    pass
        except Exception as e:
            logger.warning(f"Cache cleanup error: {e}")
        
        if removed > 0:
            logger.info(f"Removed {removed} expired cache entries")
        
        return removed
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            cache_files = list(self.cache_dir.glob("*.json"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            return {
                "entries": len(cache_files),
                "size_bytes": total_size,
                "ttl_minutes": self.ttl_seconds // 60,
            }
        except Exception as e:
            logger.warning(f"Cache stats error: {e}")
            return {"entries": 0, "size_bytes": 0, "ttl_minutes": self.ttl_seconds // 60}
