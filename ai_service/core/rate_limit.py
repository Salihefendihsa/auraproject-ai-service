"""
Rate Limiting Module (v2.0.0)
IP-based and user-based rate limiting with sliding window.
"""
import time
import logging
from collections import defaultdict
from typing import Optional, Dict, Tuple
from dataclasses import dataclass, field
from fastapi import Request, HTTPException

from ai_service.core.auth import User, get_user_concurrent_jobs

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration for an endpoint."""
    requests_per_minute: int = 5
    requests_per_hour: int = 100
    max_concurrent_jobs: int = 2


# Default configurations
RATE_LIMITS: Dict[str, RateLimitConfig] = {
    "/ai/outfit": RateLimitConfig(requests_per_minute=5, max_concurrent_jobs=2),
    "/ai/wardrobe/items": RateLimitConfig(requests_per_minute=10, max_concurrent_jobs=2),
    "default": RateLimitConfig(requests_per_minute=30, max_concurrent_jobs=2),
}


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter with IP and user tracking.
    """
    
    def __init__(self):
        # Structure: {key: [(timestamp, count), ...]}
        self._requests: Dict[str, list] = defaultdict(list)
        self._lock_time = 60  # Window size in seconds
    
    def _cleanup_old_requests(self, key: str, window_seconds: int) -> None:
        """Remove requests older than window."""
        cutoff = time.time() - window_seconds
        self._requests[key] = [
            (ts, count) for ts, count in self._requests[key]
            if ts > cutoff
        ]
    
    def _count_requests(self, key: str, window_seconds: int) -> int:
        """Count requests in the current window."""
        self._cleanup_old_requests(key, window_seconds)
        return sum(count for _, count in self._requests[key])
    
    def _add_request(self, key: str) -> None:
        """Record a new request."""
        self._requests[key].append((time.time(), 1))
    
    def check_rate_limit(
        self,
        ip: str,
        user_id: Optional[str],
        endpoint: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if request should be rate limited.
        
        Args:
            ip: Client IP address
            user_id: Authenticated user ID (if any)
            endpoint: Request endpoint path
        
        Returns:
            Tuple of (allowed, error_message)
        """
        config = RATE_LIMITS.get(endpoint, RATE_LIMITS["default"])
        
        # Check IP-based limit
        ip_key = f"ip:{ip}:{endpoint}"
        ip_count = self._count_requests(ip_key, 60)
        
        if ip_count >= config.requests_per_minute:
            remaining = 60 - (time.time() - self._requests[ip_key][0][0])
            return False, f"Rate limit exceeded. Try again in {int(remaining)}s"
        
        # Check user-based limit (if authenticated)
        if user_id:
            user_key = f"user:{user_id}:{endpoint}"
            user_count = self._count_requests(user_key, 60)
            
            if user_count >= config.requests_per_minute:
                remaining = 60 - (time.time() - self._requests[user_key][0][0])
                return False, f"User rate limit exceeded. Try again in {int(remaining)}s"
        
        return True, None
    
    def record_request(self, ip: str, user_id: Optional[str], endpoint: str) -> None:
        """Record a request for rate limiting."""
        ip_key = f"ip:{ip}:{endpoint}"
        self._add_request(ip_key)
        
        if user_id:
            user_key = f"user:{user_id}:{endpoint}"
            self._add_request(user_key)
    
    def get_remaining(self, ip: str, user_id: Optional[str], endpoint: str) -> int:
        """Get remaining requests in current window."""
        config = RATE_LIMITS.get(endpoint, RATE_LIMITS["default"])
        
        ip_key = f"ip:{ip}:{endpoint}"
        ip_count = self._count_requests(ip_key, 60)
        
        remaining = config.requests_per_minute - ip_count
        
        if user_id:
            user_key = f"user:{user_id}:{endpoint}"
            user_count = self._count_requests(user_key, 60)
            user_remaining = config.requests_per_minute - user_count
            remaining = min(remaining, user_remaining)
        
        return max(0, remaining)


# Global rate limiter instance
rate_limiter = SlidingWindowRateLimiter()


def get_client_ip(request: Request) -> str:
    """
    Extract client IP from request.
    Handles X-Forwarded-For for reverse proxy setups.
    """
    # Check for forwarded IP
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # Take first IP in chain
        return forwarded.split(",")[0].strip()
    
    # Fall back to direct client
    if request.client:
        return request.client.host
    
    return "unknown"


async def check_rate_limit(
    request: Request,
    user: Optional[User] = None
) -> None:
    """
    FastAPI dependency for rate limiting.
    
    Usage:
        @router.post("/endpoint")
        async def endpoint(request: Request, _: None = Depends(check_rate_limit)):
            ...
    
    Raises:
        HTTPException 429: If rate limit exceeded
    """
    ip = get_client_ip(request)
    user_id = user.user_id if user else None
    endpoint = request.url.path
    
    allowed, error_msg = rate_limiter.check_rate_limit(ip, user_id, endpoint)
    
    if not allowed:
        logger.warning(f"Rate limit exceeded: IP={ip}, user={user_id}, endpoint={endpoint}")
        raise HTTPException(
            status_code=429,
            detail=error_msg,
            headers={"Retry-After": "60"}
        )
    
    # Record the request
    rate_limiter.record_request(ip, user_id, endpoint)


async def check_concurrent_jobs(user: User) -> None:
    """
    Check if user has reached concurrent job limit.
    
    Raises:
        HTTPException 429: If concurrent job limit reached
    """
    config = RATE_LIMITS.get("/ai/outfit", RATE_LIMITS["default"])
    current_jobs = get_user_concurrent_jobs(user.user_id)
    
    if current_jobs >= config.max_concurrent_jobs:
        logger.warning(f"Concurrent job limit reached: user={user.user_id}, jobs={current_jobs}")
        raise HTTPException(
            status_code=429,
            detail=f"Maximum concurrent jobs ({config.max_concurrent_jobs}) reached. Wait for current jobs to complete.",
            headers={"Retry-After": "30"}
        )


def get_rate_limit_headers(request: Request, user: Optional[User] = None) -> Dict[str, str]:
    """
    Generate rate limit headers for response.
    
    Returns:
        Dict with X-RateLimit-* headers
    """
    ip = get_client_ip(request)
    user_id = user.user_id if user else None
    endpoint = request.url.path
    
    config = RATE_LIMITS.get(endpoint, RATE_LIMITS["default"])
    remaining = rate_limiter.get_remaining(ip, user_id, endpoint)
    
    return {
        "X-RateLimit-Limit": str(config.requests_per_minute),
        "X-RateLimit-Remaining": str(remaining),
        "X-RateLimit-Reset": str(int(time.time()) + 60)
    }
