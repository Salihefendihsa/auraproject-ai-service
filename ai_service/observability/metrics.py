"""
Metrics Module (v1.5.0)
Track request counts, cache performance, and LLM costs.
"""
import threading
from typing import Dict, Any

# Thread-safe metrics storage
_lock = threading.Lock()
_metrics = {
    "total_requests": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "total_tokens": 0,
    "total_cost_usd": 0.0,
    "requests_by_provider": {},
    "errors": 0
}


def increment_request(provider: str, cache_hit: bool, tokens: int = 0, cost_usd: float = 0.0, error: bool = False):
    """
    Record a request in metrics.
    
    Args:
        provider: LLM provider used
        cache_hit: Whether it was a cache hit
        tokens: Token count
        cost_usd: Estimated cost
        error: Whether request failed
    """
    with _lock:
        _metrics["total_requests"] += 1
        
        if cache_hit:
            _metrics["cache_hits"] += 1
        else:
            _metrics["cache_misses"] += 1
        
        _metrics["total_tokens"] += tokens
        _metrics["total_cost_usd"] += cost_usd
        
        if provider:
            _metrics["requests_by_provider"][provider] = _metrics["requests_by_provider"].get(provider, 0) + 1
        
        if error:
            _metrics["errors"] += 1


def get_metrics() -> Dict[str, Any]:
    """Get current metrics snapshot."""
    with _lock:
        total = _metrics["total_requests"]
        hits = _metrics["cache_hits"]
        
        return {
            "total_requests": total,
            "cache_hits": hits,
            "cache_misses": _metrics["cache_misses"],
            "cache_hit_ratio": round(hits / total, 3) if total > 0 else 0.0,
            "total_tokens": _metrics["total_tokens"],
            "total_cost_usd": round(_metrics["total_cost_usd"], 4),
            "requests_by_provider": dict(_metrics["requests_by_provider"]),
            "errors": _metrics["errors"]
        }


def reset_metrics():
    """Reset all metrics (for testing)."""
    global _metrics
    with _lock:
        _metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "requests_by_provider": {},
            "errors": 0
        }


# Token estimation constants (approximate)
TOKENS_PER_OUTFIT_REQUEST = 1500  # Input + output tokens estimate

# Cost per 1K tokens (approximate, as of late 2024)
COST_PER_1K_TOKENS = {
    "openai": 0.002,  # GPT-4o-mini
    "gemini": 0.0005  # Gemini Flash
}


def estimate_cost(provider: str, tokens: int = TOKENS_PER_OUTFIT_REQUEST) -> tuple:
    """
    Estimate cost for a request.
    
    Returns:
        (tokens, cost_usd)
    """
    rate = COST_PER_1K_TOKENS.get(provider, 0.001)
    cost = (tokens / 1000) * rate
    return tokens, round(cost, 6)
