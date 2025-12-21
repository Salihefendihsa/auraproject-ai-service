"""
Request Logger (v1.5.0)
Structured logging for request tracking and observability.
"""
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

# Ensure logs directory exists
LOGS_DIR = Path(__file__).parent.parent.parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

REQUEST_LOG_FILE = LOGS_DIR / "requests.log"

# Configure request logger
request_logger = logging.getLogger("aura.requests")
request_logger.setLevel(logging.INFO)

# File handler for requests
file_handler = logging.FileHandler(REQUEST_LOG_FILE, encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(message)s"))
request_logger.addHandler(file_handler)

# Prevent propagation to root logger
request_logger.propagate = False


def log_request(
    job_id: str,
    provider_used: str,
    cache_hit: bool,
    latency_ms: int,
    status: str,
    error: Optional[str] = None,
    tokens: int = 0,
    cost_usd: float = 0.0
):
    """
    Log a structured request entry.
    
    Args:
        job_id: Unique job identifier
        provider_used: LLM provider (openai/gemini/cached)
        cache_hit: Whether response was from cache
        latency_ms: Request latency in milliseconds
        status: success or fail
        error: Error message if failed
        tokens: Estimated token usage
        cost_usd: Estimated cost in USD
    """
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "job_id": job_id,
        "provider": provider_used,
        "cache_hit": cache_hit,
        "latency_ms": latency_ms,
        "status": status,
        "tokens": tokens,
        "cost_usd": round(cost_usd, 6)
    }
    
    if error:
        entry["error"] = error
    
    request_logger.info(json.dumps(entry))


def is_logging_enabled() -> bool:
    """Check if request logging is enabled."""
    return os.getenv("AURA_LOGGING_ENABLED", "true").lower() == "true"
