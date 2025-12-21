"""
Providers Module (v1.4.1)
LLM provider availability and priority management.
"""
import logging
from typing import Optional, List, Dict, Any

from ai_service.config.settings import get_settings

logger = logging.getLogger(__name__)


SUPPORTED_PROVIDERS = ["openai", "gemini"]


def get_provider_availability() -> Dict[str, bool]:
    """Get availability status for each provider."""
    settings = get_settings()
    return {
        "openai": settings.has_openai(),
        "gemini": settings.has_gemini(),
    }


def get_active_provider() -> Optional[str]:
    """
    Get the active primary provider based on settings and availability.
    
    Returns:
        Provider name or None if no provider available
    """
    settings = get_settings()
    
    if not settings.llm_enabled:
        logger.warning("LLM is disabled via AURA_LLM_ENABLED")
        return None
    
    availability = get_provider_availability()
    
    # Try primary provider first
    if settings.llm_primary in SUPPORTED_PROVIDERS:
        if availability.get(settings.llm_primary):
            return settings.llm_primary
        else:
            logger.warning(
                f"Primary provider '{settings.llm_primary}' not available, "
                f"checking secondary..."
            )
    
    # Try secondary provider
    if settings.llm_secondary in SUPPORTED_PROVIDERS:
        if availability.get(settings.llm_secondary):
            logger.info(f"Using secondary provider: {settings.llm_secondary}")
            return settings.llm_secondary
    
    # No provider available
    logger.error("No LLM provider available")
    return None


def get_fallback_provider() -> Optional[str]:
    """
    Get the fallback provider (secondary or any available).
    
    Returns:
        Provider name or None if no fallback available
    """
    settings = get_settings()
    availability = get_provider_availability()
    
    if not settings.llm_enabled:
        return None
    
    active = get_active_provider()
    
    # Try secondary if it's different from active
    if settings.llm_secondary != active:
        if availability.get(settings.llm_secondary):
            return settings.llm_secondary
    
    # Try any other available provider
    for provider in SUPPORTED_PROVIDERS:
        if provider != active and availability.get(provider):
            return provider
    
    return None


def get_provider_status() -> Dict[str, Any]:
    """
    Get complete provider status for health endpoint.
    
    Returns:
        Dict with enabled, primary, secondary, availability, daily_limit
    """
    settings = get_settings()
    availability = get_provider_availability()
    
    return {
        "enabled": settings.llm_enabled,
        "primary": settings.llm_primary,
        "secondary": settings.llm_secondary,
        "availability": availability,
        "active_provider": get_active_provider(),
        "fallback_provider": get_fallback_provider(),
        "daily_limit": settings.llm_daily_limit,
    }


def validate_provider_config() -> List[str]:
    """
    Validate provider configuration and return warnings.
    
    Returns:
        List of warning messages
    """
    settings = get_settings()
    availability = get_provider_availability()
    warnings = []
    
    if not settings.llm_enabled:
        warnings.append("LLM is disabled - outfit planning will fail")
    
    if not any(availability.values()):
        warnings.append("No LLM provider configured - set OPENAI_API_KEY or GEMINI_API_KEY")
    
    if settings.llm_primary not in SUPPORTED_PROVIDERS:
        warnings.append(f"Unknown primary provider: {settings.llm_primary}")
    
    if settings.llm_secondary not in SUPPORTED_PROVIDERS:
        warnings.append(f"Unknown secondary provider: {settings.llm_secondary}")
    
    if not availability.get(settings.llm_primary):
        warnings.append(f"Primary provider '{settings.llm_primary}' not configured")
    
    return warnings
