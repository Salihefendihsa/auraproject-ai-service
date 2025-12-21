"""
LLM Router (v1.3.0)
Orchestrates the hybrid LLM brain with OpenAI + Gemini.
"""
import logging
from typing import Dict, Any, Optional

from ai_service.llm import openai_client, gemini_client

logger = logging.getLogger(__name__)


def get_provider_status() -> Dict[str, bool]:
    """Get status of LLM providers."""
    return {
        "openai": openai_client.is_configured(),
        "gemini": gemini_client.is_configured()
    }


async def plan_outfits(
    detected_items: Dict[str, Any],
    user_note: Optional[str] = None,
    season_hint: Optional[str] = None
) -> list:
    """
    Plan outfits using hybrid LLM approach.
    
    Flow:
    1. If Gemini is available, get style context/trends
    2. Call OpenAI with context injected
    3. If Gemini fails, continue with OpenAI only
    4. If OpenAI fails, raise error
    
    Args:
        detected_items: Dict of detected clothing
        user_note: Optional user notes
        season_hint: Optional season for Gemini
        
    Returns:
        List of 5 outfit dicts
    """
    style_context = None
    
    # Step 1: Try to get context from Gemini (advisory)
    if gemini_client.is_configured():
        logger.info("Gemini available - getting style context...")
        try:
            style_context = await gemini_client.get_style_context(
                detected_items=detected_items,
                season_hint=season_hint
            )
            if style_context:
                logger.info("✓ Gemini context acquired")
            else:
                logger.info("Gemini returned no context")
        except Exception as e:
            logger.warning(f"Gemini failed (continuing without): {e}")
            style_context = None
    else:
        logger.info("Gemini not configured - using OpenAI only")
    
    # Step 2: Call OpenAI (primary decision maker)
    if not openai_client.is_configured():
        raise ValueError("OPENAI_API_KEY not set - cannot generate outfits")
    
    logger.info("Calling OpenAI for outfit planning...")
    
    outfits = await openai_client.plan_outfits(
        detected_items=detected_items,
        user_note=user_note,
        style_context=style_context
    )
    
    return outfits
