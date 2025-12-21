"""
LLM Router (v1.4.1)
Orchestrates the hybrid LLM brain with config-based provider management.
"""
import logging
from typing import Dict, Any, Optional

from ai_service.llm import openai_client, gemini_client
from ai_service.config import (
    get_settings,
    get_active_provider,
    get_fallback_provider,
    get_provider_status,
)

logger = logging.getLogger(__name__)


async def plan_outfits(
    detected_items: Dict[str, Any],
    user_note: Optional[str] = None,
    season_hint: Optional[str] = None
) -> list:
    """
    Plan outfits using hybrid LLM approach with config-based routing.
    
    Flow:
    1. Check if LLM is enabled
    2. Get active provider from config
    3. If Gemini available and secondary, get style context
    4. Call primary provider for outfit planning
    5. Fallback to secondary if primary fails
    
    Args:
        detected_items: Dict of detected clothing
        user_note: Optional user notes
        season_hint: Optional season for Gemini
        
    Returns:
        List of 5 outfit dicts
    """
    settings = get_settings()
    
    # Check if LLM is enabled
    if not settings.llm_enabled:
        raise ValueError("LLM is disabled via AURA_LLM_ENABLED=false")
    
    active = get_active_provider()
    fallback = get_fallback_provider()
    
    if not active:
        raise ValueError("No LLM provider available - check API keys")
    
    style_context = None
    
    # Step 1: Try to get context from Gemini (if configured and secondary)
    if gemini_client.is_configured():
        logger.info("Gemini available - getting style context...")
        try:
            style_context = await gemini_client.get_style_context(
                detected_items=detected_items,
                season_hint=season_hint
            )
            if style_context:
                logger.info("✓ Gemini context acquired")
        except Exception as e:
            logger.warning(f"Gemini failed (continuing without): {e}")
            style_context = None
    else:
        logger.info("Gemini not configured - skipping style context")
    
    # Step 2: Call primary provider
    logger.info(f"Using primary provider: {active}")
    
    try:
        if active == "openai":
            outfits = await openai_client.plan_outfits(
                detected_items=detected_items,
                user_note=user_note,
                style_context=style_context
            )
        elif active == "gemini":
            # Use Gemini as primary planner (without context injection)
            outfits = await _plan_with_gemini(detected_items, user_note)
        else:
            raise ValueError(f"Unknown provider: {active}")
        
        return outfits
        
    except Exception as e:
        logger.error(f"Primary provider failed: {e}")
        
        # Step 3: Try fallback if available
        if fallback and fallback != active:
            logger.info(f"Trying fallback provider: {fallback}")
            try:
                if fallback == "openai":
                    return await openai_client.plan_outfits(
                        detected_items=detected_items,
                        user_note=user_note,
                        style_context=style_context
                    )
                elif fallback == "gemini":
                    return await _plan_with_gemini(detected_items, user_note)
            except Exception as fallback_error:
                logger.error(f"Fallback provider also failed: {fallback_error}")
        
        raise


async def _plan_with_gemini(
    detected_items: Dict[str, Any],
    user_note: Optional[str] = None
) -> list:
    """Use Gemini as primary outfit planner (basic implementation)."""
    import os
    import json
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")
    
    import google.generativeai as genai
    
    genai.configure(api_key=api_key)
    
    # Build context
    detected_context = []
    missing_parts = []
    
    for category, item in detected_items.items():
        if item.get("present"):
            item_type = item.get("type", "unknown")
            item_color = item.get("color", "unknown")
            detected_context.append(f"{category}: {item_color} {item_type}")
        else:
            missing_parts.append(category)
    
    prompt = f"""Generate exactly 5 outfit recommendations as JSON.

User has: {', '.join(detected_context) if detected_context else 'no items detected'}
Missing: {', '.join(missing_parts) if missing_parts else 'none'}
{f'Note: {user_note}' if user_note else ''}

Return JSON format:
{{"outfits": [
  {{"rank": 1, "style_tag": "...", "items": {{
    "top": {{"name": "...", "color": "...", "source": "user or suggested"}},
    "bottom": {{"name": "...", "color": "...", "source": "user or suggested"}},
    "outerwear": {{"name": "...", "color": "...", "source": "user or suggested"}},
    "shoes": {{"name": "...", "color": "...", "source": "user or suggested"}}
  }}, "explanation": "..."}}
]}}

Return ONLY valid JSON."""

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    
    text = response.text.strip()
    
    # Parse JSON from response
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    
    data = json.loads(text)
    return data.get("outfits", [])
