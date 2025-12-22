"""
LLM Router (v2.3.0)
Orchestrates the hybrid LLM brain with event/weather/wardrobe context.
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

# Event context descriptions for LLM
EVENT_CONTEXTS = {
    "business": "professional business meeting or interview - formal, conservative colors",
    "casual": "everyday casual outing - relaxed, comfortable",
    "sport": "athletic activity or gym - activewear, breathable",
    "wedding": "wedding event - elegant, formal, avoid white",
    "party": "party or night out - stylish, trendy, statement pieces",
    "date": "romantic date - attractive, put-together, confident"
}


def build_context_prompt(
    event: Optional[str] = None,
    weather_context: Optional[str] = None,
    user_note: Optional[str] = None,
    wardrobe_context: Optional[str] = None
) -> str:
    """
    Build context string for LLM prompt.
    
    Args:
        event: Event type (business, casual, etc.)
        weather_context: Weather info string
        user_note: User's custom note
        wardrobe_context: User's wardrobe items context (v2.3)
    
    Returns:
        Context string for LLM
    """
    parts = []
    
    if wardrobe_context:
        parts.append(f"PRIORITIZE USER'S WARDROBE: {wardrobe_context}")
    
    if event and event in EVENT_CONTEXTS:
        parts.append(f"Event context: {EVENT_CONTEXTS[event]}")
    
    if weather_context:
        parts.append(weather_context)
    
    if user_note:
        parts.append(f"User preference: {user_note}")
    
    return " | ".join(parts) if parts else ""


async def plan_outfits(
    detected_items: Dict[str, Any],
    user_note: Optional[str] = None,
    season_hint: Optional[str] = None,
    event: Optional[str] = None,
    weather_context: Optional[str] = None,
    wardrobe_context: Optional[str] = None
) -> list:
    """
    Plan outfits using hybrid LLM approach with full context.
    
    v2.3.0: Added wardrobe context for user-owned items preference.
    
    Args:
        detected_items: Dict of detected clothing
        user_note: Optional user notes
        season_hint: Optional season for Gemini
        event: Event type (business, casual, etc.)
        weather_context: Weather context string
        wardrobe_context: User's wardrobe items (v2.3)
        
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
    
    # Build combined context with wardrobe priority
    combined_context = build_context_prompt(event, weather_context, user_note, wardrobe_context)
    
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
    
    # Combine style context with event/weather context
    if style_context and combined_context:
        full_context = f"{style_context} | {combined_context}"
    elif combined_context:
        full_context = combined_context
    else:
        full_context = style_context
    
    # Step 2: Call primary provider
    logger.info(f"Using primary provider: {active}")
    if event:
        logger.info(f"Event context: {event}")
    if weather_context:
        logger.info(f"Weather context: {weather_context[:50]}...")
    
    try:
        if active == "openai":
            outfits = await openai_client.plan_outfits(
                detected_items=detected_items,
                user_note=combined_context or user_note,
                style_context=full_context
            )
        elif active == "gemini":
            # Use Gemini as primary planner
            outfits = await _plan_with_gemini(detected_items, combined_context or user_note, event, weather_context)
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
                        user_note=combined_context or user_note,
                        style_context=full_context
                    )
                elif fallback == "gemini":
                    return await _plan_with_gemini(detected_items, combined_context or user_note, event, weather_context)
            except Exception as fallback_error:
                logger.error(f"Fallback provider also failed: {fallback_error}")
        
        raise


async def _plan_with_gemini(
    detected_items: Dict[str, Any],
    user_note: Optional[str] = None,
    event: Optional[str] = None,
    weather_context: Optional[str] = None
) -> list:
    """Use Gemini as primary outfit planner with event/weather support."""
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
    
    # Build event/weather context for prompt
    context_parts = []
    if event and event in EVENT_CONTEXTS:
        context_parts.append(f"Occasion: {EVENT_CONTEXTS[event]}")
    if weather_context:
        context_parts.append(weather_context)
    if user_note:
        context_parts.append(f"User preference: {user_note}")
    
    context_str = "\n".join(context_parts) if context_parts else ""
    
    prompt = f"""Generate exactly 5 outfit recommendations as JSON.

User has: {', '.join(detected_context) if detected_context else 'no items detected'}
Missing: {', '.join(missing_parts) if missing_parts else 'none'}

{context_str}

IMPORTANT:
- Recommend currently trending fashion styles (e.g., quiet luxury, street heritage, monochrome minimal)
- Consider the occasion/event when selecting formality level
- Consider weather when recommending layers and fabric weight
- Prioritize appropriateness while maintaining a "fashion-forward" look

Return JSON format:
{{"outfits": [
  {{"rank": 1, "style_tag": "...", "items": {{
    "top": {{"name": "...", "color": "...", "source": "user or suggested"}},
    "bottom": {{"name": "...", "color": "...", "source": "user or suggested"}},
    "outerwear": {{"name": "...", "color": "...", "source": "user or suggested"}},
    "shoes": {{"name": "...", "color": "...", "source": "user or suggested"}}
  }}, "explanation": "...", "occasion_fit": "how this suits the occasion"}}
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
