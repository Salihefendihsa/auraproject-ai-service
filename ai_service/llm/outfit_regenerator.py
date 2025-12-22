"""
Outfit Regenerator (v2.4.0)
OpenAI-based outfit regeneration for replacing low-quality try-ons.
"""
import os
import json
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


REGENERATE_SYSTEM_PROMPT = """You are a fashion stylist AI. Generate a REPLACEMENT outfit.

The previous outfit attempt had quality issues. Generate a NEW outfit that:
1. Avoids the specific issues mentioned
2. Works with the user's existing items
3. Considers the event/weather context
4. Uses items from the user's wardrobe when possible

Return ONLY valid JSON in this format:
{
    "outfit": {
        "rank": <number>,
        "style_tag": "style name",
        "items": {
            "top": {"name": "...", "color": "...", "source": "wardrobe or suggested"},
            "bottom": {"name": "...", "color": "...", "source": "wardrobe or suggested"},
            "outerwear": {"name": "...", "color": "...", "source": "wardrobe or suggested"},
            "shoes": {"name": "...", "color": "...", "source": "wardrobe or suggested"}
        },
        "explanation": "Why this outfit works better"
    }
}"""


def is_configured() -> bool:
    """Check if OpenAI is configured for regeneration."""
    return bool(os.getenv("OPENAI_API_KEY"))


async def regenerate_outfit(
    outfit_index: int,
    detected_items: Dict[str, Any],
    issues: List[str],
    event: Optional[str] = None,
    weather_context: Optional[str] = None,
    wardrobe_context: Optional[str] = None,
    original_outfit: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Generate a replacement outfit using OpenAI.
    
    Args:
        outfit_index: Rank of the outfit being replaced
        detected_items: User's detected clothing items
        issues: List of issues from Gemini judge
        event: Event context (business, casual, etc.)
        weather_context: Weather context string
        wardrobe_context: User's wardrobe items
        original_outfit: The original outfit that failed
    
    Returns:
        New outfit dict or None
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OpenAI not configured for regeneration")
        return None
    
    try:
        from openai import AsyncOpenAI
        
        client = AsyncOpenAI(api_key=api_key)
        
        # Build context
        prompt_parts = [
            f"REGENERATE OUTFIT #{outfit_index}",
            "",
            "QUALITY ISSUES TO AVOID:",
            "- " + "\n- ".join(issues) if issues else "General improvement needed",
            ""
        ]
        
        # Add detected items
        detected_context = []
        for category, item in detected_items.items():
            if item.get("present"):
                item_type = item.get("type", "unknown")
                item_color = item.get("color", "unknown")
                detected_context.append(f"{category}: {item_color} {item_type}")
        
        if detected_context:
            prompt_parts.extend([
                "USER CURRENTLY WEARING:",
                "\n".join(detected_context),
                ""
            ])
        
        # Add wardrobe
        if wardrobe_context:
            prompt_parts.extend([
                "USER'S WARDROBE (PREFER THESE ITEMS):",
                wardrobe_context,
                ""
            ])
        
        # Add context
        if event:
            prompt_parts.append(f"EVENT: {event}")
        if weather_context:
            prompt_parts.append(f"WEATHER: {weather_context}")
        
        # Add original outfit info
        if original_outfit:
            original_style = original_outfit.get("style_tag", "unknown")
            prompt_parts.extend([
                "",
                f"ORIGINAL STYLE (failed): {original_style}",
                "Generate a DIFFERENT style that avoids the issues above."
            ])
        
        prompt_parts.extend([
            "",
            "Generate ONE replacement outfit that will render better.",
            "Prioritize simple, classic styles that are easier to render realistically.",
            "Avoid complex patterns, textures, or unusual clothing items."
        ])
        
        user_prompt = "\n".join(prompt_parts)
        
        logger.info(f"Regenerating outfit {outfit_index} via OpenAI...")
        
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": REGENERATE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.8,  # Slightly higher for variety
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        data = json.loads(content)
        
        outfit = data.get("outfit", {})
        
        if outfit:
            # Ensure rank is set
            outfit["rank"] = outfit_index
            outfit["regenerated"] = True
            outfit["regeneration_reason"] = issues
            
            logger.info(f"Regenerated outfit {outfit_index}: {outfit.get('style_tag', 'unknown')}")
            return outfit
        
        return None
        
    except Exception as e:
        logger.error(f"Outfit regeneration failed: {e}")
        return None


async def regenerate_worst_outfits(
    worst_results: List[Dict[str, Any]],
    detected_items: Dict[str, Any],
    outfits: List[Dict[str, Any]],
    event: Optional[str] = None,
    weather_context: Optional[str] = None,
    wardrobe_context: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Regenerate all worst outfits.
    
    Args:
        worst_results: List of JudgeResult dicts for worst outfits
        detected_items: User's detected clothing
        outfits: Original outfits list
        event, weather_context, wardrobe_context: Context
    
    Returns:
        Updated outfits list with replacements
    """
    updated_outfits = list(outfits)
    
    for worst in worst_results:
        outfit_index = worst.get("outfit_index", 1)
        issues = worst.get("issues", [])
        
        # Get original outfit
        original = None
        if 0 < outfit_index <= len(outfits):
            original = outfits[outfit_index - 1]
        
        # Regenerate
        new_outfit = await regenerate_outfit(
            outfit_index=outfit_index,
            detected_items=detected_items,
            issues=issues,
            event=event,
            weather_context=weather_context,
            wardrobe_context=wardrobe_context,
            original_outfit=original
        )
        
        if new_outfit and 0 < outfit_index <= len(updated_outfits):
            updated_outfits[outfit_index - 1] = new_outfit
            logger.info(f"Replaced outfit {outfit_index} with regenerated version")
    
    return updated_outfits
