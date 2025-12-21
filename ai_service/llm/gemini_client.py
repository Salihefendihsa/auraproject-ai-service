"""
Gemini Client (v1.3.0)
Context and trend advisor for outfit planning.
"""
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def is_configured() -> bool:
    """Check if Gemini API key is configured."""
    return bool(os.getenv("GEMINI_API_KEY"))


async def get_style_context(
    detected_items: Dict[str, Any],
    season_hint: Optional[str] = None
) -> Optional[str]:
    """
    Get style context and trend advice from Gemini.
    
    Args:
        detected_items: Dict of detected clothing with type/color/style
        season_hint: Optional season hint
        
    Returns:
        Short context string (1-3 sentences) or None on failure
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    
    try:
        import google.generativeai as genai
        
        genai.configure(api_key=api_key)
        
        # Build context about detected items
        items_desc = []
        detected_styles = []
        detected_colors = []
        
        for category, item in detected_items.items():
            if item.get("present"):
                item_type = item.get("type", "unknown")
                item_color = item.get("color", "unknown")
                item_style = item.get("style", "")
                items_desc.append(f"{item_color} {item_type}")
                if item_style:
                    detected_styles.append(item_style)
                if item_color and item_color != "unknown":
                    detected_colors.append(item_color)
        
        if not items_desc:
            items_desc = ["no specific items detected"]
        
        prompt = f"""You are a fashion trend advisor. Be very concise (1-2 sentences max).

The user is wearing: {', '.join(items_desc)}
{f"Season: {season_hint}" if season_hint else ""}

Give a brief style recommendation considering current fashion trends.
Focus on what colors or styles would complement their existing items.
Maximum 2 sentences."""

        logger.info("Calling Gemini for style context...")
        
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        
        context = response.text.strip()
        
        # Ensure it's concise
        if len(context) > 300:
            context = context[:300] + "..."
        
        logger.info(f"Gemini context: {context[:100]}...")
        
        return context
        
    except Exception as e:
        logger.warning(f"Gemini error (non-fatal): {e}")
        return None
