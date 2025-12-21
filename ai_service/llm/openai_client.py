"""
OpenAI Client (v1.3.0)
Primary LLM for outfit planning decisions.
"""
import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a fashion stylist AI. Generate outfit recommendations.

RULES:
1. Return EXACTLY 5 outfits as JSON
2. Each outfit has: top, bottom, outerwear, shoes
3. If user already has an item (provided below), keep it with source="user"
4. For missing items, suggest new ones with source="suggested"
5. Match the user's detected style and colors where appropriate
6. Return ONLY valid JSON, no markdown

OUTPUT FORMAT:
{
  "outfits": [
    {
      "rank": 1,
      "style_tag": "style name",
      "items": {
        "top": {"name": "...", "color": "...", "source": "user or suggested"},
        "bottom": {"name": "...", "color": "...", "source": "user or suggested"},
        "outerwear": {"name": "...", "color": "...", "source": "user or suggested"},
        "shoes": {"name": "...", "color": "...", "source": "user or suggested"}
      },
      "explanation": "Brief explanation"
    }
  ]
}"""


def is_configured() -> bool:
    """Check if OpenAI API key is configured."""
    return bool(os.getenv("OPENAI_API_KEY"))


async def plan_outfits(
    detected_items: Dict[str, Any],
    user_note: Optional[str] = None,
    style_context: Optional[str] = None
) -> list:
    """
    Generate 5 outfits using OpenAI.
    
    Args:
        detected_items: Dict of detected clothing with type/color/style
        user_note: Optional user notes
        style_context: Optional context from Gemini advisor
        
    Returns:
        List of 5 outfit dicts
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    
    # Build context from detected items
    detected_context = []
    missing_parts = []
    
    for category, item in detected_items.items():
        if item.get("present"):
            item_type = item.get("type", "unknown")
            item_color = item.get("color", "unknown")
            item_style = item.get("style", "casual")
            detected_context.append(
                f"{category}: {item_color} {item_type} ({item_style} style)"
            )
        else:
            missing_parts.append(category)
    
    # Build user prompt
    prompt_parts = [
        "Generate 5 complete outfits.",
        "",
        "DETECTED on user (keep these, mark source=\"user\"):",
        '\n'.join(detected_context) if detected_context else 'None detected',
        "",
        "MISSING (must suggest, mark source=\"suggested\"):",
        ', '.join(missing_parts) if missing_parts else 'None missing',
    ]
    
    # Add style context from Gemini if available
    if style_context:
        prompt_parts.extend([
            "",
            "STYLE ADVISOR CONTEXT:",
            style_context,
        ])
    
    # Add user note if provided
    if user_note:
        prompt_parts.extend([
            "",
            f"User note: {user_note}",
        ])
    
    prompt_parts.extend([
        "",
        "Return ONLY valid JSON with exactly 5 outfits."
    ])
    
    user_prompt = '\n'.join(prompt_parts)
    
    try:
        from openai import AsyncOpenAI
        
        client = AsyncOpenAI(api_key=api_key)
        
        logger.info("Calling OpenAI for outfit planning...")
        
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=2500,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        data = json.loads(content)
        
        outfits = data.get("outfits", [])
        logger.info(f"OpenAI returned {len(outfits)} outfits")
        
        return outfits
        
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        raise
