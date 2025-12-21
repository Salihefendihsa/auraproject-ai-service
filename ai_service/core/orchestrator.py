"""
Pipeline Orchestrator (v1.1.0)
Coordinates segmentation and LLM-based outfit planning.
"""
import os
import json
import time
import logging
from typing import Dict, Any, Optional

from ai_service.core.storage import storage
from ai_service.vision.segmenter import segmenter

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a fashion stylist AI. Generate outfit recommendations.

RULES:
1. Return EXACTLY 5 outfits as JSON
2. Each outfit has: top, bottom, outerwear, shoes
3. If user already has an item (provided below), keep it as source="user"
4. For missing items, suggest new ones with source="suggested"
5. Return ONLY valid JSON, no markdown

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


async def run_pipeline(
    job_id: str,
    image_path: str,
    user_note: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run the full analysis pipeline.
    
    Steps:
    1. Run segmentation
    2. Save masks
    3. Call LLM with detected clothing context
    4. Return complete result
    """
    start_time = time.time()
    
    result = {
        "job_id": job_id,
        "detected_clothing": {
            "top": False,
            "bottom": False,
            "outerwear": False,
            "shoes": False
        },
        "masks": {},
        "raw_labels": [],
        "outfits": [],
        "status": "completed",
        "error": None
    }
    
    job_path = storage.get_job_path(job_id)
    masks_dir = str(job_path / "masks")
    
    # ================================================
    # STEP 1: Segmentation
    # ================================================
    logger.info(f"[{job_id}] Running segmentation...")
    
    try:
        seg_result = segmenter.segment(image_path)
        
        result["detected_clothing"]["top"] = seg_result.get("top", False)
        result["detected_clothing"]["bottom"] = seg_result.get("bottom", False)
        result["detected_clothing"]["outerwear"] = seg_result.get("outerwear", False)
        result["detected_clothing"]["shoes"] = seg_result.get("shoes", False)
        result["raw_labels"] = seg_result.get("raw_labels", [])
        
        # Save masks
        masks = seg_result.get("masks", {})
        if masks:
            saved_masks = segmenter.save_masks(masks, masks_dir)
            result["masks"] = saved_masks
        
        if seg_result.get("error"):
            logger.warning(f"Segmentation warning: {seg_result['error']}")
            
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        # Continue - LLM can still generate outfits
    
    # ================================================
    # STEP 2: LLM Outfit Generation
    # ================================================
    logger.info(f"[{job_id}] Generating outfits with LLM...")
    
    try:
        outfits = await generate_outfits_llm(
            detected_clothing=result["detected_clothing"],
            user_note=user_note
        )
        result["outfits"] = outfits
        logger.info(f"[{job_id}] Generated {len(outfits)} outfits")
        
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        result["error"] = str(e)
        result["status"] = "partial"
    
    processing_time = int((time.time() - start_time) * 1000)
    result["processing_time_ms"] = processing_time
    
    logger.info(f"[{job_id}] Pipeline complete in {processing_time}ms")
    
    return result


async def generate_outfits_llm(
    detected_clothing: Dict[str, bool],
    user_note: Optional[str] = None
) -> list:
    """Generate 5 outfits using OpenAI LLM."""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    
    # Build user prompt with detected clothing context
    detected_parts = [k for k, v in detected_clothing.items() if v]
    missing_parts = [k for k, v in detected_clothing.items() if not v]
    
    user_prompt = f"""Generate 5 complete outfits.

DETECTED on user (keep these, mark source="user"):
{', '.join(detected_parts) if detected_parts else 'None detected'}

MISSING (must suggest, mark source="suggested"):
{', '.join(missing_parts) if missing_parts else 'None missing'}

{f'User note: {user_note}' if user_note else ''}

Return ONLY valid JSON with exactly 5 outfits."""

    try:
        from openai import AsyncOpenAI
        
        client = AsyncOpenAI(api_key=api_key)
        
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
        
        return data.get("outfits", [])
        
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        raise
