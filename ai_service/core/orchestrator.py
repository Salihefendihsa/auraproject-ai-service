"""
Pipeline Orchestrator (v1.2.0)
Coordinates segmentation + attributes + LLM + try-on rendering.
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


async def run_pipeline(
    job_id: str,
    image_path: str,
    user_note: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run the full analysis and rendering pipeline.
    
    Steps (v1.2.0):
    1. Run segmentation with attribute extraction
    2. Save masks
    3. Call LLM with detected items context
    4. Render try-on images for each outfit
    5. Return complete result with render URLs
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
        "detected_items": {},
        "masks": {},
        "raw_labels": [],
        "outfits": [],
        "renders": {},
        "status": "completed",
        "error": None
    }
    
    job_path = storage.get_job_path(job_id)
    masks_dir = str(job_path / "masks")
    renders_dir = str(job_path / "renders")
    
    # ================================================
    # STEP 1: Segmentation with Attributes
    # ================================================
    logger.info(f"[{job_id}] Running segmentation with attribute extraction...")
    
    try:
        seg_result = segmenter.segment(image_path, extract_attributes=True)
        
        result["detected_clothing"]["top"] = seg_result.get("top", False)
        result["detected_clothing"]["bottom"] = seg_result.get("bottom", False)
        result["detected_clothing"]["outerwear"] = seg_result.get("outerwear", False)
        result["detected_clothing"]["shoes"] = seg_result.get("shoes", False)
        result["raw_labels"] = seg_result.get("raw_labels", [])
        result["detected_items"] = seg_result.get("detected_items", {})
        
        # Save masks
        masks = seg_result.get("masks", {})
        if masks:
            saved_masks = segmenter.save_masks(masks, masks_dir)
            result["masks"] = saved_masks
        
        if seg_result.get("error"):
            logger.warning(f"Segmentation warning: {seg_result['error']}")
            
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        for cat in ["top", "bottom", "outerwear", "shoes"]:
            result["detected_items"][cat] = {"present": False}
    
    # ================================================
    # STEP 2: LLM Outfit Generation
    # ================================================
    logger.info(f"[{job_id}] Generating outfits with LLM...")
    
    try:
        outfits = await generate_outfits_llm(
            detected_items=result["detected_items"],
            user_note=user_note
        )
        result["outfits"] = outfits
        logger.info(f"[{job_id}] Generated {len(outfits)} outfits")
        
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        result["error"] = str(e)
        result["status"] = "partial"
    
    # ================================================
    # STEP 3: Try-On Rendering
    # ================================================
    if result["outfits"]:
        logger.info(f"[{job_id}] Rendering try-on images...")
        
        try:
            from ai_service.renderer.tryon import render_all_outfits
            
            render_result = render_all_outfits(
                input_image_path=image_path,
                masks_dir=masks_dir,
                outfits=result["outfits"],
                output_dir=renders_dir
            )
            
            result["renders"] = render_result.get("renders", {})
            
            # Add render_url and tryon_method to each outfit
            for i, outfit in enumerate(result["outfits"], start=1):
                if i in result["renders"]:
                    outfit["render_url"] = f"/ai/assets/jobs/{job_id}/renders/{result['renders'][i]}"
                    outfit["tryon_method"] = "inpainting"
            
            logger.info(
                f"[{job_id}] Try-on complete: "
                f"{render_result.get('success_count', 0)} rendered, "
                f"{render_result.get('fallback_count', 0)} fallback"
            )
            
        except ImportError as e:
            logger.warning(f"Try-on renderer not available: {e}")
            _create_fallback_renders(image_path, renders_dir, result)
        except Exception as e:
            logger.error(f"Try-on rendering failed: {e}")
            _create_fallback_renders(image_path, renders_dir, result)
    
    processing_time = int((time.time() - start_time) * 1000)
    result["processing_time_ms"] = processing_time
    
    logger.info(f"[{job_id}] Pipeline complete in {processing_time}ms")
    
    return result


def _create_fallback_renders(image_path: str, renders_dir: str, result: Dict):
    """Create fallback renders by copying input image."""
    import shutil
    from pathlib import Path
    
    try:
        output_path = Path(renders_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i, outfit in enumerate(result["outfits"], start=1):
            render_file = f"outfit_{i}.png"
            shutil.copy(image_path, output_path / render_file)
            result["renders"][i] = render_file
            outfit["render_url"] = f"/ai/assets/jobs/{result['job_id']}/renders/{render_file}"
            outfit["tryon_method"] = "fallback"
        
        logger.info(f"Created {len(result['outfits'])} fallback renders")
    except Exception as e:
        logger.error(f"Fallback render creation failed: {e}")


async def generate_outfits_llm(
    detected_items: Dict[str, Any],
    user_note: Optional[str] = None
) -> list:
    """Generate 5 outfits using OpenAI LLM with detected item context."""
    
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
    
    user_prompt = f"""Generate 5 complete outfits.

DETECTED on user (keep these, mark source="user"):
{chr(10).join(detected_context) if detected_context else 'None detected'}

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
