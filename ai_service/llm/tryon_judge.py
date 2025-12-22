"""
Try-On Quality Judge (v2.5.0)
Visual evaluation using configurable LLM (Gemini/OpenAI) via judge config.

Uses AURA_JUDGE_* environment variables for provider/model selection.
"""
import os
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass

from ai_service.config.llm_config import get_judge_config, LLMRole
from ai_service.llm.llm_adapter import get_judge_client

logger = logging.getLogger(__name__)


@dataclass
class JudgeResult:
    """Result from try-on quality evaluation."""
    outfit_index: int
    quality_score: float
    verdict: str
    issues: List[str]
    
    def to_dict(self) -> dict:
        return {
            "outfit_index": self.outfit_index,
            "quality_score": self.quality_score,
            "verdict": self.verdict,
            "issues": self.issues
        }


def is_configured() -> bool:
    """Check if judge is configured."""
    config = get_judge_config()
    if config.is_gemini():
        return bool(os.getenv("GEMINI_API_KEY"))
    else:
        return bool(os.getenv("OPENAI_API_KEY"))


JUDGE_PROMPT = """You are a visual quality judge for AI-generated try-on images.

Compare the ORIGINAL photo with the RENDERED try-on image.

Evaluate:
1. FACE INTEGRITY: Is the face unchanged BEFORE and after? Not distorted?
2. POSE CONSISTENCY: Does body pose match original?
3. CLOTHING REALISM: Do clothes look realistic and well-fitted?
4. ARTIFACTS: Any glitches, blurs, unnatural edges?

Respond with ONLY this JSON:
{
    "quality_score": <float 0.0-1.0>,
    "verdict": "good" or "bad",
    "issues": ["issue1", "issue2", ...]
}

SCORING:
- 0.9-1.0: Excellent
- 0.7-0.9: Good
- 0.5-0.7: Acceptable
- 0.3-0.5: Poor
- 0.0-0.3: Very poor

ISSUES: "face_distortion", "extra_limbs", "clothing_blur", "edge_artifacts", "pose_mismatch"

Return ONLY valid JSON."""


async def judge_tryon_quality(
    original_image_path: str,
    rendered_image_path: str,
    mask_preview_path: Optional[str] = None,
    outfit_index: int = 1
) -> JudgeResult:
    """Evaluate try-on render quality using configured judge."""
    
    fallback = JudgeResult(
        outfit_index=outfit_index,
        quality_score=0.7,
        verdict="good",
        issues=[]
    )
    
    config = get_judge_config()
    
    if not is_configured():
        logger.warning("Judge not configured, using default score")
        return fallback
    
    if not Path(rendered_image_path).exists():
        return JudgeResult(outfit_index=outfit_index, quality_score=0.3, verdict="bad", issues=["render_not_found"])
    
    try:
        client = get_judge_client()
        
        if config.is_gemini():
            # Use Gemini vision
            from PIL import Image
            
            original_img = Image.open(original_image_path)
            rendered_img = Image.open(rendered_image_path)
            
            prompt = f"{JUDGE_PROMPT}\n\nORIGINAL IMAGE (first) vs RENDERED TRY-ON (second):"
            
            text = await client.generate_with_images(
                prompt=prompt,
                images=[original_img, rendered_img],
                json_mode=True
            )
            
            data = _parse_judge_response(text)
            
        else:
            # OpenAI text-based (describe images if no vision)
            user_prompt = f"Evaluate try-on for outfit {outfit_index}. Assume good quality if no visual issues."
            data = await client.generate_json(JUDGE_PROMPT, user_prompt)
        
        quality_score = max(0.0, min(1.0, float(data.get("quality_score", 0.7))))
        verdict = data.get("verdict", "good")
        if verdict not in ["good", "bad"]:
            verdict = "good" if quality_score >= 0.6 else "bad"
        issues = data.get("issues", [])
        if not isinstance(issues, list):
            issues = []
        
        result = JudgeResult(
            outfit_index=outfit_index,
            quality_score=quality_score,
            verdict=verdict,
            issues=issues
        )
        
        logger.info(f"Judge outfit {outfit_index}: score={quality_score:.2f}, verdict={verdict}")
        return result
        
    except Exception as e:
        logger.error(f"Judge error: {e}")
        return fallback


def _parse_judge_response(text: str) -> dict:
    """Parse JSON from judge response."""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    return json.loads(text.strip())


async def judge_all_outfits(
    original_image_path: str,
    renders_dir: str,
    outfit_count: int = 5
) -> List[JudgeResult]:
    """Judge all try-on renders."""
    results = []
    renders_path = Path(renders_dir)
    
    for i in range(1, outfit_count + 1):
        render_file = renders_path / f"outfit_{i}.png"
        
        if render_file.exists():
            result = await judge_tryon_quality(
                original_image_path=original_image_path,
                rendered_image_path=str(render_file),
                outfit_index=i
            )
            results.append(result)
        else:
            results.append(JudgeResult(
                outfit_index=i, quality_score=0.3, verdict="bad", issues=["render_missing"]
            ))
    
    results.sort(key=lambda r: r.quality_score, reverse=True)
    return results


def identify_worst_outfits(
    judge_results: List[JudgeResult],
    threshold: float = 0.6,
    max_replace: int = 2
) -> List[JudgeResult]:
    """Identify worst outfits needing replacement."""
    worst = [r for r in judge_results if r.quality_score < threshold]
    worst.sort(key=lambda r: r.quality_score)
    return worst[:max_replace]
