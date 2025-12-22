"""
Virtual Try-On Integration via Replicate (v1.0.0)

Uses Replicate's IDM-VTON model for reliable virtual try-on.
Cost: ~$0.024 per run (free trial credits available).

Setup:
1. Sign up at https://replicate.com
2. Get API token from https://replicate.com/account/api-tokens
3. Set environment variable: REPLICATE_API_TOKEN=your_token
"""
import os
import logging
import asyncio
from pathlib import Path
from typing import Optional
from PIL import Image
import base64
import io

logger = logging.getLogger(__name__)

# Default Replicate model for try-on
DEFAULT_TRYON_MODEL = "cuuupid/idm-vton:c871bb9b046f6d0a3d78549b8d1f7cf9bbf2909dd0b1e8c30ef38f7632920523"


def is_replicate_configured() -> bool:
    """Check if Replicate API token is configured."""
    return bool(os.getenv("REPLICATE_API_TOKEN"))


def image_to_data_uri(image_path: str) -> str:
    """Convert image file to data URI for API upload."""
    path = Path(image_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    with open(path, "rb") as f:
        data = f.read()
    
    # Determine MIME type
    ext = path.suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp"
    }
    mime_type = mime_types.get(ext, "image/png")
    
    # Encode as base64 data URI
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


async def replicate_try_on(
    person_image: str,
    garment_image: str,
    output_path: str,
    category: str = "upper_body",
    denoise_steps: int = 30,
    seed: int = 42
) -> bool:
    """
    Perform virtual try-on using Replicate IDM-VTON.
    
    Args:
        person_image: Path to person photo
        garment_image: Path to garment/clothing image
        output_path: Where to save the result
        category: "upper_body", "lower_body", or "dresses"
        denoise_steps: Quality steps (15-50, higher = better but slower)
        seed: Random seed for reproducibility
        
    Returns:
        True if successful, False otherwise
    """
    if not is_replicate_configured():
        logger.error("REPLICATE_API_TOKEN not set. Get one at https://replicate.com/account/api-tokens")
        return False
    
    try:
        import replicate
        
        logger.info(f"Starting Replicate try-on...")
        logger.info(f"  Person: {person_image}")
        logger.info(f"  Garment: {garment_image}")
        logger.info(f"  Category: {category}")
        
        # Convert images to data URIs
        human_uri = image_to_data_uri(person_image)
        garment_uri = image_to_data_uri(garment_image)
        
        # Call Replicate API
        output = await asyncio.to_thread(
            replicate.run,
            DEFAULT_TRYON_MODEL,
            input={
                "human_img": human_uri,
                "garm_img": garment_uri,
                "category": category,
                "denoise_steps": denoise_steps,
                "seed": seed
            }
        )
        
        if output:
            # Output is typically a URL to the generated image
            logger.info(f"Got result URL: {output}")
            
            # Download the result
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.get(str(output))
                if response.status_code == 200:
                    # Save to output path
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(output_path, "wb") as f:
                        f.write(response.content)
                    
                    logger.info(f"✓ Try-on saved to: {output_path}")
                    return True
                else:
                    logger.error(f"Failed to download result: {response.status_code}")
                    return False
        
        logger.error("Replicate returned empty output")
        return False
        
    except Exception as e:
        logger.error(f"Replicate try-on failed: {e}")
        return False


async def batch_try_on(
    person_image: str,
    garment_images: list,
    output_dir: str,
    category: str = "upper_body"
) -> dict:
    """
    Try on multiple garments on the same person.
    
    Args:
        person_image: Path to person photo
        garment_images: List of garment image paths
        output_dir: Directory to save results
        category: Garment category
        
    Returns:
        Dict mapping garment path to output path (or None if failed)
    """
    results = {}
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for i, garment in enumerate(garment_images):
        out_file = output_path / f"tryon_{i+1}.png"
        
        success = await replicate_try_on(
            person_image=person_image,
            garment_image=garment,
            output_path=str(out_file),
            category=category
        )
        
        results[garment] = str(out_file) if success else None
    
    return results
