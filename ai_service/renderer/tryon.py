"""
Virtual Try-On Renderer (v1.2.0)
Uses Stable Diffusion Inpainting to render user wearing suggested outfits.

Model: runwayml/stable-diffusion-inpainting
"""
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class TryOnRenderer:
    """
    Singleton class for virtual try-on rendering using SD Inpainting.
    Lazy loading - model only loads on first use.
    """
    
    _instance = None
    _pipeline = None
    _is_loaded = False
    _load_failed = False
    _device = "cpu"
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _load_model(self):
        """Load SD Inpainting model on first use."""
        if self._is_loaded or self._load_failed:
            return
        
        try:
            logger.info("Loading Stable Diffusion Inpainting model...")
            logger.info("First time may download ~2-3GB from Hugging Face")
            
            import torch
            from diffusers import StableDiffusionInpaintPipeline
            
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self._device}")
            
            # Load pipeline
            self._pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            self._pipeline.to(self._device)
            
            # Optimize for CPU if needed
            if self._device == "cpu":
                self._pipeline.enable_attention_slicing()
            
            self._is_loaded = True
            logger.info("✓ SD Inpainting model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load SD Inpainting model: {e}")
            self._load_failed = True
    
    def render_outfit(
        self,
        input_image_path: str,
        masks_dir: str,
        outfit: Dict[str, Any],
        output_path: str
    ) -> bool:
        """
        Render a single outfit try-on.
        
        Args:
            input_image_path: Path to original user image
            masks_dir: Directory containing segmentation masks
            outfit: Outfit dict with items (each has source field)
            output_path: Path to save rendered image
            
        Returns:
            True if rendering succeeded, False if fallback was used
        """
        try:
            # Try to load model
            self._load_model()
            
            if not self._is_loaded:
                logger.warning("Model not loaded, using fallback")
                return self._fallback_copy(input_image_path, output_path)
            
            # Load input image
            input_image = Image.open(input_image_path).convert("RGB")
            original_size = input_image.size
            
            # Resize for SD (must be 512x512)
            input_image_resized = input_image.resize((512, 512), Image.Resampling.LANCZOS)
            
            # Build combined mask for suggested items only
            combined_mask = self._build_combined_mask(masks_dir, outfit, original_size)
            
            if combined_mask is None:
                logger.info("No suggested items to render, using fallback")
                return self._fallback_copy(input_image_path, output_path)
            
            # Resize mask to 512x512
            combined_mask_resized = combined_mask.resize((512, 512), Image.Resampling.NEAREST)
            
            # Build prompt
            prompt = self._build_prompt(outfit)
            
            logger.info(f"Rendering with prompt: {prompt[:100]}...")
            
            # Run inpainting
            import torch
            
            with torch.no_grad():
                result = self._pipeline(
                    prompt=prompt,
                    image=input_image_resized,
                    mask_image=combined_mask_resized,
                    num_inference_steps=20 if self._device == "cpu" else 30,
                    guidance_scale=7.5
                )
            
            # Get result and resize back
            output_image = result.images[0]
            output_image = output_image.resize(original_size, Image.Resampling.LANCZOS)
            
            # Save
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            output_image.save(output_path)
            
            logger.info(f"Saved render: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Rendering failed: {e}")
            return self._fallback_copy(input_image_path, output_path)
    
    def _build_combined_mask(
        self,
        masks_dir: str,
        outfit: Dict[str, Any],
        original_size: tuple
    ) -> Optional[Image.Image]:
        """Build combined mask from suggested items only."""
        masks_path = Path(masks_dir)
        combined = None
        
        items = outfit.get("items", {})
        
        for category in ["top", "bottom", "outerwear", "shoes"]:
            item = items.get(category, {})
            
            # Only include suggested items (not user-owned)
            if item.get("source") == "suggested":
                mask_file = masks_path / f"mask_{category}.png"
                
                if mask_file.exists():
                    mask = Image.open(mask_file).convert("L")
                    mask = mask.resize(original_size, Image.Resampling.NEAREST)
                    mask_array = np.array(mask)
                    
                    if combined is None:
                        combined = mask_array
                    else:
                        combined = np.maximum(combined, mask_array)
        
        if combined is not None:
            # Ensure binary mask (0 or 255)
            combined = (combined > 0).astype(np.uint8) * 255
            return Image.fromarray(combined)
        
        return None
    
    def _build_prompt(self, outfit: Dict[str, Any]) -> str:
        """Build inpainting prompt from outfit items."""
        items = outfit.get("items", {})
        suggested_parts = []
        
        for category in ["top", "bottom", "outerwear", "shoes"]:
            item = items.get(category, {})
            if item.get("source") == "suggested":
                color = item.get("color", "")
                name = item.get("name", category)
                suggested_parts.append(f"{color} {name}".strip())
        
        if not suggested_parts:
            return "A realistic photo of the same person, keep original face, body, pose and lighting."
        
        clothing_desc = ", ".join(suggested_parts)
        
        return (
            f"A realistic photo of the same person wearing {clothing_desc}. "
            f"Keep original face, body, pose and lighting. "
            f"Photorealistic, high quality, natural lighting."
        )
    
    def _fallback_copy(self, input_path: str, output_path: str) -> bool:
        """Fallback: copy input image to output path."""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(input_path, output_path)
            logger.info(f"Fallback: copied input to {output_path}")
            return False
        except Exception as e:
            logger.error(f"Fallback copy failed: {e}")
            return False


def render_all_outfits(
    input_image_path: str,
    masks_dir: str,
    outfits: List[Dict[str, Any]],
    output_dir: str
) -> Dict[str, Any]:
    """
    Render try-on images for all outfits.
    
    Args:
        input_image_path: Path to original user image
        masks_dir: Directory containing segmentation masks
        outfits: List of outfit dicts
        output_dir: Directory to save rendered images
        
    Returns:
        Dict with render paths and success info
    """
    renderer = TryOnRenderer()
    results = {
        "renders": {},
        "success_count": 0,
        "fallback_count": 0
    }
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for i, outfit in enumerate(outfits[:5], start=1):
        render_filename = f"outfit_{i}.png"
        render_path = output_path / render_filename
        
        success = renderer.render_outfit(
            input_image_path=input_image_path,
            masks_dir=masks_dir,
            outfit=outfit,
            output_path=str(render_path)
        )
        
        results["renders"][i] = render_filename
        
        if success:
            results["success_count"] += 1
        else:
            results["fallback_count"] += 1
    
    logger.info(
        f"Rendering complete: {results['success_count']} rendered, "
        f"{results['fallback_count']} fallback"
    )
    
    return results


# Global instance
tryon_renderer = TryOnRenderer()
