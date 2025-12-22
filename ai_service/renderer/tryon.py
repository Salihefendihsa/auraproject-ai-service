"""
Virtual Try-On Renderer (v2.6.0)
Enhanced SD Inpainting with optional ControlNet pose locking.

Model: runwayml/stable-diffusion-inpainting
ControlNet: Optional pose/edge/depth conditioning
"""
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageFont

from ai_service.config.llm_config import get_controlnet_config

logger = logging.getLogger(__name__)


# ==================== PROMPT DISCIPLINE ====================

# Fixed positive prompt components for realistic fabric rendering
POSITIVE_PROMPT_BASE = (
    "realistic fabric texture, natural lighting, same pose, high quality clothing, "
    "well fitted outfit, photorealistic, professional photography, detailed fabric"
)

# Fixed negative prompt to avoid common SD issues
NEGATIVE_PROMPT = (
    "distorted face, extra limbs, blurry, low quality, cartoon, unrealistic body, "
    "bad anatomy, deformed, disfigured, mutation, extra fingers, missing limbs, "
    "floating limbs, disconnected limbs, malformed hands, bad hands, "
    "watermark, text, signature, logo"
)

# Resolution settings
STAGE1_SIZE = 512   # Fast preview
STAGE2_SIZE = 768   # High quality refine (CPU compatible)


# ==================== MASK PROCESSING ====================

def process_mask_quality(
    mask: Image.Image,
    blur_sigma: float = 4.0,
    dilate_pixels: int = 3,
    erode_pixels: int = 2
) -> Image.Image:
    """
    Apply mask quality improvements for smoother inpainting edges.
    
    Args:
        mask: Binary mask (0/255)
        blur_sigma: Gaussian blur sigma (3-5 recommended)
        dilate_pixels: Dilation amount for edge expansion
        erode_pixels: Erosion amount for edge cleanup
        
    Returns:
        Processed mask with feathered edges
    """
    mask_array = np.array(mask)
    
    # Step 1: Morphological dilation (expand mask slightly)
    if dilate_pixels > 0:
        from scipy import ndimage
        mask_array = ndimage.binary_dilation(
            mask_array > 127,
            iterations=dilate_pixels
        ).astype(np.uint8) * 255
    
    # Step 2: Morphological erosion (clean up edges)
    if erode_pixels > 0:
        from scipy import ndimage
        mask_array = ndimage.binary_erosion(
            mask_array > 127,
            iterations=erode_pixels
        ).astype(np.uint8) * 255
    
    # Step 3: Gaussian blur for edge feathering
    mask_pil = Image.fromarray(mask_array)
    if blur_sigma > 0:
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
    
    return mask_pil


def exclude_face_hands_from_mask(
    mask: Image.Image,
    face_region: Optional[Tuple[int, int, int, int]] = None,
    protection_margin: int = 30
) -> Image.Image:
    """
    Ensure face and hands are NEVER masked (protected regions).
    
    For now, uses a simple top-portion exclusion for face protection.
    In future versions, can integrate face detection.
    
    Args:
        mask: Input mask
        face_region: Optional (x1, y1, x2, y2) face bounding box
        protection_margin: Extra margin around protected regions
        
    Returns:
        Mask with face/hand regions excluded
    """
    mask_array = np.array(mask)
    h, w = mask_array.shape
    
    # Simple heuristic: protect top 25% of image (usually contains face)
    face_cutoff = int(h * 0.25)
    mask_array[:face_cutoff, :] = 0
    
    # If explicit face region provided, use it
    if face_region:
        x1, y1, x2, y2 = face_region
        # Expand with margin
        x1 = max(0, x1 - protection_margin)
        y1 = max(0, y1 - protection_margin)
        x2 = min(w, x2 + protection_margin)
        y2 = min(h, y2 + protection_margin)
        mask_array[y1:y2, x1:x2] = 0
    
    return Image.fromarray(mask_array)


def add_fallback_watermark(image: Image.Image, text: str = "Preview") -> Image.Image:
    """Add watermark to fallback images."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    
    # Simple text watermark
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    w, h = img.size
    text_position = (10, h - 30)
    
    # Semi-transparent background
    draw.rectangle([5, h - 35, 80, h - 10], fill=(128, 128, 128, 180))
    draw.text(text_position, text, fill=(255, 255, 255), font=font)
    
    return img


# ==================== TRYON RENDERER ====================

class TryOnRenderer:
    """
    Enhanced Try-On Renderer v2.6.0.
    
    Features:
    - Mask quality improvements (blur, dilation, erosion)
    - Fixed positive/negative prompts
    - Face/hand protection
    - Two-stage rendering (512 preview, 768 refine)
    - ControlNet pose locking (optional, v2.6.0)
    - Fallback with watermark
    """
    
    _instance = None
    _pipeline = None
    _controlnet_pipeline = None
    _controlnet = None
    _is_loaded = False
    _controlnet_loaded = False
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
    
    def _load_controlnet(self):
        """Load ControlNet model if enabled (v2.6.0)."""
        if self._controlnet_loaded or not get_controlnet_config().enabled:
            return
        
        if not self._is_loaded or self._device == "cpu":
            logger.warning("ControlNet requires GPU and loaded SD pipeline")
            return
        
        try:
            import torch
            from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline
            
            config = get_controlnet_config()
            model_id = config.get_model_id()
            
            logger.info(f"Loading ControlNet: {model_id}")
            
            self._controlnet = ControlNetModel.from_pretrained(
                model_id,
                torch_dtype=torch.float16
            ).to(self._device)
            
            # Create ControlNet-enabled inpainting pipeline
            self._controlnet_pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                controlnet=self._controlnet,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self._device)
            
            self._controlnet_loaded = True
            logger.info(f"✓ ControlNet loaded: {config.control_type.value}")
            
        except Exception as e:
            logger.warning(f"ControlNet load failed (using fallback): {e}")
            self._controlnet = None
            self._controlnet_pipeline = None
    
    def _get_controlnet_condition(self, input_image_path: str) -> Optional[Image.Image]:
        """Get ControlNet conditioning image for pose lock (v2.6.0)."""
        config = get_controlnet_config()
        
        if not config.enabled:
            return None
        
        # Load ControlNet if not loaded
        if not self._controlnet_loaded:
            self._load_controlnet()
        
        if self._controlnet_pipeline is None:
            return None
        
        try:
            from ai_service.renderer.controlnet import load_controlnet_condition
            
            condition = load_controlnet_condition(input_image_path)
            if condition and condition.ready:
                return condition.condition_image
            
            return None
            
        except Exception as e:
            logger.warning(f"ControlNet condition failed (using fallback): {e}")
            return None
    
    def render_outfit(
        self,
        input_image_path: str,
        masks_dir: str,
        outfit: Dict[str, Any],
        output_path: str,
        resolution: int = STAGE1_SIZE,
        num_inference_steps: Optional[int] = None
    ) -> bool:
        """
        Render a single outfit try-on with quality improvements.
        
        v2.2.0 Changes:
        - Mask feathering with blur/dilation/erosion
        - Face/hand protection
        - Fixed positive/negative prompts
        - Configurable resolution for two-stage
        """
        try:
            # Try to load model
            self._load_model()
            
            if not self._is_loaded:
                logger.warning("Model not loaded, using fallback")
                return self._fallback_with_watermark(input_image_path, output_path)
            
            # Load input image
            input_image = Image.open(input_image_path).convert("RGB")
            original_size = input_image.size
            
            # Resize for SD
            input_image_resized = input_image.resize(
                (resolution, resolution), 
                Image.Resampling.LANCZOS
            )
            
            # Build combined mask for suggested items only
            combined_mask = self._build_combined_mask(masks_dir, outfit, original_size)
            
            if combined_mask is None:
                logger.info("No suggested items to render, using fallback")
                return self._fallback_with_watermark(input_image_path, output_path)
            
            # Apply mask quality improvements
            processed_mask = process_mask_quality(
                combined_mask,
                blur_sigma=4.0,
                dilate_pixels=3,
                erode_pixels=2
            )
            
            # Exclude face/hands from mask
            processed_mask = exclude_face_hands_from_mask(processed_mask)
            
            # Resize mask to target resolution
            mask_resized = processed_mask.resize(
                (resolution, resolution), 
                Image.Resampling.LANCZOS  # Use LANCZOS for smoother mask edges
            )
            
            # Build prompt with discipline
            clothing_prompt = self._build_clothing_prompt(outfit)
            # v2.8.0 Enhancement: High-fidelity texture and preservation tags
            quality_tags = "highly detailed, 8k uhd, photorealistic, realistic fabric texture, intricate details, preservation of original person features"
            full_prompt = f"{clothing_prompt}. {quality_tags}. {POSITIVE_PROMPT_BASE}"
            
            logger.info(f"Rendering at {resolution}x{resolution}: {clothing_prompt[:60]}...")
            
            # Configure inference steps based on device and resolution
            if num_inference_steps is None:
                # v2.8.0: Increased steps for better quality (approximating VITON)
                if self._device == "cpu":
                    num_inference_steps = 25 if resolution <= 512 else 35
                else:
                    num_inference_steps = 40 if resolution <= 512 else 50
            
            # Run inpainting with quality settings
            import torch
            
            # v2.6.0: Try ControlNet conditioning if enabled
            controlnet_condition = self._get_controlnet_condition(input_image_path)
            
            with torch.no_grad():
                if controlnet_condition is not None and self._controlnet_pipeline is not None:
                    # ControlNet-enhanced rendering
                    logger.info("Using ControlNet pose conditioning")
                    result = self._controlnet_pipeline(
                        prompt=full_prompt,
                        negative_prompt=NEGATIVE_PROMPT,
                        image=input_image_resized,
                        mask_image=mask_resized,
                        control_image=controlnet_condition,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=7.5,
                        controlnet_conditioning_scale=get_controlnet_config().conditioning_scale,
                        strength=0.85
                    )
                else:
                    # Standard SD Inpainting fallback
                    result = self._pipeline(
                        prompt=full_prompt,
                        negative_prompt=NEGATIVE_PROMPT,
                        image=input_image_resized,
                        mask_image=mask_resized,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=7.5,
                        strength=0.85  # Preserve more of original
                    )
            
            # Get result and resize back
            output_image = result.images[0]
            output_image = output_image.resize(original_size, Image.Resampling.LANCZOS)
            
            # Save
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            output_image.save(output_path, quality=95)
            
            logger.info(f"Saved render: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Rendering failed: {e}")
            return self._fallback_with_watermark(input_image_path, output_path)
    
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
        
        # Only include clothing regions, NEVER face or hands
        clothing_categories = ["top", "bottom", "outerwear", "shoes"]
        
        for category in clothing_categories:
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
    
    def _build_clothing_prompt(self, outfit: Dict[str, Any]) -> str:
        """Build clothing-focused prompt from outfit items."""
        items = outfit.get("items", {})
        suggested_parts = []
        
        for category in ["top", "bottom", "outerwear", "shoes"]:
            item = items.get(category, {})
            if item.get("source") == "suggested":
                color = item.get("color", "")
                name = item.get("name", category)
                suggested_parts.append(f"{color} {name}".strip())
        
        if not suggested_parts:
            return "A realistic photo of the same person"
        
        clothing_desc = ", ".join(suggested_parts)
        
        return f"A person wearing {clothing_desc}"
    
    def _fallback_with_watermark(self, input_path: str, output_path: str) -> bool:
        """Fallback: copy input image with watermark."""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Load and add watermark
            image = Image.open(input_path).convert("RGB")
            watermarked = add_fallback_watermark(image, "Fallback")
            watermarked.save(output_path, quality=95)
            
            logger.info(f"Fallback with watermark: {output_path}")
            return False
        except Exception as e:
            logger.error(f"Fallback save failed: {e}")
            # Last resort: just copy
            try:
                shutil.copy(input_path, output_path)
            except:
                pass
            return False


# ==================== TWO-STAGE RENDERING ====================

def render_all_outfits(
    input_image_path: str,
    masks_dir: str,
    outfits: List[Dict[str, Any]],
    output_dir: str,
    enable_two_stage: bool = True,
    refine_top_n: int = 2
) -> Dict[str, Any]:
    """
    Render try-on images for all outfits with two-stage quality.
    
    v2.2.0 Two-Stage Rendering:
    - Stage 1: Render all 5 outfits at 512x512 (fast)
    - Stage 2: Refine top 1-2 outfits at 768x768 (quality)
    
    Args:
        input_image_path: Path to original user image
        masks_dir: Directory containing segmentation masks
        outfits: List of outfit dicts
        output_dir: Directory to save rendered images
        enable_two_stage: Whether to do two-stage rendering
        refine_top_n: How many top outfits to refine (1-2)
        
    Returns:
        Dict with render paths and success info
    """
    renderer = TryOnRenderer()
    results = {
        "renders": {},
        "success_count": 0,
        "fallback_count": 0,
        "refined": [],
        "stage1_resolution": STAGE1_SIZE,
        "stage2_resolution": STAGE2_SIZE
    }
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Stage 1: Render all at 512x512
    logger.info(f"Stage 1: Rendering {len(outfits[:5])} outfits at {STAGE1_SIZE}x{STAGE1_SIZE}")
    
    for i, outfit in enumerate(outfits[:5], start=1):
        render_filename = f"outfit_{i}.png"
        render_path = output_path / render_filename
        
        success = renderer.render_outfit(
            input_image_path=input_image_path,
            masks_dir=masks_dir,
            outfit=outfit,
            output_path=str(render_path),
            resolution=STAGE1_SIZE
        )
        
        results["renders"][i] = render_filename
        
        if success:
            results["success_count"] += 1
        else:
            results["fallback_count"] += 1
    
    # Stage 2: Refine top N outfits at higher resolution
    if enable_two_stage and results["success_count"] > 0 and refine_top_n > 0:
        # Refine the first N successful renders (typically rank 1-2)
        refined_count = 0
        
        for i in range(1, min(refine_top_n + 1, 6)):
            if results["renders"].get(i):
                render_filename = f"outfit_{i}.png"
                render_path = output_path / render_filename
                
                # Check if this was a successful render (not fallback)
                if render_path.exists():
                    logger.info(f"Stage 2: Refining outfit {i} at {STAGE2_SIZE}x{STAGE2_SIZE}")
                    
                    success = renderer.render_outfit(
                        input_image_path=input_image_path,
                        masks_dir=masks_dir,
                        outfit=outfits[i - 1],
                        output_path=str(render_path),
                        resolution=STAGE2_SIZE
                    )
                    
                    if success:
                        results["refined"].append(i)
                        refined_count += 1
                    
                    if refined_count >= refine_top_n:
                        break
    
    # Ensure we always have 5 render URLs
    for i in range(1, 6):
        if i not in results["renders"]:
            # Create fallback for missing renders
            render_filename = f"outfit_{i}.png"
            render_path = output_path / render_filename
            renderer._fallback_with_watermark(input_image_path, str(render_path))
            results["renders"][i] = render_filename
            results["fallback_count"] += 1
    
    logger.info(
        f"Rendering complete: {results['success_count']} rendered, "
        f"{results['fallback_count']} fallback, {len(results['refined'])} refined"
    )
    
    return results


# Global instance
tryon_renderer = TryOnRenderer()
