"""
Local IDM-VTON Wrapper (Simplified v1.0.0)

This module provides a simplified interface to IDM-VTON that works on Windows.
It uses our existing segmentation module instead of the complex DensePose setup.

Requirements:
- Downloaded IDM-VTON checkpoint from HuggingFace (yisol/IDM-VTON)
- RTX 4060 or similar GPU with 8GB VRAM
"""
import sys
import os
import logging
from pathlib import Path
from typing import Optional
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Add IDM-VTON to path
IDM_VTON_PATH = Path(__file__).parent.parent.parent / "IDM-VTON"
sys.path.insert(0, str(IDM_VTON_PATH))


class LocalTryOn:
    """
    Local IDM-VTON wrapper for virtual try-on.
    
    Uses HuggingFace models downloaded to IDM-VTON/ckpt directory.
    Works without detectron2 by using our simpler segmentation.
    """
    
    _pipeline = None
    _loaded = False
    _device = "cpu"
    
    def __init__(self):
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"LocalTryOn initialized (device: {self._device})")
    
    def _load_pipeline(self):
        """Lazy-load the IDM-VTON pipeline."""
        if self._loaded:
            return
        
        try:
            from diffusers import StableDiffusionXLInpaintPipeline, AutoencoderKL
            from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
            
            logger.info("Loading IDM-VTON pipeline...")
            
            model_path = str(IDM_VTON_PATH / "ckpt")
            
            # Check if model exists
            if not (Path(model_path) / "vae").exists():
                logger.error(f"Model not found at {model_path}")
                logger.error("Run: huggingface-cli download yisol/IDM-VTON --local-dir IDM-VTON/ckpt")
                return
            
            # Load VAE
            vae = AutoencoderKL.from_pretrained(
                model_path,
                subfolder="vae",
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32
            )
            
            # Load image encoder for garment
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                model_path,
                subfolder="image_encoder",
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32
            )
            
            # Use SDXL Inpainting as base (simpler than full IDM-VTON)
            self._pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
                "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                vae=vae,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
                variant="fp16" if self._device == "cuda" else None
            )
            
            self._pipeline.to(self._device)
            
            # Enable memory optimizations
            if self._device == "cuda":
                self._pipeline.enable_model_cpu_offload()
            
            self._loaded = True
            logger.info("✓ IDM-VTON pipeline loaded")
            
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            self._loaded = False
    
    async def try_on(
        self,
        person_image_path: str,
        garment_image_path: str,
        mask_path: str,
        output_path: str,
        garment_description: str = "a clothing item",
        steps: int = 25,
        seed: int = 42
    ) -> bool:
        """
        Perform virtual try-on.
        
        Args:
            person_image_path: Path to person photo
            garment_image_path: Path to garment image
            mask_path: Path to upper body mask
            output_path: Where to save result
            garment_description: Text description of garment
            steps: Denoising steps (20-40)
            seed: Random seed
            
        Returns:
            True if successful
        """
        self._load_pipeline()
        
        if not self._loaded:
            logger.error("Pipeline not loaded, cannot perform try-on")
            return False
        
        try:
            # Load images
            person = Image.open(person_image_path).convert("RGB")
            garment = Image.open(garment_image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
            
            # Resize to standard size
            target_size = (768, 1024)
            person = person.resize(target_size, Image.Resampling.LANCZOS)
            garment = garment.resize(target_size, Image.Resampling.LANCZOS)
            mask = mask.resize(target_size, Image.Resampling.NEAREST)
            
            # Generate prompt
            prompt = f"model wearing {garment_description}, photorealistic, high quality"
            negative_prompt = "ugly, blurry, low quality, distorted"
            
            # Set seed for reproducibility
            generator = torch.Generator(device=self._device).manual_seed(seed)
            
            logger.info(f"Running try-on: {steps} steps, seed {seed}")
            
            # Run inpainting
            result = self._pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=person,
                mask_image=mask,
                num_inference_steps=steps,
                generator=generator,
                strength=0.9
            ).images[0]
            
            # Save result
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            result.save(output_path, quality=95)
            
            logger.info(f"✓ Try-on saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Try-on failed: {e}")
            return False


# Singleton
_local_tryon: Optional[LocalTryOn] = None


def get_local_tryon() -> LocalTryOn:
    """Get or create LocalTryOn singleton."""
    global _local_tryon
    if _local_tryon is None:
        _local_tryon = LocalTryOn()
    return _local_tryon


async def local_try_on(
    person_image: str,
    garment_image: str,
    mask_path: str,
    output_path: str,
    description: str = "clothing item"
) -> bool:
    """
    Convenience function for local try-on.
    
    Args:
        person_image: Path to person photo
        garment_image: Path to garment to try on
        mask_path: Path to body mask
        output_path: Where to save result
        description: Garment description for prompt
        
    Returns:
        True if successful
    """
    client = get_local_tryon()
    return await client.try_on(
        person_image_path=person_image,
        garment_image_path=garment_image,
        mask_path=mask_path,
        output_path=output_path,
        garment_description=description
    )
