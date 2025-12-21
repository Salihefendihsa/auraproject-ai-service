"""
Clothing Segmenter Module (v1.1.0)
Uses SegFormer model from Hugging Face for clothing detection.
"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# Label mapping for mattmdjaga/segformer_b2_clothes
ATR_LABELS = [
    "Background", "Hat", "Hair", "Sunglasses", "Upper-clothes", "Skirt",
    "Pants", "Dress", "Belt", "Left-shoe", "Right-shoe", "Face",
    "Left-leg", "Right-leg", "Left-arm", "Right-arm", "Bag", "Scarf"
]

# Map model labels to our categories
CLOTHING_LABEL_MAP = {
    "Upper-clothes": "top",
    "Dress": "top",
    "Pants": "bottom",
    "Skirt": "bottom",
    "Left-shoe": "shoes",
    "Right-shoe": "shoes",
    "Scarf": "outerwear",
    # Note: This model doesn't have explicit outerwear - we handle this gracefully
}


class ClothingSegmenter:
    """Singleton class for clothing segmentation using SegFormer."""
    
    _instance = None
    _pipeline = None
    _is_loaded = False
    _load_failed = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _load_model(self):
        """Load segmentation model on first use."""
        if self._is_loaded or self._load_failed:
            return
        
        try:
            logger.info("Loading SegFormer model (first time may download ~300MB)...")
            
            from transformers import pipeline
            import torch
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            self._pipeline = pipeline(
                "image-segmentation",
                model="mattmdjaga/segformer_b2_clothes",
                device=device
            )
            
            self._is_loaded = True
            logger.info("✓ SegFormer model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load segmentation model: {e}")
            self._load_failed = True
    
    def segment(self, image_path: str) -> Dict[str, Any]:
        """
        Segment clothing from image.
        
        Returns:
            Dict with detected_clothing booleans, raw_labels, masks, error
        """
        result = {
            "top": False,
            "bottom": False,
            "outerwear": False,
            "shoes": False,
            "masks": {},
            "raw_labels": [],
            "error": None
        }
        
        # Try to load model
        self._load_model()
        
        if not self._is_loaded:
            result["error"] = "Segmentation model not available"
            return result
        
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Run segmentation
            outputs = self._pipeline(image)
            
            # Process results
            for segment in outputs:
                label = segment.get("label", "")
                result["raw_labels"].append(label)
                
                # Map to our categories
                category = CLOTHING_LABEL_MAP.get(label)
                if category:
                    result[category] = True
                    
                    # Get mask
                    mask = segment.get("mask")
                    if mask:
                        # Combine masks for same category (e.g., left+right shoe)
                        if category in result["masks"]:
                            existing = result["masks"][category]
                            combined = np.maximum(np.array(existing), np.array(mask))
                            result["masks"][category] = Image.fromarray(combined.astype(np.uint8))
                        else:
                            result["masks"][category] = mask
            
            logger.info(f"Segmentation complete: {result['raw_labels']}")
            
        except Exception as e:
            logger.error(f"Segmentation error: {e}")
            result["error"] = str(e)
        
        return result
    
    def save_masks(self, masks: Dict[str, Image.Image], output_dir: str) -> Dict[str, str]:
        """Save masks to disk as PNG files."""
        saved = {}
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for category, mask in masks.items():
            filename = f"mask_{category}.png"
            filepath = output_path / filename
            try:
                mask.save(filepath)
                saved[category] = filename
                logger.info(f"Saved mask: {filename}")
            except Exception as e:
                logger.error(f"Failed to save mask {filename}: {e}")
        
        return saved


# Global instance
segmenter = ClothingSegmenter()
