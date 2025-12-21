"""
Clothing Attributes Extractor (v1.1.1)
Uses CLIP models to extract type, color, and style from cropped clothing images.

Models:
- openai/clip-vit-large-patch14 (type + color)
- patrickjohncyh/fashion-clip (style)
"""
import logging
from typing import Dict, Any, Optional
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


# Clothing type candidates per category
CLOTHING_TYPES = {
    "top": [
        "t-shirt", "shirt", "blouse", "sweater", "hoodie", 
        "tank top", "polo shirt", "cardigan", "turtleneck", "crop top"
    ],
    "bottom": [
        "jeans", "pants", "trousers", "shorts", "skirt",
        "leggings", "chinos", "joggers", "cargo pants", "dress pants"
    ],
    "outerwear": [
        "jacket", "coat", "blazer", "hoodie", "cardigan",
        "bomber jacket", "denim jacket", "leather jacket", "parka", "windbreaker"
    ],
    "shoes": [
        "sneakers", "boots", "loafers", "sandals", "heels",
        "flats", "oxford shoes", "running shoes", "ankle boots", "slip-ons"
    ]
}

# Color candidates
COLORS = [
    "white", "black", "gray", "navy blue", "light blue", "red", "pink",
    "green", "olive", "brown", "beige", "cream", "yellow", "orange",
    "purple", "burgundy", "teal", "coral", "maroon", "khaki"
]

# Style candidates
STYLES = ["casual", "smart", "sporty", "street", "minimal", "formal", "bohemian"]


class ClothingAttributeExtractor:
    """
    Singleton class for extracting clothing attributes using CLIP models.
    Lazy loading - models only load on first use.
    """
    
    _instance = None
    _clip_model = None
    _clip_processor = None
    _fashion_clip_model = None
    _fashion_clip_processor = None
    _is_loaded = False
    _load_failed = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _load_models(self):
        """Load CLIP models on first use."""
        if self._is_loaded or self._load_failed:
            return
        
        try:
            logger.info("Loading CLIP models for attribute extraction...")
            
            import torch
            from transformers import CLIPProcessor, CLIPModel
            
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self._device}")
            
            # Load OpenAI CLIP for type and color
            logger.info("Loading openai/clip-vit-large-patch14...")
            self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
            self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self._clip_model.to(self._device)
            self._clip_model.eval()
            
            # Load Fashion-CLIP for style
            logger.info("Loading patrickjohncyh/fashion-clip...")
            self._fashion_clip_model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
            self._fashion_clip_processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
            self._fashion_clip_model.to(self._device)
            self._fashion_clip_model.eval()
            
            self._is_loaded = True
            logger.info("✓ CLIP models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load CLIP models: {e}")
            self._load_failed = True
    
    def _classify_with_clip(
        self,
        image: Image.Image,
        candidates: list,
        use_fashion_clip: bool = False
    ) -> str:
        """Classify image against candidate labels using CLIP."""
        import torch
        
        model = self._fashion_clip_model if use_fashion_clip else self._clip_model
        processor = self._fashion_clip_processor if use_fashion_clip else self._clip_processor
        
        # Prepare inputs
        inputs = processor(
            text=candidates,
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits_per_image
            probs = logits.softmax(dim=1)
            
        # Get best match
        best_idx = probs.argmax().item()
        return candidates[best_idx]
    
    def extract_attributes(
        self,
        cropped_image: Image.Image,
        category: str
    ) -> Dict[str, Any]:
        """
        Extract type, color, and style from a cropped clothing image.
        
        Args:
            cropped_image: PIL Image of the cropped clothing item
            category: Clothing category (top, bottom, outerwear, shoes)
            
        Returns:
            Dict with type, color, style
        """
        result = {
            "type": "unknown",
            "color": "unknown",
            "style": "casual"
        }
        
        # Try to load models
        self._load_models()
        
        if not self._is_loaded:
            logger.warning("CLIP models not available, returning defaults")
            return result
        
        try:
            # Get type candidates for this category
            type_candidates = CLOTHING_TYPES.get(category, CLOTHING_TYPES["top"])
            
            # Classify type
            result["type"] = self._classify_with_clip(
                cropped_image,
                type_candidates,
                use_fashion_clip=False
            )
            
            # Classify color
            color_prompts = [f"a {c} colored clothing item" for c in COLORS]
            color_result = self._classify_with_clip(
                cropped_image,
                color_prompts,
                use_fashion_clip=False
            )
            # Extract color name from prompt
            result["color"] = COLORS[color_prompts.index(color_result)]
            
            # Classify style using Fashion-CLIP
            style_prompts = [f"a {s} style outfit" for s in STYLES]
            style_result = self._classify_with_clip(
                cropped_image,
                style_prompts,
                use_fashion_clip=True
            )
            result["style"] = STYLES[style_prompts.index(style_result)]
            
            logger.info(f"Attributes for {category}: {result}")
            
        except Exception as e:
            logger.error(f"Attribute extraction failed: {e}")
        
        return result


def crop_clothing_from_mask(
    image: Image.Image,
    mask: Image.Image
) -> Optional[Image.Image]:
    """
    Crop the clothing region from image using the mask.
    
    Args:
        image: Original PIL Image
        mask: Binary mask (PIL Image)
        
    Returns:
        Cropped clothing region as PIL Image
    """
    try:
        # Convert mask to numpy
        mask_array = np.array(mask.convert("L"))
        
        # Find bounding box of non-zero pixels
        rows = np.any(mask_array > 0, axis=1)
        cols = np.any(mask_array > 0, axis=0)
        
        if not rows.any() or not cols.any():
            return None
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Add some padding
        padding = 10
        rmin = max(0, rmin - padding)
        rmax = min(mask_array.shape[0], rmax + padding)
        cmin = max(0, cmin - padding)
        cmax = min(mask_array.shape[1], cmax + padding)
        
        # Resize mask to match image if needed
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
            mask_array = np.array(mask.convert("L"))
        
        # Crop the image
        cropped = image.crop((cmin, rmin, cmax, rmax))
        
        return cropped
        
    except Exception as e:
        logger.error(f"Failed to crop clothing: {e}")
        return None


# Global instance
attribute_extractor = ClothingAttributeExtractor()
