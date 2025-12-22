"""
ControlNet Module (v2.6.0)
Pose-locking conditioning for improved try-on stability.

Lazy-loads ControlNet models. Fails gracefully if GPU unavailable.
"""
import logging
from typing import Optional, Any, Dict
from pathlib import Path
from dataclasses import dataclass

from ai_service.config.llm_config import get_controlnet_config, ControlNetType

logger = logging.getLogger(__name__)


@dataclass
class ConditioningData:
    """ControlNet conditioning data."""
    condition_image: Any  # PIL Image or tensor
    control_type: str
    scale: float
    ready: bool = True
    error: Optional[str] = None


class ControlNetProcessor:
    """
    Singleton ControlNet processor with lazy loading.
    
    Extracts pose/edge/depth from input images for conditioning.
    """
    
    _instance = None
    _controlnet = None
    _processor = None
    _is_loaded = False
    _load_failed = False
    _device = "cpu"
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def is_enabled(self) -> bool:
        """Check if ControlNet is enabled via config."""
        config = get_controlnet_config()
        return config.enabled
    
    def _load_models(self):
        """Lazy-load ControlNet models."""
        if self._is_loaded or self._load_failed:
            return
        
        config = get_controlnet_config()
        if not config.enabled:
            return
        
        try:
            logger.info("Loading ControlNet models (requires GPU)...")
            
            import torch
            from diffusers import ControlNetModel
            
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            
            if self._device == "cpu":
                logger.warning("ControlNet requires GPU. Disabling.")
                self._load_failed = True
                return
            
            model_id = config.get_model_id()
            logger.info(f"Loading ControlNet: {model_id}")
            
            self._controlnet = ControlNetModel.from_pretrained(
                model_id,
                torch_dtype=torch.float16
            ).to(self._device)
            
            # Load processor based on type
            if config.control_type == ControlNetType.POSE:
                self._load_pose_processor()
            elif config.control_type == ControlNetType.EDGE:
                self._load_edge_processor()
            elif config.control_type == ControlNetType.DEPTH:
                self._load_depth_processor()
            
            self._is_loaded = True
            logger.info("✓ ControlNet loaded successfully")
            
        except Exception as e:
            logger.error(f"ControlNet load failed: {e}")
            self._load_failed = True
    
    def _load_pose_processor(self):
        """Load OpenPose processor."""
        try:
            from controlnet_aux import OpenposeDetector
            self._processor = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
            logger.info("✓ OpenPose processor loaded")
        except ImportError:
            logger.warning("controlnet_aux not installed, using fallback")
            self._processor = None
    
    def _load_edge_processor(self):
        """Load Canny edge processor."""
        # Canny is built-in, no external model needed
        self._processor = "canny"
        logger.info("✓ Canny edge processor ready")
    
    def _load_depth_processor(self):
        """Load depth estimator."""
        try:
            from transformers import pipeline
            self._processor = pipeline("depth-estimation")
            logger.info("✓ Depth processor loaded")
        except Exception as e:
            logger.warning(f"Depth processor failed: {e}")
            self._processor = None
    
    def extract_condition(self, image_path: str) -> Optional[ConditioningData]:
        """
        Extract conditioning data from input image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            ConditioningData or None if failed/disabled
        """
        config = get_controlnet_config()
        
        if not config.enabled:
            return None
        
        self._load_models()
        
        if self._load_failed or not self._is_loaded:
            return None
        
        try:
            from PIL import Image
            image = Image.open(image_path).convert("RGB")
            
            if config.control_type == ControlNetType.POSE:
                condition = self._extract_pose(image)
            elif config.control_type == ControlNetType.EDGE:
                condition = self._extract_edge(image)
            elif config.control_type == ControlNetType.DEPTH:
                condition = self._extract_depth(image)
            else:
                return None
            
            if condition is None:
                return None
            
            return ConditioningData(
                condition_image=condition,
                control_type=config.control_type.value,
                scale=config.conditioning_scale
            )
            
        except Exception as e:
            logger.error(f"Condition extraction failed: {e}")
            return ConditioningData(
                condition_image=None,
                control_type=config.control_type.value,
                scale=config.conditioning_scale,
                ready=False,
                error=str(e)
            )
    
    def _extract_pose(self, image) -> Optional[Any]:
        """Extract pose using OpenPose."""
        if self._processor is None:
            return None
        try:
            pose_image = self._processor(image)
            return pose_image
        except Exception as e:
            logger.error(f"Pose extraction failed: {e}")
            return None
    
    def _extract_edge(self, image) -> Optional[Any]:
        """Extract edges using Canny."""
        try:
            import numpy as np
            import cv2
            from PIL import Image as PILImage
            
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_image = PILImage.fromarray(edges)
            return edge_image
        except Exception as e:
            logger.error(f"Edge extraction failed: {e}")
            return None
    
    def _extract_depth(self, image) -> Optional[Any]:
        """Extract depth map."""
        if self._processor is None:
            return None
        try:
            depth = self._processor(image)["depth"]
            return depth
        except Exception as e:
            logger.error(f"Depth extraction failed: {e}")
            return None
    
    def get_controlnet_model(self) -> Optional[Any]:
        """Get loaded ControlNet model."""
        if not self._is_loaded:
            self._load_models()
        return self._controlnet
    
    def get_status(self) -> dict:
        """Get ControlNet status."""
        config = get_controlnet_config()
        return {
            "enabled": config.enabled,
            "type": config.control_type.value if config.enabled else None,
            "loaded": self._is_loaded,
            "failed": self._load_failed,
            "device": self._device if self._is_loaded else None
        }


# Global instance
controlnet_processor = ControlNetProcessor()


def load_controlnet_condition(input_image: str) -> Optional[ConditioningData]:
    """
    Main entry point: Extract ControlNet conditioning from image.
    
    Args:
        input_image: Path to input image
        
    Returns:
        ConditioningData or None if disabled/failed
    """
    return controlnet_processor.extract_condition(input_image)


def is_controlnet_enabled() -> bool:
    """Check if ControlNet is enabled."""
    return controlnet_processor.is_enabled()


def get_controlnet_status() -> dict:
    """Get ControlNet status for health endpoint."""
    return controlnet_processor.get_status()
