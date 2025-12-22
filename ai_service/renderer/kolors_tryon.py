"""
Kolors Virtual Try-On Integration (v1.0.0)
Uses HuggingFace Gradio Client to access free Kolors Try-On API.

This replaces the local SD Inpainting renderer with cloud-based try-on.
"""
import logging
import asyncio
from pathlib import Path
from typing import Optional
from PIL import Image

logger = logging.getLogger(__name__)

# HuggingFace Space for Kolors Virtual Try-On
KOLORS_SPACE = "Kwai-Kolors/Kolors-Virtual-Try-On"


class KolorsTryOn:
    """
    Kolors Virtual Try-On via HuggingFace Gradio Client.
    
    Free tier - may have queue wait times.
    """
    
    _client = None
    _initialized = False
    
    def __init__(self):
        self._init_client()
    
    def _init_client(self):
        """Initialize Gradio client on first use."""
        if self._initialized:
            return
        
        try:
            from gradio_client import Client
            
            logger.info(f"Connecting to Kolors Try-On Space: {KOLORS_SPACE}")
            self._client = Client(KOLORS_SPACE)
            self._initialized = True
            logger.info("✓ Kolors Try-On client initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kolors client: {e}")
            self._initialized = False
    
    async def try_on(
        self,
        person_image_path: str,
        garment_image_path: str,
        output_path: str
    ) -> bool:
        """
        Perform virtual try-on using Kolors API.
        
        Args:
            person_image_path: Path to the person/model image
            garment_image_path: Path to the garment/clothing image
            output_path: Where to save the result
            
        Returns:
            True if successful, False otherwise
        """
        if not self._initialized:
            self._init_client()
        
        if not self._client:
            logger.error("Kolors client not available")
            return False
        
        try:
            logger.info(f"Starting Kolors try-on...")
            logger.info(f"  Person: {person_image_path}")
            logger.info(f"  Garment: {garment_image_path}")
            
            # Run in thread to avoid blocking
            result = await asyncio.to_thread(
                self._client.predict,
                person_image_path,  # Person image
                garment_image_path,  # Garment image
                api_name="/tryon"
            )
            
            if result:
                # Result is typically the path to the generated image
                result_path = Path(result)
                
                if result_path.exists():
                    # Copy to output path
                    output = Path(output_path)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    
                    img = Image.open(result_path)
                    img.save(output_path, quality=95)
                    
                    logger.info(f"✓ Try-on saved to: {output_path}")
                    return True
                else:
                    # Result might be the image data directly
                    logger.warning(f"Unexpected result type: {type(result)}")
                    return False
            
            return False
            
        except Exception as e:
            logger.error(f"Kolors try-on failed: {e}")
            return False


# Singleton instance
_kolors_client: Optional[KolorsTryOn] = None


def get_kolors_client() -> KolorsTryOn:
    """Get or create Kolors client singleton."""
    global _kolors_client
    
    if _kolors_client is None:
        _kolors_client = KolorsTryOn()
    
    return _kolors_client


async def kolors_try_on(
    person_image: str,
    garment_image: str,
    output_path: str
) -> bool:
    """
    Convenience function for Kolors try-on.
    
    Args:
        person_image: Path to person photo
        garment_image: Path to garment to try on
        output_path: Where to save result
        
    Returns:
        True if successful
    """
    client = get_kolors_client()
    return await client.try_on(person_image, garment_image, output_path)
