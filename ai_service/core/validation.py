"""
Input Validation Module (v2.0.0)
Validates uploaded images before processing.
"""
import io
import logging
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image, ExifTags

logger = logging.getLogger(__name__)

# Configuration
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class ValidationError(Exception):
    """Custom exception for validation errors."""
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


def validate_file_size(content: bytes) -> None:
    """
    Check if file size is within limits.
    
    Raises:
        ValidationError: If file exceeds MAX_FILE_SIZE_MB
    """
    size_mb = len(content) / (1024 * 1024)
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise ValidationError(
            f"File too large: {size_mb:.1f}MB (max {MAX_FILE_SIZE_MB}MB)",
            status_code=413
        )
    logger.debug(f"File size OK: {size_mb:.2f}MB")


def validate_mime_type(content_type: Optional[str]) -> None:
    """
    Check if MIME type is allowed.
    
    Raises:
        ValidationError: If MIME type is not in ALLOWED_MIME_TYPES
    """
    if content_type is None:
        raise ValidationError("Missing Content-Type header", status_code=415)
    
    # Normalize content type (remove charset etc.)
    mime = content_type.split(";")[0].strip().lower()
    
    if mime not in ALLOWED_MIME_TYPES:
        raise ValidationError(
            f"Unsupported file type: {mime}. Allowed: {', '.join(ALLOWED_MIME_TYPES)}",
            status_code=415
        )
    logger.debug(f"MIME type OK: {mime}")


def decode_image(content: bytes) -> Image.Image:
    """
    Decode image bytes to PIL Image.
    
    Raises:
        ValidationError: If image cannot be decoded
    """
    try:
        image = Image.open(io.BytesIO(content))
        image.load()  # Force load to catch truncated images
        return image
    except Exception as e:
        raise ValidationError(
            f"Cannot decode image: {str(e)}",
            status_code=400
        )


def fix_exif_orientation(image: Image.Image) -> Image.Image:
    """
    Fix image orientation based on EXIF data.
    
    Returns:
        Properly oriented PIL Image
    """
    try:
        # Get EXIF orientation tag
        exif = image._getexif()
        if exif is None:
            return image
        
        # Find orientation tag
        orientation_key = None
        for tag, name in ExifTags.TAGS.items():
            if name == 'Orientation':
                orientation_key = tag
                break
        
        if orientation_key is None or orientation_key not in exif:
            return image
        
        orientation = exif[orientation_key]
        
        # Apply rotation/flip based on orientation
        if orientation == 2:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            image = image.rotate(180)
        elif orientation == 4:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        elif orientation == 5:
            image = image.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            image = image.rotate(-90, expand=True)
        elif orientation == 7:
            image = image.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            image = image.rotate(90, expand=True)
        
        logger.debug(f"Fixed EXIF orientation: {orientation}")
        return image
        
    except Exception as e:
        logger.warning(f"Could not fix EXIF orientation: {e}")
        return image


def sanitize_asset_path(path: str, base_dir: Path) -> Path:
    """
    Sanitize asset path to prevent path traversal attacks.
    
    Args:
        path: Requested file path
        base_dir: Base directory for assets
    
    Returns:
        Safe absolute path
    
    Raises:
        ValidationError: If path traversal detected
    """
    # Normalize path
    requested_path = Path(path).resolve()
    base_resolved = base_dir.resolve()
    
    # Check if path is within base directory
    try:
        requested_path.relative_to(base_resolved)
    except ValueError:
        raise ValidationError(
            "Access denied: path traversal detected",
            status_code=403
        )
    
    # Additional checks for suspicious patterns
    suspicious_patterns = ['..', '~', '$', '%']
    for pattern in suspicious_patterns:
        if pattern in path:
            raise ValidationError(
                f"Access denied: suspicious path pattern '{pattern}'",
                status_code=403
            )
    
    return requested_path


async def validate_image_upload(file, content_type: Optional[str]) -> Tuple[bytes, Image.Image]:
    """
    Complete validation pipeline for uploaded images.
    
    Args:
        file: File-like object (UploadFile.file)
        content_type: MIME type from request
    
    Returns:
        Tuple of (validated bytes, PIL Image)
    
    Raises:
        ValidationError: If any validation fails
    """
    # Read file content
    content = await file.read()
    await file.seek(0)  # Reset for potential re-read
    
    # Validate size
    validate_file_size(content)
    
    # Validate MIME type
    validate_mime_type(content_type)
    
    # Decode image
    image = decode_image(content)
    
    # Fix orientation
    image = fix_exif_orientation(image)
    
    logger.info(f"Image validated: {image.size[0]}x{image.size[1]}, {image.mode}")
    
    return content, image


def validate_image_upload_sync(content: bytes, content_type: Optional[str]) -> Image.Image:
    """
    Synchronous version of validate_image_upload.
    
    Args:
        content: File bytes
        content_type: MIME type
    
    Returns:
        Validated PIL Image
    """
    validate_file_size(content)
    validate_mime_type(content_type)
    image = decode_image(content)
    image = fix_exif_orientation(image)
    
    logger.info(f"Image validated: {image.size[0]}x{image.size[1]}, {image.mode}")
    return image


# ==================== OUTFIT SEED VALIDATION ====================

ALLOWED_GENDERS = {"male", "female"}
ALLOWED_EVENTS = {"work", "date", "party", "casual"}
ALLOWED_SEASONS = {"summer", "winter"}
# ISOLATED EXTENSION: user_photo_tryon added without modifying existing modes
ALLOWED_MODES = {"mock", "partial_tryon", "full_tryon", "user_photo_tryon"}


def validate_outfit_seed_input(
    gender: Optional[str],
    seed_image_provided: bool,
    person_image_provided: bool,
    event: Optional[str] = None,
    season: Optional[str] = None,
    mode: Optional[str] = None
) -> dict:
    """
    Validate input for POST /ai/outfit-seed endpoint.
    
    Args:
        gender: Required gender field
        seed_image_provided: Whether seed_image was uploaded
        person_image_provided: Whether person_image was uploaded
        event: Optional event type
        season: Optional season
        mode: Optional mode (defaults to "mock")
    
    Returns:
        Dict with validated/normalized values
    
    Raises:
        ValidationError: If validation fails
    """
    errors = []
    
    # Gender is mandatory
    if not gender:
        errors.append("gender is required")
    elif gender.lower() not in ALLOWED_GENDERS:
        errors.append(f"gender must be one of: {', '.join(ALLOWED_GENDERS)}")
    
    # At least one image required
    if not seed_image_provided and not person_image_provided:
        errors.append("At least one of seed_image or person_image is required")
    
    # Validate event if provided
    if event and event.lower() not in ALLOWED_EVENTS:
        errors.append(f"event must be one of: {', '.join(ALLOWED_EVENTS)}")
    
    # Validate season if provided
    if season and season.lower() not in ALLOWED_SEASONS:
        errors.append(f"season must be one of: {', '.join(ALLOWED_SEASONS)}")
    
    # Validate mode
    normalized_mode = (mode or "mock").lower()
    if normalized_mode not in ALLOWED_MODES:
        errors.append(f"mode must be one of: {', '.join(ALLOWED_MODES)}")
    
    if errors:
        raise ValidationError("; ".join(errors), status_code=400)
    
    return {
        "gender": gender.lower(),
        "event": event.lower() if event else None,
        "season": season.lower() if season else None,
        "mode": normalized_mode
    }
