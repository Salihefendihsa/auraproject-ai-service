"""
User Photo Detection Module (v1.0.0)

ISOLATED EXTENSION: Validates that uploaded image is a human photo.
Does NOT modify any existing try-on logic.

This module is ONLY used when mode == "user_photo_tryon".

Detection Methods:
1. MediaPipe Pose - Checks for body keypoints
2. Face detection fallback - Checks for face presence
3. Aspect ratio heuristics - Basic shape validation

Returns:
{
  "is_person": true | false,
  "confidence": float (0-1),
  "reason": string,
  "keypoints_detected": int (optional)
}

Rules:
- confidence < 0.6 → FAIL
- On FAIL → caller should return HTTP 400 or fallback to mannequin
"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Detection thresholds
PERSON_CONFIDENCE_THRESHOLD = 0.6
MIN_KEYPOINTS_FOR_PERSON = 8  # At least 8 out of 17 keypoints needed


def detect_person_mediapipe(image_path: str) -> Dict[str, Any]:
    """
    Detect if image contains a person using MediaPipe Pose.
    
    This is the PRIMARY detection method - most reliable for full-body photos.
    
    Returns:
        Detection result dict
    """
    try:
        import mediapipe as mp
        from PIL import Image
        import numpy as np
        
        # Load image
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)
        
        # Initialize MediaPipe Pose
        mp_pose = mp.solutions.pose
        
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5
        ) as pose:
            results = pose.process(img_array)
            
            if not results.pose_landmarks:
                return {
                    "is_person": False,
                    "confidence": 0.0,
                    "reason": "No body pose detected in image",
                    "keypoints_detected": 0,
                    "method": "mediapipe_pose"
                }
            
            # Count visible keypoints
            landmarks = results.pose_landmarks.landmark
            visible_keypoints = sum(
                1 for lm in landmarks 
                if lm.visibility > 0.5
            )
            
            # Calculate confidence based on keypoint visibility
            confidence = min(visible_keypoints / 17.0, 1.0)  # 17 keypoints max
            is_person = visible_keypoints >= MIN_KEYPOINTS_FOR_PERSON
            
            if is_person:
                reason = f"Full-body pose detected ({visible_keypoints}/17 keypoints)"
            else:
                reason = f"Partial pose detected ({visible_keypoints}/17 keypoints) - need at least {MIN_KEYPOINTS_FOR_PERSON}"
            
            logger.info(f"[user_photo_detection] MediaPipe: {reason}")
            
            return {
                "is_person": is_person,
                "confidence": round(confidence, 3),
                "reason": reason,
                "keypoints_detected": visible_keypoints,
                "method": "mediapipe_pose"
            }
            
    except ImportError:
        logger.debug("MediaPipe not available for person detection")
        return {
            "is_person": False,
            "confidence": 0.0,
            "reason": "MediaPipe not available",
            "method": "mediapipe_unavailable"
        }
    except Exception as e:
        logger.warning(f"[user_photo_detection] MediaPipe error: {e}")
        return {
            "is_person": False,
            "confidence": 0.0,
            "reason": f"Detection error: {str(e)}",
            "method": "mediapipe_error"
        }


def detect_person_face(image_path: str) -> Dict[str, Any]:
    """
    Fallback: Detect person by face detection.
    
    Less reliable than pose but still useful for partial body shots.
    
    Returns:
        Detection result dict
    """
    try:
        import mediapipe as mp
        from PIL import Image
        import numpy as np
        
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)
        
        mp_face = mp.solutions.face_detection
        
        with mp_face.FaceDetection(
            model_selection=1,  # Full range model
            min_detection_confidence=0.5
        ) as face_detection:
            results = face_detection.process(img_array)
            
            if not results.detections:
                return {
                    "is_person": False,
                    "confidence": 0.0,
                    "reason": "No face detected in image",
                    "method": "face_detection"
                }
            
            # Use highest confidence face
            best_confidence = max(d.score[0] for d in results.detections)
            
            # Face alone isn't enough for try-on, but indicates person
            # Give lower confidence than pose detection
            adjusted_confidence = best_confidence * 0.5
            
            return {
                "is_person": True,
                "confidence": round(adjusted_confidence, 3),
                "reason": f"Face detected (confidence={best_confidence:.2f}) - body pose needed for try-on",
                "method": "face_detection"
            }
            
    except ImportError:
        return {
            "is_person": False,
            "confidence": 0.0,
            "reason": "MediaPipe not available for face detection",
            "method": "face_unavailable"
        }
    except Exception as e:
        logger.warning(f"[user_photo_detection] Face detection error: {e}")
        return {
            "is_person": False,
            "confidence": 0.0,
            "reason": f"Face detection error: {str(e)}",
            "method": "face_error"
        }


def detect_person_heuristic(image_path: str) -> Dict[str, Any]:
    """
    Simple heuristic fallback: Check image aspect ratio.
    
    Full-body photos typically have portrait orientation (taller than wide).
    
    Returns:
        Detection result dict
    """
    try:
        from PIL import Image
        
        img = Image.open(image_path)
        width, height = img.size
        aspect_ratio = width / height
        
        # Portrait orientation (height > width) is more likely a person
        if 0.4 <= aspect_ratio <= 0.8:
            # Strong portrait - likely standing person
            confidence = 0.4
            is_person = True
            reason = f"Portrait aspect ratio ({aspect_ratio:.2f}) suggests standing person"
        elif 0.8 < aspect_ratio <= 1.2:
            # Square-ish - could be person
            confidence = 0.25
            is_person = False
            reason = f"Square aspect ratio ({aspect_ratio:.2f}) - uncertain if person"
        else:
            # Landscape - less likely full-body person
            confidence = 0.1
            is_person = False
            reason = f"Landscape aspect ratio ({aspect_ratio:.2f}) - unlikely full-body photo"
        
        return {
            "is_person": is_person,
            "confidence": confidence,
            "reason": reason,
            "method": "aspect_ratio_heuristic"
        }
        
    except Exception as e:
        return {
            "is_person": False,
            "confidence": 0.0,
            "reason": f"Heuristic check failed: {str(e)}",
            "method": "heuristic_error"
        }


def detect_user_photo(image_path: str) -> Dict[str, Any]:
    """
    Main entry point: Detect if image is a valid human photo for try-on.
    
    Tries detection methods in order of reliability:
    1. MediaPipe Pose (best - detects full body)
    2. Face detection (fallback - at least confirms person)
    3. Aspect ratio heuristic (last resort)
    
    Args:
        image_path: Path to the uploaded user photo
        
    Returns:
        Detection result dict with:
        - is_person: bool
        - confidence: float (0-1)
        - reason: str
        - method: str
        - is_valid_for_tryon: bool (confidence >= threshold)
    """
    if not image_path or not Path(image_path).exists():
        return {
            "is_person": False,
            "confidence": 0.0,
            "reason": "Image file not found",
            "method": "validation",
            "is_valid_for_tryon": False
        }
    
    # Try MediaPipe Pose first (most reliable)
    result = detect_person_mediapipe(image_path)
    
    if result["is_person"] and result["confidence"] >= PERSON_CONFIDENCE_THRESHOLD:
        result["is_valid_for_tryon"] = True
        logger.info(f"[user_photo_detection] Valid person detected: {result['reason']}")
        return result
    
    # Try face detection as fallback
    face_result = detect_person_face(image_path)
    if face_result["confidence"] > result["confidence"]:
        result = face_result
    
    # Try heuristic as last resort
    if result["confidence"] < 0.3:
        heuristic_result = detect_person_heuristic(image_path)
        if heuristic_result["confidence"] > result["confidence"]:
            result = heuristic_result
    
    # Final validation
    result["is_valid_for_tryon"] = (
        result["is_person"] and 
        result["confidence"] >= PERSON_CONFIDENCE_THRESHOLD
    )
    
    if result["is_valid_for_tryon"]:
        logger.info(f"[user_photo_detection] Valid for try-on: {result['reason']}")
    else:
        logger.warning(f"[user_photo_detection] Not valid for try-on: {result['reason']}")
    
    return result


def validate_user_photo_for_tryon(image_path: str) -> Tuple[bool, str, float]:
    """
    Convenience function: Validate user photo and return simple result.
    
    Args:
        image_path: Path to user photo
        
    Returns:
        Tuple of (is_valid, reason, confidence)
    """
    result = detect_user_photo(image_path)
    return (
        result["is_valid_for_tryon"],
        result["reason"],
        result["confidence"]
    )
