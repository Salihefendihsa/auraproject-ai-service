"""
Pose Estimation Module (v3.0.0)

Extracts body keypoints for garment warping and alignment.
Uses MediaPipe when available, falls back to rule-based estimation.
"""
import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any

try:
    from PIL import Image
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Try to import MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

logger = logging.getLogger(__name__)

# Standard pose keypoint names (COCO-style)
KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye", 
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def estimate_pose_mediapipe(image_path: str) -> Optional[Dict[str, Tuple[float, float, float]]]:
    """
    Estimate pose using MediaPipe.
    
    Returns:
        Dict mapping keypoint names to (x, y, confidence) tuples.
        Coordinates are normalized [0, 1].
    """
    if not MEDIAPIPE_AVAILABLE or not PIL_AVAILABLE:
        return None
    
    try:
        # Initialize MediaPipe Pose
        mp_pose = mp.solutions.pose
        
        # Load image
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)
        
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5
        ) as pose:
            results = pose.process(img_array)
            
            if not results.pose_landmarks:
                logger.warning("MediaPipe: No pose detected")
                return None
            
            landmarks = results.pose_landmarks.landmark
            
            # Map MediaPipe landmarks to our keypoint names
            # MediaPipe uses different indices
            mp_to_coco = {
                0: "nose",
                2: "left_eye",
                5: "right_eye",
                7: "left_ear",
                8: "right_ear",
                11: "left_shoulder",
                12: "right_shoulder",
                13: "left_elbow",
                14: "right_elbow",
                15: "left_wrist",
                16: "right_wrist",
                23: "left_hip",
                24: "right_hip",
                25: "left_knee",
                26: "right_knee",
                27: "left_ankle",
                28: "right_ankle",
            }
            
            keypoints = {}
            for mp_idx, name in mp_to_coco.items():
                if mp_idx < len(landmarks):
                    lm = landmarks[mp_idx]
                    keypoints[name] = (lm.x, lm.y, lm.visibility)
            
            logger.info(f"MediaPipe detected {len(keypoints)} keypoints")
            return keypoints
            
    except Exception as e:
        logger.error(f"MediaPipe pose estimation failed: {e}")
        return None


def estimate_pose_simple(image_path: str) -> Dict[str, Tuple[float, float, float]]:
    """
    Simple rule-based pose estimation (fallback).
    
    Uses typical standing pose proportions.
    Good for mannequins and standardized poses.
    
    Returns:
        Dict mapping keypoint names to (x, y, confidence) tuples.
        Coordinates are normalized [0, 1].
    """
    # Standard standing pose proportions
    # Based on typical human body ratios
    keypoints = {
        "nose": (0.5, 0.08, 1.0),
        "left_eye": (0.45, 0.06, 1.0),
        "right_eye": (0.55, 0.06, 1.0),
        "left_ear": (0.40, 0.08, 1.0),
        "right_ear": (0.60, 0.08, 1.0),
        "left_shoulder": (0.30, 0.18, 1.0),
        "right_shoulder": (0.70, 0.18, 1.0),
        "left_elbow": (0.18, 0.35, 1.0),
        "right_elbow": (0.82, 0.35, 1.0),
        "left_wrist": (0.15, 0.50, 1.0),
        "right_wrist": (0.85, 0.50, 1.0),
        "left_hip": (0.38, 0.55, 1.0),
        "right_hip": (0.62, 0.55, 1.0),
        "left_knee": (0.35, 0.75, 1.0),
        "right_knee": (0.65, 0.75, 1.0),
        "left_ankle": (0.33, 0.95, 1.0),
        "right_ankle": (0.67, 0.95, 1.0),
    }
    
    logger.info("Using simple pose estimation (fallback)")
    return keypoints


def estimate_pose(image_path: str) -> Dict[str, Tuple[float, float, float]]:
    """
    Estimate pose keypoints for an image.
    
    Tries MediaPipe first, falls back to simple estimation.
    
    Returns:
        Dict mapping keypoint names to (x, y, confidence) tuples.
    """
    # Try MediaPipe first
    if MEDIAPIPE_AVAILABLE:
        result = estimate_pose_mediapipe(image_path)
        if result:
            return result
    
    # Fallback to simple estimation
    return estimate_pose_simple(image_path)


def get_torso_keypoints(pose: Dict[str, Tuple[float, float, float]]) -> Dict[str, Tuple[float, float]]:
    """Extract key torso points for garment warping."""
    return {
        "left_shoulder": (pose["left_shoulder"][0], pose["left_shoulder"][1]),
        "right_shoulder": (pose["right_shoulder"][0], pose["right_shoulder"][1]),
        "left_hip": (pose["left_hip"][0], pose["left_hip"][1]),
        "right_hip": (pose["right_hip"][0], pose["right_hip"][1]),
    }


def get_arm_keypoints(pose: Dict[str, Tuple[float, float, float]], side: str) -> Dict[str, Tuple[float, float]]:
    """Extract arm keypoints for sleeve warping."""
    prefix = f"{side}_"
    return {
        "shoulder": (pose[f"{prefix}shoulder"][0], pose[f"{prefix}shoulder"][1]),
        "elbow": (pose[f"{prefix}elbow"][0], pose[f"{prefix}elbow"][1]),
        "wrist": (pose[f"{prefix}wrist"][0], pose[f"{prefix}wrist"][1]),
    }


def get_leg_keypoints(pose: Dict[str, Tuple[float, float, float]], side: str) -> Dict[str, Tuple[float, float]]:
    """Extract leg keypoints for pants warping."""
    prefix = f"{side}_"
    return {
        "hip": (pose[f"{prefix}hip"][0], pose[f"{prefix}hip"][1]),
        "knee": (pose[f"{prefix}knee"][0], pose[f"{prefix}knee"][1]),
        "ankle": (pose[f"{prefix}ankle"][0], pose[f"{prefix}ankle"][1]),
    }


def calculate_body_measurements(
    pose: Dict[str, Tuple[float, float, float]],
    image_size: Tuple[int, int]
) -> Dict[str, float]:
    """
    Calculate body measurements from pose keypoints.
    
    Returns:
        Dict with measurements in pixels.
    """
    width, height = image_size
    
    def dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        dx = (p1[0] - p2[0]) * width
        dy = (p1[1] - p2[1]) * height
        return (dx**2 + dy**2) ** 0.5
    
    ls = (pose["left_shoulder"][0], pose["left_shoulder"][1])
    rs = (pose["right_shoulder"][0], pose["right_shoulder"][1])
    lh = (pose["left_hip"][0], pose["left_hip"][1])
    rh = (pose["right_hip"][0], pose["right_hip"][1])
    
    shoulder_width = dist(ls, rs)
    hip_width = dist(lh, rh)
    torso_height = dist(
        ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2),
        ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
    )
    
    return {
        "shoulder_width": shoulder_width,
        "hip_width": hip_width,
        "torso_height": torso_height,
        "torso_center_x": (ls[0] + rs[0]) / 2 * width,
        "torso_center_y": ((ls[1] + rs[1]) / 2 + (lh[1] + rh[1]) / 2) / 2 * height,
    }
