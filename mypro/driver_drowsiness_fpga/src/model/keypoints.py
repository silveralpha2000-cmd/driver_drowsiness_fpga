"""
Shared seven-keypoint configuration for DriverPoseNet7KP.

The seven keypoints MUST follow this exact order everywhere:
  0: Nose
  1: Left Eye
  2: Right Eye
  3: Left Ear
  4: Right Ear
  5: Left Shoulder
  6: Right Shoulder

Do not silently change the order. Import KEYPOINT_NAMES / KEYPOINT_INDEX
from this module in dataset, training, decoding, HLS, RTL, and app code.
"""

from typing import Dict, List, Tuple

KEYPOINT_NAMES: List[str] = [
    "Nose",
    "Left Eye",
    "Right Eye",
    "Left Ear",
    "Right Ear",
    "Left Shoulder",
    "Right Shoulder",
]

KEYPOINT_INDEX: Dict[str, int] = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

NUM_KEYPOINTS: int = len(KEYPOINT_NAMES)

# COCO 17-keypoint indices -> our 7-keypoint indices
# COCO: 0 nose, 1 leye, 2 reye, 3 lear, 4 rear, 5 lsho, 6 rsho, ...
COCO_TO_7KP: Dict[int, int] = {
    0: 0,  # nose
    1: 1,  # left eye
    2: 2,  # right eye
    3: 3,  # left ear
    4: 4,  # right ear
    5: 5,  # left shoulder
    6: 6,  # right shoulder
}

# Skeleton connections (pairs of 7-keypoint indices) for visualization
SKELETON_EDGES: List[Tuple[int, int]] = [
    (0, 1),  # nose - left eye
    (0, 2),  # nose - right eye
    (1, 3),  # left eye - left ear
    (2, 4),  # right eye - right ear
    (1, 5),  # left eye - left shoulder
    (2, 6),  # right eye - right shoulder
    (5, 6),  # left shoulder - right shoulder
]


def validate_keypoint_order(names: List[str]) -> None:
    """Assert that a keypoint name list matches the canonical order exactly."""
    if names != KEYPOINT_NAMES:
        raise ValueError(
            f"Keypoint order mismatch.\n"
            f"Expected: {KEYPOINT_NAMES}\n"
            f"Got:      {names}"
        )


def num_keypoints() -> int:
    return NUM_KEYPOINTS


__all__ = [
    "KEYPOINT_NAMES",
    "KEYPOINT_INDEX",
    "NUM_KEYPOINTS",
    "COCO_TO_7KP",
    "SKELETON_EDGES",
    "validate_keypoint_order",
    "num_keypoints",
]
