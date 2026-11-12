"""
Gaussian heatmap generation for seven-keypoint pose estimation.

Generates 64x48 heatmaps from 2D keypoint coordinates in 256x192 image space.
Output stride: 4 (256/64 = 4, 192/48 = 4).
"""

import numpy as np
from typing import List, Tuple, Optional

OUTPUT_STRIDE: int = 4


def generate_gaussian_heatmap(
    heatmap_size: Tuple[int, int],
    center: Tuple[float, float],
    sigma: float = 2.0,
) -> np.ndarray:
    """
    Generate a 2D Gaussian heatmap centered at ``center``.

    Args:
        heatmap_size: (H, W) of the heatmap.
        center: (x, y) center in heatmap coordinates.
        sigma: Gaussian standard deviation in heatmap pixels.

    Returns:
        Heatmap array of shape (H, W), dtype float32, values in [0, 1].
    """
    h, w = heatmap_size
    cx, cy = center

    # If center is outside the heatmap, return zeros
    if cx < 0 or cx >= w or cy < 0 or cy >= h:
        return np.zeros((h, w), dtype=np.float32)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    heatmap = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    heatmap = heatmap.astype(np.float32)

    # Normalize peak to 1
    heatmap /= heatmap.max() + 1e-8
    return heatmap


def generate_heatmaps(
    keypoints: np.ndarray,
    heatmap_size: Tuple[int, int] = (64, 48),
    sigma: float = 2.0,
    image_size: Tuple[int, int] = (256, 192),
) -> np.ndarray:
    """
    Generate heatmaps for all seven keypoints.

    Args:
        keypoints: Array of shape (7, 3) with [x, y, visibility] in image-space
                   coordinates (0..255, 0..191). Visibility: 0=unlabeled,
                   1=labeled-not-visible, 2=visible.
        heatmap_size: (H, W) of output heatmaps.
        sigma: Gaussian sigma in heatmap pixels.
        image_size: (H, W) of the input image.

    Returns:
        Heatmaps array of shape (7, H, W), dtype float32.
    """
    num_kps = keypoints.shape[0]
    h, w = heatmap_size
    heatmaps = np.zeros((num_kps, h, w), dtype=np.float32)

    img_h, img_w = image_size

    for kp_idx in range(num_kps):
        x, y, vis = keypoints[kp_idx]

        # Skip invisible/unlabeled keypoints (leave heatmap as zeros)
        if vis <= 0:
            continue

        # Map image-space coordinates to heatmap-space coordinates
        hx = x / img_w * w
        hy = y / img_h * h

        heatmaps[kp_idx] = generate_gaussian_heatmap(heatmap_size, (hx, hy), sigma)

    return heatmaps


def decode_heatmaps(
    heatmaps: np.ndarray,
    image_size: Tuple[int, int] = (256, 192),
) -> np.ndarray:
    """
    Decode heatmaps back to keypoint coordinates in image space.

    Args:
        heatmaps: Array of shape (7, H, W).
        image_size: (H, W) of the output image space.

    Returns:
        Array of shape (7, 3) with [x, y, confidence] in image-space coordinates.
    """
    num_kps, h, w = heatmaps.shape
    img_h, img_w = image_size
    keypoints = np.zeros((num_kps, 3), dtype=np.float32)

    for kp_idx in range(num_kps):
        hm = heatmaps[kp_idx]
        max_idx = np.argmax(hm)
        cy, cx = np.unravel_index(max_idx, hm.shape)
        confidence = float(hm[cy, cx])

        # Map heatmap-space back to image-space
        x = cx / w * img_w
        y = cy / h * img_h

        keypoints[kp_idx] = [x, y, confidence]

    return keypoints
