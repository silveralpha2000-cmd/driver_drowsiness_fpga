"""
Visualization utilities for seven-keypoint pose estimation dataset.
"""

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from ..model.keypoints import KEYPOINT_NAMES, SKELETON_EDGES


def _denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert [3, H, W] tensor in [0,1] to HWC uint8 BGR for OpenCV."""
    image = image_tensor.permute(1, 2, 0).numpy()
    image = (image * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def visualize_sample(
    image: torch.Tensor,
    heatmaps: torch.Tensor,
    keypoints: torch.Tensor,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Create a composite visualization of a dataset sample.

    Layout (2 rows x 4 columns):
        Row 1: Original image | Resized image | Keypoints overlay | Skeleton overlay
        Row 2: Heatmap 0 (Nose) | Heatmap 1 (Left Eye) | Heatmap 2 (Right Eye) | Heatmap 3 (Left Ear)
        Row 3: Heatmap 4 (Right Ear) | Heatmap 5 (Left Shoulder) | Heatmap 6 (Right Shoulder) | Heatmap maxima overlay

    Args:
        image: [3, 256, 192] tensor in [0, 1].
        heatmaps: [7, 64, 48] tensor.
        keypoints: [7, 3] tensor (x, y, visibility) in 256x192 space.
        save_path: If provided, save the composite to this path.

    Returns:
        Composite BGR image (uint8).
    """
    img = _denormalize_image(image)
    h, w = img.shape[:2]

    # Resize heatmaps to image size for overlay
    heatmaps_np = heatmaps.numpy()
    keypoints_np = keypoints.numpy()

    # --- Keypoints overlay ---
    kp_img = img.copy()
    for i in range(heatmaps_np.shape[0]):
        x, y, vis = keypoints_np[i]
        if vis > 0:
            cx, cy = int(x), int(y)
            color = (0, 255, 0) if vis > 0 else (0, 0, 255)
            cv2.circle(kp_img, (cx, cy), 4, color, -1)
            cv2.putText(
                kp_img, str(i), (cx + 5, cy - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )

    # --- Skeleton overlay ---
    skel_img = img.copy()
    for a, b in SKELETON_EDGES:
        x1, y1, v1 = keypoints_np[a]
        x2, y2, v2 = keypoints_np[b]
        if v1 > 0 and v2 > 0:
            cv2.line(skel_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
    for i in range(heatmaps_np.shape[0]):
        x, y, vis = keypoints_np[i]
        if vis > 0:
            cv2.circle(skel_img, (int(x), int(y)), 4, (0, 255, 0), -1)

    # --- Heatmaps ---
    heatmap_imgs = []
    for i in range(heatmaps_np.shape[0]):
        hm = heatmaps_np[i]
        hm_resized = cv2.resize(hm, (w, h), interpolation=cv2.INTER_LINEAR)
        hm_colored = cv2.applyColorMap((hm_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        # Blend with original
        blended = cv2.addWeighted(img, 0.6, hm_colored, 0.4, 0)
        cv2.putText(
            blended, KEYPOINT_NAMES[i], (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
        heatmap_imgs.append(blended)

    # --- Heatmap maxima overlay ---
    maxima_img = img.copy()
    for i in range(heatmaps_np.shape[0]):
        hm = heatmaps_np[i]
        max_idx = np.argmax(hm)
        cy, cx = np.unravel_index(max_idx, hm.shape)
        # Map heatmap coords to image coords
        mx = cx / hm.shape[1] * w
        my = cy / hm.shape[0] * h
        cv2.circle(maxima_img, (int(mx), int(my)), 5, (0, 255, 0), -1)
        cv2.putText(
            maxima_img, str(i), (int(mx) + 5, int(my) - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

    # --- Build composite ---
    # Row 1: original, resized(=original), keypoints, skeleton
    # Row 2-3: heatmaps (7 of them)
    # Row 4: maxima overlay + padding

    # Pad all to same size
    pad = 10
    cell_w = w
    cell_h = h

    # Row 1
    row1 = np.hstack([img, img, kp_img, skel_img])

    # Rows 2 and 3: 4 heatmaps each
    row2 = np.hstack(heatmap_imgs[:4])
    row3 = np.hstack(heatmap_imgs[4:] + [np.zeros_like(img)])  # pad last cell

    # Row 4: maxima overlay + blank
    row4 = np.hstack([maxima_img, np.zeros_like(img), np.zeros_like(img), np.zeros_like(img)])

    composite = np.vstack([row1, row2, row3, row4])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, composite)

    return composite


def save_visualization(
    image: torch.Tensor,
    heatmaps: torch.Tensor,
    keypoints: torch.Tensor,
    output_dir: str,
    prefix: str = "sample",
) -> str:
    """
    Save a visualization composite to disk.

    Returns the saved file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{prefix}.jpg")
    visualize_sample(image, heatmaps, keypoints, save_path=save_path)
    return save_path
