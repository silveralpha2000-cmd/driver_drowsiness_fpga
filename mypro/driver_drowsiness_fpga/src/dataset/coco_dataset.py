"""
COCO keypoint dataset for DriverPoseNet7KP.

Loads COCO person keypoint annotations, maps COCO 17-keypoint format to the
seven-keypoint representation, resizes images to 256x192, and generates
64x48 Gaussian heatmaps.

Sample validity threshold (Stage 4 redesign):
    A sample is VALID only if:
        - At least `min_facial_keypoints` of {Nose, Left Eye, Right Eye,
          Left Ear, Right Ear} are visible, AND
        - Both {Left Shoulder, Right Shoulder} are visible (if require_both_shoulders=True).
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..model.keypoints import COCO_TO_7KP, KEYPOINT_NAMES, NUM_KEYPOINTS
from .heatmap import generate_heatmaps

# Default validity thresholds (mirrors configs/dataset.yaml)
DEFAULT_MIN_TOTAL_KEYPOINTS: int = 5
DEFAULT_MIN_FACIAL_KEYPOINTS: int = 3
DEFAULT_MIN_UPPERBODY_KEYPOINTS: int = 2
DEFAULT_FACIAL_INDICES: List[int] = [0, 1, 2, 3, 4]   # Nose, Left Eye, Right Eye, Left Ear, Right Ear
DEFAULT_UPPERBODY_INDICES: List[int] = [5, 6]          # Left Shoulder, Right Shoulder


class COCOKeypointDataset(Dataset):
    """
    PyTorch Dataset for COCO person keypoints mapped to seven keypoints.

    Each sample returns:
        image:       FloatTensor [3, 256, 192] normalized to [0, 1]
        heatmaps:    FloatTensor [7, 64, 48]
        keypoints:   FloatTensor [7, 3] (x, y, visibility) in 256x192 image space
        image_id:    int COCO image id
        file_name:   str original file name
        is_valid:    bool whether sample meets the keypoint validity threshold
    """

    def __init__(
        self,
        images_dir: str,
        annotations_file: str,
        input_size: Tuple[int, int] = (256, 192),
        heatmap_size: Tuple[int, int] = (64, 48),
        gaussian_sigma: float = 2.0,
        max_samples: Optional[int] = None,
        min_total_keypoints: int = DEFAULT_MIN_TOTAL_KEYPOINTS,
        min_facial_keypoints: int = DEFAULT_MIN_FACIAL_KEYPOINTS,
        min_upperbody_keypoints: int = DEFAULT_MIN_UPPERBODY_KEYPOINTS,
        facial_indices: Optional[List[int]] = None,
        upperbody_indices: Optional[List[int]] = None,
        filter_invalid: bool = True,
    ):
        """
        Args:
            images_dir: Directory containing COCO images.
            annotations_file: Path to COCO person_keypoints JSON.
            input_size: (H, W) for resized images.
            heatmap_size: (H, W) for output heatmaps.
            gaussian_sigma: Gaussian sigma in heatmap pixels.
            max_samples: If set, limit to this many samples (for debugging).
            min_facial_keypoints: Min visible facial keypoints for a valid sample.
            require_both_shoulders: If True, both shoulders must be visible.
            facial_indices: List of facial keypoint indices (default: [0,1,2,3,4]).
            shoulder_indices: List of shoulder keypoint indices (default: [5,6]).
            filter_invalid: If True, drop samples that do not meet the validity threshold.
        """
        self.images_dir = images_dir
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.gaussian_sigma = gaussian_sigma
        self.min_total_keypoints = min_total_keypoints
        self.min_facial_keypoints = min_facial_keypoints
        self.min_upperbody_keypoints = min_upperbody_keypoints
        self.facial_indices = facial_indices if facial_indices is not None else DEFAULT_FACIAL_INDICES
        self.upperbody_indices = upperbody_indices if upperbody_indices is not None else DEFAULT_UPPERBODY_INDICES
        self.filter_invalid = filter_invalid

        with open(annotations_file, "r") as f:
            coco_data = json.load(f)

        # Build image id -> image info map
        self.images: Dict[int, dict] = {}
        for img_info in coco_data["images"]:
            self.images[img_info["id"]] = img_info

        # Build image_id -> list of annotations
        self.img_to_anns: Dict[int, List[dict]] = {}
        for ann in coco_data["annotations"]:
            if ann.get("iscrowd", 0) == 1:
                continue
            img_id = ann["image_id"]
            self.img_to_anns.setdefault(img_id, []).append(ann)

        # Only keep images that have at least one person annotation
        all_image_ids: List[int] = sorted(
            img_id for img_id in self.images if img_id in self.img_to_anns
        )

        # Evaluate validity per image (using best annotation)
        self.valid_flags: Dict[int, bool] = {}
        self.filtered_count: int = 0
        self.valid_count: int = 0

        valid_image_ids: List[int] = []
        for img_id in all_image_ids:
            anns = self.img_to_anns[img_id]
            best_ann = self._select_best_annotation(anns)
            keypoints = self._extract_seven_keypoints(best_ann)
            is_valid = self._is_valid_keypoints(keypoints)
            self.valid_flags[img_id] = is_valid
            if is_valid:
                valid_image_ids.append(img_id)
                self.valid_count += 1
            else:
                self.filtered_count += 1

        self.image_ids = valid_image_ids if filter_invalid else all_image_ids

        if max_samples is not None:
            self.image_ids = self.image_ids[:max_samples]

        print(
            f"[Dataset] Total annotated images: {len(all_image_ids)} | "
            f"Valid (threshold met): {self.valid_count} | "
            f"Filtered out: {self.filtered_count}"
        )

    def __len__(self) -> int:
        return len(self.image_ids)

    def _is_valid_keypoints(self, keypoints: np.ndarray) -> bool:
        """
        Check if a keypoint set meets the Stage 4 validity threshold.

        A sample is VALID only if:
            - At least `min_total_keypoints` total keypoints are visible, AND
            - At least `min_facial_keypoints` facial keypoints are visible, AND
            - At least `min_upperbody_keypoints` upper-body (shoulder) keypoints are visible.

        Args:
            keypoints: Array of shape (7, 3) with [x, y, visibility].

        Returns:
            True if the sample meets the threshold.
        """
        total_visible = int(np.sum(keypoints[:, 2] > 0))
        if total_visible < self.min_total_keypoints:
            return False

        facial_visible = sum(
            1 for idx in self.facial_indices if keypoints[idx, 2] > 0
        )
        if facial_visible < self.min_facial_keypoints:
            return False

        upperbody_visible = sum(
            1 for idx in self.upperbody_indices if keypoints[idx, 2] > 0
        )
        if upperbody_visible < self.min_upperbody_keypoints:
            return False

        return True

    def _select_best_annotation(self, anns: List[dict]) -> dict:
        """Select the annotation with the most visible seven keypoints."""
        best_ann = anns[0]
        best_visible = -1
        for ann in anns:
            kps = ann["keypoints"]  # 51 values: x,y,v x 17
            visible_count = 0
            for coco_idx in range(7):  # only first 7 COCO keypoints
                v = kps[coco_idx * 3 + 2]
                if v > 0:
                    visible_count += 1
            if visible_count > best_visible:
                best_visible = visible_count
                best_ann = ann
        return best_ann

    def _extract_seven_keypoints(self, ann: dict) -> np.ndarray:
        """
        Extract seven keypoints from a COCO annotation.

        Returns:
            Array of shape (7, 3): [x, y, visibility] in original image space.
            Visibility: 0=missing, 1=present (v=1 or v=2 both treated as present).
        """
        kps = ann["keypoints"]  # 51 values
        keypoints = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
        for our_idx, coco_idx in COCO_TO_7KP.items():
            x = kps[coco_idx * 3 + 0]
            y = kps[coco_idx * 3 + 1]
            v = kps[coco_idx * 3 + 2]
            keypoints[our_idx] = [x, y, 1.0 if v > 0 else 0.0]
        return keypoints

    def _resize_keypoints(
        self,
        keypoints: np.ndarray,
        orig_size: Tuple[int, int],
        target_size: Tuple[int, int],
    ) -> np.ndarray:
        """Rescale keypoints from original image size to target size."""
        orig_h, orig_w = orig_size
        tgt_h, tgt_w = target_size
        scale_x = tgt_w / orig_w
        scale_y = tgt_h / orig_h
        kps = keypoints.copy()
        kps[:, 0] *= scale_x
        kps[:, 1] *= scale_y
        return kps

    def __getitem__(self, idx: int) -> dict:
        image_id = self.image_ids[idx]
        img_info = self.images[image_id]
        file_name = img_info["file_name"]
        orig_w = img_info["width"]
        orig_h = img_info["height"]

        # Load image
        img_path = os.path.join(self.images_dir, file_name)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Select best annotation
        anns = self.img_to_anns[image_id]
        ann = self._select_best_annotation(anns)

        # Extract keypoints in original image space
        keypoints_orig = self._extract_seven_keypoints(ann)

        # Resize image
        image_resized = cv2.resize(image, (self.input_size[1], self.input_size[0]))
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1)  # [3, H, W]

        # Resize keypoints to input size
        keypoints_resized = self._resize_keypoints(
            keypoints_orig, (orig_h, orig_w), self.input_size
        )

        # Generate heatmaps
        heatmaps = generate_heatmaps(
            keypoints_resized,
            heatmap_size=self.heatmap_size,
            sigma=self.gaussian_sigma,
            image_size=self.input_size,
        )
        heatmaps_tensor = torch.from_numpy(heatmaps)  # [7, 64, 48]
        keypoints_tensor = torch.from_numpy(keypoints_resized)  # [7, 3]

        return {
            "image": image_tensor,
            "heatmaps": heatmaps_tensor,
            "keypoints": keypoints_tensor,
            "image_id": image_id,
            "file_name": file_name,
            "is_valid": self.valid_flags.get(image_id, False),
        }


def collate_fn(batch: list) -> dict:
    """
    Collate a list of samples into a batch.

    Returns dict with batched tensors.
    """
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "heatmaps": torch.stack([b["heatmaps"] for b in batch]),
        "keypoints": torch.stack([b["keypoints"] for b in batch]),
        "image_id": [b["image_id"] for b in batch],
        "file_name": [b["file_name"] for b in batch],
        "is_valid": [b["is_valid"] for b in batch],
    }
