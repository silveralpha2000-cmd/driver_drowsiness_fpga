"""
Evaluation metrics for DriverPoseNet7KP.

Sample validity threshold (Stage 4 redesign):
    A sample is VALID only if:
        - At least `min_facial_keypoints` of {Nose, Left Eye, Right Eye,
          Left Ear, Right Ear} are visible, AND
        - Both {Left Shoulder, Right Shoulder} are visible.

Metrics:
- Heatmap MSE
- Keypoint coordinate error (L2 distance in pixels)
- PCK@threshold (Percentage of Correct Keypoints)
- Per-keypoint breakdown
"""

import csv
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset.coco_dataset import collate_fn

# Default validity thresholds (mirrors configs/dataset.yaml)
DEFAULT_MIN_TOTAL_KEYPOINTS: int = 5
DEFAULT_MIN_FACIAL_KEYPOINTS: int = 3
DEFAULT_MIN_UPPERBODY_KEYPOINTS: int = 2
DEFAULT_FACIAL_INDICES: List[int] = [0, 1, 2, 3, 4]   # Nose, Left Eye, Right Eye, Left Ear, Right Ear
DEFAULT_UPPERBODY_INDICES: List[int] = [5, 6]          # Left Shoulder, Right Shoulder


def is_sample_valid(
    keypoints: np.ndarray,
    min_total_keypoints: int = DEFAULT_MIN_TOTAL_KEYPOINTS,
    min_facial_keypoints: int = DEFAULT_MIN_FACIAL_KEYPOINTS,
    min_upperbody_keypoints: int = DEFAULT_MIN_UPPERBODY_KEYPOINTS,
    facial_indices: Optional[List[int]] = None,
    upperbody_indices: Optional[List[int]] = None,
) -> bool:
    """
    Check if a single sample meets the Stage 4 validity threshold.

    A sample is VALID only if:
        - At least `min_total_keypoints` total keypoints are visible, AND
        - At least `min_facial_keypoints` facial keypoints are visible, AND
        - At least `min_upperbody_keypoints` upper-body (shoulder) keypoints are visible.

    Args:
        keypoints: Array of shape (7, 3) with [x, y, visibility].
        min_total_keypoints: Min total visible keypoints for a valid sample.
        min_facial_keypoints: Min visible facial keypoints for a valid sample.
        min_upperbody_keypoints: Min visible upper-body keypoints for a valid sample.
        facial_indices: List of facial keypoint indices.
        upperbody_indices: List of upper-body keypoint indices.

    Returns:
        True if the sample meets the threshold.
    """
    if facial_indices is None:
        facial_indices = DEFAULT_FACIAL_INDICES
    if upperbody_indices is None:
        upperbody_indices = DEFAULT_UPPERBODY_INDICES

    total_visible = int(np.sum(keypoints[:, 2] > 0))
    if total_visible < min_total_keypoints:
        return False

    facial_visible = sum(1 for idx in facial_indices if keypoints[idx, 2] > 0)
    if facial_visible < min_facial_keypoints:
        return False

    upperbody_visible = sum(1 for idx in upperbody_indices if keypoints[idx, 2] > 0)
    if upperbody_visible < min_upperbody_keypoints:
        return False

    return True


def compute_keypoint_errors(
    pred_keypoints: np.ndarray,
    gt_keypoints: np.ndarray,
    image_size: tuple = (256, 192),
    sample_valid_mask: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute per-keypoint L2 errors between predictions and ground truth.

    Args:
        pred_keypoints: [N, 7, 3] (x, y, confidence)
        gt_keypoints: [N, 7, 3] (x, y, visibility)
        image_size: (H, W) for normalization
        sample_valid_mask: [N] boolean mask of valid samples (Stage 4 threshold).
                           If None, all samples are included.

    Returns:
        Dict with:
            - errors: [N, 7] L2 distance in pixels (nan for invalid)
            - valid_mask: [N, 7] boolean (gt visibility > 0)
            - sample_valid_mask: [N] boolean (sample passes threshold)
            - mean_error: [7] mean error over valid samples
            - pck_at_5: [7] PCK@5 pixels
            - pck_at_10: [7] PCK@10 pixels
    """
    N, num_kps, _ = pred_keypoints.shape
    pred_xy = pred_keypoints[:, :, :2]  # [N, 7, 2]
    gt_xy = gt_keypoints[:, :, :2]  # [N, 7, 2]
    valid = gt_keypoints[:, :, 2] > 0  # [N, 7]

    # Sample-level validity
    if sample_valid_mask is None:
        sample_valid_mask = np.ones(N, dtype=bool)

    # L2 distance per keypoint
    errors = np.linalg.norm(pred_xy - gt_xy, axis=2)  # [N, 7]
    # Mask out invalid samples entirely
    errors = np.where(sample_valid_mask[:, None] & valid, errors, np.nan)

    mean_error = np.nanmean(errors, axis=0)  # [7]

    pck_at_5 = np.nanmean(errors <= 5.0, axis=0)  # [7]
    pck_at_10 = np.nanmean(errors <= 10.0, axis=0)  # [7]

    return {
        "errors": errors,
        "valid_mask": valid,
        "sample_valid_mask": sample_valid_mask,
        "mean_error": mean_error,
        "pck_at_5": pck_at_5,
        "pck_at_10": pck_at_10,
    }


def evaluate_model(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    image_size: tuple = (256, 192),
    min_total_keypoints: int = DEFAULT_MIN_TOTAL_KEYPOINTS,
    min_facial_keypoints: int = DEFAULT_MIN_FACIAL_KEYPOINTS,
    min_upperbody_keypoints: int = DEFAULT_MIN_UPPERBODY_KEYPOINTS,
    facial_indices: Optional[List[int]] = None,
    upperbody_indices: Optional[List[int]] = None,
) -> Dict:
    """
    Run full evaluation on the validation set.

    Only samples meeting the Stage 4 validity threshold are included in metrics.

    Returns dict with aggregate and per-keypoint metrics, plus predictions/GT.
    """
    model.eval()

    all_pred_keypoints: List[np.ndarray] = []
    all_gt_keypoints: List[np.ndarray] = []
    all_sample_valid: List[bool] = []
    total_mse = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            heatmaps_gt = batch["heatmaps"].to(device)
            keypoints_gt = batch["keypoints"].numpy()  # [B, 7, 3]
            is_valid_list = batch.get("is_valid", [True] * len(keypoints_gt))

            # Forward
            heatmaps_pred = model(images)  # [B, 7, 64, 48]

            # MSE loss
            mse = torch.nn.functional.mse_loss(heatmaps_pred, heatmaps_gt)
            total_mse += mse.item()
            num_batches += 1

            # Decode heatmaps to keypoints
            heatmaps_np = heatmaps_pred.cpu().numpy()  # [B, 7, 64, 48]
            batch_pred = []
            for b in range(heatmaps_np.shape[0]):
                kps = decode_batch_heatmaps(heatmaps_np[b], image_size=image_size)
                batch_pred.append(kps)
            all_pred_keypoints.append(np.stack(batch_pred))
            all_gt_keypoints.append(keypoints_gt)
            all_sample_valid.extend([bool(v) for v in is_valid_list])

    pred_keypoints = np.concatenate(all_pred_keypoints, axis=0)  # [N, 7, 3]
    gt_keypoints = np.concatenate(all_gt_keypoints, axis=0)  # [N, 7, 3]
    sample_valid_mask = np.array(all_sample_valid, dtype=bool)  # [N]

    # Apply Stage 4 validity threshold (re-check from GT, not just dataset flag)
    gt_sample_valid = np.array([
        is_sample_valid(
            gt_keypoints[i],
            min_total_keypoints=min_total_keypoints,
            min_facial_keypoints=min_facial_keypoints,
            min_upperbody_keypoints=min_upperbody_keypoints,
            facial_indices=facial_indices,
            upperbody_indices=upperbody_indices,
        )
        for i in range(len(gt_keypoints))
    ], dtype=bool)

    # Use the stricter of dataset flag and GT re-check
    final_valid_mask = sample_valid_mask & gt_sample_valid

    # Compute errors (only valid samples)
    error_metrics = compute_keypoint_errors(
        pred_keypoints, gt_keypoints, image_size, sample_valid_mask=final_valid_mask
    )

    n_valid = int(final_valid_mask.sum())
    n_total = len(pred_keypoints)
    n_filtered = n_total - n_valid

    results = {
        "mean_heatmap_mse": total_mse / max(num_batches, 1),
        "num_samples": n_total,
        "num_valid_samples": n_valid,
        "num_filtered_samples": n_filtered,
        "validity_threshold": {
            "min_total_keypoints": min_total_keypoints,
            "min_facial_keypoints": min_facial_keypoints,
            "min_upperbody_keypoints": min_upperbody_keypoints,
            "facial_indices": facial_indices if facial_indices else DEFAULT_FACIAL_INDICES,
            "upperbody_indices": upperbody_indices if upperbody_indices else DEFAULT_UPPERBODY_INDICES,
        },
        "mean_error_all": float(np.nanmean(error_metrics["errors"])),
        "mean_error_per_keypoint": error_metrics["mean_error"].tolist(),
        "pck_at_5_per_keypoint": error_metrics["pck_at_5"].tolist(),
        "pck_at_10_per_keypoint": error_metrics["pck_at_10"].tolist(),
        "valid_count_per_keypoint": error_metrics["valid_mask"].sum(axis=0).tolist(),
    }

    return results, pred_keypoints, gt_keypoints


def decode_batch_heatmaps(
    heatmaps: np.ndarray,
    image_size: tuple = (256, 192),
) -> np.ndarray:
    """
    Decode a batch of heatmaps to keypoint coordinates.

    Args:
        heatmaps: [7, 64, 48]
        image_size: (H, W)

    Returns:
        [7, 3] array of (x, y, confidence)
    """
    num_kps, h, w = heatmaps.shape
    img_h, img_w = image_size
    keypoints = np.zeros((num_kps, 3), dtype=np.float32)

    for kp_idx in range(num_kps):
        hm = heatmaps[kp_idx]
        max_idx = np.argmax(hm)
        cy, cx = np.unravel_index(max_idx, hm.shape)
        confidence = float(hm[cy, cx])

        x = cx / w * img_w
        y = cy / h * img_h
        keypoints[kp_idx] = [x, y, confidence]

    return keypoints


def save_evaluation_results(
    results: Dict,
    output_dir: str = "results/accuracy",
    filename: str = "evaluation_results.json",
):
    """Save evaluation results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Evaluation results saved: {path}")


def save_evaluation_csv(
    results: Dict,
    output_dir: str = "results/accuracy",
    filename: str = "evaluation_results.csv",
):
    """Save per-keypoint metrics to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    from src.model.keypoints import KEYPOINT_NAMES

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["keypoint", "mean_error_px", "pck_at_5", "pck_at_10", "valid_count"]
        )
        for i, name in enumerate(KEYPOINT_NAMES):
            writer.writerow(
                [
                    name,
                    f"{results['mean_error_per_keypoint'][i]:.3f}",
                    f"{results['pck_at_5_per_keypoint'][i]:.4f}",
                    f"{results['pck_at_10_per_keypoint'][i]:.4f}",
                    int(results["valid_count_per_keypoint"][i]),
                ]
            )
        writer.writerow([])
        writer.writerow(
            [
                "ALL",
                f"{results['mean_error_all']:.3f}",
                "",
                "",
                int(sum(results["valid_count_per_keypoint"])),
            ]
        )
        writer.writerow([])
        writer.writerow(["Validity threshold summary"])
        writer.writerow(["Total samples", results["num_samples"]])
        writer.writerow(["Valid samples", results["num_valid_samples"]])
        writer.writerow(["Filtered samples", results["num_filtered_samples"]])
        writer.writerow(["Min total keypoints", results["validity_threshold"]["min_total_keypoints"]])
        writer.writerow(["Min facial keypoints", results["validity_threshold"]["min_facial_keypoints"]])
        writer.writerow(["Min upperbody keypoints", results["validity_threshold"]["min_upperbody_keypoints"]])
    print(f"Evaluation CSV saved: {path}")


def print_evaluation_summary(results: Dict):
    """Print evaluation summary to console."""
    print("\n=== Evaluation Results (Stage 4 Validity Threshold) ===")
    print(f"Total samples: {results['num_samples']}")
    print(f"Valid samples: {results['num_valid_samples']}")
    print(f"Filtered samples: {results['num_filtered_samples']}")
    print(f"Validity rule: >= {results['validity_threshold']['min_total_keypoints']} total + "
          f">= {results['validity_threshold']['min_facial_keypoints']} facial + "
          f">= {results['validity_threshold']['min_upperbody_keypoints']} upperbody")
    print(f"Mean heatmap MSE: {results['mean_heatmap_mse']:.6f}")
    print(f"Mean keypoint error (all valid): {results['mean_error_all']:.3f} px")
    print()
    print(f"{'Keypoint':<16} {'MeanErr(px)':>12} {'PCK@5':>10} {'PCK@10':>10} {'Valid':>8}")
    print("-" * 60)
    from src.model.keypoints import KEYPOINT_NAMES

    for i, name in enumerate(KEYPOINT_NAMES):
        print(
            f"{name:<16} "
            f"{results['mean_error_per_keypoint'][i]:>12.3f} "
            f"{results['pck_at_5_per_keypoint'][i]:>10.4f} "
            f"{results['pck_at_10_per_keypoint'][i]:>10.4f} "
            f"{int(results['valid_count_per_keypoint'][i]):>8}"
        )
    print("=" * 60)
