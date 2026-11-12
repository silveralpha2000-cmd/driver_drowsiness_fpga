"""
Stage 2 verification: test COCO dataset pipeline.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import COCOKeypointDataset, collate_fn
from src.dataset.visualize import save_visualization
from torch.utils.data import DataLoader


def main():
    images_dir = r"G:\mypro\driver_drowsiness_fpga\datasets\coco\val2017"
    ann_file = r"G:\mypro\driver_drowsiness_fpga\datasets\coco\annotations\person_keypoints_val2017.json"

    print("=== Stage 2: Dataset Preparation Verification ===")
    print(f"Images dir: {images_dir}")
    print(f"Annotations: {ann_file}")

    # Create dataset
    dataset = COCOKeypointDataset(
        images_dir=images_dir,
        annotations_file=ann_file,
        input_size=(256, 192),
        heatmap_size=(64, 48),
        gaussian_sigma=2.0,
        max_samples=100,  # limit for quick verification
    )
    print(f"\nDataset size (sampled): {len(dataset)}")

    # Test single sample
    sample = dataset[0]
    print(f"\n--- Sample 0 ---")
    print(f"image shape:      {tuple(sample['image'].shape)}  (expected [3, 256, 192])")
    print(f"heatmaps shape:   {tuple(sample['heatmaps'].shape)}  (expected [7, 64, 48])")
    print(f"keypoints shape:  {tuple(sample['keypoints'].shape)}  (expected [7, 3])")
    print(f"image_id:         {sample['image_id']}")
    print(f"file_name:        {sample['file_name']}")

    assert tuple(sample["image"].shape) == (3, 256, 192), "Image shape mismatch"
    assert tuple(sample["heatmaps"].shape) == (7, 64, 48), "Heatmaps shape mismatch"
    assert tuple(sample["keypoints"].shape) == (7, 3), "Keypoints shape mismatch"
    print("All shape assertions PASSED")

    # Test DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(loader))
    print(f"\n--- Batch ---")
    print(f"batch image:      {tuple(batch['image'].shape)}")
    print(f"batch heatmaps:   {tuple(batch['heatmaps'].shape)}")
    print(f"batch keypoints:  {tuple(batch['keypoints'].shape)}")
    assert tuple(batch["image"].shape) == (4, 3, 256, 192)
    assert tuple(batch["heatmaps"].shape) == (4, 7, 64, 48)
    print("Batch shape assertions PASSED")

    # Generate visualization samples
    out_dir = r"G:\mypro\driver_drowsiness_fpga\results\training\dataset_visualization"
    print(f"\nGenerating visualization samples in {out_dir} ...")
    for i in range(5):
        sample = dataset[i]
        save_visualization(
            sample["image"],
            sample["heatmaps"],
            sample["keypoints"],
            output_dir=out_dir,
            prefix=f"sample_{i:03d}_id{sample['image_id']}",
        )
    print("Visualization samples saved.")

    # Keypoint visibility stats
    print(f"\n--- Keypoint visibility stats (first 100 samples) ---")
    vis_counts = [0] * 7
    for i in range(len(dataset)):
        kps = dataset[i]["keypoints"]
        for j in range(7):
            if kps[j, 2] > 0:
                vis_counts[j] += 1
    from src.model.keypoints import KEYPOINT_NAMES
    for j, name in enumerate(KEYPOINT_NAMES):
        print(f"  {j}: {name} visible in {vis_counts[j]}/{len(dataset)} samples ({100*vis_counts[j]/len(dataset):.1f}%)")

    print("\n=== Stage 2 Verification Complete ===")


if __name__ == "__main__":
    main()
