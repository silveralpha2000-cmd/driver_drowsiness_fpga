"""
Generate keypoint visualization images for Stage 4 verification.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.dataset import COCOKeypointDataset
from src.dataset.visualize import save_visualization


def main():
    images_dir = r"G:\mypro\driver_drowsiness_fpga\datasets\coco\val2017"
    ann_file = r"G:\mypro\driver_drowsiness_fpga\datasets\coco\annotations\person_keypoints_val2017.json"
    out_dir = r"G:\mypro\driver_drowsiness_fpga\results\training\keypoint_visualization"

    print("=== Keypoint Visualization Generation ===")

    dataset = COCOKeypointDataset(
        images_dir=images_dir,
        annotations_file=ann_file,
        input_size=(256, 192),
        heatmap_size=(64, 48),
        gaussian_sigma=2.0,
        min_total_keypoints=5,
        min_facial_keypoints=3,
        min_upperbody_keypoints=2,
        facial_indices=[0, 1, 2, 3, 4],
        upperbody_indices=[5, 6],
        filter_invalid=True,
        max_samples=None,
    )

    print(f"Valid dataset size: {len(dataset)}")
    print(f"Saving keypoint visualizations to: {out_dir}")

    os.makedirs(out_dir, exist_ok=True)

    # Generate visualizations for first 10 valid samples
    for i in range(min(10, len(dataset))):
        sample = dataset[i]
        saved_path = save_visualization(
            sample["image"],
            sample["heatmaps"],
            sample["keypoints"],
            output_dir=out_dir,
            prefix=f"kp_sample_{i:03d}_id{sample['image_id']}",
        )
        print(f"  Saved: {saved_path}")

    print("\n=== Keypoint Visualization Complete ===")


if __name__ == "__main__":
    main()
