"""
Apply train/val split (90/10) to the full COCO dataset (5,000 images).

Saves:
- results/training/dataset_split/train_ids.json  (list of image IDs for training)
- results/training/dataset_split/val_ids.json    (list of image IDs for validation)
- results/training/dataset_split/split_manifest.json  (split metadata)
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset.coco_dataset import COCOKeypointDataset


def main():
    images_dir = r"G:\mypro\driver_drowsiness_fpga\datasets\coco\val2017"
    ann_file = r"G:\mypro\driver_drowsiness_fpga\datasets\coco\annotations\person_keypoints_val2017.json"
    out_dir = r"G:\mypro\driver_drowsiness_fpga\results\training\dataset_split"
    train_ratio = 0.9
    seed = 42

    print("=== Stage 2: Train/Val Split (Full Dataset) ===")
    print(f"Images dir: {images_dir}")
    print(f"Annotations: {ann_file}")
    print(f"Train ratio: {train_ratio}")
    print(f"Seed: {seed}")

    # Load dataset to get valid image IDs (those with annotations)
    dataset = COCOKeypointDataset(
        images_dir=images_dir,
        annotations_file=ann_file,
        input_size=(256, 192),
        heatmap_size=(64, 48),
        gaussian_sigma=2.0,
    )

    total = len(dataset)
    print(f"\nTotal images with annotations: {total}")

    # Shuffle and split
    random.seed(seed)
    image_ids = dataset.image_ids.copy()
    random.shuffle(image_ids)

    train_count = int(total * train_ratio)
    train_ids = image_ids[:train_count]
    val_ids = image_ids[train_count:]

    print(f"\nTrain samples: {len(train_ids)} ({100*len(train_ids)/total:.1f}%)")
    print(f"Val samples:   {len(val_ids)} ({100*len(val_ids)/total:.1f}%)")

    # Save splits
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "train_ids.json"), "w") as f:
        json.dump(train_ids, f)
    with open(os.path.join(out_dir, "val_ids.json"), "w") as f:
        json.dump(val_ids, f)

    manifest = {
        "total_images": total,
        "train_ratio": train_ratio,
        "seed": seed,
        "train_count": len(train_ids),
        "val_count": len(val_ids),
        "train_file": "train_ids.json",
        "val_file": "val_ids.json",
        "annotations_file": ann_file,
        "images_dir": images_dir,
    }
    with open(os.path.join(out_dir, "split_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSplit files saved to: {out_dir}")
    print("  - train_ids.json")
    print("  - val_ids.json")
    print("  - split_manifest.json")
    print("\n=== Split Complete ===")


if __name__ == "__main__":
    main()
