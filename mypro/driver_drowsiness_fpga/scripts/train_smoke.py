"""
Stage 3: Train DriverPoseNet7KP on COCO seven-keypoint dataset.

Smoke test: 10 epochs to verify end-to-end training pipeline.
Full training can be run by increasing epochs in configs/training.yaml.
"""

import os
import sys
import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.driver_pose_net import DriverPoseNet7KP
from src.model.keypoints import NUM_KEYPOINTS
from src.training import PoseTrainer, build_dataloaders, load_split_ids, set_seed, get_device
from src.evaluation import evaluate_model, print_evaluation_summary, save_evaluation_results, save_evaluation_csv


def main():
    print("=== Stage 3: CNN Training (Smoke Test) ===")

    # Load configs
    with open("configs/training.yaml", "r") as f:
        train_config = yaml.safe_load(f)["training"]
    with open("configs/dataset.yaml", "r") as f:
        dataset_config = yaml.safe_load(f)["dataset"]

    # Keypoint validity threshold (Stage 4 redesign)
    kp_threshold = dataset_config.get("keypoint_threshold", {})
    min_total = kp_threshold.get("min_total_keypoints", 5)
    min_facial = kp_threshold.get("min_facial_keypoints", 3)
    min_upperbody = kp_threshold.get("min_upperbody_keypoints", 2)
    facial_indices = kp_threshold.get("facial_indices", [0, 1, 2, 3, 4])
    upperbody_indices = kp_threshold.get("upperbody_indices", [5, 6])

    print(f"Keypoint threshold: >= {min_total} total + >= {min_facial} facial + >= {min_upperbody} upperbody")

    # Override for smoke test
    smoke_epochs = 2
    print(f"Smoke test epochs: {smoke_epochs} (full: {train_config['epochs']})")

    # Set seed
    set_seed(42)

    # Device
    device = get_device(train_config["device"])
    print(f"Device: {device}")

    # Load split
    split_dir = "results/training/dataset_split"
    train_ids, val_ids = load_split_ids(split_dir)
    # Limit for smoke test speed
    train_ids = train_ids[:100]
    val_ids = val_ids[:20]
    print(f"Train IDs (smoke): {len(train_ids)}, Val IDs (smoke): {len(val_ids)}")

    # Build dataloaders
    train_loader, val_loader = build_dataloaders(
        images_dir=dataset_config["images_dir"],
        annotations_file=dataset_config["annotations_file"],
        train_ids=train_ids,
        val_ids=val_ids,
        batch_size=int(train_config["batch_size"]),
        input_size=tuple(dataset_config["input_size"]),
        heatmap_size=tuple(dataset_config["heatmap_size"]),
        gaussian_sigma=dataset_config["gaussian_sigma"],
        num_workers=0,
        min_total_keypoints=min_total,
        min_facial_keypoints=min_facial,
        min_upperbody_keypoints=min_upperbody,
        facial_indices=facial_indices,
        upperbody_indices=upperbody_indices,
        filter_invalid=True,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Build model
    model = DriverPoseNet7KP(num_keypoints=NUM_KEYPOINTS)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Train (smoke test)
    trainer = PoseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
        epochs=smoke_epochs,
        warmup_epochs=min(2, smoke_epochs),
        patience=int(train_config["patience"]),
        checkpoint_dir=train_config["checkpoint_dir"],
        best_model_file=train_config["best_model_file"],
        save_every=int(train_config["save_every"]),
        log_every=int(train_config["log_every"]),
        history_csv=train_config["history_csv"],
    )

    results = trainer.fit()
    print(f"\nSmoke test results: {results}")

    # Evaluate
    print("\n=== Evaluation ===")
    eval_results, pred_kps, gt_kps = evaluate_model(
        model, val_loader, device,
        image_size=tuple(dataset_config["input_size"]),
        min_total_keypoints=min_total,
        min_facial_keypoints=min_facial,
        min_upperbody_keypoints=min_upperbody,
        facial_indices=facial_indices,
        upperbody_indices=upperbody_indices,
    )
    print_evaluation_summary(eval_results)
    save_evaluation_results(eval_results, output_dir="results/accuracy")
    save_evaluation_csv(eval_results, output_dir="results/accuracy")

    print("\n=== Stage 3 Smoke Test Complete ===")


if __name__ == "__main__":
    main()
