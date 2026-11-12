"""
Training engine for DriverPoseNet7KP.

Implements:
- AdamW optimizer with weight decay
- Cosine LR scheduler with warmup
- Heatmap MSE loss
- Checkpoint saving (every N epochs + best model)
- Early stopping with patience
- Training history CSV
- Reproducible random seeds
- GPU auto / CPU fallback
"""

import csv
import json
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset.coco_dataset import (
    COCOKeypointDataset,
    collate_fn,
    DEFAULT_MIN_TOTAL_KEYPOINTS,
    DEFAULT_MIN_FACIAL_KEYPOINTS,
    DEFAULT_MIN_UPPERBODY_KEYPOINTS,
    DEFAULT_FACIAL_INDICES,
    DEFAULT_UPPERBODY_INDICES,
)
from src.model.driver_pose_net import DriverPoseNet7KP
from src.model.keypoints import NUM_KEYPOINTS


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: str = "auto") -> torch.device:
    """Select training device."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class PoseTrainer:
    """
    Trainer for DriverPoseNet7KP heatmap regression.

    Args:
        model: DriverPoseNet7KP model.
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        device: torch device.
        learning_rate: Initial learning rate.
        weight_decay: Weight decay for AdamW.
        epochs: Total epochs.
        warmup_epochs: Warmup epochs for cosine scheduler.
        patience: Early stopping patience.
        checkpoint_dir: Directory to save checkpoints.
        best_model_file: Path to save best model.
        save_every: Save checkpoint every N epochs.
        log_every: Log every N batches.
        history_csv: Path to training history CSV.
    """

    def __init__(
        self,
        model: DriverPoseNet7KP,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 100,
        warmup_epochs: int = 5,
        patience: int = 10,
        checkpoint_dir: str = "results/training",
        best_model_file: str = "results/training/best_model.pth",
        save_every: int = 10,
        log_every: int = 50,
        history_csv: str = "results/training/history.csv",
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = int(epochs)
        self.warmup_epochs = int(warmup_epochs)
        self.patience = int(patience)
        self.checkpoint_dir = checkpoint_dir
        self.best_model_file = best_model_file
        self.save_every = int(save_every)
        self.log_every = int(log_every)
        self.history_csv = history_csv

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.dirname(history_csv), exist_ok=True)

        # Loss
        self.criterion = nn.MSELoss()

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )

        # Scheduler: cosine with warmup
        self.scheduler = self._build_scheduler(learning_rate, epochs, warmup_epochs)

        # History
        self.history: list = []
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.epochs_no_improve = 0

    def _build_scheduler(
        self, lr: float, epochs: int, warmup_epochs: int
    ) -> optim.lr_scheduler._LRScheduler:
        """Build cosine annealing scheduler with linear warmup."""

        def lr_lambda(epoch: int):
            if epoch < warmup_epochs:
                return (epoch + 1) / max(warmup_epochs, 1)
            progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        running_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device)
            heatmaps = batch["heatmaps"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, heatmaps)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            num_batches += 1

            if batch_idx % self.log_every == 0:
                print(
                    f"  Epoch {epoch} | Batch {batch_idx}/{len(self.train_loader)} "
                    f"| Loss: {loss.item():.6f}"
                )

        avg_loss = running_loss / max(num_batches, 1)
        self.scheduler.step()
        return {"train_loss": avg_loss}

    def validate(self, epoch: int) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        running_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device)
                heatmaps = batch["heatmaps"].to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, heatmaps)
                running_loss += loss.item()
                num_batches += 1

        avg_loss = running_loss / max(num_batches, 1)
        return {"val_loss": avg_loss}

    def save_checkpoint(
        self, epoch: int, train_loss: float, val_loss: float, is_best: bool = False
    ):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }

        # Regular checkpoint
        if epoch % self.save_every == 0:
            ckpt_path = os.path.join(
                self.checkpoint_dir, f"checkpoint_epoch{epoch}.pth"
            )
            torch.save(checkpoint, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

        # Best model
        if is_best:
            torch.save(checkpoint, self.best_model_file)
            print(f"  Best model saved: {self.best_model_file} (val_loss={val_loss:.6f})")

    def save_history(self):
        """Save training history to CSV."""
        if not self.history:
            return
        keys = self.history[0].keys()
        with open(self.history_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.history)
        print(f"  Training history saved: {self.history_csv}")

    def plot_history(self, plot_path: str = "results/training/loss_plot.png"):
        """Plot training and validation loss curves."""
        if not self.history:
            return
        epochs = [h["epoch"] for h in self.history]
        train_losses = [h["train_loss"] for h in self.history]
        val_losses = [h["val_loss"] for h in self.history]

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_losses, label="Train Loss", marker="o")
        plt.plot(epochs, val_losses, label="Val Loss", marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("DriverPoseNet7KP Training History")
        plt.legend()
        plt.grid(True, alpha=0.3)
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Loss plot saved: {plot_path}")

    def fit(self) -> Dict[str, float]:
        """Full training loop."""
        print(f"\n=== Training DriverPoseNet7KP ===")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.epochs}")
        print(f"Warmup: {self.warmup_epochs}")
        print(f"Patience: {self.patience}")
        print(f"Best model: {self.best_model_file}")

        for epoch in range(1, self.epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)

            entry = {"epoch": epoch, **train_metrics, **val_metrics}
            self.history.append(entry)

            train_loss = train_metrics["train_loss"]
            val_loss = val_metrics["val_loss"]

            print(
                f"Epoch {epoch:3d} | train_loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f} | lr={self.optimizer.param_groups[0]['lr']:.2e}"
            )

            # Early stopping / best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            self.save_checkpoint(epoch, train_loss, val_loss, is_best=is_best)

            if self.epochs_no_improve >= self.patience:
                print(
                    f"\nEarly stopping at epoch {epoch} "
                    f"(no improvement for {self.patience} epochs)."
                )
                break

        self.save_history()
        self.plot_history()

        print(f"\n=== Training Complete ===")
        print(f"Best epoch: {self.best_epoch}")
        print(f"Best val loss: {self.best_val_loss:.6f}")

        return {
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "final_train_loss": self.history[-1]["train_loss"],
            "final_val_loss": self.history[-1]["val_loss"],
        }


def load_split_ids(split_dir: str) -> Tuple[list, list]:
    """Load train/val image ID lists from split directory."""
    with open(os.path.join(split_dir, "train_ids.json"), "r") as f:
        train_ids = json.load(f)
    with open(os.path.join(split_dir, "val_ids.json"), "r") as f:
        val_ids = json.load(f)
    return train_ids, val_ids


def build_dataloaders(
    images_dir: str,
    annotations_file: str,
    train_ids: list,
    val_ids: list,
    batch_size: int = 16,
    input_size: Tuple[int, int] = (256, 192),
    heatmap_size: Tuple[int, int] = (64, 48),
    gaussian_sigma: float = 2.0,
    num_workers: int = 0,
    min_total_keypoints: int = DEFAULT_MIN_TOTAL_KEYPOINTS,
    min_facial_keypoints: int = DEFAULT_MIN_FACIAL_KEYPOINTS,
    min_upperbody_keypoints: int = DEFAULT_MIN_UPPERBODY_KEYPOINTS,
    facial_indices: Optional[List[int]] = None,
    upperbody_indices: Optional[List[int]] = None,
    filter_invalid: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and val DataLoaders from split IDs."""
    full_dataset = COCOKeypointDataset(
        images_dir=images_dir,
        annotations_file=annotations_file,
        input_size=input_size,
        heatmap_size=heatmap_size,
        gaussian_sigma=gaussian_sigma,
        min_total_keypoints=min_total_keypoints,
        min_facial_keypoints=min_facial_keypoints,
        min_upperbody_keypoints=min_upperbody_keypoints,
        facial_indices=facial_indices,
        upperbody_indices=upperbody_indices,
        filter_invalid=filter_invalid,
    )

    # Map IDs to dataset indices
    id_to_idx = {img_id: i for i, img_id in enumerate(full_dataset.image_ids)}
    train_indices = [id_to_idx[i] for i in train_ids if i in id_to_idx]
    val_indices = [id_to_idx[i] for i in val_ids if i in id_to_idx]

    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
    )

    return train_loader, val_loader


def main():
    """Run training with config from configs/training.yaml and dataset split."""
    import yaml

    # Load config
    with open("configs/training.yaml", "r") as f:
        train_config = yaml.safe_load(f)["training"]
    with open("configs/dataset.yaml", "r") as f:
        dataset_config = yaml.safe_load(f)["dataset"]

    # Set seed
    set_seed(42)

    # Device
    device = get_device(train_config["device"])
    print(f"Using device: {device}")

    # Load split
    split_dir = "results/training/dataset_split"
    train_ids, val_ids = load_split_ids(split_dir)
    print(f"Train IDs: {len(train_ids)}, Val IDs: {len(val_ids)}")

    # Build dataloaders
    train_loader, val_loader = build_dataloaders(
        images_dir=dataset_config["images_dir"],
        annotations_file=dataset_config["annotations_file"],
        train_ids=train_ids,
        val_ids=val_ids,
        batch_size=train_config["batch_size"],
        input_size=tuple(dataset_config["input_size"]),
        heatmap_size=tuple(dataset_config["heatmap_size"]),
        gaussian_sigma=dataset_config["gaussian_sigma"],
        num_workers=0,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Build model
    model = DriverPoseNet7KP(num_keypoints=NUM_KEYPOINTS)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Train
    trainer = PoseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
        epochs=train_config["epochs"],
        warmup_epochs=train_config["warmup_epochs"],
        patience=train_config["patience"],
        checkpoint_dir=train_config["checkpoint_dir"],
        best_model_file=train_config["best_model_file"],
        save_every=train_config["save_every"],
        log_every=train_config["log_every"],
        history_csv=train_config["history_csv"],
    )

    results = trainer.fit()
    print(f"\nTraining results: {results}")


if __name__ == "__main__":
    main()
