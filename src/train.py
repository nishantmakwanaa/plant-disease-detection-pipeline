"""
Full training pipeline for the Plant Disease Detection model.

Features:
- Cross-entropy loss with class weights
- AdamW optimizer + CosineAnnealingLR with linear warmup
- Mixed-precision training (torch.cuda.amp)
- Gradient clipping
- Early stopping (monitor val_acc or val_loss)
- Best model checkpoint saving (top-k)
- Training curve plotting
- Backbone freeze/unfreeze schedule
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model import PlantDiseaseClassifier, build_model, PLANT_DISEASE_CLASSES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Stop training when a monitored metric stops improving.

    Args:
        patience: Number of epochs with no improvement before stopping.
        min_delta: Minimum change to qualify as improvement.
        mode: 'max' for accuracy-like metrics, 'min' for loss-like.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.triggered = False

    def step(self, score: float) -> bool:
        """
        Call at end of each epoch.

        Returns:
            True if training should stop, False otherwise.
        """
        improved = False
        if self.best_score is None:
            improved = True
        elif self.mode == "max" and score > self.best_score + self.min_delta:
            improved = True
        elif self.mode == "min" and score < self.best_score - self.min_delta:
            improved = True

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True

        return self.triggered


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


# ---------------------------------------------------------------------------
# Single epoch train / eval passes
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: PlantDiseaseClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    grad_clip: float = 1.0,
    print_every: int = 50,
    use_amp: bool = True,
) -> Dict[str, float]:
    """Run a single training epoch."""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = max(len(loader), 1)
    amp_enabled = use_amp and device.type == "cuda"

    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        batch_acc = _accuracy(logits.detach(), labels)
        total_loss += loss.item()
        total_acc += batch_acc

        if (i + 1) % print_every == 0 or i == len(loader) - 1:
            logger.debug(
                "  batch [%d/%d]  loss=%.4f  acc=%.4f",
                i + 1,
                len(loader),
                total_loss / (i + 1),
                total_acc / (i + 1),
            )

    return {
        "loss": total_loss / n_batches,
        "acc": total_acc / n_batches,
    }


@torch.no_grad()
def evaluate_epoch(
    model: PlantDiseaseClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
) -> Dict[str, float]:
    """Run a single evaluation epoch."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = max(len(loader), 1)
    amp_enabled = use_amp and device.type == "cuda"

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with autocast(enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)

        total_loss += loss.item()
        total_acc += _accuracy(logits, labels)

    return {
        "loss": total_loss / n_batches,
        "acc": total_acc / n_batches,
    }


# ---------------------------------------------------------------------------
# Checkpoint manager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """Save top-k model checkpoints keyed by validation accuracy."""

    def __init__(self, checkpoint_dir: str | Path, top_k: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self._records: List[Tuple[float, Path]] = []  # (val_acc, path)

    def save(
        self,
        model: PlantDiseaseClassifier,
        epoch: int,
        val_acc: float,
        val_loss: float,
        extra: Optional[Dict] = None,
    ) -> Path:
        """Save checkpoint; prune if more than top_k checkpoints exist."""
        fname = f"ckpt_epoch{epoch:03d}_acc{val_acc:.4f}.pth"
        path = self.checkpoint_dir / fname
        metadata = {"epoch": epoch, "val_acc": val_acc, "val_loss": val_loss}
        if extra:
            metadata.update(extra)
        model.save(path, metadata=metadata)

        self._records.append((val_acc, path))
        self._records.sort(key=lambda x: x[0], reverse=True)

        # Remove worst checkpoints beyond top_k
        while len(self._records) > self.top_k:
            _, old_path = self._records.pop()
            if old_path.exists():
                old_path.unlink()
                logger.debug("Pruned checkpoint: %s", old_path)

        # Always keep a 'best.pth' symlink / copy
        best_path = self.checkpoint_dir / "best.pth"
        best_acc_path = self._records[0][1]
        # Copy the best file to best.pth
        import shutil
        shutil.copy2(best_acc_path, best_path)

        return path

    @property
    def best_path(self) -> Optional[Path]:
        best = self.checkpoint_dir / "best.pth"
        return best if best.exists() else None


# ---------------------------------------------------------------------------
# Curve plotting
# ---------------------------------------------------------------------------

def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str | Path,
) -> None:
    """
    Plot and save training/validation loss & accuracy curves.

    Args:
        history: Dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
        save_path: Output PNG file path.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", linewidth=2, linestyle="--")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train Acc", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], label="Val Acc", linewidth=2, linestyle="--")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Training curves saved → %s", save_path)


# ---------------------------------------------------------------------------
# Main Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Orchestrates the full training pipeline.

    Args:
        model: PlantDiseaseClassifier instance.
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        class_weights: Optional class weight tensor for loss.
        epochs: Maximum training epochs.
        lr: Initial learning rate.
        weight_decay: AdamW weight decay.
        warmup_epochs: Number of linear warm-up epochs.
        grad_clip: Gradient clipping norm.
        early_stopping_patience: Epochs without improvement before stopping.
        checkpoint_dir: Where to save checkpoints.
        log_dir: Where to save logs and plots.
        device: Training device.
        mixed_precision: Enable AMP.
        freeze_backbone_epochs: Epochs to keep backbone frozen.
    """

    def __init__(
        self,
        model: PlantDiseaseClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: Optional[torch.Tensor] = None,
        epochs: int = 50,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        eta_min: float = 1e-6,
        grad_clip: float = 1.0,
        early_stopping_patience: int = 10,
        early_stopping_min_delta: float = 1e-4,
        early_stopping_monitor: str = "val_acc",
        checkpoint_dir: str | Path = "./checkpoints",
        save_top_k: int = 3,
        log_dir: str | Path = "./logs",
        device: Optional[torch.device] = None,
        mixed_precision: bool = True,
        freeze_backbone_epochs: int = 5,
        print_every: int = 50,
        save_curves: bool = True,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.eta_min = eta_min
        self.grad_clip = grad_clip
        self.mixed_precision = mixed_precision
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.early_stopping_monitor = early_stopping_monitor
        self.print_every = print_every
        self.save_curves = save_curves
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(self.device)
        logger.info("Training on device: %s", self.device)

        # Loss
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

        # Optimizer (all params; backbone may be frozen)
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Scheduler: optional linear warmup -> cosine annealing
        if warmup_epochs > 0 and epochs > warmup_epochs:
            warmup_scheduler = LinearLR(
                self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
            )
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=eta_min
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=max(epochs, 1), eta_min=eta_min
            )

        # AMP
        self.scaler = GradScaler(enabled=mixed_precision and self.device.type == "cuda")

        # Early stopping
        es_mode = "min" if early_stopping_monitor == "val_loss" else "max"
        self.early_stopper = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            mode=es_mode,
        )

        # Checkpoint manager
        self.ckpt_manager = CheckpointManager(checkpoint_dir, top_k=save_top_k)

        # History
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []
        }

    # ------------------------------------------------------------------
    def _maybe_unfreeze(self, epoch: int) -> None:
        """Unfreeze backbone after warm-up phase and rebuild optimizer/scheduler."""
        if epoch == self.freeze_backbone_epochs and self.freeze_backbone_epochs > 0:
            logger.info("Epoch %d: unfreezing backbone for full fine-tuning.", epoch)
            self.model.unfreeze_backbone()
            lr = self.optimizer.param_groups[0]["lr"]
            weight_decay = self.optimizer.param_groups[0]["weight_decay"]
            self.optimizer = AdamW(
                [
                    {"params": self.model.backbone.parameters(), "lr": lr * 0.1},
                    {"params": self.model.head.parameters(), "lr": lr},
                ],
                weight_decay=weight_decay,
            )
            remaining = max(self.epochs - epoch + 1, 1)
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=remaining, eta_min=self.eta_min
            )

    def fit(self) -> Dict[str, List[float]]:
        """
        Run the full training loop.

        Returns:
            Training history dict.
        """
        logger.info(
            "Starting training: %d epochs, device=%s", self.epochs, self.device
        )

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            # Possibly unfreeze backbone
            self._maybe_unfreeze(epoch)

            # --- Train ---
            train_metrics = train_one_epoch(
                self.model,
                self.train_loader,
                self.optimizer,
                self.criterion,
                self.scaler,
                self.device,
                self.grad_clip,
                print_every=self.print_every,
                use_amp=self.mixed_precision,
            )

            # --- Validate ---
            val_metrics = evaluate_epoch(
                self.model,
                self.val_loader,
                self.criterion,
                self.device,
                use_amp=self.mixed_precision,
            )

            self.scheduler.step()

            # Record
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["acc"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["acc"])

            elapsed = time.time() - epoch_start
            lr_now = self.optimizer.param_groups[-1]["lr"]
            logger.info(
                "Epoch [%03d/%03d] | "
                "train_loss=%.4f  train_acc=%.4f | "
                "val_loss=%.4f  val_acc=%.4f | "
                "lr=%.2e | %.1fs",
                epoch,
                self.epochs,
                train_metrics["loss"],
                train_metrics["acc"],
                val_metrics["loss"],
                val_metrics["acc"],
                lr_now,
                elapsed,
            )

            # Checkpoint
            self.ckpt_manager.save(
                self.model,
                epoch=epoch,
                val_acc=val_metrics["acc"],
                val_loss=val_metrics["loss"],
            )

            # Early stopping
            monitored_value = (
                val_metrics["loss"]
                if self.early_stopping_monitor == "val_loss"
                else val_metrics["acc"]
            )
            if self.early_stopper.step(monitored_value):
                logger.info(
                    "Early stopping at epoch %d (best %s=%.4f).",
                    epoch,
                    self.early_stopping_monitor,
                    self.early_stopper.best_score,
                )
                break

        # Save final curves
        if self.save_curves and self.history["train_loss"]:
            plot_training_curves(
                self.history,
                save_path=self.log_dir / "training_curves.png",
            )
        return self.history
