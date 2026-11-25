"""
PlantVillage dataset handler.

Supports:
- Local directory in ImageFolder format (one sub-folder per class).
- Stratified train / val / test splits (default 70/15/15).
- Strong data augmentation via torchvision transforms.
- Class-weight computation for imbalanced datasets.
- Optional albumentations-based augmentation pipeline.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Default ImageNet normalisation stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Transform builders
# ---------------------------------------------------------------------------

def build_train_transforms(image_size: int = 300) -> transforms.Compose:
    """Build augmentation pipeline for training."""
    return transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        ]
    )


def build_eval_transforms(image_size: int = 300) -> transforms.Compose:
    """Build deterministic transforms for validation / test / inference."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_inference_transforms(image_size: int = 300) -> transforms.Compose:
    """Same as eval but also returns the original PIL image alongside."""
    return build_eval_transforms(image_size)


# ---------------------------------------------------------------------------
# Subset dataset wrapper (index-based)
# ---------------------------------------------------------------------------

class SubsetDataset(Dataset):
    """Thin wrapper that restricts an ImageFolder to a list of indices and
    applies a custom transform."""

    def __init__(
        self,
        base_dataset: datasets.ImageFolder,
        indices: List[int],
        transform: Optional[Callable] = None,
    ):
        self.base = base_dataset
        self.indices = indices
        self.transform = transform
        self.classes = base_dataset.classes
        self.class_to_idx = base_dataset.class_to_idx

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        orig_idx = self.indices[idx]
        path, label = self.base.samples[orig_idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ---------------------------------------------------------------------------
# PlantVillage dataset loader
# ---------------------------------------------------------------------------

class PlantVillageDataset:
    """
    Manages train / val / test splits of the PlantVillage dataset.

    Directory layout expected (ImageFolder format)::

        data/
        ├── Apple___Apple_scab/
        │   ├── image001.jpg
        │   └── ...
        ├── Apple___Black_rot/
        │   └── ...
        └── ...

    Args:
        root: Path to the root data directory.
        image_size: Spatial size to resize images to.
        train_split: Fraction for training (default 0.70).
        val_split: Fraction for validation (default 0.15).
        test_split: Fraction for test (default 0.15).
        seed: Random seed for reproducible splits.
    """

    def __init__(
        self,
        root: str | Path,
        image_size: int = 300,
        train_split: float = 0.70,
        val_split: float = 0.15,
        test_split: float = 0.15,
        seed: int = 42,
    ):
        assert math.isclose(train_split + val_split + test_split, 1.0, abs_tol=1e-6), (
            "Splits must sum to 1.0"
        )

        self.root = Path(root)
        self.image_size = image_size
        self.seed = seed

        # Build the full ImageFolder (no transform — transforms applied per split)
        base = datasets.ImageFolder(str(self.root))
        self.classes: List[str] = base.classes
        self.class_to_idx: Dict[str, int] = base.class_to_idx
        self.num_classes: int = len(self.classes)

        logger.info(
            "PlantVillage root=%s | %d classes | %d total images",
            self.root,
            self.num_classes,
            len(base),
        )

        # Stratified split
        indices = list(range(len(base)))
        labels = [base.samples[i][1] for i in indices]

        train_idx, temp_idx, train_lbl, temp_lbl = train_test_split(
            indices,
            labels,
            test_size=(val_split + test_split),
            stratify=labels,
            random_state=seed,
        )
        relative_val = val_split / (val_split + test_split)
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=1 - relative_val,
            stratify=temp_lbl,
            random_state=seed,
        )

        logger.info(
            "Split sizes → train=%d | val=%d | test=%d",
            len(train_idx),
            len(val_idx),
            len(test_idx),
        )

        # Build subset datasets
        self.train_dataset = SubsetDataset(
            base, train_idx, transform=build_train_transforms(image_size)
        )
        self.val_dataset = SubsetDataset(
            base, val_idx, transform=build_eval_transforms(image_size)
        )
        self.test_dataset = SubsetDataset(
            base, test_idx, transform=build_eval_transforms(image_size)
        )

        # Cache class weights for imbalanced training
        self._train_labels = [labels[i] for i in train_idx]
        self._class_weights: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Class weights
    # ------------------------------------------------------------------
    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for the training split.

        Returns:
            FloatTensor of shape (num_classes,).
        """
        if self._class_weights is not None:
            return self._class_weights

        counts = np.bincount(self._train_labels, minlength=self.num_classes).astype(float)
        counts = np.where(counts == 0, 1.0, counts)  # avoid division by zero
        weights = 1.0 / counts
        weights = weights / weights.sum() * self.num_classes  # normalize
        self._class_weights = torch.FloatTensor(weights)
        return self._class_weights

    def get_sampler(self) -> WeightedRandomSampler:
        """
        Build a WeightedRandomSampler for balanced training batch sampling.

        Returns:
            WeightedRandomSampler instance.
        """
        class_weights = self.get_class_weights()
        sample_weights = [class_weights[lbl].item() for lbl in self._train_labels]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    # ------------------------------------------------------------------
    # DataLoader factories
    # ------------------------------------------------------------------
    def get_loaders(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        use_sampler: bool = True,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train / val / test DataLoaders.

        Args:
            batch_size: Images per batch.
            num_workers: Parallel data loading workers.
            pin_memory: Pin memory for faster GPU transfer.
            use_sampler: Use WeightedRandomSampler for training (overrides shuffle).

        Returns:
            (train_loader, val_loader, test_loader)
        """
        sampler = self.get_sampler() if use_sampler else None

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=len(self.train_dataset) >= batch_size,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Single-image loader (for inference)
# ---------------------------------------------------------------------------

def load_image_for_inference(
    image_path: str | Path,
    image_size: int = 300,
) -> Tuple[torch.Tensor, Image.Image]:
    """
    Load a single image from disk and prepare it for model inference.

    Args:
        image_path: Path to the image file (jpg/png/etc.).
        image_size: Spatial size to resize to.

    Returns:
        (tensor, pil_image) — tensor shape (1, 3, H, W), un-normalised PIL image.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    pil_image = Image.open(path).convert("RGB")
    transform = build_inference_transforms(image_size)
    tensor = transform(pil_image).unsqueeze(0)  # add batch dim
    return tensor, pil_image


# ---------------------------------------------------------------------------
# Quick dataset summary helper
# ---------------------------------------------------------------------------

def dataset_summary(root: str | Path) -> Dict:
    """
    Print and return a summary of an ImageFolder dataset.

    Args:
        root: Root directory.

    Returns:
        Dict with 'num_classes', 'total_images', 'per_class' counts.
    """
    root = Path(root)
    base = datasets.ImageFolder(str(root))
    labels = [lbl for _, lbl in base.samples]
    counts = np.bincount(labels, minlength=len(base.classes))
    summary = {
        "num_classes": len(base.classes),
        "total_images": len(base),
        "per_class": {cls: int(cnt) for cls, cnt in zip(base.classes, counts)},
    }
    logger.info(
        "Dataset summary: %d classes, %d images total",
        summary["num_classes"],
        summary["total_images"],
    )
    return summary
