"""
CLI training script for the Plant Disease Detection Pipeline.

Example:
    python scripts/train.py --data_dir ./data --epochs 50 --batch_size 32
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import PlantVillageDataset
from src.model import build_model
from src.train import Trainer


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Plant Disease Detection classifier")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--data_dir", type=str, default=None, help="Override dataset root directory")
    parser.add_argument("--arch", type=str, default=None, help="Backbone architecture (e.g., efficientnet_b3, resnet50)")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--image_size", type=int, default=None, help="Override input image size")
    parser.add_argument("--num_workers", type=int, default=None, help="Override DataLoader workers")
    parser.add_argument("--output_dir", type=str, default=None, help="Override checkpoint output directory")
    parser.add_argument("--log_dir", type=str, default=None, help="Override logs directory")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--no_sampler", action="store_true", help="Disable weighted random sampler")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    cfg = load_config(args.config)

    train_cfg = cfg.get("training", {})
    opt_cfg = cfg.get("optimizer", {})
    sched_cfg = cfg.get("scheduler", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("dataset", {})
    es_cfg = cfg.get("early_stopping", {})
    ckpt_cfg = cfg.get("checkpoint", {})
    log_cfg = cfg.get("logging", {})

    seed = int(data_cfg.get("seed", 42))
    set_seed(seed)

    data_dir = args.data_dir or data_cfg.get("root", "./data")
    image_size = args.image_size or int(data_cfg.get("image_size", 300))
    epochs = args.epochs or int(train_cfg.get("epochs", 50))
    batch_size = args.batch_size or int(train_cfg.get("batch_size", 32))
    num_workers = args.num_workers if args.num_workers is not None else int(train_cfg.get("num_workers", 4))
    pin_memory = bool(train_cfg.get("pin_memory", True)) and torch.cuda.is_available()
    lr = args.lr or float(opt_cfg.get("lr", 3e-4))
    weight_decay = float(opt_cfg.get("weight_decay", 1e-4))
    arch = args.arch or model_cfg.get("architecture", "efficientnet_b3")
    dropout = float(model_cfg.get("dropout", 0.4))
    pretrained = bool(model_cfg.get("pretrained", True))
    freeze_epochs = int(model_cfg.get("freeze_backbone_epochs", 5))
    checkpoint_dir = args.output_dir or ckpt_cfg.get("dir", "./checkpoints")
    log_dir = args.log_dir or log_cfg.get("log_dir", "./logs")
    mixed_precision = False if args.no_amp else bool(train_cfg.get("mixed_precision", True))

    dataset = PlantVillageDataset(
        root=data_dir,
        image_size=image_size,
        train_split=float(data_cfg.get("train_split", 0.70)),
        val_split=float(data_cfg.get("val_split", 0.15)),
        test_split=float(data_cfg.get("test_split", 0.15)),
        seed=seed,
    )

    train_loader, val_loader, _ = dataset.get_loaders(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        use_sampler=not args.no_sampler,
    )

    num_classes = dataset.num_classes or int(model_cfg.get("num_classes", 38))
    model = build_model(
        architecture=arch,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=freeze_epochs > 0,
    )
    model.class_names = list(dataset.classes)

    class_weights = dataset.get_class_weights() if data_cfg.get("use_class_weights", True) else None
    device = torch.device(args.device) if args.device else None

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        warmup_epochs=int(sched_cfg.get("warmup_epochs", 5)),
        eta_min=float(sched_cfg.get("eta_min", 1e-6)),
        grad_clip=float(train_cfg.get("gradient_clip", 1.0)),
        early_stopping_patience=int(es_cfg.get("patience", 10)),
        early_stopping_min_delta=float(es_cfg.get("min_delta", 1e-4)),
        early_stopping_monitor=str(es_cfg.get("monitor", "val_acc")),
        checkpoint_dir=checkpoint_dir,
        save_top_k=int(ckpt_cfg.get("save_top_k", 3)),
        log_dir=log_dir,
        device=device,
        mixed_precision=mixed_precision,
        freeze_backbone_epochs=freeze_epochs,
        print_every=int(log_cfg.get("print_every", 50)),
        save_curves=bool(log_cfg.get("save_curves", True)),
    )
    trainer.fit()


if __name__ == "__main__":
    main()
