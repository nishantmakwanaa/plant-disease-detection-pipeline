"""
CLI evaluation script for the Plant Disease Detection Pipeline.

Example:
    python scripts/evaluate.py --checkpoint ./checkpoints/best.pth --data_dir ./data
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import PlantVillageDataset
from src.evaluate import evaluate_model
from src.model import PlantDiseaseClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Plant Disease Detection checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset root (ImageFolder)")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--split", type=str, choices=["val", "test"], default="test", help="Split to evaluate")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--num_workers", type=int, default=None, help="Override DataLoader workers")
    parser.add_argument("--output_dir", type=str, default="./evaluation", help="Directory for evaluation artifacts")
    parser.add_argument("--max_gradcam", type=int, default=12, help="Max misclassified samples for Grad-CAM")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()

    cfg = {}
    if Path(args.config).exists():
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    data_cfg = cfg.get("dataset", {})
    train_cfg = cfg.get("training", {})
    inf_cfg = cfg.get("inference", {})

    image_size = int(data_cfg.get("image_size", 300))
    batch_size = args.batch_size or int(train_cfg.get("batch_size", 32))
    num_workers = args.num_workers if args.num_workers is not None else int(train_cfg.get("num_workers", 4))
    seed = int(data_cfg.get("seed", 42))

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = PlantVillageDataset(
        root=args.data_dir,
        image_size=image_size,
        train_split=float(data_cfg.get("train_split", 0.70)),
        val_split=float(data_cfg.get("val_split", 0.15)),
        test_split=float(data_cfg.get("test_split", 0.15)),
        seed=seed,
    )
    _, val_loader, test_loader = dataset.get_loaders(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        use_sampler=False,
    )
    loader = val_loader if args.split == "val" else test_loader

    model = PlantDiseaseClassifier.load(args.checkpoint, device=device)
    summary = evaluate_model(
        model=model,
        loader=loader,
        class_names=dataset.classes,
        output_dir=args.output_dir,
        device=device,
        max_gradcam_samples=args.max_gradcam,
        cam_alpha=float(inf_cfg.get("cam_alpha", 0.5)),
    )

    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
