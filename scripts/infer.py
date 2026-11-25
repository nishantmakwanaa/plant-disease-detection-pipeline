"""
CLI inference script for the Plant Disease Detection Pipeline.

Examples:
    python scripts/infer.py --image leaf.jpg --checkpoint ./checkpoints/best.pth
    python scripts/infer.py --image leaf.jpg --checkpoint ./checkpoints/best.pth --gradcam --output cam.jpg
    python scripts/infer.py --dir ./test_images --checkpoint ./checkpoints/best.pth --json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import PlantDiseasePredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Plant Disease Detection inference")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to a single leaf image")
    group.add_argument("--dir", type=str, help="Path to a directory of leaf images")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--top_k", type=int, default=None, help="Number of top predictions to return")
    parser.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM overlay (single image)")
    parser.add_argument("--cam_method", type=str, choices=["gradcam", "gradcam++"], default="gradcam")
    parser.add_argument("--output", type=str, default=None, help="Output path for Grad-CAM image")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    parser.add_argument("--json", action="store_true", help="Print output as JSON")
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
    inf_cfg = cfg.get("inference", {})

    image_size = int(data_cfg.get("image_size", 300))
    top_k = args.top_k or int(inf_cfg.get("top_k", 3))
    cam_alpha = float(inf_cfg.get("cam_alpha", 0.5))

    predictor = PlantDiseasePredictor(
        checkpoint_path=args.checkpoint,
        image_size=image_size,
        device=args.device,
    )

    if args.image:
        cam_out = args.output
        if args.gradcam and cam_out is None:
            img_p = Path(args.image)
            cam_out = str(img_p.with_name(f"{img_p.stem}_gradcam.jpg"))

        result = predictor.predict_image(
            image_path=args.image,
            top_k=top_k,
            generate_cam=args.gradcam,
            cam_output_path=cam_out,
            cam_alpha=cam_alpha,
            cam_method=args.cam_method,
        )
        # Drop non-serializable PIL image before printing
        printable = {k: v for k, v in result.items() if k != "gradcam_image"}
        if args.json:
            print(json.dumps(printable, indent=2))
        else:
            print(f"\nImage: {printable['image_path']}")
            for pred in printable["predictions"]:
                print(
                    f"  #{pred['rank']} {pred['crop']} — {pred['condition']} "
                    f"({pred['confidence']:.2%})"
                )
            if "gradcam_path" in printable:
                print(f"  Grad-CAM saved: {printable['gradcam_path']}")
    else:
        results = predictor.predict_directory(args.dir, top_k=top_k)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            for res in results:
                top = res["top_prediction"]
                print(
                    f"{res['image_path']} -> {top['crop']} — {top['condition']} "
                    f"({top['confidence']:.2%})"
                )


if __name__ == "__main__":
    main()
