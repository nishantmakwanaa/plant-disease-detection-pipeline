"""
Inference module for Plant Disease Detection.

Supports:
- Loading a trained PlantDiseaseClassifier checkpoint
- Single-image and batch/directory predictions with top-k confidence scores
- Optional Grad-CAM heatmap generation & overlay export
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from PIL import Image

from src.dataset import build_inference_transforms, load_image_for_inference
from src.gradcam import GradCAM, GradCAMPlusPlus, overlay_heatmap, save_cam_image
from src.model import PLANT_DISEASE_CLASSES, PlantDiseaseClassifier

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _format_label(class_name: str) -> Dict[str, str]:
    """Split PlantVillage 'Crop___Disease' naming into readable crop/condition."""
    if "___" in class_name:
        crop, condition = class_name.split("___", 1)
    else:
        crop, condition = "Unknown", class_name
    return {
        "crop": crop.replace("_", " "),
        "condition": condition.replace("_", " "),
    }


class PlantDiseasePredictor:
    """High-level inference wrapper around PlantDiseaseClassifier."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        image_size: int = 300,
        device: Optional[str | torch.device] = None,
        class_names: Optional[List[str]] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.image_size = image_size
        self.model = PlantDiseaseClassifier.load(checkpoint_path, device=self.device)
        self.transform = build_inference_transforms(image_size)
        self.class_names = (
            class_names
            or getattr(self.model, "class_names", None)
            or PLANT_DISEASE_CLASSES[: self.model.num_classes]
        )

    def predict_image(
        self,
        image_path: str | Path,
        top_k: int = 3,
        generate_cam: bool = False,
        cam_output_path: Optional[str | Path] = None,
        cam_alpha: float = 0.5,
        cam_method: str = "gradcam",
    ) -> Dict:
        """
        Predict disease class for a single leaf image.

        Args:
            image_path: Path to input image.
            top_k: Number of top predictions to return.
            generate_cam: Whether to compute a Grad-CAM overlay.
            cam_output_path: Optional path to save the Grad-CAM image.
            cam_alpha: Heatmap blend weight.
            cam_method: 'gradcam' or 'gradcam++'.

        Returns:
            Dict containing 'image_path', 'predictions', 'top_prediction', and
            optionally 'gradcam_image' / 'gradcam_path'.
        """
        image_path = Path(image_path)
        tensor, pil_image = load_image_for_inference(image_path, image_size=self.image_size)
        tensor = tensor.to(self.device)

        preds = self.model.predict(tensor, top_k=top_k, class_names=self.class_names)[0]
        for pred in preds:
            pred.update(_format_label(pred["class_name"]))

        result: Dict = {
            "image_path": str(image_path),
            "top_prediction": preds[0],
            "predictions": preds,
        }

        if generate_cam:
            cam_cls = GradCAMPlusPlus if cam_method == "gradcam++" else GradCAM
            cam_extractor = cam_cls(self.model)
            try:
                heatmap, _ = cam_extractor(tensor, class_idx=preds[0]["class_id"])
            finally:
                cam_extractor.remove_hooks()

            blended = overlay_heatmap(heatmap, pil_image, alpha=cam_alpha)
            result["gradcam_image"] = blended

            if cam_output_path is not None:
                saved_path = save_cam_image(
                    heatmap,
                    pil_image,
                    save_path=cam_output_path,
                    alpha=cam_alpha,
                    class_name=preds[0]["condition"],
                    confidence=preds[0]["confidence"],
                )
                result["gradcam_path"] = str(saved_path)

        return result

    def predict_batch(
        self,
        image_paths: Sequence[str | Path],
        top_k: int = 3,
        batch_size: int = 32,
    ) -> List[Dict]:
        """Predict disease classes for a sequence of image paths."""
        paths = [Path(p) for p in image_paths]
        results: List[Dict] = []

        for start in range(0, len(paths), batch_size):
            chunk_paths = paths[start : start + batch_size]
            tensors = []
            for p in chunk_paths:
                if not p.exists():
                    raise FileNotFoundError(f"Image not found: {p}")
                img = Image.open(p).convert("RGB")
                tensors.append(self.transform(img))

            batch_tensor = torch.stack(tensors, dim=0).to(self.device)
            batch_preds = self.model.predict(
                batch_tensor, top_k=top_k, class_names=self.class_names
            )
            for p, preds in zip(chunk_paths, batch_preds):
                for pred in preds:
                    pred.update(_format_label(pred["class_name"]))
                results.append(
                    {
                        "image_path": str(p),
                        "top_prediction": preds[0],
                        "predictions": preds,
                    }
                )

        return results

    def predict_directory(
        self,
        directory: str | Path,
        top_k: int = 3,
        batch_size: int = 32,
    ) -> List[Dict]:
        """Predict all images inside a directory (non-recursive)."""
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        image_paths = sorted(
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            logger.warning("No images found in %s", directory)
            return []
        return self.predict_batch(image_paths, top_k=top_k, batch_size=batch_size)
