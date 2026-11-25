"""
Grad-CAM implementation for plant disease model explainability.

Provides:
- GradCAM: standard Grad-CAM using the last convolutional layer.
- GradCAMPlusPlus: improved Grad-CAM++ variant.
- overlay_heatmap: blend a CAM with the original PIL image.
- save_cam_image: save the overlaid image to disk.

References:
  - Selvaraju et al. (2017) "Grad-CAM: Visual Explanations from Deep Networks"
  - Chattopadhyay et al. (2018) "Grad-CAM++: Improved Visual Explanations"
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hook-based feature / gradient storage
# ---------------------------------------------------------------------------

class _HookManager:
    """Stores forward activations and backward gradients via PyTorch hooks."""

    def __init__(self):
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handles: List = []

    def register(self, layer: nn.Module) -> None:
        """Register forward & backward hooks on *layer*."""
        self._handles.append(
            layer.register_forward_hook(self._forward_hook)
        )
        self._handles.append(
            layer.register_full_backward_hook(self._backward_hook)
        )

    def _forward_hook(self, _module, _input, output):
        self.activations = output.detach()

    def _backward_hook(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ---------------------------------------------------------------------------
# Base CAM class
# ---------------------------------------------------------------------------

class _BaseCAM:
    """Base class for CAM-family methods."""

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.target_layer = target_layer or self._find_last_conv(model)
        self._hooks = _HookManager()
        self._hooks.register(self.target_layer)
        logger.debug("CAM target layer: %s", self.target_layer)

    @staticmethod
    def _find_last_conv(model: nn.Module) -> nn.Module:
        last_conv = None
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv is None:
            raise RuntimeError("No Conv2d layer found in model.")
        return last_conv

    def _get_weights(
        self,
        activations: torch.Tensor,
        gradients: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def __call__(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Compute Grad-CAM heatmap.

        Args:
            input_tensor: Preprocessed image tensor (1, 3, H, W).
            class_idx: Target class index. Uses predicted class if None.

        Returns:
            (heatmap, class_idx): numpy array in [0, 1] of shape (H, W),
            and the class index used.
        """
        self.model.eval()
        self.model.zero_grad()

        try:
            model_device = next(self.model.parameters()).device
            input_tensor = input_tensor.to(model_device)
        except StopIteration:
            pass

        output = self.model(input_tensor)  # (1, C)

        if class_idx is None:
            class_idx = int(output.argmax(dim=-1).item())

        score = output[0, class_idx]
        score.backward()

        activations = self._hooks.activations   # (1, C, h, w)
        gradients = self._hooks.gradients       # (1, C, h, w)

        weights = self._get_weights(activations, gradients)  # (1, C, 1, 1)

        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Resize to input resolution
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze(0).squeeze(0).cpu().numpy()  # (H, W)

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, class_idx

    def remove_hooks(self) -> None:
        """Clean up registered hooks."""
        self._hooks.remove()


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

class GradCAM(_BaseCAM):
    """
    Standard Grad-CAM.

    Weights = global average of gradients over spatial dims.
    """

    def _get_weights(self, activations, gradients) -> torch.Tensor:
        return gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)


# ---------------------------------------------------------------------------
# Grad-CAM++
# ---------------------------------------------------------------------------

class GradCAMPlusPlus(_BaseCAM):
    """
    Grad-CAM++ — better localisation for multiple instances of same class.

    Weights account for second-order derivatives of ReLU activations.
    """

    def _get_weights(self, activations, gradients) -> torch.Tensor:
        # alpha computation (Chattopadhyay et al. 2018, Eq. 19)
        grad_sq = gradients ** 2
        grad_cu = gradients ** 3
        sum_act = activations.sum(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        alpha_denom = 2.0 * grad_sq + sum_act * grad_cu + 1e-7
        alpha = grad_sq / alpha_denom
        weights = (alpha * F.relu(gradients)).mean(dim=(2, 3), keepdim=True)
        return weights


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------

def overlay_heatmap(
    cam: np.ndarray,
    pil_image: Image.Image,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> Image.Image:
    """
    Blend a Grad-CAM heatmap on top of the original image.

    Args:
        cam: Normalised heatmap (H, W) in [0, 1].
        pil_image: Original PIL image (any size).
        alpha: Heatmap blend weight (0=image only, 1=heatmap only).
        colormap: OpenCV colourmap constant (default JET).

    Returns:
        Blended PIL image (RGB).
    """
    img_np = np.array(pil_image.convert("RGB"))
    h, w = img_np.shape[:2]

    # Resize heatmap to match image
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)  # BGR
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # → RGB

    blended = np.clip(
        (1 - alpha) * img_np.astype(np.float32) + alpha * heatmap.astype(np.float32),
        0,
        255,
    ).astype(np.uint8)

    return Image.fromarray(blended)


def save_cam_image(
    cam: np.ndarray,
    pil_image: Image.Image,
    save_path: str | Path,
    alpha: float = 0.5,
    class_name: str = "",
    confidence: float = 0.0,
) -> Path:
    """
    Overlay Grad-CAM on the image, optionally add a text label, and save.

    Args:
        cam: Normalised heatmap (H, W).
        pil_image: Original PIL image.
        save_path: Output file path.
        alpha: Blend factor.
        class_name: Class name to annotate (optional).
        confidence: Confidence score to annotate (optional).

    Returns:
        Path to the saved image.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    blended = overlay_heatmap(cam, pil_image, alpha)

    if class_name:
        # Annotate the image with cv2 for speed
        img_cv = cv2.cvtColor(np.array(blended), cv2.COLOR_RGB2BGR)
        label = f"{class_name} ({confidence:.2%})" if confidence else class_name
        cv2.putText(
            img_cv,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        blended = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    blended.save(str(save_path))
    logger.info("CAM image saved → %s", save_path)
    return save_path


# ---------------------------------------------------------------------------
# Batch CAM for misclassified samples
# ---------------------------------------------------------------------------

def generate_cams_for_samples(
    model: nn.Module,
    samples: List[Tuple[torch.Tensor, int, int]],  # (tensor, true_label, pred_label)
    class_names: List[str],
    output_dir: str | Path,
    alpha: float = 0.5,
    method: str = "gradcam",
) -> List[Path]:
    """
    Generate and save Grad-CAM images for a list of (image, true, pred) samples.

    Args:
        model: Trained PlantDiseaseClassifier.
        samples: List of (preprocessed_tensor (1,3,H,W), true_label, pred_label).
        class_names: Class name list.
        output_dir: Directory to save CAM images.
        alpha: Heatmap blend weight.
        method: 'gradcam' or 'gradcam++'.

    Returns:
        List of saved file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not samples:
        return []

    cam_cls = GradCAM if method == "gradcam" else GradCAMPlusPlus
    cam_obj = cam_cls(model)

    saved_paths: List[Path] = []
    try:
        for i, (tensor, true_lbl, pred_lbl) in enumerate(samples):
            # Generate CAM for the predicted class
            heatmap, _ = cam_obj(tensor, class_idx=pred_lbl)

            # Reconstruct a rough PIL image by un-normalising
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_tensor = tensor.squeeze(0).cpu() * std + mean
            img_tensor = img_tensor.clamp(0, 1)
            pil_img = Image.fromarray((img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))

            true_name = class_names[true_lbl] if true_lbl < len(class_names) else str(true_lbl)
            pred_name = class_names[pred_lbl] if pred_lbl < len(class_names) else str(pred_lbl)
            fname = f"sample_{i:04d}_true_{true_name[:20]}_pred_{pred_name[:20]}.jpg"
            for bad_char in ("/", "\\", ":", " "):
                fname = fname.replace(bad_char, "_")

            path = save_cam_image(
                heatmap,
                pil_img,
                output_dir / fname,
                alpha=alpha,
                class_name=f"Pred: {pred_name}",
            )
            saved_paths.append(path)
    finally:
        cam_obj.remove_hooks()

    return saved_paths
