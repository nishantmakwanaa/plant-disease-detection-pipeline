"""
Model definition for Plant Disease Detection.

Uses EfficientNet-B3 (or ResNet-50) pre-trained on ImageNet as the backbone,
with a custom classification head for 38 PlantVillage disease classes.

Architecture highlights:
- EfficientNet-B3 backbone (timm)
- Global Average Pooling → Dropout → FC(38)
- Grad-CAM-compatible hook API
- Mixed-precision compatible
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 38 PlantVillage class names
# ---------------------------------------------------------------------------
PLANT_DISEASE_CLASSES: List[str] = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

NUM_CLASSES = len(PLANT_DISEASE_CLASSES)  # 38


# ---------------------------------------------------------------------------
# Classification head
# ---------------------------------------------------------------------------
class ClassificationHead(nn.Module):
    """Custom head: AdaptiveAvgPool → Flatten → Dropout → Linear."""

    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, H, W) → (B, num_classes)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class PlantDiseaseClassifier(nn.Module):
    """
    EfficientNet-B3 (or ResNet-50) based plant disease classifier.

    Args:
        architecture: timm model name, e.g. 'efficientnet_b3' or 'resnet50'.
        num_classes: Number of output classes (default: 38).
        pretrained: Whether to load ImageNet weights (default: True).
        dropout: Dropout rate before the final FC layer (default: 0.4).
        freeze_backbone: If True, freeze all backbone parameters (useful for
            warm-up phase where only the head is trained).
    """

    def __init__(
        self,
        architecture: str = "efficientnet_b3",
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        dropout: float = 0.4,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.architecture = architecture
        self.num_classes = num_classes
        self.dropout = dropout
        self.class_names: List[str] = (
            PLANT_DISEASE_CLASSES[:num_classes]
            if num_classes <= len(PLANT_DISEASE_CLASSES)
            else [str(i) for i in range(num_classes)]
        )

        # Load backbone (without the original classification head)
        self.backbone = timm.create_model(
            architecture,
            pretrained=pretrained,
            num_classes=0,       # remove original head
            global_pool="",      # keep spatial feature maps
        )

        # Determine backbone output channels
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 300, 300)
            feat = self.backbone(dummy)
            in_features = feat.shape[1]  # channel dim

        logger.info(
            "Backbone '%s' | in_features=%d | pretrained=%s",
            architecture,
            in_features,
            pretrained,
        )

        # Custom classification head
        self.head = ClassificationHead(in_features, num_classes, dropout)

        if freeze_backbone:
            self.freeze_backbone()

        # Grad-CAM hook handles
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None
        self._register_cam_hooks()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, 3, H, W).
        Returns:
            Logits of shape (B, num_classes).
        """
        features = self.backbone(x)   # (B, C, H', W')
        logits = self.head(features)  # (B, num_classes)
        return logits

    # ------------------------------------------------------------------
    # Convenience inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        top_k: int = 3,
        class_names: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Run inference and return top-k predictions.

        Args:
            x: Preprocessed image tensor (1, 3, H, W) or batch (B, 3, H, W).
            top_k: Number of top predictions to return.
            class_names: List of class name strings. Defaults to PLANT_DISEASE_CLASSES.

        Returns:
            List of dicts per image: [{'rank': 1, 'class_id': int, 'class_name': str, 'confidence': float}, ...]
        """
        if class_names is None:
            class_names = PLANT_DISEASE_CLASSES

        self.eval()
        logits = self(x)
        probs = F.softmax(logits, dim=-1)
        k = min(top_k, probs.shape[-1])
        topk_probs, topk_ids = probs.topk(k, dim=-1)

        results = []
        for b in range(x.shape[0]):
            preds = []
            for rank, (prob, cls_id) in enumerate(
                zip(topk_probs[b].tolist(), topk_ids[b].tolist()), start=1
            ):
                preds.append(
                    {
                        "rank": rank,
                        "class_id": cls_id,
                        "class_name": class_names[cls_id] if cls_id < len(class_names) else str(cls_id),
                        "confidence": round(prob, 4),
                    }
                )
            results.append(preds)
        return results

    # ------------------------------------------------------------------
    # Backbone freeze / unfreeze
    # ------------------------------------------------------------------
    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (train head only)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen.")

    def unfreeze_backbone(self, lr_scale: float = 0.1) -> None:
        """Unfreeze backbone for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen (lr_scale hint=%.2f).", lr_scale)

    # ------------------------------------------------------------------
    # Grad-CAM support
    # ------------------------------------------------------------------
    def _get_last_conv_layer(self) -> nn.Module:
        """Auto-detect the last convolutional layer in the backbone."""
        last_conv = None
        for module in self.backbone.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is None:
            raise RuntimeError("No Conv2d found in backbone.")
        return last_conv

    def _register_cam_hooks(self) -> None:
        """Register forward/backward hooks on the last conv layer."""
        target_layer = self._get_last_conv_layer()

        def _save_activation(_, __, output):
            self._activations = output.detach()

        def _save_gradient(_, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        target_layer.register_forward_hook(_save_activation)
        target_layer.register_full_backward_hook(_save_gradient)

    def get_cam(
        self,
        x: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Compute Grad-CAM heatmap for a single image.

        Args:
            x: Preprocessed image tensor (1, 3, H, W).
            class_idx: Target class index. If None, uses the predicted class.

        Returns:
            cam: Heatmap tensor of shape (H, W) with values in [0, 1].
            class_idx: The target class index used.
        """
        self.eval()
        self.zero_grad()

        logits = self(x)  # (1, num_classes)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=-1).item())

        # Backprop w.r.t. the target class score
        score = logits[0, class_idx]
        score.backward()

        # Grad-CAM formula
        gradients = self._gradients   # (1, C, H', W')
        activations = self._activations  # (1, C, H', W')

        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        cam = F.relu(cam)

        # Resize to input spatial size
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(0).squeeze(0)  # (H, W)

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam, class_idx

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    def save(self, path: str | Path, metadata: Optional[dict] = None) -> None:
        """Save model weights and metadata to a checkpoint file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "architecture": self.architecture,
            "num_classes": self.num_classes,
            "dropout": self.dropout,
            "class_names": self.class_names,
            "state_dict": self.state_dict(),
        }
        if metadata:
            payload.update(metadata)
        torch.save(payload, path)
        logger.info("Model saved → %s", path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str | torch.device = "cpu",
        dropout: float = 0.4,
    ) -> "PlantDiseaseClassifier":
        """
        Load a PlantDiseaseClassifier from a checkpoint file.

        Args:
            path: Path to the .pth checkpoint.
            device: Target device.
            dropout: Dropout rate (fallback if not stored in checkpoint).

        Returns:
            Loaded PlantDiseaseClassifier in eval mode.
        """
        path = Path(path)
        payload = torch.load(path, map_location=device, weights_only=False)

        architecture = payload.get("architecture", "efficientnet_b3")
        num_classes = payload.get("num_classes", NUM_CLASSES)
        ckpt_dropout = payload.get("dropout", dropout)

        model = cls(
            architecture=architecture,
            num_classes=num_classes,
            pretrained=False,
            dropout=ckpt_dropout,
        )
        model.load_state_dict(payload["state_dict"])
        if "class_names" in payload and payload["class_names"]:
            model.class_names = list(payload["class_names"])
        model.to(device)
        model.eval()
        logger.info("Model loaded ← %s  (device=%s)", path, device)
        return model


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------
def build_model(
    architecture: str = "efficientnet_b3",
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    dropout: float = 0.4,
    freeze_backbone: bool = False,
) -> PlantDiseaseClassifier:
    """Convenience factory for building a PlantDiseaseClassifier."""
    return PlantDiseaseClassifier(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    )
