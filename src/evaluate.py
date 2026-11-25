"""
Evaluation pipeline for the Plant Disease Detection model.

Generates:
- Overall & per-class classification report (precision, recall, F1, support)
- Confusion matrix visualization (counts + normalized)
- Multi-class One-vs-Rest ROC curves (macro & micro AUC)
- Grad-CAM explainability overlays for misclassified samples
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

from src.gradcam import generate_cams_for_samples
from src.model import PLANT_DISEASE_CLASSES, PlantDiseaseClassifier

logger = logging.getLogger(__name__)


@torch.no_grad()
def collect_predictions(
    model: PlantDiseaseClassifier,
    loader: DataLoader,
    device: torch.device,
    max_misclassified: int = 16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[torch.Tensor, int, int]]]:
    """
    Run inference across a DataLoader and collect labels, probabilities, and
    misclassified samples for Grad-CAM inspection.
    """
    model.eval()
    model.to(device)

    all_targets: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []
    misclassified: List[Tuple[torch.Tensor, int, int]] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=-1)
        targets = labels.numpy()

        all_targets.append(targets)
        all_preds.append(preds)
        all_probs.append(probs)

        if len(misclassified) < max_misclassified:
            for idx in range(len(targets)):
                if preds[idx] != targets[idx]:
                    misclassified.append(
                        (images[idx : idx + 1].detach().cpu(), int(targets[idx]), int(preds[idx]))
                    )
                    if len(misclassified) >= max_misclassified:
                        break

    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)
    return y_true, y_pred, y_prob, misclassified


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: str | Path,
    normalize: bool = True,
) -> Path:
    """Plot and save a confusion matrix heatmap."""
    labels_idx = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
        fmt = ".2f"
        title = "Normalized Confusion Matrix"
    else:
        cm_plot = cm
        fmt = "d"
        title = "Confusion Matrix"

    fig_size = max(12, len(class_names) * 0.42)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    short_names = [name.split("___")[-1][:22] for name in class_names]
    sns.heatmap(
        cm_plot,
        annot=len(class_names) <= 20,
        fmt=fmt,
        cmap="Blues",
        xticklabels=short_names,
        yticklabels=short_names,
        ax=ax,
        cbar=True,
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved -> %s", save_path)
    return save_path


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    save_path: str | Path,
    max_class_curves: int = 10,
) -> Dict[str, Optional[float]]:
    """
    Plot macro-average, micro-average, and representative per-class OvR ROC curves.
    """
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    if n_classes == 2 and y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])

    fpr: Dict[int | str, np.ndarray] = {}
    tpr: Dict[int | str, np.ndarray] = {}
    roc_auc: Dict[int | str, float] = {}

    valid_classes: List[int] = []
    for c in range(n_classes):
        positives = int(y_bin[:, c].sum())
        if 0 < positives < len(y_true):
            fpr[c], tpr[c], _ = roc_curve(y_bin[:, c], y_prob[:, c])
            roc_auc[c] = float(auc(fpr[c], tpr[c]))
            valid_classes.append(c)

    # Micro-average ROC
    micro_auc: Optional[float] = None
    if len(np.unique(y_bin)) > 1:
        fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_prob.ravel())
        micro_auc = float(auc(fpr["micro"], tpr["micro"]))
        roc_auc["micro"] = micro_auc

    # Macro-average ROC
    macro_auc: Optional[float] = None
    if valid_classes:
        all_fpr = np.unique(np.concatenate([fpr[c] for c in valid_classes]))
        mean_tpr = np.zeros_like(all_fpr)
        for c in valid_classes:
            mean_tpr += np.interp(all_fpr, fpr[c], tpr[c])
        mean_tpr /= len(valid_classes)
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        macro_auc = float(auc(fpr["macro"], tpr["macro"]))
        roc_auc["macro"] = macro_auc

    fig, ax = plt.subplots(figsize=(9, 7))
    if "micro" in fpr:
        ax.plot(
            fpr["micro"],
            tpr["micro"],
            label=f"Micro-average ROC (AUC = {roc_auc['micro']:.4f})",
            linestyle=":",
            linewidth=2.5,
        )
    if "macro" in fpr:
        ax.plot(
            fpr["macro"],
            tpr["macro"],
            label=f"Macro-average ROC (AUC = {roc_auc['macro']:.4f})",
            linestyle="-",
            linewidth=2.5,
        )

    # Plot lowest-AUC classes first so potential weak spots are visible
    for c in sorted(valid_classes, key=lambda idx: roc_auc[idx])[:max_class_curves]:
        short_name = class_names[c].split("___")[-1][:24]
        ax.plot(
            fpr[c],
            tpr[c],
            alpha=0.55,
            linewidth=1.2,
            label=f"{short_name} (AUC = {roc_auc[c]:.3f})",
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Multi-Class One-vs-Rest ROC Curves")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROC curves saved -> %s", save_path)

    return {"micro_auc": micro_auc, "macro_auc": macro_auc}


def evaluate_model(
    model: PlantDiseaseClassifier,
    loader: DataLoader,
    class_names: Optional[List[str]] = None,
    output_dir: str | Path = "./evaluation",
    device: Optional[torch.device] = None,
    max_gradcam_samples: int = 12,
    cam_alpha: float = 0.5,
) -> Dict:
    """
    Run full evaluation on a DataLoader and export metrics + visualizations.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if class_names is None:
        class_names = (
            list(model.class_names)
            if getattr(model, "class_names", None)
            else PLANT_DISEASE_CLASSES[: model.num_classes]
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true, y_pred, y_prob, misclassified = collect_predictions(
        model, loader, device=device, max_misclassified=max_gradcam_samples
    )

    labels_idx = list(range(len(class_names)))
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels_idx,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels_idx,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")

    metrics = {
        "num_samples": int(len(y_true)),
        "num_misclassified": int(np.sum(y_true != y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    cm_path = plot_confusion_matrix(
        y_true, y_pred, class_names, output_dir / "confusion_matrix.png", normalize=True
    )
    roc_stats = plot_roc_curves(
        y_true, y_prob, class_names, output_dir / "roc_curves.png"
    )
    metrics.update(roc_stats)

    gradcam_paths: List[str] = []
    if misclassified:
        saved = generate_cams_for_samples(
            model=model,
            samples=misclassified,
            class_names=class_names,
            output_dir=output_dir / "misclassified_gradcam",
            alpha=cam_alpha,
        )
        gradcam_paths = [str(p) for p in saved]

    summary = {
        "metrics": metrics,
        "artifacts": {
            "classification_report": str(output_dir / "classification_report.txt"),
            "confusion_matrix": str(cm_path),
            "roc_curves": str(output_dir / "roc_curves.png"),
            "misclassified_gradcam": gradcam_paths,
        },
        "per_class": report_dict,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    logger.info(
        "Evaluation complete | acc=%.4f | macro_f1=%.4f | weighted_f1=%.4f | misclassified=%d/%d",
        metrics["accuracy"],
        metrics["macro_f1"],
        metrics["weighted_f1"],
        metrics["num_misclassified"],
        metrics["num_samples"],
    )
    return summary
