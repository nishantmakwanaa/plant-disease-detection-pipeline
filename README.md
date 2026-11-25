# Plant Disease Detection Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![timm](https://img.shields.io/badge/timm-EfficientNet--B3-orange.svg)](https://huggingface.co/docs/timm)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end deep learning pipeline for automated plant leaf disease classification across **38 PlantVillage classes** (14 crop species, 26 diseases + 12 healthy classes) with **Grad-CAM / Grad-CAM++** visual explainability.

---

## Key Highlights

- **Transfer Learning Backbone:** Uses `timm` **EfficientNet-B3** (or configurable `resnet50`) pre-trained on ImageNet with a custom `AdaptiveAvgPool2d → Dropout → Linear` classification head.
- **Two-Stage Fine-Tuning:** Freezes the backbone during the warm-up phase (`freeze_backbone_epochs: 5`), then unfreezes all layers with discriminative learning rates (`0.1x` backbone LR, `1.0x` head LR) and cosine annealing.
- **Imbalance & Regularization:** Stratified `70 / 15 / 15` train/val/test split, inverse-frequency class weights, `WeightedRandomSampler`, label smoothing (`0.1`), mixed precision (`torch.cuda.amp`), gradient clipping, and early stopping.
- **Full Evaluation Suite:** Per-class precision/recall/F1 report, normalized confusion matrix, macro/micro One-vs-Rest ROC-AUC curves, and automatic Grad-CAM inspection on misclassified samples.
- **Explainable Inference:** Top-k predictions (`top_k=3`) for single images or directories with optional Grad-CAM / Grad-CAM++ heatmap overlays.

---

## Model Architecture

```text
Input Image (B, 3, 300, 300)
        │
        ▼
EfficientNet-B3 Backbone (timm, ImageNet pretrained, num_classes=0, global_pool="")
        │   ──▶ (B, 1536, 10, 10) spatial feature maps (hooked for Grad-CAM)
        ▼
ClassificationHead
  ├── AdaptiveAvgPool2d(1) ──▶ (B, 1536, 1, 1)
  ├── Flatten              ──▶ (B, 1536)
  ├── Dropout(p=0.4)
  └── Linear(1536, 38)     ──▶ Logits (B, 38)
```

---

## Dataset (PlantVillage)

The pipeline expects the **PlantVillage** dataset (38 classes, ~54K–87K leaf images depending on augmented vs. raw split) organized in standard `torchvision.datasets.ImageFolder` layout:

```text
data/
├── Apple___Apple_scab/
│   ├── image_0001.jpg
│   └── ...
├── Apple___Black_rot/
├── ...
└── Tomato___healthy/
```

All 38 class labels are defined in [`PLANT_DISEASE_CLASSES`](src/model.py). Splits are created reproducibly (`seed: 42`) with stratified sampling:
- **Train:** 70% (random crop, H/V flips, rotation ±30°, color jitter, random grayscale, random erasing)
- **Validation:** 15% (deterministic resize & ImageNet normalization)
- **Test:** 15% (deterministic resize & ImageNet normalization)

---

## Repository Structure

```text
plant-disease-detection-pipeline/
├── configs/
│   └── default.yaml          # Training, model, dataset, and inference hyperparameters
├── scripts/
│   ├── train.py              # CLI training launcher
│   ├── evaluate.py           # CLI test/val evaluation & artifact export
│   └── infer.py              # CLI single-image / directory inference + Grad-CAM
├── src/
│   ├── __init__.py           # Package exports
│   ├── dataset.py            # PlantVillageDataset, transforms, stratified splits, sampler
│   ├── model.py              # PlantDiseaseClassifier, ClassificationHead, 38 classes
│   ├── gradcam.py            # GradCAM, GradCAMPlusPlus, heatmap overlay utilities
│   ├── train.py              # Trainer, EarlyStopping, CheckpointManager, curve plotting
│   ├── evaluate.py           # Classification report, confusion matrix, ROC curves, CAMs
│   └── inference.py          # PlantDiseasePredictor for single/batch/directory inference
├── requirements.txt          # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

## Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --data_dir ./data \
  --epochs 50 \
  --batch_size 32
```

Artifacts saved:
- Checkpoints (top-3 + `best.pth`) in `./checkpoints/`
- Training loss/accuracy curves in `./logs/training_curves.png`

### 3. Evaluate

```bash
python scripts/evaluate.py \
  --checkpoint ./checkpoints/best.pth \
  --data_dir ./data \
  --output_dir ./evaluation
```

Exports:
- `evaluation/classification_report.txt`
- `evaluation/metrics.json`
- `evaluation/confusion_matrix.png`
- `evaluation/roc_curves.png`
- `evaluation/misclassified_gradcam/*.jpg`

### 4. Inference & Grad-CAM

```bash
# Single image with Grad-CAM++ heatmap overlay
python scripts/infer.py \
  --image leaf.jpg \
  --checkpoint ./checkpoints/best.pth \
  --gradcam \
  --cam_method gradcam++ \
  --output leaf_gradcam.jpg

# Directory batch inference (JSON output)
python scripts/infer.py \
  --dir ./sample_leaves \
  --checkpoint ./checkpoints/best.pth \
  --json
```

---

## Target Benchmark Metrics

Expected benchmark targets for EfficientNet-B3 fine-tuned on the 38-class PlantVillage dataset (before local dataset training is executed):

| Metric | Target Benchmark |
| :--- | :---: |
| **Top-1 Test Accuracy** | ≥ 96.0% |
| **Macro F1 Score** | ≥ 95.5% |
| **Weighted F1 Score** | ≥ 96.0% |
| **Macro OvR ROC-AUC** | ≥ 0.995 |

| Crop Group (Examples) | Classes Included | Target F1 |
| :--- | :---: | :---: |
| Apple (`Apple_scab`, `Black_rot`, `Cedar_apple_rust`, `healthy`) | 4 | ≥ 96% |
| Corn (`Cercospora_leaf_spot`, `Common_rust_`, `Northern_Leaf_Blight`, `healthy`) | 4 | ≥ 95% |
| Grape (`Black_rot`, `Esca`, `Leaf_blight`, `healthy`) | 4 | ≥ 96% |
| Potato (`Early_blight`, `Late_blight`, `healthy`) | 3 | ≥ 96% |
| Tomato (`Bacterial_spot`, `Early_blight`, `Late_blight`, `Leaf_Mold`, `Septoria_leaf_spot`, `Spider_mites`, `Target_Spot`, `Yellow_Leaf_Curl_Virus`, `mosaic_virus`, `healthy`) | 10 | ≥ 94% |
| Other Crops (`Blueberry`, `Cherry`, `Orange`, `Peach`, `Pepper`, `Raspberry`, `Soybean`, `Squash`, `Strawberry`) | 13 | ≥ 96% |
