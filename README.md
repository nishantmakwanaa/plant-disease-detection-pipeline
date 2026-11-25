# Plant Disease Detection Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![timm](https://img.shields.io/badge/timm-EfficientNet--B3-orange.svg)](https://huggingface.co/docs/timm)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end computer vision pipeline for classifying **38 plant leaf diseases and healthy classes** across **14 crop species** using transfer learning (**EfficientNet-B3** / **ResNet-50** via `timm`) with **Grad-CAM / Grad-CAM++** visual explainability.

---

## Dataset & Source

- **Dataset:** **PlantVillage** (originally introduced by Hughes & Salathé, 2015; commonly obtained from Kaggle [`emmarex/plantdisease`](https://www.kaggle.com/datasets/emmarex/plantdisease) / [`abdallahalidev/plantvillage-dataset`](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) or Hugging Face PlantVillage mirrors).
- **Scale & Classes:** **38 classes** across **14 crops** (26 diseased + 12 healthy leaf categories; ~54,303 original images or ~87K augmented images depending on distribution). All 38 canonical labels are listed in `PLANT_DISEASE_CLASSES` inside [`src/model.py`](src/model.py).
- **Expected Local Format (`torchvision.datasets.ImageFolder`):**

```text
data/
├── Apple___Apple_scab/
│   ├── image_0001.jpg
│   └── ...
├── Apple___Black_rot/
├── ...
└── Tomato___healthy/
```

- **Splits & Preprocessing ([`src/dataset.py`](src/dataset.py)):**
  - Stratified split (`seed=42`): **70% Train / 15% Validation / 15% Test**.
  - **Train transforms (`300x300`):** Resize(`332x332`) → RandomCrop(`300`) → RandomHorizontalFlip(`0.5`) → RandomVerticalFlip(`0.5`) → RandomRotation(`±30°`) → ColorJitter(`0.3, 0.3, 0.3, 0.1`) → RandomGrayscale(`0.05`) → ImageNet normalization → RandomErasing(`0.1`).
  - **Val / Test / Inference transforms (`300x300`):** Deterministic Resize(`300x300`) + ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
  - **Class Imbalance Handling:** Computes inverse-frequency class weights (`get_class_weights()`) and balanced batch sampling via `WeightedRandomSampler`.

---

## Model & Architecture

Implemented in [`src/model.py`](src/model.py) (`PlantDiseaseClassifier`) and [`src/gradcam.py`](src/gradcam.py):

- **Model:** **EfficientNet-B3** (`timm.create_model("efficientnet_b3", pretrained=True, num_classes=0, global_pool="")`) initialized with **ImageNet-1K pre-trained weights**. Supports switching to **`resnet50`** (or any `timm` CNN backbone) via `--arch` or [`configs/default.yaml`](configs/default.yaml).
- **Architecture Flow:**

```text
Input Leaf Image (B, 3, 300, 300)
        │
        ▼
EfficientNet-B3 Backbone (timm, ImageNet pretrained, spatial feature output)
        │   ──▶ (B, 1536, 10, 10) feature maps (last Conv2d hooked for Grad-CAM)
        ▼
ClassificationHead
  ├── AdaptiveAvgPool2d(1) ──▶ (B, 1536, 1, 1)
  ├── Flatten              ──▶ (B, 1536)
  ├── Dropout(p=0.4)
  └── Linear(1536, 38)     ──▶ Class Logits (B, 38)
```

### How the Model Is Trained, Evaluated & Used

1. **Two-Stage Transfer Learning ([`src/train.py`](src/train.py)):**
   - **Warm-up Stage (Epochs 1–4):** Backbone weights are frozen (`freeze_backbone_epochs: 5`); only `ClassificationHead` trains with `AdamW` (`lr=3e-4`, `weight_decay=1e-4`).
   - **Fine-Tuning Stage (Epoch 5+):** Backbone unfreezes with discriminative learning rates (**`0.1x` LR** for backbone parameters, **`1.0x` LR** for head) and `CosineAnnealingLR` (`eta_min=1e-6`).
   - **Regularization & Stability:** Weighted `CrossEntropyLoss(label_smoothing=0.1)`, AMP mixed precision (`torch.cuda.amp`), gradient clipping (`1.0`), early stopping (`patience=10` on `val_acc`), and top-3 checkpointing (`checkpoints/best.pth`).
2. **Evaluation & Error Analysis ([`src/evaluate.py`](src/evaluate.py)):**
   - Generates per-class precision/recall/F1 (`classification_report.txt`, `metrics.json`), normalized `confusion_matrix.png`, macro/micro One-vs-Rest `roc_curves.png`, and Grad-CAM heatmaps for misclassified test images.
3. **Inference & Explainability ([`src/inference.py`](src/inference.py)):**
   - [`PlantDiseasePredictor`](src/inference.py) loads `best.pth`, runs single-image or directory batch inference, parses `Crop___Disease` labels into human-readable top-3 predictions with confidence scores, and overlays **Grad-CAM** or **Grad-CAM++** heatmaps onto the original leaf image.

---

## Repository Structure

```text
plant-disease-detection-pipeline/
├── configs/
│   └── default.yaml          # Model, dataset, augmentation, optimizer, and CAM config
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

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/nishantmakwanaa/plant-disease-detection-pipeline.git
cd plant-disease-detection-pipeline
pip install -r requirements.txt
```

### 2. Prepare PlantVillage Data

Download the PlantVillage dataset (e.g., from Kaggle) and place the 38 class folders inside `./data/`.

### 3. Train

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --data_dir ./data \
  --epochs 50 \
  --batch_size 32
```

Outputs: top-3 checkpoints + `./checkpoints/best.pth` and `./logs/training_curves.png`.

### 4. Evaluate

```bash
python scripts/evaluate.py \
  --checkpoint ./checkpoints/best.pth \
  --data_dir ./data \
  --output_dir ./evaluation
```

### 5. Inference & Grad-CAM

```bash
# Single image with Grad-CAM++ overlay
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

Expected benchmark targets for EfficientNet-B3 fine-tuned on the 38-class PlantVillage dataset (no dataset or trained checkpoint is bundled in this repository):

| Metric | Target Benchmark |
| :--- | :---: |
| **Top-1 Test Accuracy** | ≥ 96.0% |
| **Macro F1 Score** | ≥ 95.5% |
| **Weighted F1 Score** | ≥ 96.0% |
| **Macro OvR ROC-AUC** | ≥ 0.995 |
