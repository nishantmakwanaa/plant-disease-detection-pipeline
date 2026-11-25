# Plant Disease Detection Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![CNN](https://img.shields.io/badge/CNN-EfficientNet--B3%20%7C%20ResNet--50-orange.svg)](https://huggingface.co/docs/timm)
[![Explainability](https://img.shields.io/badge/XAI-Grad--CAM%20%2B%2B-brightgreen.svg)](https://arxiv.org/abs/1710.11063)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end deep learning pipeline for classifying **38 plant leaf diseases and healthy classes** across **14 crop species** using transfer learning (**EfficientNet-B3** / **ResNet-50** via `timm`) and **Grad-CAM / Grad-CAM++** visual explainability.

---

## Proof & Reference Links (Datasets, Papers, Code & Configs)

| Category | Resource | Link |
| :--- | :--- | :--- |
| **Dataset Paper** | Hughes & Salathé (2015), *An open access repository of images on plant health to enable the development of mobile disease diagnostics* | [`arXiv:1511.08060`](https://arxiv.org/abs/1511.08060) |
| **Dataset Mirrors** | PlantVillage Dataset on Kaggle | [`emmarex/plantdisease`](https://www.kaggle.com/datasets/emmarex/plantdisease) · [`abdallahalidev/plantvillage-dataset`](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) |
| **Primary CNN Paper** | Tan & Le (2019), *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks* | [`arXiv:1905.11946`](https://arxiv.org/abs/1905.11946) |
| **Alternative CNN Paper** | He et al. (2015), *Deep Residual Learning for Image Recognition* (ResNet-50) | [`arXiv:1512.03385`](https://arxiv.org/abs/1512.03385) |
| **Model Library Docs** | PyTorch Image Models (`timm`) | [Hugging Face `timm` Docs](https://huggingface.co/docs/timm) |
| **Grad-CAM Paper** | Selvaraju et al. (2017), *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization* | [`arXiv:1610.02391`](https://arxiv.org/abs/1610.02391) |
| **Grad-CAM++ Paper** | Chattopadhyay et al. (2018), *Grad-CAM++: Generalized Gradient-Based Visual Explanations* | [`arXiv:1710.11063`](https://arxiv.org/abs/1710.11063) |
| **Model Implementation** | `PlantDiseaseClassifier`, `ClassificationHead`, 38 classes | [`src/model.py`](src/model.py) |
| **Data & Augmentation** | `PlantVillageDataset`, stratified splits, class weights, sampler | [`src/dataset.py`](src/dataset.py) |
| **Explainability Code** | `GradCAM`, `GradCAMPlusPlus`, heatmap blending | [`src/gradcam.py`](src/gradcam.py) |
| **Training & Evaluation** | Two-stage trainer, AMP, ROC-AUC, confusion matrix | [`src/train.py`](src/train.py) · [`src/evaluate.py`](src/evaluate.py) |
| **Hyperparameters** | Full YAML training & inference config | [`configs/default.yaml`](configs/default.yaml) |

---

## Dataset & Source (PlantVillage)

- **Source:** **PlantVillage** ([`arXiv:1511.08060`](https://arxiv.org/abs/1511.08060)), downloadable from Kaggle ([`emmarex/plantdisease`](https://www.kaggle.com/datasets/emmarex/plantdisease) or [`abdallahalidev/plantvillage-dataset`](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)).
- **Classes & Scale:** **38 leaf classes** across **14 crops** (26 diseased + 12 healthy classes; ~54,303 original images or ~87K augmented images depending on release). Every class string is defined in `PLANT_DISEASE_CLASSES` in [`src/model.py`](src/model.py).
- **Expected Directory Format (`torchvision.datasets.ImageFolder`):**

```text
data/
├── Apple___Apple_scab/
│   ├── image_0001.jpg
│   └── ...
├── Apple___Black_rot/
├── ...
└── Tomato___healthy/
```

- **Stratified Splitting & Preprocessing ([`src/dataset.py`](src/dataset.py)):**
  - **Splits (`seed=42`):** **70% Train / 15% Validation / 15% Test** via `sklearn.model_selection.train_test_split(..., stratify=labels)`.
  - **Train Augmentations (`300×300`):** `Resize(332×332) → RandomCrop(300) → RandomHorizontalFlip(0.5) → RandomVerticalFlip(0.5) → RandomRotation(±30°) → ColorJitter(0.3, 0.3, 0.3, 0.1) → RandomGrayscale(0.05) → ImageNet Normalization → RandomErasing(0.1)`.
  - **Val / Test / Inference (`300×300`):** Deterministic `Resize(300×300)` + ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
  - **Class Imbalance Handling:** Inverse-frequency class weights (`get_class_weights()`) combined with `WeightedRandomSampler`.

---

## CNN Architecture & Explainability

Implemented in [`PlantDiseaseClassifier`](src/model.py) ([`src/model.py`](src/model.py)) and [`src/gradcam.py`](src/gradcam.py):

- **Primary CNN — EfficientNet-B3 (`timm`, ImageNet-1K pretrained):**
  - Compound-scaled mobile inverted bottleneck CNN (**MBConv** blocks with depthwise separable convolutions, **Squeeze-and-Excitation** channel attention, and **SiLU/Swish** activations; [`arXiv:1905.11946`](https://arxiv.org/abs/1905.11946)).
  - Loaded with `num_classes=0, global_pool=""` so spatial feature maps `(B, 1536, 10, 10)` are preserved for Grad-CAM.
- **Alternative CNN — ResNet-50 (`--arch resnet50`):**
  - 50-layer residual bottleneck CNN ([`arXiv:1512.03385`](https://arxiv.org/abs/1512.03385)) outputting `(B, 2048, 10, 10)` spatial feature maps; head input dimension is inferred dynamically.
- **Custom Classification Head (`ClassificationHead`):**
  - `AdaptiveAvgPool2d(1) → Flatten → Dropout(p=0.4) → Linear(1536, 38)`.
- **Grad-CAM & Grad-CAM++ Hooks:**
  - Automatically locates the final `nn.Conv2d` layer in the backbone, registers forward activation and full backward gradient hooks, and renders class-discriminative lesion heatmaps (`overlay_heatmap()`, `save_cam_image()`).

```text
Input Leaf Image (B, 3, 300, 300)
        │
        ▼
EfficientNet-B3 CNN Backbone (timm, ImageNet-1K pretrained, MBConv + SE blocks)
        │   ──▶ (B, 1536, 10, 10) spatial feature maps ──► Grad-CAM / Grad-CAM++ hooks
        ▼
ClassificationHead
  ├── AdaptiveAvgPool2d(1) ──▶ (B, 1536, 1, 1)
  ├── Flatten              ──▶ (B, 1536)
  ├── Dropout(p=0.4)
  └── Linear(1536, 38)     ──▶ 38-Class Logits (B, 38)
```

### How the Model Is Trained, Evaluated & Used

1. **Two-Stage Transfer Learning ([`src/train.py`](src/train.py)):**
   - **Stage 1 (Warm-up, Epochs 1–4):** Backbone frozen (`freeze_backbone_epochs: 5`); trains only `ClassificationHead` (`AdamW`, `lr=3e-4`, `weight_decay=1e-4`).
   - **Stage 2 (Fine-Tuning, Epoch 5+):** Unfreezes backbone with discriminative learning rates (**`0.1×` LR** for backbone, **`1.0×` LR** for head) and `CosineAnnealingLR` (`eta_min=1e-6`).
   - **Loss & Regularization:** Weighted `CrossEntropyLoss(label_smoothing=0.1)`, AMP mixed precision (`torch.cuda.amp`), gradient clipping (`1.0`), early stopping (`patience=10` on `val_acc`), top-3 checkpoint pruning (`checkpoints/best.pth`), and loss/accuracy curve plotting.
2. **Evaluation ([`src/evaluate.py`](src/evaluate.py)):**
   - Outputs per-class precision/recall/F1 (`classification_report.txt`, `metrics.json`), normalized `confusion_matrix.png`, macro/micro One-vs-Rest `roc_curves.png`, and Grad-CAM images for misclassified samples.
3. **Inference ([`src/inference.py`](src/inference.py)):**
   - [`PlantDiseasePredictor`](src/inference.py) loads a `.pth` checkpoint, runs single-image or directory batch inference, returns top-3 `{crop, condition, confidence}` predictions, and optionally saves **Grad-CAM** / **Grad-CAM++** overlays.

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

```bash
# 1. Clone repository & install dependencies
git clone https://github.com/nishantmakwanaa/plant-disease-detection-pipeline.git
cd plant-disease-detection-pipeline
pip install -r requirements.txt

# 2. Place PlantVillage class folders under ./data/, then train
python scripts/train.py --config configs/default.yaml --data_dir ./data --epochs 50 --batch_size 32

# 3. Evaluate on the test split and export Grad-CAM error analysis
python scripts/evaluate.py --checkpoint ./checkpoints/best.pth --data_dir ./data --output_dir ./evaluation

# 4. Run single-image inference with Grad-CAM++ overlay (or batch --dir)
python scripts/infer.py --image leaf.jpg --checkpoint ./checkpoints/best.pth --gradcam --cam_method gradcam++ --output leaf_gradcam.jpg
python scripts/infer.py --dir ./sample_leaves --checkpoint ./checkpoints/best.pth --json
```

---

## Target Benchmark Metrics

Expected benchmark targets for EfficientNet-B3 fine-tuned on the 38-class PlantVillage dataset (no dataset or pre-trained checkpoint is bundled in this repository):

| Metric | Target Benchmark |
| :--- | :---: |
| **Top-1 Test Accuracy** | ≥ 96.0% |
| **Macro F1 Score** | ≥ 95.5% |
| **Weighted F1 Score** | ≥ 96.0% |
| **Macro OvR ROC-AUC** | ≥ 0.995 |
