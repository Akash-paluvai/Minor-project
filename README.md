# Structure-Constrained Learning for Vision Transformers in Medical Image Analysis

A research pipeline that learns structural interaction patterns between image patches using unsupervised methods, then constrains ViT attention to anatomically plausible interactions for CheXpert chest X-ray multi-label classification.

## Architecture

```
Image (224×224)
    ↓
Patch Embedding (16×16 → 196 patches)
    ↓
DINO Feature Extractor (frozen, pretrained) → patch features (196 × 768)
    ↓
Patch Similarity Graph (k-NN, cosine similarity)
    ↓
Spectral Clustering → K clusters of patches
    ↓
Structural Feasibility Matrix (197 × 197)
    ↓
Structure-Constrained ViT (masked attention)
    ↓
Multi-label Prediction (5 pathologies)
```

## Setup

```bash
pip install -r requirements.txt
```

Download the CheXpert-v1.0-small dataset from [Kaggle](https://www.kaggle.com/datasets/ashery/chexpert) and place it in `data/CheXpert-v1.0-small/`.

## Quick Start

### 1. Train Baseline ViT
```bash
python train_baseline.py --config configs/default.yaml
# Dry-run to verify pipeline:
python train_baseline.py --config configs/default.yaml --dry-run
```

### 2. Build Structure (Unsupervised)
```bash
python build_structure.py --config configs/default.yaml --visualize
```

### 3. Train Constrained ViT
```bash
python train_constrained.py --config configs/default.yaml
```

### 4. Evaluate & Compare
```bash
python evaluate.py --config configs/default.yaml --visualize-attention
```

### 5. Launch Web Research Dashboard
```bash
python gui_dashboard.py
```
Open `http://127.0.0.1:7860` in your browser to view:
- **Experiment Results**: Model metrics table and comparison charts (AUROC, F1, Entropy, Noise Robustness)
- **Visualization Gallery**: Browser for clustering, graph topology, feasibility matrices, and dataset samples
- **Test on Images**: Multi-label pathology prediction and calibrated inference
- **Architecture**: Heuristic anatomy priors and structural attention pipeline

## Project Structure

```
├── configs/default.yaml           # All hyperparameters
├── src/
│   ├── data/
│   │   ├── chexpert_dataset.py    # Dataset with uncertainty handling
│   │   └── transforms.py         # Train/val/SimCLR augmentations
│   ├── models/
│   │   ├── vit.py                 # Baseline Vision Transformer
│   │   ├── constrained_vit.py     # Structure-constrained ViT
│   │   └── feature_extractor.py   # DINO patch feature extractor
│   ├── structure/
│   │   ├── graph_builder.py       # k-NN patch similarity graph
│   │   ├── clustering.py          # Spectral clustering
│   │   └── feasibility.py         # Feasibility matrix builder
│   └── training/
│       ├── trainer.py             # Training loop (AMP, cosine LR)
│       ├── metrics.py             # AUROC, AUPRC, F1
│       └── evaluate.py            # Evaluation + attention viz
├── train_baseline.py              # Entry: baseline ViT training
├── build_structure.py             # Entry: structure discovery
├── train_constrained.py           # Entry: constrained ViT training
└── evaluate.py                    # Entry: comparison & evaluation
```

## Key Features

- **Uncertainty handling**: U-Ones, U-Zeros, U-Ignore policies for CheXpert labels
- **Pretrained weights**: timm ViT-B/16 initialization
- **Mixed precision**: AMP training for memory efficiency
- **Structure discovery**: DINO → k-NN graph → spectral clustering → feasibility matrix
- **Soft masking**: Inter-cluster attention scaled (not just binary blocked)
- **Visualization**: Cluster maps, feasibility matrices, attention overlays, comparison charts

## Metrics

- **Primary**: Mean AUROC across 5 pathologies (CheXpert competition standard)
- **Secondary**: AUPRC, F1 score
- **Per-class**: Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion
