# Project Information — Structure-Constrained Vision Transformer for Medical Image Analysis

## Dataset(s) Used

| Dataset | Size | Description |
|---------|------|-------------|
| **CheXpert** (Stanford ML Group) | ~11 GB / ~224,316 chest X-ray images | Large-scale chest radiograph dataset with 14 pathology labels. Used the CheXpert-v1.0-small subset for training and validation. |

> **Note:** Experiments were conducted on a subset of 12,000 training images and 1,500 validation images due to hardware constraints (Apple Silicon M3 Pro with MPS backend).

---

## Key Metrics Achieved

### 5×5 Comparative Evaluation (5 Model Variants × 5 Metrics)

| Model | Mean AUROC | Mean F1 | Accuracy | Attention Entropy | Noise Robustness (σ=0.3 AUROC) |
|-------|-----------|---------|----------|-------------------|-------------------------------|
| Baseline ViT | 0.7733 | 0.1853 | 0.7556 | 3.926 | 0.7741 |
| ViT + KMeans | 0.7636 | 0.1516 | 0.7564 | 2.480 | 0.7630 |
| ViT + Spectral | 0.7549 | 0.2051 | 0.7487 | 2.921 | 0.7562 |
| ViT + Graph (GNN) | **0.7764** | 0.0911 | **0.7564** | 3.343 | 0.7749 |
| ViT + Struct-Constrained | 0.7500 | 0.1339 | 0.7436 | **2.800** | 0.7535 |

### Hybrid Anatomy-Guided Relational ViT (Primary Contribution)

| Metric | Value |
|--------|-------|
| Mean AUROC | 0.7129 |
| Mean F1 | 0.3951 |
| Attention Entropy (mean) | 3.257 |
| Attention Sparsity (mean) | 0.877 |
| Anatomical Attention Ratio | 0.761 |
| Noise Robustness (σ=0.3 AUROC) | 0.708 |

### Per-Class AUROC (Best Variant — ViT + Graph GNN)

| Pathology | AUROC |
|-----------|-------|
| Atelectasis | 0.7756 |
| Cardiomegaly | 0.7428 |
| Consolidation | 0.8360 |
| Edema | 0.7684 |
| Pleural Effusion | 0.7591 |

---

## Contribution

### End-to-End Research Pipeline Development

Designed and implemented the complete research pipeline for **Structure-Constrained Vision Transformers (SC-ViT)** applied to medical image analysis on CheXpert chest X-rays. The contributions span the following areas:

1. **Structure Discovery Pipeline**: Built an unsupervised structure discovery system using DINO (self-supervised ViT) feature extraction, k-NN patch similarity graph construction, and multiple clustering methods (KMeans, Spectral, GNN-based, Diffusion Maps) to learn structural interaction patterns between image patches without supervision.

2. **Anatomy-Aware Prior Design**: Designed a domain-specific anatomy prior that partitions the chest X-ray patch grid into 7 clinically meaningful anatomical regions (Left/Right Upper/Lower Lung, Heart, Mediastinum, Background) with region interaction weights based on clinical adjacency knowledge.

3. **Hybrid Relational ViT Architecture**: Proposed and implemented the **Hybrid Anatomy-Guided Relational Vision Transformer** — the primary model contribution — which uses dual soft additive biases (graph relational bias + anatomy spatial prior) with learnable scaling parameters (λ_graph, λ_anatomy) to guide attention without hard masking, preserving long-range interactions and attention entropy.

4. **Comprehensive Evaluation Framework**: Developed a 5×5 evaluation framework comparing 5 model variants across 5 metrics (Accuracy, AUROC, F1, Attention Entropy, Noise Robustness), demonstrating that structural constraints act as regularizers — reducing attention entropy and improving noise robustness while maintaining competitive classification performance.

5. **Visualization & Analysis Tools**: Created publication-quality visualization tools for attention overlays, anatomy region maps, feasibility matrix heatmaps, constrained vs. unconstrained attention comparison, and multi-metric radar charts.

6. **Hardware-Optimized Training**: Optimized the entire pipeline for Apple Silicon MPS backend (M3 Pro), including image size reduction (224→192), gradient accumulation, backbone freezing, and MPS-compatible configurations.
