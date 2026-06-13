#!/usr/bin/env python3
"""
Extract Patch Embeddings from CheXpert Images.

Uses a pretrained self-supervised ViT (DINO, MAE, or HuggingFace ViT)
to convert each image into patch-level embeddings φ_i ∈ R^d.

Output:
  For each image → (N_patches, embed_dim) tensor
  Saved as a .pt file for downstream clustering.

Usage:
    python extract_features.py --config configs/default.yaml
    python extract_features.py --model-type dino --num-samples 1000
    python extract_features.py --model-type mae --num-samples 500
    python extract_features.py --model-type huggingface --num-samples 500
"""

import os
import argparse
import time
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.data.chexpert_dataset import CheXpertDataset, get_dataset_path
from src.data.transforms import get_val_transforms
from src.models.feature_extractor import (
    create_feature_extractor,
    save_embeddings,
    load_embeddings,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract Patch Embeddings")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--model-type", type=str, default="dino",
        choices=["dino", "mae", "huggingface"],
        help="Pretrained model backend"
    )
    parser.add_argument(
        "--model-name", type=str, default=None,
        help="Specific model name (overrides default for chosen type)"
    )
    parser.add_argument(
        "--num-samples", type=int, default=1000,
        help="Number of images to extract features from"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for extraction"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for embeddings .pt file"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Generate embedding visualizations"
    )
    return parser.parse_args()


def visualize_embeddings(embeddings: torch.Tensor, output_dir: str):
    """
    Visualize extracted patch embeddings.

    Creates:
    1. Patch embedding similarity heatmap for sample images
    2. Mean embedding norm per patch position
    3. t-SNE / PCA of patch embeddings
    """
    os.makedirs(output_dir, exist_ok=True)
    N, num_patches, embed_dim = embeddings.shape
    grid_size = int(np.sqrt(num_patches))

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ---- Plot 1: Cosine similarity matrix for first image ----
    patch_feat = embeddings[0]  # (196, 768)
    patch_norm = torch.nn.functional.normalize(patch_feat, dim=-1)
    sim_matrix = (patch_norm @ patch_norm.T).numpy()

    im0 = axes[0, 0].imshow(sim_matrix, cmap="viridis", vmin=-0.2, vmax=1.0)
    axes[0, 0].set_title("Patch Cosine Similarity (Image 0)", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Patch index")
    axes[0, 0].set_ylabel("Patch index")
    plt.colorbar(im0, ax=axes[0, 0], shrink=0.8)

    # ---- Plot 2: Mean embedding norm per patch position ----
    norms = torch.norm(embeddings, dim=-1)  # (N, 196)
    mean_norms = norms.mean(dim=0).numpy()  # (196,)
    norm_grid = mean_norms.reshape(grid_size, grid_size)

    im1 = axes[0, 1].imshow(norm_grid, cmap="hot", interpolation="bilinear")
    axes[0, 1].set_title("Mean Embedding Norm per Patch", fontsize=12, fontweight="bold")
    plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)

    # ---- Plot 3: Embedding variance per patch ----
    variance = embeddings.var(dim=0).mean(dim=-1).numpy()  # (196,)
    var_grid = variance.reshape(grid_size, grid_size)

    im2 = axes[1, 0].imshow(var_grid, cmap="plasma", interpolation="bilinear")
    axes[1, 0].set_title("Embedding Variance per Patch (across images)", fontsize=12, fontweight="bold")
    plt.colorbar(im2, ax=axes[1, 0], shrink=0.8)

    # ---- Plot 4: PCA projection of patch embeddings ----
    from sklearn.decomposition import PCA

    # Sample patches from multiple images for PCA
    sample_count = min(20, N)
    sample_patches = embeddings[:sample_count].reshape(-1, embed_dim).numpy()

    pca = PCA(n_components=2)
    projected = pca.fit_transform(sample_patches)

    # Color by patch position
    patch_positions = np.tile(np.arange(num_patches), sample_count)
    scatter = axes[1, 1].scatter(
        projected[:, 0], projected[:, 1],
        c=patch_positions, cmap="tab20", s=2, alpha=0.5
    )
    axes[1, 1].set_title(
        f"PCA of Patch Embeddings (n={sample_count} images)",
        fontsize=12, fontweight="bold"
    )
    axes[1, 1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    axes[1, 1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    plt.colorbar(scatter, ax=axes[1, 1], shrink=0.8, label="Patch index")

    plt.suptitle("Patch Embedding Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_path = os.path.join(output_dir, "embedding_analysis.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved embedding visualization to {save_path}")

    # ---- Additional: Per-patch feature heatmap (first 3 PCA components as RGB) ----
    fig2, axes2 = plt.subplots(1, 4, figsize=(20, 5))

    # Average embeddings across all images
    mean_embedding = embeddings.mean(dim=0).numpy()  # (196, 768)
    pca3 = PCA(n_components=3)
    rgb_features = pca3.fit_transform(mean_embedding)  # (196, 3)

    # Normalize to [0, 1] for RGB display
    rgb_min = rgb_features.min(axis=0)
    rgb_max = rgb_features.max(axis=0)
    rgb_norm = (rgb_features - rgb_min) / (rgb_max - rgb_min + 1e-8)
    rgb_grid = rgb_norm.reshape(grid_size, grid_size, 3)

    axes2[0].imshow(rgb_grid)
    axes2[0].set_title("PCA RGB (3 components)", fontweight="bold")

    for i, (comp_name, cmap) in enumerate([("PC1", "Reds"), ("PC2", "Greens"), ("PC3", "Blues")]):
        comp = rgb_features[:, i].reshape(grid_size, grid_size)
        im = axes2[i + 1].imshow(comp, cmap=cmap, interpolation="bilinear")
        axes2[i + 1].set_title(f"{comp_name} ({pca3.explained_variance_ratio_[i]:.1%})", fontweight="bold")
        plt.colorbar(im, ax=axes2[i + 1], shrink=0.8)

    for ax in axes2:
        ax.axis("off")

    plt.suptitle("Population-Average Patch Features (PCA)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_path2 = os.path.join(output_dir, "patch_pca_rgb.png")
    plt.savefig(save_path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved PCA RGB visualization to {save_path2}")


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Patch Embedding Extraction")
    print("=" * 60)
    print(f"  Model type:   {args.model_type}")
    print(f"  Model name:   {args.model_name or '(default)'}")
    print(f"  Num samples:  {args.num_samples}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Device:       {device}")

    # ---- Step 1: Setup dataset ----
    print("\n[1/4] Loading dataset...")
    dataset_dir = data_cfg["dataset_dir"]
    if not os.path.exists(dataset_dir):
        dataset_dir = get_dataset_path()

    transform = get_val_transforms(data_cfg.get("image_size", 224))
    train_csv = os.path.join(dataset_dir, data_cfg["train_csv"])

    dataset = CheXpertDataset(
        csv_path=train_csv,
        dataset_dir=dataset_dir,
        uncertainty_policy="ones",
        transform=transform,
    )

    from torch.utils.data import DataLoader
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 2),
        pin_memory=True,
    )

    print(f"  Dataset size: {len(dataset):,}")
    print(f"  Extracting from: {args.num_samples} images")

    # ---- Step 2: Load feature extractor ----
    print(f"\n[2/4] Loading {args.model_type.upper()} feature extractor...")
    extractor = create_feature_extractor(
        model_type=args.model_type,
        model_name=args.model_name,
        device=str(device),
    )
    print(f"  Output dim: {extractor.get_output_dim()}")
    print(f"  Num patches: {extractor.get_num_patches()}")

    # ---- Step 3: Extract features ----
    print(f"\n[3/4] Extracting patch embeddings...")
    start_time = time.time()

    embeddings = extractor.extract_from_dataloader(
        dataloader=loader,
        num_samples=args.num_samples,
        device=device,
        show_progress=True,
    )

    elapsed = time.time() - start_time
    print(f"\n  Extraction complete!")
    print(f"  Shape: {list(embeddings.shape)}")
    print(f"  Time:  {elapsed:.1f}s ({elapsed / embeddings.shape[0] * 1000:.1f}ms/image)")

    # ---- Step 4: Save ----
    print(f"\n[4/4] Saving embeddings...")
    output_path = args.output or f"outputs/embeddings/{args.model_type}_embeddings.pt"

    save_embeddings(
        embeddings,
        save_path=output_path,
        metadata={
            "model_type": args.model_type,
            "model_name": args.model_name or "default",
            "num_samples": embeddings.shape[0],
            "num_patches": embeddings.shape[1],
            "embed_dim": embeddings.shape[2],
            "dataset": "CheXpert-v1.0-small",
        },
    )

    # ---- Visualize ----
    if args.visualize:
        print("\nGenerating visualizations...")
        visualize_embeddings(
            embeddings,
            output_dir=f"outputs/embeddings/{args.model_type}_analysis",
        )

    print("\n" + "=" * 60)
    print("Feature extraction complete!")
    print("=" * 60)
    print(f"\n  Embeddings saved to: {output_path}")
    print(f"  Each image → {embeddings.shape[1]} patches × {embeddings.shape[2]}-dim embeddings")
    print(f"  Ready for clustering: feed to build_structure.py")


if __name__ == "__main__":
    main()
