#!/usr/bin/env python3
"""
Build Structural Feasibility Matrix.

Pipeline:
1. Load DINO feature extractor
2. Extract patch features from a sample of training images
3. Build patch similarity graph (k-NN)
4. Perform spectral clustering on the graph
5. Build and save the feasibility matrix

Usage:
    python build_structure.py --config configs/default.yaml
    python build_structure.py --config configs/default.yaml --visualize
"""

import os
import argparse
import yaml
import torch
import numpy as np

from src.data.chexpert_dataset import get_chexpert_dataloaders
from src.data.transforms import get_val_transforms
from src.models.feature_extractor import DINOFeatureExtractor
from src.structure.graph_builder import PatchGraphBuilder
from src.structure.clustering import SpectralClusterer
from src.structure.feasibility import FeasibilityMatrixBuilder


def parse_args():
    parser = argparse.ArgumentParser(description="Build Structural Feasibility Matrix")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Save visualizations of clusters and feasibility matrix"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for feasibility matrix (overrides config)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    struct_cfg = config["structure"]
    data_cfg = config["data"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Structure Discovery Pipeline")
    print("=" * 60)

    # ---- Step 1: Setup data loader (no augmentation) ----
    print("\n[1/5] Loading dataset...")
    val_transform = get_val_transforms(data_cfg["image_size"])

    # Use training data for structure discovery but with val transforms
    train_loader, _ = get_chexpert_dataloaders(config, val_transform, val_transform)
    print(f"  Dataset size: {len(train_loader.dataset)}")

    # ---- Step 2: Load DINO feature extractor ----
    print("\n[2/5] Loading DINO feature extractor...")
    feature_extractor = DINOFeatureExtractor(
        model_name=config["model"].get("pretrained_model", "vit_base_patch16_224"),
    )
    feature_extractor = feature_extractor.to(device)
    print(f"  Feature dim: {feature_extractor.get_output_dim()}")

    # ---- Step 3: Build patch graph ----
    print(f"\n[3/5] Building patch similarity graph (k={struct_cfg['k_neighbors']})...")
    graph_builder = PatchGraphBuilder(
        k=struct_cfg["k_neighbors"],
        similarity_metric="cosine",
    )

    adjacency = graph_builder.build_from_dataloader(
        feature_extractor=feature_extractor,
        dataloader=train_loader,
        num_samples=struct_cfg["num_samples_for_graph"],
        device=str(device),
    )
    print(f"  Adjacency shape: {adjacency.shape}")
    print(f"  Non-zero edges: {(adjacency > 0).sum().item()}")
    print(f"  Graph density: {(adjacency > 0).float().mean().item():.4f}")

    # ---- Step 4: Spectral clustering ----
    print(f"\n[4/5] Performing spectral clustering (K={struct_cfg['n_clusters']})...")
    clusterer = SpectralClusterer(
        n_clusters=struct_cfg["n_clusters"],
        use_sklearn=True,
    )
    cluster_labels = clusterer.fit(adjacency)
    print(f"  Cluster labels shape: {cluster_labels.shape}")
    print(f"  Cluster sizes: {np.bincount(cluster_labels)}")

    # ---- Step 5: Build feasibility matrix ----
    print(f"\n[5/5] Building feasibility matrix...")
    matrix_builder = FeasibilityMatrixBuilder(
        inter_cluster_weight=struct_cfg["inter_cluster_weight"],
        masking_mode=struct_cfg["masking_mode"],
    )
    feasibility_matrix = matrix_builder.build(cluster_labels, adjacency)

    sparsity = matrix_builder.get_sparsity(feasibility_matrix)
    print(f"  Feasibility matrix shape: {feasibility_matrix.shape}")
    print(f"  Attention sparsity: {sparsity:.1%} connections blocked")

    # ---- Save ----
    output_path = args.output or struct_cfg.get(
        "feasibility_matrix_path", "outputs/feasibility_matrix.pt"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torch.save({
        "feasibility_matrix": feasibility_matrix,
        "cluster_labels": cluster_labels,
        "adjacency": adjacency,
        "config": struct_cfg,
    }, output_path)
    print(f"\n  ✓ Saved to {output_path}")

    # ---- Visualize ----
    if args.visualize:
        viz_dir = "outputs/structure_visualizations"
        os.makedirs(viz_dir, exist_ok=True)

        print("\nGenerating visualizations...")

        # Cluster map
        clusterer.visualize_clusters(
            cluster_labels,
            grid_size=int(np.sqrt(len(cluster_labels))),
            save_path=os.path.join(viz_dir, "cluster_assignments.png"),
        )

        # Feasibility matrix heatmap
        matrix_builder.visualize(
            feasibility_matrix,
            save_path=os.path.join(viz_dir, "feasibility_matrix.png"),
        )

        print(f"  ✓ Visualizations saved to {viz_dir}/")

    print("\n" + "=" * 60)
    print("Structure discovery complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
