#!/usr/bin/env python3
"""
Unsupervised Patch Clustering for Structure Discovery.

Runs 4 clustering algorithms on extracted patch embeddings and/or adjacency matrix:
  1. K-Means on raw embeddings
  2. Spectral Clustering on adjacency graph
  3. Diffusion Maps + K-Means on adjacency graph
  4. GNN Clustering (graph convolution + modularity loss)

Usage:
    # Run all 4 algorithms with comparison
    python cluster_patches.py --embeddings outputs/embeddings/dino_embeddings.pt \
                              --graph outputs/graph/cosine_graph.pt --k 16

    # Run a specific algorithm
    python cluster_patches.py --embeddings outputs/embeddings/dino_embeddings.pt \
                              --graph outputs/graph/cosine_graph.pt --algorithm spectral --k 12

    # With image overlay visualization
    python cluster_patches.py --embeddings outputs/embeddings/dino_embeddings.pt \
                              --graph outputs/graph/cosine_graph.pt --visualize --overlay-image
"""

import os
import argparse
import yaml
import torch
import numpy as np
from PIL import Image

from src.models.feature_extractor import load_embeddings
from src.structure.graph_builder import load_graph
from src.structure.clustering import (
    kmeans_clustering,
    spectral_clustering,
    diffusion_map_clustering,
    gnn_clustering,
    run_all_clustering,
    visualize_clusters,
    compare_clusterings,
    visualize_graph_clusters,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Unsupervised Patch Clustering")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--embeddings", type=str, default="outputs/embeddings/dino_embeddings.pt",
        help="Path to saved patch embeddings"
    )
    parser.add_argument(
        "--graph", type=str, default="outputs/graph/cosine_graph.pt",
        help="Path to saved adjacency matrix"
    )
    parser.add_argument("--k", type=int, default=16, help="Number of clusters")
    parser.add_argument(
        "--algorithm", type=str, default="all",
        choices=["all", "kmeans", "spectral", "diffusion", "gnn"],
        help="Which clustering algorithm(s) to run"
    )
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--overlay-image", action="store_true", help="Overlay clusters on a sample X-ray")
    parser.add_argument("--output-dir", type=str, default="outputs/clustering")
    return parser.parse_args()


def load_sample_image(config):
    """Load a sample CheXpert image for overlay visualization."""
    import pandas as pd
    data_cfg = config["data"]
    dataset_dir = data_cfg["dataset_dir"]
    train_csv = os.path.join(dataset_dir, data_cfg["train_csv"])

    df = pd.read_csv(train_csv)
    csv_path = df["Path"].iloc[0]
    for prefix in ["CheXpert-v1.0-small/", "CheXpert-v1.0/"]:
        if csv_path.startswith(prefix):
            csv_path = csv_path[len(prefix):]
            break
    img_path = os.path.join(dataset_dir, csv_path)

    img = Image.open(img_path).convert("RGB").resize((224, 224))
    return np.array(img)


def save_cluster_labels(results: dict, output_dir: str):
    """Save cluster labels to disk for downstream use."""
    os.makedirs(output_dir, exist_ok=True)
    for name, labels in results.items():
        path = os.path.join(output_dir, f"{name}_labels.npy")
        np.save(path, labels)
    # Also save as a combined .pt file
    combined_path = os.path.join(output_dir, "all_cluster_labels.pt")
    torch.save({name: torch.tensor(labels) for name, labels in results.items()}, combined_path)
    print(f"  ✓ Saved cluster labels to {output_dir}/")


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Unsupervised Patch Clustering")
    print("=" * 60)
    print(f"  Algorithm:  {args.algorithm}")
    print(f"  k:          {args.k}")

    # ── Load data ──
    print("\nLoading data...")

    embeddings = None
    adj_matrix = None

    if os.path.exists(args.embeddings):
        data = load_embeddings(args.embeddings)
        all_embeddings = data["embeddings"]  # (N_images, 196, 768)
        # Average embeddings across images → population-level
        embeddings = all_embeddings.mean(dim=0).numpy()  # (196, 768)
        print(f"  Embeddings: {embeddings.shape} (averaged from {all_embeddings.shape[0]} images)")
    else:
        print(f"  ⚠ Embeddings not found: {args.embeddings}")

    if os.path.exists(args.graph):
        graph_data = load_graph(args.graph)
        adj_matrix = graph_data["adjacency"].numpy()
        print(f"  Adjacency:  {adj_matrix.shape}")
    else:
        print(f"  ⚠ Graph not found: {args.graph}")

    # Validate we have the right inputs
    if embeddings is None and adj_matrix is None:
        raise ValueError("Need at least embeddings or adjacency matrix!")

    # If we need adjacency but don't have it, build from embeddings
    if adj_matrix is None and embeddings is not None:
        from src.structure.graph_builder import PatchGraphBuilder
        builder = PatchGraphBuilder(k=8, similarity_metric="cosine")
        adj_matrix = builder.build_population_graph(
            torch.tensor(embeddings).unsqueeze(0)
        ).numpy()

    # ── Run clustering ──
    print(f"\n{'=' * 60}")
    print("Running Clustering Algorithms")
    print(f"{'=' * 60}")

    results = {}

    if args.algorithm == "all":
        if embeddings is None:
            raise ValueError("Need embeddings for K-Means and GNN. Provide --embeddings.")
        results = run_all_clustering(embeddings, adj_matrix, k=args.k)
    elif args.algorithm == "kmeans":
        if embeddings is None:
            raise ValueError("K-Means needs embeddings.")
        results["kmeans"] = kmeans_clustering(embeddings, k=args.k)
    elif args.algorithm == "spectral":
        results["spectral"] = spectral_clustering(adj_matrix, k=args.k)
    elif args.algorithm == "diffusion":
        results["diffusion"] = diffusion_map_clustering(adj_matrix, k=args.k)
    elif args.algorithm == "gnn":
        if embeddings is None:
            raise ValueError("GNN needs embeddings.")
        results["gnn"] = gnn_clustering(adj_matrix, embeddings, k=args.k)

    # ── Save results ──
    print(f"\n{'=' * 60}")
    print("Saving Results")
    print(f"{'=' * 60}")
    save_cluster_labels(results, args.output_dir)

    # ── Visualization ──
    if args.visualize:
        print(f"\n{'=' * 60}")
        print("Generating Visualizations")
        print(f"{'=' * 60}")

        # Load sample image for overlay
        sample_image = None
        if args.overlay_image:
            try:
                sample_image = load_sample_image(config)
                print(f"  Loaded sample X-ray: {sample_image.shape}")
            except Exception as e:
                print(f"  ⚠ Could not load sample image: {e}")

        # Individual cluster maps
        for name, labels in results.items():
            save_path = os.path.join(args.output_dir, f"{name}_clusters.png")
            visualize_clusters(
                labels,
                title=f"{name.upper()} Clustering (k={args.k})",
                save_path=save_path,
                original_image=sample_image,
            )

        # Comparison chart (if multiple algorithms)
        if len(results) > 1:
            compare_clusterings(
                results,
                save_path=os.path.join(args.output_dir, "algorithm_comparison.png"),
            )

        # Graph visualization (for spectral or best algorithm)
        if adj_matrix is not None:
            best_name = list(results.keys())[0]
            visualize_graph_clusters(
                adj_matrix, results[best_name],
                title=f"Patch Graph — {best_name.upper()} Clusters",
                save_path=os.path.join(args.output_dir, "graph_clusters.png"),
            )

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("Clustering Summary")
    print(f"{'=' * 60}")
    print(f"\n  {'Algorithm':<15} {'Clusters':>10} {'Size Range':>15} {'Std':>8}")
    print(f"  {'─' * 48}")
    for name, labels in results.items():
        sizes = np.bincount(labels, minlength=args.k)
        print(f"  {name:<15} {len(np.unique(labels)):>10} "
              f"{f'[{sizes.min()}, {sizes.max()}]':>15} {sizes.std():>8.1f}")

    print(f"\n  Results saved to: {args.output_dir}/")
    print("  Next step: Build structural feasibility matrix")


if __name__ == "__main__":
    main()
