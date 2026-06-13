#!/usr/bin/env python3
"""
Build Structural Feasibility Matrix from Clustering Output.

Converts cluster assignments into a feasibility matrix:
    A_ij = P(patch_i interacts with patch_j)

Rules:
  - Same cluster → A_ij = high (1.0)
  - Different cluster, graph-connected → A_ij = moderate (0.0 - 0.5)
  - Different cluster, no connection → A_ij = 0.0

Usage:
    python build_feasibility.py --cluster-labels outputs/clustering/all_cluster_labels.pt \
                                --graph outputs/graph/cosine_graph.pt \
                                --algorithm spectral --visualize

    python build_feasibility.py --cluster-labels outputs/clustering/spectral_labels.npy \
                                --graph outputs/graph/cosine_graph.pt --visualize
"""

import os
import argparse
import yaml
import torch
import numpy as np

from src.structure.feasibility import FeasibilityMatrixBuilder, save_feasibility


def parse_args():
    parser = argparse.ArgumentParser(description="Build Structural Feasibility Matrix")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--cluster-labels", type=str, default="outputs/clustering/all_cluster_labels.pt",
        help="Path to cluster labels (.pt or .npy)"
    )
    parser.add_argument(
        "--graph", type=str, default="outputs/graph/cosine_graph.pt",
        help="Path to adjacency matrix"
    )
    parser.add_argument(
        "--algorithm", type=str, default="spectral",
        choices=["kmeans", "spectral", "diffusion", "gnn", "all"],
        help="Which clustering algorithm's labels to use"
    )
    parser.add_argument("--intra-weight", type=float, default=1.0,
                       help="Same-cluster interaction probability")
    parser.add_argument("--inter-weight", type=float, default=0.3,
                       help="Max inter-cluster interaction probability")
    parser.add_argument("--spatial-decay", type=float, default=0.0,
                       help="Spatial proximity bonus (0 = disabled)")
    parser.add_argument("--mode", type=str, default="soft",
                       choices=["soft", "hard"], help="Masking mode")
    parser.add_argument("--output-dir", type=str, default="outputs/feasibility")
    parser.add_argument("--visualize", action="store_true")
    return parser.parse_args()


def load_cluster_labels(path: str, algorithm: str = "spectral") -> dict:
    """Load cluster labels from .pt or .npy file."""
    if path.endswith(".pt"):
        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict):
            # all_cluster_labels.pt → dict of {name: tensor}
            results = {}
            for name, tensor in data.items():
                results[name] = tensor.numpy()
            return results
        else:
            return {"default": data.numpy()}
    elif path.endswith(".npy"):
        labels = np.load(path)
        name = os.path.basename(path).replace("_labels.npy", "")
        return {name: labels}
    else:
        raise ValueError(f"Unknown format: {path}")


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Structural Feasibility Matrix Builder")
    print("=" * 60)
    print(f"  Mode:           {args.mode}")
    print(f"  Intra weight:   {args.intra_weight}")
    print(f"  Inter weight:   {args.inter_weight}")
    print(f"  Spatial decay:  {args.spatial_decay}")

    # ── Load cluster labels ──
    print(f"\nLoading cluster labels from {args.cluster_labels}...")
    all_labels = load_cluster_labels(args.cluster_labels, args.algorithm)
    available = list(all_labels.keys())
    print(f"  Available algorithms: {available}")

    # ── Load adjacency matrix ──
    adjacency = None
    if os.path.exists(args.graph):
        from src.structure.graph_builder import load_graph
        graph_data = load_graph(args.graph)
        adjacency = graph_data["adjacency"]
        print(f"  Adjacency: {adjacency.shape}")

    # ── Select which labels to use ──
    if args.algorithm == "all":
        algorithms_to_run = available
    elif args.algorithm in all_labels:
        algorithms_to_run = [args.algorithm]
    else:
        print(f"  ⚠ '{args.algorithm}' not found. Using '{available[0]}'")
        algorithms_to_run = [available[0]]

    # ── Build feasibility matrix for each algorithm ──
    builder = FeasibilityMatrixBuilder(
        intra_cluster_weight=args.intra_weight,
        inter_cluster_weight=args.inter_weight,
        spatial_decay=args.spatial_decay,
        masking_mode=args.mode,
    )

    results = {}
    for algo in algorithms_to_run:
        labels = all_labels[algo]
        print(f"\n{'─' * 50}")
        print(f"Building matrix for: {algo.upper()}")
        print(f"{'─' * 50}")

        F = builder.build(labels, adjacency=adjacency)
        results[algo] = {"feasibility": F, "labels": labels}

        # Save
        save_path = os.path.join(args.output_dir, f"{algo}_feasibility.pt")
        save_feasibility(
            F, labels,
            metadata={"algorithm": algo, "mode": args.mode,
                      "intra": args.intra_weight, "inter": args.inter_weight},
            save_path=save_path,
        )

        # Visualize
        if args.visualize:
            viz_dir = os.path.join(args.output_dir, f"{algo}_viz")
            builder.visualize(F, cluster_labels=labels, save_dir=viz_dir)

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("Feasibility Matrix Summary")
    print(f"{'=' * 60}")
    print(f"\n  {'Algorithm':<15} {'Shape':>12} {'Sparsity':>10} {'Mean P':>10}")
    print(f"  {'─' * 47}")
    for algo, data in results.items():
        F = data["feasibility"]
        patch_F = F[1:, 1:]
        sparsity = (patch_F == 0).sum().item() / patch_F.numel()
        mean_p = patch_F.mean().item()
        print(f"  {algo:<15} {str(list(F.shape)):>12} {sparsity:>10.1%} {mean_p:>10.4f}")

    print(f"\n  Results saved to: {args.output_dir}/")
    print("  Next step: Train structure-constrained ViT with feasibility matrix")


if __name__ == "__main__":
    main()
