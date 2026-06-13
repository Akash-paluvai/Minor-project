#!/usr/bin/env python3
"""
Build Patch Similarity Graph from Extracted Embeddings.

Constructs A_ij similarity matrix between all patch pairs using
cosine similarity or Gaussian kernel, then sparsifies with k-NN.

Usage:
    python build_graph.py --embeddings outputs/embeddings/dino_embeddings.pt
    python build_graph.py --embeddings outputs/embeddings/dino_embeddings.pt --metric gaussian
    python build_graph.py --embeddings outputs/embeddings/dino_embeddings.pt --metric cosine --k 12
"""

import argparse
import yaml
import torch

from src.structure.graph_builder import PatchGraphBuilder, save_graph


def parse_args():
    parser = argparse.ArgumentParser(description="Build Patch Similarity Graph")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--embeddings", type=str, default="outputs/embeddings/dino_embeddings.pt",
        help="Path to saved patch embeddings (.pt file)"
    )
    parser.add_argument(
        "--metric", type=str, default="cosine",
        choices=["cosine", "gaussian"],
        help="Similarity metric: cosine or gaussian"
    )
    parser.add_argument("--k", type=int, default=8, help="k for k-NN sparsification")
    parser.add_argument("--sigma", type=float, default=None, help="Gaussian bandwidth (auto if None)")
    parser.add_argument("--output", type=str, default=None, help="Output .pt path")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument(
        "--compare-metrics", action="store_true",
        help="Build graphs with BOTH metrics and compare"
    )
    return parser.parse_args()


def build_single_graph(features, metric, k, sigma=None, visualize=False, output=None, viz_dir=None):
    """Build a single graph with given parameters and optionally visualize."""
    print(f"\n{'─' * 50}")
    print(f"Building graph: metric={metric}, k={k}, sigma={sigma}")
    print(f"{'─' * 50}")

    builder = PatchGraphBuilder(
        k=k,
        similarity_metric=metric,
        sigma=sigma,
    )

    adjacency, raw_similarity = builder.build_population_graph(
        features, return_raw_similarity=True
    )

    # Save
    if output:
        save_graph(
            adjacency, raw_similarity,
            metadata={"metric": metric, "k": k, "sigma": builder.sigma},
            save_path=output,
        )

    # Visualize
    if visualize and viz_dir:
        print("\n  Generating visualizations...")
        builder.visualize(adjacency, raw_similarity, save_dir=viz_dir)

    return adjacency, raw_similarity, builder


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("Patch Similarity Graph Construction")
    print("=" * 60)

    # Load embeddings
    print(f"\nLoading embeddings from {args.embeddings}...")
    from src.models.feature_extractor import load_embeddings
    data = load_embeddings(args.embeddings)
    features = data["embeddings"]  # (N_images, 196, 768)
    print(f"  Shape: {list(features.shape)}")

    if args.compare_metrics:
        # ── Compare both metrics side by side ──
        print("\n" + "=" * 60)
        print("COMPARING COSINE vs GAUSSIAN")
        print("=" * 60)

        adj_cos, sim_cos, builder_cos = build_single_graph(
            features, "cosine", args.k,
            visualize=True, viz_dir="outputs/graph/cosine",
            output="outputs/graph/cosine_graph.pt",
        )

        adj_gauss, sim_gauss, builder_gauss = build_single_graph(
            features, "gaussian", args.k, sigma=args.sigma,
            visualize=True, viz_dir="outputs/graph/gaussian",
            output="outputs/graph/gaussian_graph.pt",
        )

        # Comparison stats
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(f"  {'Metric':<15} {'Edges':>10} {'Density':>10} {'Weight Range':>20}")
        print(f"  {'─'*55}")
        for name, adj in [("Cosine", adj_cos), ("Gaussian", adj_gauss)]:
            edges = (adj > 0).sum().item()
            density = edges / adj.numel()
            w_min = adj[adj > 0].min().item()
            w_max = adj.max().item()
            print(f"  {name:<15} {edges:>10,} {density:>10.2%} {f'[{w_min:.4f}, {w_max:.4f}]':>20}")

    else:
        # ── Single metric ──
        output = args.output or f"outputs/graph/{args.metric}_graph.pt"
        viz_dir = f"outputs/graph/{args.metric}" if args.visualize else None

        adj, sim, builder = build_single_graph(
            features, args.metric, args.k, args.sigma,
            visualize=args.visualize, output=output, viz_dir=viz_dir,
        )

    print("\n" + "=" * 60)
    print("Graph construction complete!")
    print("=" * 60)
    print("  Next step: python build_structure.py (clustering)")


if __name__ == "__main__":
    main()
