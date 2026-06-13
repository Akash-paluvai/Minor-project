#!/usr/bin/env python3
"""
Precompute Structure Artifacts — One-Time Offline Pipeline.

Run this ONCE before training to precompute and cache all structural
feasibility matrices. Training then simply loads these from disk,
avoiding expensive DINO inference during every experiment.

Pipeline:
    1. Load CheXpert subset (configurable N images)
    2. Extract DINO patch features → (N, 144, 384)
    3. Aggregate to population embedding → (144, 384)
    4. Build cosine similarity graph → (144, 144)
    5. Spectral clustering → labels (144,)
    6. FeasibilityMatrixBuilder → F_feature (145, 145)
    7. AnatomyPriorBuilder → F_anatomy (145, 145)
    8. build_hybrid_feasibility → F_hybrid (145, 145)
    9. Save all to outputs/cache/

MPS-optimized:
    - batch_size=4 for DINO extraction
    - pin_memory=False
    - num_workers=0
    - device = mps if available else cpu

Usage:
    python scripts/precompute_structure.py
    python scripts/precompute_structure.py --config configs/default.yaml --n-images 500
    python scripts/precompute_structure.py --dry-run    # validates pipeline, no data
"""

import os
import sys
import json
import time
import argparse
import numpy as np

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from src.data.chexpert_dataset import CheXpertDataset, get_dataset_path, DEFAULT_TARGETS
from src.data.transforms import get_val_transforms
from src.models.feature_extractor import create_feature_extractor
from src.structure.graph_builder import PatchGraphBuilder
from src.structure.clustering import spectral_clustering
from src.structure.feasibility import FeasibilityMatrixBuilder, build_hybrid_feasibility, save_feasibility
from src.structure.anatomy_prior import AnatomyPriorBuilder


def get_device(config: dict) -> torch.device:
    """MPS-first device selection."""
    sys_cfg = config.get("system", {})
    if sys_cfg.get("use_mps", True) and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def create_dataloader(config: dict, n_images: int) -> DataLoader:
    """Create a small DataLoader for feature extraction."""
    data_cfg    = config["data"]
    dataset_dir = data_cfg["dataset_dir"]

    if not os.path.exists(dataset_dir):
        print("  Dataset not found locally, downloading via kagglehub...")
        dataset_dir = get_dataset_path()

    train_csv = os.path.join(dataset_dir, data_cfg["train_csv"])
    img_size  = data_cfg.get("image_size", 192)
    transform = get_val_transforms(img_size)   # no augmentation for features

    dataset = CheXpertDataset(
        csv_path=train_csv,
        dataset_dir=dataset_dir,
        target_columns=DEFAULT_TARGETS,
        uncertainty_policy="ones",
        transform=transform,
        frontal_only=False,
        subset_size=n_images,
    )

    loader = DataLoader(
        dataset,
        batch_size=4,         # Small batch for DINO on MPS
        shuffle=False,
        num_workers=0,        # 0 for MPS stability
        pin_memory=False,     # Required for MPS
    )

    return loader


def extract_population_features(
    loader: DataLoader,
    config: dict,
    device: torch.device,
) -> torch.Tensor:
    """
    Extract DINO patch features from a subset of images.

    Returns:
        mean_features: (N_patches, embed_dim) — population-averaged patch embeddings
                       N_patches = 144 for 192×192 images
    """
    struct_cfg = config.get("structure", {})
    model_type = struct_cfg.get("feature_extractor", "dino")
    model_name = struct_cfg.get("dino_model", "dino_vits16")

    print(f"  Loading feature extractor: {model_type}/{model_name}")
    extractor = create_feature_extractor(
        model_type=model_type,
        model_name=model_name,
        device=str(device),
    )
    extractor.eval()

    all_features = []
    total = 0

    print(f"  Extracting features from {len(loader.dataset)} images...")
    for batch_idx, (images, _) in enumerate(loader):
        images = images.to(device)
        with torch.no_grad():
            feats = extractor(images)   # (B, N_patches, D)
        all_features.append(feats.cpu())
        total += images.shape[0]
        if (batch_idx + 1) % 20 == 0:
            print(f"    Processed {total} / {len(loader.dataset)} images")

    # all_features: list of (B, N, D) → cat → (total, N, D)
    all_features = torch.cat(all_features, dim=0)
    print(f"  Collected features: {list(all_features.shape)}")

    # Population-level: average over all images → (N_patches, D)
    mean_features = all_features.mean(dim=0)   # (N, D)
    print(f"  Population embedding: {list(mean_features.shape)}")

    return mean_features


def build_graph_and_cluster(
    mean_features: torch.Tensor,
    config: dict,
    grid_size: int,
) -> tuple:
    """
    Build similarity graph and run spectral clustering on population features.

    Args:
        mean_features: (N_patches, embed_dim) population embedding
        config:        Full config dict
        grid_size:     Patch grid side length (12 for 192px)

    Returns:
        adj_matrix: (N, N) adjacency matrix
        labels:     (N,) cluster assignments
    """
    struct_cfg = config.get("structure", {})
    k_neighbors = struct_cfg.get("k_neighbors", 8)
    n_clusters  = struct_cfg.get("n_clusters", 9)

    print(f"\n  Building {k_neighbors}-NN similarity graph...")
    graph_builder = PatchGraphBuilder(k_neighbors=k_neighbors)
    adj_matrix = graph_builder.build(mean_features)   # (N, N)
    print(f"  Graph shape: {list(adj_matrix.shape)}")

    print(f"\n  Running spectral clustering (k={n_clusters})...")
    labels = spectral_clustering(adj_matrix, k=n_clusters)
    print(f"  Cluster sizes: {np.bincount(labels)}")

    return adj_matrix, labels


def main():
    parser = argparse.ArgumentParser(description="Precompute Structure Artifacts")
    parser.add_argument("--config",    default="configs/default.yaml",
                        help="Config YAML path")
    parser.add_argument("--n-images",  type=int, default=500,
                        help="Number of images for feature extraction (default: 500)")
    parser.add_argument("--alpha",     type=float, default=None,
                        help="Hybrid blend weight (default: from config)")
    parser.add_argument("--cache-dir", default=None,
                        help="Cache directory (default: from config)")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Run anatomy prior only (no DINO, no data needed)")
    args = parser.parse_args()

    # ── Load config ──
    with open(args.config) as f:
        config = yaml.safe_load(f)

    cache_dir = args.cache_dir or config.get("structure", {}).get("cache_dir", "outputs/cache")
    alpha     = args.alpha     or config.get("structure", {}).get("alpha", 0.7)
    img_size  = config.get("data", {}).get("image_size", 192)
    grid_size = img_size // 16   # 12 for 192px

    os.makedirs(cache_dir, exist_ok=True)
    device = get_device(config)

    print("=" * 65)
    print("  Precompute Structure Artifacts")
    print("=" * 65)
    print(f"  Config:     {args.config}")
    print(f"  Device:     {device}")
    print(f"  Image size: {img_size}×{img_size}")
    print(f"  Grid size:  {grid_size}×{grid_size} = {grid_size**2} patches")
    print(f"  N+1 tokens: {grid_size**2 + 1} (includes CLS)")
    print(f"  Alpha:      {alpha}")
    print(f"  Cache dir:  {cache_dir}")
    print(f"  Dry run:    {args.dry_run}")
    print()

    t0 = time.time()

    # ════════════════════════════════════════════════════════
    #  Step 1: Anatomy Prior (always runs — no data needed)
    # ════════════════════════════════════════════════════════
    print("[1/4] Building Anatomy Prior...")
    anat_cfg = config.get("anatomy", {})
    anatomy_builder = AnatomyPriorBuilder(
        same_region_weight = anat_cfg.get("same_region_weight", 1.0),
        adjacent_weight    = anat_cfg.get("adjacent_weight", 0.6),
        distant_weight     = anat_cfg.get("distant_weight", 0.15),
        background_weight  = anat_cfg.get("background_weight", 0.05),
    )

    F_anatomy      = anatomy_builder.build(grid_size=grid_size)   # (N+1, N+1)
    anatomy_labels = anatomy_builder.get_region_map(grid_size)    # (N,)

    # Save F_anatomy
    torch.save({
        "feasibility_matrix": F_anatomy,
        "anatomy_labels":     torch.tensor(anatomy_labels),
        "grid_size":          grid_size,
        "shape":              list(F_anatomy.shape),
        "metadata":           {"type": "anatomy_prior", "img_size": img_size},
    }, os.path.join(cache_dir, "F_anatomy.pt"))
    print(f"  ✓ F_anatomy saved: {list(F_anatomy.shape)}")

    # Save anatomy region visualization
    try:
        anatomy_builder.visualize(
            grid_size=grid_size,
            save_path=os.path.join("outputs/visualizations", "anatomy_regions.png"),
        )
    except Exception as e:
        print(f"  ⚠ Anatomy visualization failed: {e}")

    if args.dry_run:
        # In dry-run, use identity matrix as F_feature placeholder
        print("\n  Dry-run mode: using identity matrix as F_feature placeholder")
        N = grid_size ** 2
        F_feature = torch.eye(N + 1, dtype=torch.float32)
        F_feature[0, :] = 1.0
        F_feature[:, 0] = 1.0
    else:
        # ════════════════════════════════════════════════════════
        #  Step 2: Extract DINO Features
        # ════════════════════════════════════════════════════════
        print(f"\n[2/4] Extracting DINO Features ({args.n_images} images)...")
        loader = create_dataloader(config, args.n_images)
        mean_features = extract_population_features(loader, config, device)
        # mean_features: (N_patches, embed_dim)

        # ════════════════════════════════════════════════════════
        #  Step 3: Graph + Clustering
        # ════════════════════════════════════════════════════════
        print("\n[3/4] Building Graph and Running Clustering...")
        adj_matrix, cluster_labels = build_graph_and_cluster(
            mean_features, config, grid_size
        )

        # Build feature feasibility matrix
        struct_cfg   = config.get("structure", {})
        feas_builder = FeasibilityMatrixBuilder(
            intra_cluster_weight = struct_cfg.get("intra_cluster_weight", 1.0),
            inter_cluster_weight = struct_cfg.get("inter_cluster_weight", 0.3),
            masking_mode         = struct_cfg.get("masking_mode", "soft"),
        )
        F_feature = feas_builder.build(
            cluster_labels=cluster_labels,
            adjacency=adj_matrix,
            grid_size=grid_size,
        )   # (N+1, N+1)

        # Save F_feature
        torch.save({
            "feasibility_matrix": F_feature,
            "cluster_labels":     torch.tensor(cluster_labels),
            "grid_size":          grid_size,
            "shape":              list(F_feature.shape),
            "metadata":           {"type": "feature_prior", "img_size": img_size},
        }, os.path.join(cache_dir, "F_feature.pt"))
        print(f"  ✓ F_feature saved: {list(F_feature.shape)}")

        # Visualize F_feature
        try:
            os.makedirs("outputs/visualizations", exist_ok=True)
            feas_builder.visualize(F_feature, save_dir="outputs/visualizations/feasibility")
        except Exception as e:
            print(f"  ⚠ Feature feasibility visualization failed: {e}")

    # ════════════════════════════════════════════════════════
    #  Step 4: Build Hybrid Matrix
    # ════════════════════════════════════════════════════════
    print(f"\n[4/4] Building Hybrid Feasibility Matrix (alpha={alpha})...")
    F_hybrid = build_hybrid_feasibility(F_anatomy, F_feature, alpha=alpha)

    # Save F_hybrid
    torch.save({
        "feasibility_matrix": F_hybrid,
        "alpha":              alpha,
        "grid_size":          grid_size,
        "shape":              list(F_hybrid.shape),
        "metadata":           {
            "type": "hybrid_prior",
            "img_size": img_size,
            "alpha": alpha,
        },
    }, os.path.join(cache_dir, "F_hybrid.pt"))
    print(f"  ✓ F_hybrid saved: {list(F_hybrid.shape)}")

    # ── Save metadata JSON ──
    elapsed = time.time() - t0
    metadata = {
        "img_size":     img_size,
        "grid_size":    grid_size,
        "n_patches":    grid_size ** 2,
        "n_tokens":     grid_size ** 2 + 1,
        "alpha":        alpha,
        "dry_run":      args.dry_run,
        "n_images":     args.n_images if not args.dry_run else 0,
        "elapsed_sec":  round(elapsed, 1),
        "files": {
            "F_anatomy": os.path.join(cache_dir, "F_anatomy.pt"),
            "F_feature": os.path.join(cache_dir, "F_feature.pt") if not args.dry_run else None,
            "F_hybrid":  os.path.join(cache_dir, "F_hybrid.pt"),
        },
        "anatomy_labels_path": os.path.join(cache_dir, "F_anatomy.pt"),
    }

    with open(os.path.join(cache_dir, "precompute_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'=' * 65}")
    print(f"  Precomputation complete in {elapsed:.1f}s")
    print(f"  Cache saved to: {cache_dir}/")
    print(f"    F_anatomy.pt  : {list(F_anatomy.shape)}")
    if not args.dry_run:
        print(f"    F_feature.pt  : {list(F_feature.shape)}")
    print(f"    F_hybrid.pt   : {list(F_hybrid.shape)}")
    print(f"{'=' * 65}")
    print(f"\nNext step: python run_all_experiments.py")


if __name__ == "__main__":
    main()
