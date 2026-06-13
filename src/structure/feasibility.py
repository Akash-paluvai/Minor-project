"""
Structural Feasibility Matrix Builder.

Converts clustering output into a structural feasibility matrix:

    A_ij = P(patch_i interacts with patch_j)

where the probability of interaction depends on:
  - Cluster membership (same cluster → high, different → low)
  - Graph adjacency (connected clusters → moderate)
  - Optional: spatial proximity prior

Matrix dimensions: (N+1) × (N+1) where +1 is for the CLS token.
The CLS token row/column is always 1.0 (unconstrained global access).

Used to constrain attention in the Structure-Constrained ViT:
  - High A_ij → full attention weight
  - Low A_ij  → suppressed attention weight
  - Zero A_ij → blocked attention
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns


class FeasibilityMatrixBuilder:
    """
    Build a structural feasibility matrix from cluster assignments.

    Computes A_ij = P(patch_i interacts with patch_j) using:

    1. Cluster co-membership: Same cluster → high probability
    2. Cluster adjacency: Neighboring clusters on graph → moderate probability
    3. Spatial prior (optional): Nearby patches on grid → small bonus

    Args:
        intra_cluster_weight: Probability for same-cluster interactions (default 1.0)
        inter_cluster_weight: Base probability for different-cluster interactions (default 0.3)
        spatial_decay: Spatial proximity decay factor (0 = disabled)
        masking_mode: 'soft' (continuous probabilities) or 'hard' (binary 0/1)
        min_weight: Minimum non-zero weight (prevents complete blocking)
    """

    def __init__(
        self,
        intra_cluster_weight: float = 1.0,
        inter_cluster_weight: float = 0.3,
        spatial_decay: float = 0.0,
        masking_mode: str = "soft",
        min_weight: float = 0.0,
    ):
        self.intra_cluster_weight = intra_cluster_weight
        self.inter_cluster_weight = inter_cluster_weight
        self.spatial_decay = spatial_decay
        self.masking_mode = masking_mode
        self.min_weight = min_weight

    def build(
        self,
        cluster_labels: np.ndarray,
        adjacency: torch.Tensor = None,
        grid_size: int = 12,
    ) -> torch.Tensor:
        """
        Build the feasibility matrix A where A_ij = P(patch_i interacts with patch_j).

        Args:
            cluster_labels: (N,) cluster assignment for each patch
            adjacency: (N, N) optional graph adjacency for inter-cluster weights
            grid_size: Patch grid size (14 for 224/16)

        Returns:
            feasibility: (N+1, N+1) tensor, row/col 0 = CLS token
        """
        if isinstance(cluster_labels, torch.Tensor):
            cluster_labels = cluster_labels.cpu().numpy()

        N = len(cluster_labels)
        k = len(np.unique(cluster_labels))

        print(f"  Building feasibility matrix: N={N}, k={k}, mode={self.masking_mode}")

        # ── Step 1: Cluster co-membership matrix ──
        # A_ij = intra_weight if same cluster, else 0
        labels_row = cluster_labels.reshape(N, 1)
        labels_col = cluster_labels.reshape(1, N)
        same_cluster = (labels_row == labels_col).astype(np.float32)
        A = same_cluster * self.intra_cluster_weight  # (N, N)

        # ── Step 2: Inter-cluster interaction probability ──
        if adjacency is not None:
            inter_probs = self._compute_inter_cluster_probs(
                cluster_labels, adjacency, k, N
            )
            # Add inter-cluster probabilities where not same-cluster
            diff_cluster = 1.0 - same_cluster
            A = A + diff_cluster * inter_probs

        # ── Step 3: Spatial proximity prior ──
        if self.spatial_decay > 0:
            spatial = self._compute_spatial_prior(N, grid_size)
            # Blend: spatial provides a floor for nearby patches
            A = np.maximum(A, spatial)

        # ── Step 4: Apply masking mode ──
        if self.masking_mode == "hard":
            # Binary: above threshold → 1, else → 0
            threshold = self.inter_cluster_weight / 2
            A = (A > threshold).astype(np.float32)
        else:
            # Soft: keep continuous values, apply floor
            if self.min_weight > 0:
                A = np.maximum(A, self.min_weight)

        # Clip to [0, 1]
        A = np.clip(A, 0.0, 1.0)

        # ── Step 5: Add CLS token (index 0, fully unconstrained) ──
        F = np.ones((N + 1, N + 1), dtype=np.float32)
        F[1:, 1:] = A

        feasibility = torch.tensor(F, dtype=torch.float32)

        # Stats
        patch_F = feasibility[1:, 1:]
        stats = self._compute_stats(patch_F)
        print(f"  Sparsity: {stats['sparsity']:.1%} blocked")
        print(f"  Mean weight: {stats['mean_weight']:.4f}")
        print(f"  Weight distribution: 0={stats['n_zero']}, (0,0.5]={stats['n_low']}, "
              f"(0.5,1)={stats['n_mid']}, 1={stats['n_one']}")

        return feasibility

    def _compute_inter_cluster_probs(
        self, labels: np.ndarray, adjacency, k: int, N: int
    ) -> np.ndarray:
        """
        Compute inter-cluster interaction probabilities from graph adjacency.

        P(interact | different cluster) = f(graph_connectivity between clusters)

        The probability is proportional to the average edge weight between clusters.
        """
        if isinstance(adjacency, torch.Tensor):
            adj_np = adjacency.cpu().numpy()
        else:
            adj_np = np.array(adjacency)

        # Compute cluster-level connectivity
        # cluster_sim[a, b] = average edge weight between patches in cluster a and cluster b
        cluster_edge_sum = np.zeros((k, k), dtype=np.float64)
        cluster_edge_count = np.zeros((k, k), dtype=np.float64)

        for i in range(N):
            for j in range(i + 1, N):
                ci, cj = labels[i], labels[j]
                if ci != cj and adj_np[i, j] > 0:
                    cluster_edge_sum[ci, cj] += adj_np[i, j]
                    cluster_edge_sum[cj, ci] += adj_np[i, j]
                    cluster_edge_count[ci, cj] += 1
                    cluster_edge_count[cj, ci] += 1

        # Average edge weight between clusters
        cluster_sim = np.zeros((k, k), dtype=np.float32)
        mask = cluster_edge_count > 0
        cluster_sim[mask] = (cluster_edge_sum[mask] / cluster_edge_count[mask]).astype(np.float32)

        # Normalize to [0, inter_cluster_weight]
        if cluster_sim.max() > 0:
            cluster_sim = cluster_sim / cluster_sim.max() * self.inter_cluster_weight

        # Expand cluster-level probs to patch-level
        inter_probs = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in range(N):
                ci, cj = labels[i], labels[j]
                if ci != cj:
                    inter_probs[i, j] = cluster_sim[ci, cj]

        return inter_probs

    def _compute_spatial_prior(self, N: int, grid_size: int) -> np.ndarray:
        """
        Spatial proximity prior: nearby patches get a small interaction bonus.

        Uses Gaussian decay based on grid distance:
            spatial_ij = spatial_decay * exp(-dist_ij² / (2 * σ²))
        """
        coords = np.array([(i // grid_size, i % grid_size) for i in range(N)])
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist_sq = (diff ** 2).sum(axis=-1)

        sigma = grid_size / 4.0
        spatial = self.spatial_decay * np.exp(-dist_sq / (2 * sigma ** 2))
        return spatial.astype(np.float32)

    def _compute_stats(self, patch_F: torch.Tensor) -> dict:
        """Compute statistics about the feasibility matrix."""
        total = patch_F.numel()
        return {
            "sparsity": (patch_F == 0).sum().item() / total,
            "mean_weight": patch_F.mean().item(),
            "n_zero": (patch_F == 0).sum().item(),
            "n_low": ((patch_F > 0) & (patch_F <= 0.5)).sum().item(),
            "n_mid": ((patch_F > 0.5) & (patch_F < 1.0)).sum().item(),
            "n_one": (patch_F == 1.0).sum().item(),
        }

    def get_sparsity(self, feasibility: torch.Tensor) -> float:
        """Fraction of blocked attention connections."""
        return (feasibility == 0).sum().item() / feasibility.numel()

    def visualize(
        self,
        feasibility: torch.Tensor,
        cluster_labels: np.ndarray = None,
        save_dir: str = None,
        grid_size: int = 12,
    ):
        """
        Comprehensive visualization of the feasibility matrix.

        Generates:
        1. Full (N+1)×(N+1) matrix heatmap with CLS token
        2. Patch-only region
        3. Interaction probability distribution
        4. Spatial map of mean interaction per patch
        """
        F_np = feasibility.cpu().numpy()
        F_patches = F_np[1:, 1:]
        N = F_patches.shape[0]

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # ── Plot 1: Full matrix with CLS ──
        im0 = axes[0, 0].imshow(F_np, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        axes[0, 0].set_title("Full Feasibility Matrix (with CLS token)",
                            fontsize=12, fontweight="bold")
        axes[0, 0].set_xlabel("Key index (0=CLS)")
        axes[0, 0].set_ylabel("Query index (0=CLS)")
        # Mark CLS region
        axes[0, 0].axhline(y=0.5, color="cyan", linewidth=1, linestyle="--")
        axes[0, 0].axvline(x=0.5, color="cyan", linewidth=1, linestyle="--")
        plt.colorbar(im0, ax=axes[0, 0], shrink=0.8, label="P(interaction)")

        # ── Plot 2: Patch-only region ──
        im1 = axes[0, 1].imshow(F_patches, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        n_patches = F_patches.shape[0]
        axes[0, 1].set_title(f"Patch-Only Region ({n_patches}×{n_patches})",
                            fontsize=12, fontweight="bold")
        axes[0, 1].set_xlabel("Key patch")
        axes[0, 1].set_ylabel("Query patch")
        plt.colorbar(im1, ax=axes[0, 1], shrink=0.8, label="P(interaction)")

        # ── Plot 3: Interaction probability distribution ──
        flat = F_patches.flatten()
        axes[1, 0].hist(flat, bins=50, color="#2196F3", edgecolor="white", alpha=0.85)
        axes[1, 0].set_xlabel("P(interaction)", fontsize=11)
        axes[1, 0].set_ylabel("Count", fontsize=11)
        axes[1, 0].set_title("Interaction Probability Distribution",
                            fontsize=12, fontweight="bold")
        axes[1, 0].axvline(flat.mean(), color="red", linestyle="--",
                          label=f"mean={flat.mean():.3f}")
        axes[1, 0].legend()

        # ── Plot 4: Mean interaction per patch (spatial map) ──
        mean_interaction = F_patches.mean(axis=1)  # (196,)
        interaction_grid = mean_interaction.reshape(grid_size, grid_size)
        im3 = axes[1, 1].imshow(interaction_grid, cmap="hot", interpolation="bilinear")
        axes[1, 1].set_title("Mean Interaction Probability per Patch",
                            fontsize=12, fontweight="bold")
        plt.colorbar(im3, ax=axes[1, 1], shrink=0.8, label="Mean P(interact)")

        sparsity = self.get_sparsity(feasibility)
        plt.suptitle(
            f"Structural Feasibility Matrix — {sparsity:.1%} blocked, "
            f"mode={self.masking_mode}",
            fontsize=14, fontweight="bold"
        )
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, "feasibility_matrix.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  ✓ Saved: {path}")
        plt.close(fig)

        # ── Additional: Cluster-sorted view ──
        if cluster_labels is not None:
            self._visualize_sorted(F_patches, cluster_labels, save_dir)

    def _visualize_sorted(self, F_patches, labels, save_dir):
        """Visualize matrix sorted by cluster for block structure visibility."""
        sort_idx = np.argsort(labels)
        F_sorted = F_patches[sort_idx][:, sort_idx]
        labels_sorted = labels[sort_idx]

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        im = ax.imshow(F_sorted, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

        # Draw cluster boundaries
        k = len(np.unique(labels_sorted))
        boundaries = []
        for c in range(k):
            positions = np.where(labels_sorted == c)[0]
            if len(positions) > 0:
                boundaries.append(positions[-1] + 0.5)

        for b in boundaries[:-1]:
            ax.axhline(y=b, color="white", linewidth=0.8, linestyle="-")
            ax.axvline(x=b, color="white", linewidth=0.8, linestyle="-")

        ax.set_title(f"Feasibility Matrix (sorted by cluster)\n"
                    f"Block-diagonal structure = same-cluster interactions",
                    fontsize=12, fontweight="bold")
        ax.set_xlabel("Key patch (sorted)")
        ax.set_ylabel("Query patch (sorted)")
        plt.colorbar(im, ax=ax, shrink=0.8, label="P(interaction)")
        plt.tight_layout()

        if save_dir:
            path = os.path.join(save_dir, "feasibility_sorted.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  ✓ Saved: {path}")
        plt.close(fig)


# ════════════════════════════════════════════════════════════
#  Hybrid Feasibility Matrix
# ════════════════════════════════════════════════════════════

def build_hybrid_feasibility(
    F_anatomy: torch.Tensor,
    F_feature: torch.Tensor,
    alpha: float = 0.7,
) -> torch.Tensor:
    """
    Build the hybrid feasibility matrix by blending anatomy and feature priors.

    Formula:
        F_hybrid = alpha * F_anatomy + (1 - alpha) * F_feature

    Where:
        F_anatomy: (N+1, N+1) — heuristic spatial anatomy prior
        F_feature: (N+1, N+1) — learned DINO + clustering prior
        alpha:     blend weight, default 0.7 (favour anatomy)

    Args:
        F_anatomy: Anatomy prior matrix — values in [0, 1]
        F_feature: Feature prior matrix — values in [0, 1]
        alpha:     Blend weight for anatomy (1-alpha goes to feature)

    Returns:
        F_hybrid: (N+1, N+1) float32 tensor, values in [0, 1]
                  Row/col 0 (CLS token) forced to 1.0
    """
    if F_anatomy.shape != F_feature.shape:
        raise ValueError(
            f"Shape mismatch: F_anatomy={F_anatomy.shape}, F_feature={F_feature.shape}. "
            f"Ensure both are built with the same grid_size."
        )

    # Weighted blend
    F_hybrid = alpha * F_anatomy.float() + (1.0 - alpha) * F_feature.float()

    # Clamp to valid probability range
    F_hybrid = F_hybrid.clamp(0.0, 1.0)

    # CLS token (index 0) is always fully connected
    F_hybrid[0, :] = 1.0
    F_hybrid[:, 0] = 1.0

    # Stats
    patch_F = F_hybrid[1:, 1:]
    print(
        f"  F_hybrid: shape={list(F_hybrid.shape)}, alpha={alpha:.2f}, "
        f"mean={patch_F.mean():.4f}, min={patch_F.min():.4f}, max={patch_F.max():.4f}"
    )

    return F_hybrid


def save_feasibility(
    feasibility: torch.Tensor,
    cluster_labels: np.ndarray,
    metadata: dict = None,
    save_path: str = "outputs/feasibility/feasibility_matrix.pt",
):
    """Save feasibility matrix and cluster labels to disk."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data = {
        "feasibility_matrix": feasibility,
        "cluster_labels": torch.tensor(cluster_labels) if not isinstance(cluster_labels, torch.Tensor) else cluster_labels,
        "shape": list(feasibility.shape),
    }
    if metadata:
        data["metadata"] = metadata
    torch.save(data, save_path)
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"  ✓ Saved: {save_path} ({size_mb:.1f} MB)")


def load_feasibility(load_path: str) -> dict:
    """Load a saved feasibility matrix."""
    data = torch.load(load_path, map_location="cpu")
    print(f"  ✓ Loaded feasibility: {load_path}, shape={data['shape']}")
    return data
