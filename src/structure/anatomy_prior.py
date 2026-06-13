"""
Anatomy Prior Builder for Chest X-ray Patch Grids.

Implements heuristic spatial partitioning of a 12×12 patch grid into 7
clinically meaningful chest X-ray regions, producing an anatomy-aware
feasibility matrix F_anatomy ∈ R^(145×145).

The 7 anatomical regions for a 12×12 grid (192×192 image, 16×16 patches):
  0 — Left Upper Lung    rows 0-4,  cols 0-5
  1 — Left Lower Lung    rows 5-11, cols 0-5
  2 — Right Upper Lung   rows 0-4,  cols 6-11
  3 — Right Lower Lung   rows 5-11, cols 6-11
  4 — Heart Region       rows 3-8,  cols 4-7
  5 — Mediastinum        rows 1-10, cols 4-7   (overlapping centre strip)
  6 — Background         peripheral + bottom rows

Priority (for overlapping regions): Heart > Mediastinum > Lung > Background

Region interaction weights:
  Same region    → 1.0
  Adjacent       → 0.6   (e.g., left upper ↔ left lower, lung ↔ heart)
  Distant        → 0.15  (e.g., left lung ↔ right lung, lung ↔ background)
  Background     → 0.05  (background ↔ background)

Output tensor shape:
  F_anatomy: (N+1, N+1) where N = grid_size² patches
             index 0 = CLS token (fully connected → all 1.0)
             indices 1..N = spatial patches

Usage:
    builder = AnatomyPriorBuilder()
    F_anatomy = builder.build(grid_size=12)   # (145, 145)
    region_map = builder.get_region_map(12)   # (144,) int labels
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os


# ════════════════════════════════════════════════════════════
#  Region Definitions
# ════════════════════════════════════════════════════════════

REGION_NAMES = [
    "Left Upper Lung",    # 0
    "Left Lower Lung",    # 1
    "Right Upper Lung",   # 2
    "Right Lower Lung",   # 3
    "Heart",              # 4
    "Mediastinum",        # 5
    "Background",         # 6
]

NUM_REGIONS = len(REGION_NAMES)

# Adjacency matrix between regions (symmetric).
# 1.0 = same region (diagonal), handled separately.
# Values below are for DIFFERENT regions.
#
#         LUL   LLL   RUL   RLL   HRT   MED   BKG
REGION_ADJACENCY = np.array([
    # LUL
    [1.0,  0.6,  0.6,  0.15, 0.6,  0.6,  0.15],
    # LLL
    [0.6,  1.0,  0.15, 0.6,  0.6,  0.6,  0.15],
    # RUL
    [0.6,  0.15, 1.0,  0.6,  0.6,  0.6,  0.15],
    # RLL
    [0.15, 0.6,  0.6,  1.0,  0.6,  0.6,  0.15],
    # HRT
    [0.6,  0.6,  0.6,  0.6,  1.0,  0.6,  0.05],
    # MED
    [0.6,  0.6,  0.6,  0.6,  0.6,  1.0,  0.05],
    # BKG
    [0.15, 0.15, 0.15, 0.15, 0.05, 0.05, 0.05],
], dtype=np.float32)


class AnatomyPriorBuilder:
    """
    Build a heuristic anatomy-based feasibility matrix for chest X-ray patches.

    Partitions a patch grid into 7 anatomical regions using spatial heuristics
    (no segmentation required) and assigns interaction weights based on clinical
    adjacency knowledge.

    Args:
        same_region_weight:  Interaction weight for patches in the same region (default 1.0)
        adjacent_weight:     Weight for adjacent regions (default 0.6)
        distant_weight:      Weight for distant regions (default 0.15)
        background_weight:   Background-background weight (default 0.05)
    """

    def __init__(
        self,
        same_region_weight: float = 1.0,
        adjacent_weight:    float = 0.6,
        distant_weight:     float = 0.15,
        background_weight:  float = 0.05,
    ):
        self.same_region_weight  = same_region_weight
        self.adjacent_weight     = adjacent_weight
        self.distant_weight      = distant_weight
        self.background_weight   = background_weight

        # Override the global adjacency matrix with user-configured weights
        self._adj = REGION_ADJACENCY.copy()
        # Rescale based on user weights (keep relative structure)
        scale = adjacent_weight / 0.6 if 0.6 > 0 else 1.0
        self._adj = np.where(
            np.eye(NUM_REGIONS, dtype=bool),
            self.same_region_weight,
            self._adj * scale,
        )
        # Ensure background interactions are clamped
        self._adj[6, :] = np.minimum(self._adj[6, :], background_weight * 3)
        self._adj[:, 6] = self._adj[6, :]
        self._adj[6, 6] = background_weight

    def get_region_map(self, grid_size: int = 12) -> np.ndarray:
        """
        Assign each patch to one of 7 anatomical regions.

        Partitioning heuristic for a 12×12 grid (scaled for other sizes):
          - Horizontal mid-point: row >= grid_size//2 → lower lung
          - Vertical centre strip (cols 4-7 of 12): heart/mediastinum
          - Heart: rows 3-8, cols 4-7 (centre vertical band, mid rows)
          - Mediastinum: rows 1-10, cols 4-7 (broader centre strip)
          - Background: outermost rows + corners

        Priority (highest wins when overlapping):
            Heart (4) > Mediastinum (5) > Lung (0-3) > Background (6)

        Args:
            grid_size: Patch grid side length (12 for 192×192/16px patches)

        Returns:
            region_labels: (N,) int array — N = grid_size², values in 0..6
        """
        N = grid_size * grid_size
        labels = np.full(N, 6, dtype=np.int32)  # default = Background

        # Fractional thresholds (scale with grid_size)
        row_mid = grid_size // 2                   # 6 for grid_size=12
        col_left_edge  = int(grid_size * 0.33)     # col 4 for 12
        col_right_edge = int(grid_size * 0.67)     # col 8 for 12

        # Heart centre vertical band (narrower than mediastinum)
        heart_col_l = int(grid_size * 0.35)        # col 4
        heart_col_r = int(grid_size * 0.65)        # col 8
        heart_row_t = int(grid_size * 0.25)        # row 3
        heart_row_b = int(grid_size * 0.75)        # row 9

        # Mediastinum: slightly wider and taller centre strip
        med_col_l = int(grid_size * 0.33)
        med_col_r = int(grid_size * 0.67)
        med_row_t = int(grid_size * 0.10)          # row 1
        med_row_b = int(grid_size * 0.90)          # row 11

        # Background: outermost 1-2 rows/cols
        bg_border = 1

        for idx in range(N):
            row = idx // grid_size
            col = idx % grid_size

            # Start as background
            region = 6

            # Assign lung regions (overwrite background)
            if col < col_left_edge:
                # Left lung
                region = 0 if row < row_mid else 1
            elif col >= col_right_edge:
                # Right lung
                region = 2 if row < row_mid else 3
            else:
                # Centre column band — mediastinum or heart
                in_med = (med_row_t <= row < med_row_b)
                in_heart = (heart_row_t <= row < heart_row_b and
                            heart_col_l <= col < heart_col_r)
                if in_heart:
                    region = 4   # Heart (higher priority)
                elif in_med:
                    region = 5   # Mediastinum
                else:
                    region = 6   # Background (top/bottom of centre)

            # Override: true background corners/extreme edges
            is_corner = (row < bg_border and col < bg_border) or \
                        (row < bg_border and col >= grid_size - bg_border) or \
                        (row >= grid_size - bg_border and col < bg_border) or \
                        (row >= grid_size - bg_border and col >= grid_size - bg_border)
            if is_corner:
                region = 6

            labels[idx] = region

        return labels

    def build(self, grid_size: int = 12) -> torch.Tensor:
        """
        Build the anatomy feasibility matrix F_anatomy.

        Args:
            grid_size: Patch grid side length (12 for 192×192, 14 for 224×224)

        Returns:
            F_anatomy: (N+1, N+1) float32 tensor — row/col 0 = CLS token
                       All values in [0, 1].
                       F_anatomy[0, :] = F_anatomy[:, 0] = 1.0 (CLS = unconstrained)

        Tensor shape detail:
            N = grid_size² = 144 for 12×12
            Output: (145, 145)  [including CLS at index 0]
        """
        N = grid_size * grid_size   # 144 for 12×12

        # ── Step 1: Get region labels for each patch ──
        region_labels = self.get_region_map(grid_size)   # (N,) ints in 0..6

        print(f"  AnatomyPriorBuilder: grid={grid_size}×{grid_size}, N={N} patches")
        for r in range(NUM_REGIONS):
            count = (region_labels == r).sum()
            print(f"    Region {r} ({REGION_NAMES[r]}): {count} patches")

        # ── Step 2: Build patch-level interaction matrix ──
        # F[i, j] = REGION_ADJACENCY[region_i, region_j]
        # Shape: (N, N)
        A = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in range(N):
                ri = region_labels[i]
                rj = region_labels[j]
                A[i, j] = self._adj[ri, rj]

        # Clip to [0, 1]
        A = np.clip(A, 0.0, 1.0)

        # ── Step 3: Add CLS token row/col (index 0 = fully connected) ──
        # F_anatomy shape: (N+1, N+1)
        F = np.ones((N + 1, N + 1), dtype=np.float32)
        F[1:, 1:] = A   # patch-patch interactions
        # F[0, :] = 1.0  and F[:, 0] = 1.0 already set by np.ones

        F_tensor = torch.tensor(F, dtype=torch.float32)

        # Stats
        patch_A = F_tensor[1:, 1:]
        mean_w = patch_A.mean().item()
        print(f"  F_anatomy: shape={list(F_tensor.shape)}, mean_weight={mean_w:.4f}")

        return F_tensor

    def visualize(
        self,
        grid_size: int = 12,
        save_path: str = None,
        show: bool = False,
    ):
        """
        Visualize the 7-region anatomy map on the patch grid.

        Args:
            grid_size: Patch grid side length
            save_path: Path to save figure (None = don't save)
            show: Whether to call plt.show()
        """
        region_labels = self.get_region_map(grid_size)
        grid = region_labels.reshape(grid_size, grid_size)

        # Color palette for 7 regions
        colors = [
            "#4CAF50",  # 0 Left Upper Lung — green
            "#81C784",  # 1 Left Lower Lung — light green
            "#2196F3",  # 2 Right Upper Lung — blue
            "#64B5F6",  # 3 Right Lower Lung — light blue
            "#E91E63",  # 4 Heart — red/pink
            "#FF9800",  # 5 Mediastinum — orange
            "#9E9E9E",  # 6 Background — grey
        ]

        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(colors)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: region grid
        im = axes[0].imshow(grid, cmap=cmap, vmin=0, vmax=6, interpolation="nearest")
        axes[0].set_title(
            f"Anatomy Region Map — {grid_size}×{grid_size} Patch Grid",
            fontsize=13, fontweight="bold"
        )
        axes[0].set_xlabel("Patch column")
        axes[0].set_ylabel("Patch row")

        # Add grid lines
        axes[0].set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
        axes[0].set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
        axes[0].grid(which="minor", color="white", linewidth=0.8)
        axes[0].tick_params(which="minor", size=0)

        # Add region numbers in each cell
        for r in range(grid_size):
            for c in range(grid_size):
                axes[0].text(c, r, str(grid[r, c]),
                             ha="center", va="center", fontsize=8,
                             color="white", fontweight="bold")

        # Legend
        patches = [mpatches.Patch(color=colors[i], label=f"{i}: {REGION_NAMES[i]}")
                   for i in range(NUM_REGIONS)]
        axes[0].legend(handles=patches, loc="upper right",
                       bbox_to_anchor=(1.0, 1.0), fontsize=9)

        # Right: region interaction weight matrix (7×7)
        im2 = axes[1].imshow(self._adj, cmap="YlOrRd", vmin=0, vmax=1)
        axes[1].set_title("Region Interaction Weights\n(7×7 adjacency)", fontsize=13, fontweight="bold")
        axes[1].set_xticks(range(NUM_REGIONS))
        axes[1].set_xticklabels([n[:6] for n in REGION_NAMES], rotation=45, ha="right", fontsize=9)
        axes[1].set_yticks(range(NUM_REGIONS))
        axes[1].set_yticklabels([n[:6] for n in REGION_NAMES], fontsize=9)
        plt.colorbar(im2, ax=axes[1], shrink=0.8, label="Interaction weight")

        # Annotate cells
        for i in range(NUM_REGIONS):
            for j in range(NUM_REGIONS):
                axes[1].text(j, i, f"{self._adj[i, j]:.2f}",
                             ha="center", va="center", fontsize=9, fontweight="bold",
                             color="white" if self._adj[i, j] > 0.5 else "black")

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  ✓ Anatomy region map saved: {save_path}")

        if show:
            plt.show()
        plt.close(fig)
