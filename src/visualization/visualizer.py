"""
Publication-Quality Visualization for Hybrid Anatomy-Aware SC-ViT.

Generates 6 types of figures:
  1. Attention overlays on X-ray images
  2. Anatomy region maps (7-region heuristic)
  3. Feasibility matrix heatmaps
  4. Constrained vs unconstrained attention comparison
  5. Patch-region assignment visualization
  6. Hybrid structure comparison (F_anatomy, F_feature, F_hybrid)

All figures use a consistent publication-ready style.
Save to outputs/visualizations/ by default.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (safe for all environments)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import torch
from typing import Optional


# ── Global plot style ──
PLT_STYLE = {
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "axes.titleweight":  "bold",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  "white",
    "axes.facecolor":    "#f8f9fa",
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
}

REGION_COLORS = [
    "#4CAF50", "#81C784", "#2196F3", "#64B5F6",
    "#E91E63", "#FF9800", "#9E9E9E",
]
REGION_NAMES = [
    "Left Upper Lung", "Left Lower Lung",
    "Right Upper Lung", "Right Lower Lung",
    "Heart", "Mediastinum", "Background",
]


class ExperimentVisualizer:
    """
    Publication-quality visualization for all experiment outputs.

    Args:
        output_dir: Base directory for saving figures
        dpi:        Figure DPI (default 150 for publication quality)
    """

    def __init__(self, output_dir: str = "outputs/visualizations", dpi: int = 150):
        self.output_dir = output_dir
        self.dpi        = dpi
        os.makedirs(output_dir, exist_ok=True)

    def _save(self, fig, filename: str):
        """Save figure and close."""
        path = os.path.join(self.output_dir, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  ✓ Saved: {path}")
        return path

    # ────────────────────────────────────────────────────────
    #  1. Attention Overlay
    # ────────────────────────────────────────────────────────

    def plot_attention_overlay(
        self,
        image: np.ndarray,
        attn_weights: torch.Tensor,
        title: str = "Attention Map",
        save_name: str = "attention_overlay.png",
        grid_size: int = 12,
        layer_idx: int = -1,
    ) -> str:
        """
        Overlay CLS-token attention on the original X-ray image.

        Args:
            image:        (H, W, 3) or (H, W) numpy array — original image
            attn_weights: (B, heads, N+1, N+1) or (heads, N+1, N+1)
            title:        Plot title
            save_name:    Output filename
            grid_size:    Patch grid side (12 for 192px)
            layer_idx:    Which layer (-1 = last)
        """
        with plt.rc_context(PLT_STYLE):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Process image
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)

            # Extract CLS attention (row 0 = CLS query, cols 1.. = patch keys)
            if attn_weights.dim() == 4:
                attn = attn_weights[0]         # (heads, N+1, N+1)
            else:
                attn = attn_weights            # (heads, N+1, N+1)

            # Mean over heads, CLS token row (index 0), patch columns
            cls_attn = attn[:, 0, 1:].mean(dim=0).cpu().numpy()   # (N,)
            attn_grid = cls_attn.reshape(grid_size, grid_size)

            # Upsample to image size
            from PIL import Image as PILImage
            h, w = image.shape[:2]
            attn_img = PILImage.fromarray(
                (attn_grid / (attn_grid.max() + 1e-8) * 255).astype(np.uint8)
            ).resize((w, h), PILImage.BICUBIC)
            attn_img = np.array(attn_img) / 255.0

            # Plot 1: Original
            axes[0].imshow(image, cmap="gray" if image.mean() < 0.5 else None)
            axes[0].set_title("Original X-ray")
            axes[0].axis("off")

            # Plot 2: Attention heatmap
            im = axes[1].imshow(attn_img, cmap="hot")
            axes[1].set_title(f"CLS Attention\n(mean over heads, {attn.shape[0]} heads)")
            axes[1].axis("off")
            plt.colorbar(im, ax=axes[1], shrink=0.8)

            # Plot 3: Overlay
            axes[2].imshow(image, cmap="gray" if image.mean() < 0.5 else None)
            axes[2].imshow(attn_img, cmap="jet", alpha=0.5)
            axes[2].set_title("Attention Overlay")
            axes[2].axis("off")

            fig.suptitle(title, fontsize=14, fontweight="bold")
            plt.tight_layout()
            return self._save(fig, save_name)

    # ────────────────────────────────────────────────────────
    #  2. Anatomy Region Map
    # ────────────────────────────────────────────────────────

    def plot_anatomy_regions(
        self,
        anatomy_labels: np.ndarray,
        grid_size: int = 12,
        save_name: str = "anatomy_regions.png",
    ) -> str:
        """
        Visualize the 7-region anatomy map on the patch grid.

        Args:
            anatomy_labels: (N,) int array — region per patch
            grid_size:      Patch grid side length (12 for 192px)
            save_name:      Output filename
        """
        with plt.rc_context(PLT_STYLE):
            fig, ax = plt.subplots(figsize=(8, 7))
            cmap = ListedColormap(REGION_COLORS)

            grid = anatomy_labels.reshape(grid_size, grid_size)
            im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=6, interpolation="nearest")

            # Grid lines
            ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=1.2)
            ax.tick_params(which="minor", size=0)
            ax.set_xticks(range(grid_size))
            ax.set_yticks(range(grid_size))
            ax.set_xticklabels([str(i) for i in range(grid_size)], fontsize=9)
            ax.set_yticklabels([str(i) for i in range(grid_size)], fontsize=9)

            # Region ID labels in cells
            for r in range(grid_size):
                for c in range(grid_size):
                    ax.text(c, r, str(grid[r, c]),
                            ha="center", va="center", fontsize=9,
                            color="white", fontweight="bold")

            # Legend
            patches = [
                mpatches.Patch(color=REGION_COLORS[i], label=f"{i}: {REGION_NAMES[i]}")
                for i in range(7)
            ]
            ax.legend(handles=patches, loc="upper right",
                      bbox_to_anchor=(1.42, 1.0), fontsize=9)

            ax.set_title(f"Anatomy Region Map — {grid_size}×{grid_size} Patch Grid\n"
                         f"(192×192 image, 16×16 patches, 7 regions)",
                         fontsize=13, fontweight="bold")
            ax.set_xlabel("Patch column")
            ax.set_ylabel("Patch row")

            plt.tight_layout()
            return self._save(fig, save_name)

    # ────────────────────────────────────────────────────────
    #  3. Feasibility Matrix Heatmap
    # ────────────────────────────────────────────────────────

    def plot_feasibility_heatmap(
        self,
        F: torch.Tensor,
        title: str = "Feasibility Matrix",
        save_name: str = "feasibility.png",
    ) -> str:
        """
        Publication-quality heatmap of a feasibility matrix.

        Args:
            F:         (N+1, N+1) feasibility matrix (includes CLS at idx 0)
            title:     Plot title
            save_name: Output filename
        """
        with plt.rc_context(PLT_STYLE):
            F_np = F.cpu().numpy()
            N    = F_np.shape[0]

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Full matrix
            im0 = axes[0].imshow(F_np, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
            axes[0].set_title(f"Full Matrix (N+1={N}, includes CLS at row/col 0)")
            axes[0].set_xlabel("Key token index (0=CLS)")
            axes[0].set_ylabel("Query token index (0=CLS)")
            axes[0].axhline(y=0.5, color="cyan", linewidth=1, linestyle="--", alpha=0.7)
            axes[0].axvline(x=0.5, color="cyan", linewidth=1, linestyle="--", alpha=0.7)
            plt.colorbar(im0, ax=axes[0], shrink=0.85, label="Interaction weight")

            # Patch-only (exclude CLS)
            F_patches = F_np[1:, 1:]
            n_patches = F_patches.shape[0]
            grid_size = int(n_patches ** 0.5)
            im1 = axes[1].imshow(F_patches, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
            axes[1].set_title(f"Patch-Only ({n_patches}×{n_patches}, {grid_size}×{grid_size} grid)\n"
                              f"Mean={F_patches.mean():.3f}")
            axes[1].set_xlabel("Key patch")
            axes[1].set_ylabel("Query patch")
            plt.colorbar(im1, ax=axes[1], shrink=0.85, label="Interaction weight")

            fig.suptitle(title, fontsize=14, fontweight="bold")
            plt.tight_layout()
            return self._save(fig, save_name)

    # ────────────────────────────────────────────────────────
    #  4. Constrained vs Unconstrained Comparison
    # ────────────────────────────────────────────────────────

    def plot_constrained_vs_unconstrained(
        self,
        attn_baseline:    torch.Tensor,
        attn_constrained: torch.Tensor,
        image:            np.ndarray,
        grid_size: int = 12,
        save_name: str = "constrained_vs_unconstrained.png",
    ) -> str:
        """
        Side-by-side CLS attention comparison: Baseline vs Constrained.

        Args:
            attn_baseline:    (heads, N+1, N+1) baseline attention
            attn_constrained: (heads, N+1, N+1) constrained attention
            image:            (H, W, 3) original image
            grid_size:        Patch grid size (12 for 192px)
            save_name:        Output filename
        """
        from PIL import Image as PILImage

        def get_cls_attn(attn, gs):
            cls = attn[:, 0, 1:].mean(dim=0).cpu().numpy()   # (N,)
            return cls.reshape(gs, gs)

        with plt.rc_context(PLT_STYLE):
            fig, axes = plt.subplots(1, 4, figsize=(18, 5))

            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            h, w = image.shape[:2]

            def upsample(grid):
                img = PILImage.fromarray(
                    (grid / (grid.max() + 1e-8) * 255).astype(np.uint8)
                ).resize((w, h), PILImage.BICUBIC)
                return np.array(img) / 255.0

            # Original
            axes[0].imshow(image, cmap="gray" if image.mean() < 0.5 else None)
            axes[0].set_title("Input X-ray")
            axes[0].axis("off")

            # Baseline
            base_grid = get_cls_attn(attn_baseline, grid_size)
            axes[1].imshow(upsample(base_grid), cmap="hot", vmin=0, vmax=1)
            axes[1].set_title("Baseline ViT\nAttention (CLS)")
            axes[1].axis("off")

            # Constrained
            cons_grid = get_cls_attn(attn_constrained, grid_size)
            axes[2].imshow(upsample(cons_grid), cmap="hot", vmin=0, vmax=1)
            axes[2].set_title("Hybrid SC-ViT\nAttention (CLS)")
            axes[2].axis("off")

            # Difference map
            diff = cons_grid - base_grid
            diff_img = upsample(np.abs(diff))
            im3 = axes[3].imshow(diff_img, cmap="RdBu_r", vmin=-1, vmax=1)
            axes[3].set_title("Difference\n(Constrained - Baseline)")
            axes[3].axis("off")
            plt.colorbar(im3, ax=axes[3], shrink=0.8)

            fig.suptitle("Constrained vs Unconstrained Attention Comparison",
                         fontsize=14, fontweight="bold")
            plt.tight_layout()
            return self._save(fig, save_name)

    # ────────────────────────────────────────────────────────
    #  5. Patch-Region Visualization
    # ────────────────────────────────────────────────────────

    def plot_patch_region_map(
        self,
        anatomy_labels: np.ndarray,
        grid_size: int = 12,
        save_name: str = "patch_region_map.png",
    ) -> str:
        """Plot which anatomical region each patch belongs to (alias)."""
        return self.plot_anatomy_regions(anatomy_labels, grid_size, save_name)

    # ────────────────────────────────────────────────────────
    #  6. Hybrid Structure Comparison
    # ────────────────────────────────────────────────────────

    def plot_hybrid_structure(
        self,
        F_anatomy: torch.Tensor,
        F_feature: torch.Tensor,
        F_hybrid:  torch.Tensor,
        alpha:     float = 0.7,
        save_name: str = "hybrid_structure_comparison.png",
    ) -> str:
        """
        3-panel comparison: F_anatomy | F_feature | F_hybrid.

        Shows how the hybrid matrix blends anatomical priors with learned features.

        Args:
            F_anatomy: (N+1, N+1) anatomy prior
            F_feature: (N+1, N+1) feature prior
            F_hybrid:  (N+1, N+1) hybrid blend
            alpha:     Blend weight (shown in title)
            save_name: Output filename
        """
        with plt.rc_context(PLT_STYLE):
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            cmap = "RdYlGn"

            matrices = [F_anatomy, F_feature, F_hybrid]
            titles   = [
                f"F_anatomy\n(heuristic spatial prior)",
                f"F_feature\n(DINO + clustering)",
                f"F_hybrid = {alpha:.1f}·F_anatomy + {1-alpha:.1f}·F_feature",
            ]

            for ax, F, title in zip(axes, matrices, titles):
                F_np = F.cpu().numpy()[1:, 1:]   # patch-only region
                im   = ax.imshow(F_np, cmap=cmap, vmin=0, vmax=1, aspect="auto")
                n    = F_np.shape[0]
                ax.set_title(f"{title}\nmean={F_np.mean():.3f}", fontsize=11)
                ax.set_xlabel(f"Key patch (N={n})")
                ax.set_ylabel("Query patch")
                plt.colorbar(im, ax=ax, shrink=0.85)

            fig.suptitle(
                "Hybrid Feasibility Matrix: Anatomy + Feature Priors",
                fontsize=14, fontweight="bold"
            )
            plt.tight_layout()
            return self._save(fig, save_name)

    # ────────────────────────────────────────────────────────
    #  Generate All Publication Figures
    # ────────────────────────────────────────────────────────

    def generate_all_results_figures(self, all_results: list) -> list:
        """
        Generate all publication-quality comparison figures from experiment results.

        Args:
            all_results: List of result dicts from run_all_experiments.py

        Returns:
            List of saved file paths
        """
        with plt.rc_context(PLT_STYLE):
            saved = []
            model_names  = [r["model_name"] for r in all_results]
            short_names  = [n.replace("Hybrid ", "").replace("Anatomy ", "")
                            .replace("Feature ", "").replace("Baseline", "Base")
                            for n in model_names]
            colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]

            TARGET_NAMES = ["Atelectasis", "Cardiomegaly", "Consolidation",
                            "Edema", "Pleural Effusion"]

            # ── Main metrics bar chart ──
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))

            metrics_map = [
                ("auroc", "mean", "Mean AUROC", "AUROC"),
                ("f1",    "mean", "Mean F1 Score", "F1"),
                ("accuracy", "mean", "Mean Accuracy", "Accuracy"),
            ]

            for ax, (key, sub, title, ylabel) in zip(axes, metrics_map):
                vals = [r.get(key, {}).get(sub, 0.0) for r in all_results]
                bars = ax.bar(short_names, vals, color=colors, edgecolor="white",
                              linewidth=0.8)
                ax.set_title(title)
                ymin = max(0, min(vals) - 0.05)
                ax.set_ylim(ymin, max(vals) + 0.04)
                ax.set_ylabel(ylabel)
                for bar, val in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.005,
                            f"{val:.4f}", ha="center", va="bottom",
                            fontsize=9, fontweight="bold")
                ax.tick_params(axis="x", rotation=20)

            plt.suptitle("Model Comparison — 4 Model Variants", fontsize=14, fontweight="bold")
            plt.tight_layout()
            saved.append(self._save(fig, "results/main_metrics_comparison.png"))

            # ── Per-class AUROC heatmap ──
            fig, ax = plt.subplots(figsize=(10, 5))
            auroc_matrix = np.array([
                [r.get("auroc", {}).get(c, 0.0) for c in TARGET_NAMES]
                for r in all_results
            ])
            im = ax.imshow(auroc_matrix, cmap="YlOrRd", aspect="auto", vmin=0.5, vmax=1.0)
            ax.set_xticks(range(len(TARGET_NAMES)))
            ax.set_xticklabels(TARGET_NAMES, rotation=30, ha="right")
            ax.set_yticks(range(len(model_names)))
            ax.set_yticklabels(short_names)
            ax.set_title("Per-Class AUROC Comparison", fontsize=13, fontweight="bold")
            for i in range(len(model_names)):
                for j in range(len(TARGET_NAMES)):
                    ax.text(j, i, f"{auroc_matrix[i, j]:.3f}",
                            ha="center", va="center", fontsize=9, fontweight="bold",
                            color="white" if auroc_matrix[i, j] > 0.75 else "black")
            plt.colorbar(im, ax=ax, label="AUROC", shrink=0.8)
            plt.tight_layout()
            saved.append(self._save(fig, "results/per_class_auroc_heatmap.png"))

            # ── Attention entropy comparison ──
            fig, ax = plt.subplots(figsize=(9, 5))
            entropy_vals = [r.get("attention_entropy", {}).get("mean", 0.0) for r in all_results]
            entropy_stds = [r.get("attention_entropy", {}).get("std", 0.0) for r in all_results]
            bars = ax.bar(short_names, entropy_vals, color=colors,
                          edgecolor="white", yerr=entropy_stds, capsize=5)
            ax.set_title("Attention Entropy (lower = more focused)", fontweight="bold")
            ax.set_ylabel("Mean Entropy (nats)")
            for bar, val in zip(bars, entropy_vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
            ax.tick_params(axis="x", rotation=20)
            plt.tight_layout()
            saved.append(self._save(fig, "results/attention_entropy_comparison.png"))

            # ── Anatomical Attention Ratio ──
            if any("anatomical_attention_ratio" in r for r in all_results):
                fig, ax = plt.subplots(figsize=(9, 5))
                aar_vals = [r.get("anatomical_attention_ratio", {}).get("mean", 0.0)
                            for r in all_results]
                bars = ax.bar(short_names, aar_vals, color=colors, edgecolor="white")
                ax.set_title("Anatomical Attention Ratio\n(higher = more anatomy-aligned)",
                             fontweight="bold")
                ax.set_ylabel("AAR")
                ax.set_ylim(0, max(aar_vals) + 0.1)
                for bar, val in zip(bars, aar_vals):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.005,
                            f"{val:.3f}", ha="center", va="bottom", fontsize=9)
                ax.tick_params(axis="x", rotation=20)
                plt.tight_layout()
                saved.append(self._save(fig, "results/anatomical_attention_ratio.png"))

            # ── Noise robustness curves ──
            fig, ax = plt.subplots(figsize=(9, 5))
            for i, r in enumerate(all_results):
                noise_data = r.get("noise_robustness", {})
                if not noise_data:
                    continue
                noise_levels = sorted(noise_data.keys())
                aurocs       = [noise_data[s] for s in noise_levels]
                ax.plot(noise_levels, aurocs, marker="o", linewidth=2,
                        label=short_names[i], color=colors[i], markersize=6)
            ax.set_xlabel("Noise Level (σ)")
            ax.set_ylabel("AUROC")
            ax.set_title("Noise Robustness (higher = more robust)", fontweight="bold")
            ax.legend(loc="lower left", framealpha=0.9)
            plt.tight_layout()
            saved.append(self._save(fig, "results/noise_robustness_curves.png"))

            # ── Radar chart ──
            fig = plt.figure(figsize=(8, 8))
            ax  = fig.add_subplot(111, polar=True)
            metric_labels = ["AUROC", "F1", "Accuracy", "Entropy↓\n(norm)", "Noise\nRobust"]
            n_metrics = len(metric_labels)
            angles    = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
            angles   += angles[:1]

            max_ent = max(
                (r.get("attention_entropy", {}).get("mean", 0.01) for r in all_results),
                default=1.0
            )

            for i, r in enumerate(all_results):
                ent      = r.get("attention_entropy", {}).get("mean", 0.0)
                noise_0  = r.get("noise_robustness", {}).get(0.0, 0.7)
                noise_03 = r.get("noise_robustness", {}).get(0.3, 0.6)
                retention = noise_03 / max(noise_0, 1e-8)

                values = [
                    r.get("auroc",    {}).get("mean", 0.0),
                    r.get("f1",       {}).get("mean", 0.0),
                    r.get("accuracy", {}).get("mean", 0.0),
                    1.0 - ent / max(max_ent, 1e-8),   # inverted entropy
                    retention,
                ]
                values += values[:1]

                ax.plot(angles, values, linewidth=2, label=short_names[i], color=colors[i])
                ax.fill(angles, values, alpha=0.1, color=colors[i])

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metric_labels, fontsize=10)
            ax.set_ylim(0, 1.1)
            ax.set_title("Multi-Metric Comparison", fontweight="bold", pad=20)
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), framealpha=0.9)
            plt.tight_layout()
            saved.append(self._save(fig, "results/radar_chart.png"))

            print(f"\n  ✓ Generated {len(saved)} publication figures in {self.output_dir}/results/")
            return saved
