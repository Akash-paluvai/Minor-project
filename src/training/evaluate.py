"""
Evaluation utilities: model evaluation, attention visualization, and comparison.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from .metrics import MetricTracker


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader,
    device: str = "cuda",
    feasibility_matrix: torch.Tensor = None,
    target_names: list = None,
    use_amp: bool = True,
):
    """
    Evaluate a model on a dataset.

    Args:
        model: Trained model
        dataloader: Validation/test DataLoader
        device: Device
        feasibility_matrix: Optional for constrained ViT
        target_names: Pathology names for reporting
        use_amp: Use mixed precision

    Returns:
        results: dict of metrics (auroc, auprc, f1)
    """
    model = model.to(device)
    model.eval()

    if feasibility_matrix is not None:
        feasibility_matrix = feasibility_matrix.to(device)

    tracker = MetricTracker(target_names)

    for images, targets in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device)
        targets = targets.to(device)

        with autocast(enabled=use_amp):
            if feasibility_matrix is not None:
                logits = model(images, feasibility_matrix=feasibility_matrix)
            else:
                logits = model(images)

        tracker.update(targets, logits)

    results = tracker.compute()

    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(tracker.summary_string(results))

    return results


def visualize_attention(
    model: nn.Module,
    image: torch.Tensor,
    original_image: np.ndarray = None,
    feasibility_matrix: torch.Tensor = None,
    layer_idx: int = -1,
    head_idx: int = 0,
    device: str = "cuda",
    save_path: str = None,
):
    """
    Visualize attention maps from a ViT model.

    Args:
        model: ViT model (baseline or constrained)
        image: (1, 3, 224, 224) preprocessed image
        original_image: (H, W, 3) original image for overlay (uint8)
        feasibility_matrix: Optional for constrained model
        layer_idx: Which transformer layer (default: last)
        head_idx: Which attention head
        device: Device
        save_path: Path to save the visualization
    """
    model = model.to(device)
    model.eval()

    image = image.to(device)
    if feasibility_matrix is not None:
        feasibility_matrix = feasibility_matrix.to(device)

    # Forward with attention
    with torch.no_grad():
        if feasibility_matrix is not None:
            _, attn_weights = model(
                image, feasibility_matrix=feasibility_matrix, return_attention=True
            )
        else:
            _, attn_weights = model(image, return_attention=True)

    # Get attention from specified layer
    attn = attn_weights[layer_idx]  # (1, heads, N, N)
    attn = attn[0, head_idx]  # (N, N) for specified head

    # Extract CLS attention to all patches
    cls_attn = attn[0, 1:]  # (num_patches,) — CLS attending to patches

    # Reshape to grid
    grid_size = int(np.sqrt(cls_attn.shape[0]))
    attn_map = cls_attn.cpu().numpy().reshape(grid_size, grid_size)

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Attention heatmap
    im0 = axes[0].imshow(attn_map, cmap="hot", interpolation="bilinear")
    axes[0].set_title(f"CLS Attention (Layer {layer_idx}, Head {head_idx})")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    # Plot 2: Full attention matrix
    full_attn = attn.cpu().numpy()
    im1 = axes[1].imshow(full_attn, cmap="viridis", aspect="auto")
    axes[1].set_title("Full Attention Matrix")
    axes[1].set_xlabel("Key")
    axes[1].set_ylabel("Query")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    # Plot 3: Overlay on original image (or upscaled attention)
    if original_image is not None:
        axes[2].imshow(original_image)
        # Upscale attention map to image size
        from PIL import Image
        attn_resized = np.array(
            Image.fromarray((attn_map * 255).astype(np.uint8)).resize(
                (original_image.shape[1], original_image.shape[0]),
                Image.BILINEAR,
            )
        ) / 255.0
        axes[2].imshow(attn_resized, alpha=0.5, cmap="hot")
        axes[2].set_title("Attention Overlay")
    else:
        # Show upscaled heatmap
        im2 = axes[2].imshow(attn_map, cmap="hot", interpolation="bilinear")
        axes[2].set_title("Attention Heatmap (upscaled)")
        plt.colorbar(im2, ax=axes[2], shrink=0.8)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved attention visualization to {save_path}")

    plt.close(fig)
    return fig


def compare_models(
    results_baseline: dict,
    results_constrained: dict,
    target_names: list = None,
    save_path: str = None,
):
    """
    Create a comparison table and chart between baseline and constrained ViT.

    Args:
        results_baseline: Metrics dict from baseline model
        results_constrained: Metrics dict from constrained model
        target_names: Pathology names
        save_path: Path to save comparison chart
    """
    if target_names is None:
        target_names = [
            "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"
        ]

    # Print comparison table
    print("\n" + "=" * 70)
    print(f"{'Pathology':<20} {'Baseline AUROC':>15} {'Constrained AUROC':>18} {'Δ':>8}")
    print("-" * 70)

    baseline_auroc = results_baseline["auroc"]
    constrained_auroc = results_constrained["auroc"]

    for name in target_names:
        b = baseline_auroc.get(name, 0.0)
        c = constrained_auroc.get(name, 0.0)
        delta = c - b
        sign = "+" if delta > 0 else ""
        print(f"{name:<20} {b:>15.4f} {c:>18.4f} {sign}{delta:>7.4f}")

    b_mean = baseline_auroc.get("mean", 0.0)
    c_mean = constrained_auroc.get("mean", 0.0)
    delta_mean = c_mean - b_mean
    sign = "+" if delta_mean > 0 else ""
    print("-" * 70)
    print(f"{'Mean':<20} {b_mean:>15.4f} {c_mean:>18.4f} {sign}{delta_mean:>7.4f}")
    print("=" * 70)

    # Create comparison chart
    if save_path:
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(target_names))
        width = 0.35

        baseline_scores = [baseline_auroc.get(n, 0) for n in target_names]
        constrained_scores = [constrained_auroc.get(n, 0) for n in target_names]

        bars1 = ax.bar(x - width / 2, baseline_scores, width, label="Baseline ViT",
                       color="#2196F3", alpha=0.8)
        bars2 = ax.bar(x + width / 2, constrained_scores, width, label="Constrained ViT",
                       color="#4CAF50", alpha=0.8)

        ax.set_xlabel("Pathology", fontsize=12)
        ax.set_ylabel("AUROC", fontsize=12)
        ax.set_title("Baseline vs Structure-Constrained ViT — AUROC Comparison", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(target_names, rotation=15, ha="right")
        ax.legend(fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved comparison chart to {save_path}")
