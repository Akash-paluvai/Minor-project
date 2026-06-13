#!/usr/bin/env python3
"""
Comprehensive Experiment Runner — 5 Models × 5 Metrics.

Models:
  1. Baseline ViT
  2. ViT + KMeans clusters
  3. ViT + Spectral clusters
  4. ViT + Graph (GNN) clustering
  5. ViT + Structure constrained (Diffusion Maps)

Metrics:
  - Accuracy (multi-label subset accuracy + per-class)
  - AUROC (per-class and mean)
  - F1 Score (per-class and mean)
  - Attention Entropy (measures attention distribution spread)
  - Robustness to Noise (AUROC degradation under Gaussian noise)

Outputs:
  - Result tables (CSV + printed)
  - Comparison plots (bar charts, radar chart, noise robustness curves)
  - All saved to outputs/experiments/

Usage:
  python run_experiments.py --epochs 5 --subset 5000 --batch-size 16
  python run_experiments.py --epochs 10 --subset 10000
"""

import os
import sys
import argparse
import json
import time
import yaml

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# ── Project imports ──
from src.data.chexpert_dataset import CheXpertDataset, get_dataset_path, DEFAULT_TARGETS
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.vit import VisionTransformer
from src.models.constrained_vit import ConstrainedVisionTransformer
from src.training.trainer import Trainer
from src.training.metrics import compute_auroc, compute_f1, MetricTracker


# ════════════════════════════════════════════════════════════
#  Configuration
# ════════════════════════════════════════════════════════════

EXPERIMENT_DIR = "outputs/experiments"

MODELS = {
    "Baseline ViT": {
        "feasibility": None,
        "description": "Standard ViT: Attention = softmax(QK^T/√d)",
    },
    "ViT + KMeans": {
        "feasibility": "outputs/feasibility/kmeans_feasibility.pt",
        "description": "Constrained ViT with KMeans feasibility matrix",
    },
    "ViT + Spectral": {
        "feasibility": "outputs/feasibility/spectral_feasibility.pt",
        "description": "Constrained ViT with Spectral clustering feasibility matrix",
    },
    "ViT + Graph (GNN)": {
        "feasibility": "outputs/feasibility/gnn_feasibility.pt",
        "description": "Constrained ViT with GNN clustering feasibility matrix",
    },
    "ViT + Struct-Constrained": {
        "feasibility": "outputs/feasibility/diffusion_feasibility.pt",
        "description": "Constrained ViT with Diffusion Maps feasibility matrix",
    },
}

NOISE_LEVELS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

TARGET_NAMES = DEFAULT_TARGETS


# ════════════════════════════════════════════════════════════
#  Data Loading
# ════════════════════════════════════════════════════════════

def create_dataloaders(config, subset_size=0):
    """Create train/val dataloaders."""
    data_cfg = config["data"]
    train_cfg = config["training"]

    dataset_dir = data_cfg["dataset_dir"]
    if not os.path.exists(dataset_dir):
        dataset_dir = get_dataset_path()

    train_transform = get_train_transforms(data_cfg["image_size"])
    val_transform = get_val_transforms(data_cfg["image_size"])

    train_dataset = CheXpertDataset(
        csv_path=os.path.join(dataset_dir, data_cfg["train_csv"]),
        dataset_dir=dataset_dir,
        target_columns=TARGET_NAMES,
        uncertainty_policy="ones",
        transform=train_transform,
    )

    val_dataset = CheXpertDataset(
        csv_path=os.path.join(dataset_dir, data_cfg["valid_csv"]),
        dataset_dir=dataset_dir,
        target_columns=TARGET_NAMES,
        uncertainty_policy="ones",
        transform=val_transform,
    )

    if subset_size > 0 and subset_size < len(train_dataset):
        np.random.seed(42)
        indices = np.random.choice(len(train_dataset), subset_size, replace=False)
        train_dataset = Subset(train_dataset, indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=False,
    )

    return train_loader, val_loader


# ════════════════════════════════════════════════════════════
#  Metric Computation Functions
# ════════════════════════════════════════════════════════════

def compute_accuracy(targets, predictions, threshold=0.5):
    """Compute multi-label accuracy (exact match + per-class)."""
    binary_preds = (predictions >= threshold).astype(np.float32)

    # Per-class accuracy
    per_class = {}
    n_classes = targets.shape[1]
    for i in range(n_classes):
        correct = (binary_preds[:, i] == targets[:, i]).sum()
        per_class[TARGET_NAMES[i]] = float(correct) / len(targets)

    # Subset accuracy (exact match)
    exact_match = np.all(binary_preds == targets, axis=1).mean()

    per_class["mean"] = np.mean(list(per_class.values()))
    per_class["subset_accuracy"] = float(exact_match)

    return per_class


def compute_attention_entropy(model, val_loader, device, feasibility_matrix=None):
    """
    Compute mean attention entropy across all layers and heads.

    Entropy of attention distribution measures how spread out attention is:
      - High entropy → uniform (spread out) attention
      - Low entropy → concentrated (focused) attention
    """
    model.eval()
    all_entropies = []

    with torch.no_grad():
        for images, _ in tqdm(val_loader, desc="Computing attention entropy", leave=False):
            images = images.to(device)

            if feasibility_matrix is not None:
                logits, attn_weights_list = model(
                    images, feasibility_matrix=feasibility_matrix, return_attention=True
                )
            else:
                logits, attn_weights_list = model(images, return_attention=True)

            # attn_weights_list is list of (B, heads, N, N)
            for attn_w in attn_weights_list:
                # Clamp to avoid log(0)
                attn_w = attn_w.clamp(min=1e-8)
                # Entropy per head per token: -sum(p * log(p))
                entropy = -(attn_w * attn_w.log()).sum(dim=-1)  # (B, heads, N)
                # Average over tokens, heads, batch
                all_entropies.append(entropy.mean().item())

    mean_entropy = np.mean(all_entropies)
    std_entropy = np.std(all_entropies)
    return {
        "mean": mean_entropy,
        "std": std_entropy,
        "per_layer": all_entropies,
    }


def compute_noise_robustness(model, val_loader, device, feasibility_matrix=None,
                              noise_levels=None):
    """
    Compute AUROC at different noise levels to measure robustness.

    Adds Gaussian noise with increasing std to input images.
    """
    if noise_levels is None:
        noise_levels = NOISE_LEVELS

    model.eval()
    results = {}

    for noise_std in noise_levels:
        metric_tracker = MetricTracker(TARGET_NAMES)

        with torch.no_grad():
            for images, targets in tqdm(
                val_loader, desc=f"Noise σ={noise_std:.2f}", leave=False
            ):
                images = images.to(device)
                targets = targets.to(device)

                # Add Gaussian noise
                if noise_std > 0:
                    noise = torch.randn_like(images) * noise_std
                    noisy_images = images + noise
                else:
                    noisy_images = images

                # Forward pass
                if feasibility_matrix is not None:
                    logits = model(noisy_images, feasibility_matrix=feasibility_matrix)
                else:
                    logits = model(noisy_images)

                metric_tracker.update(targets, logits)

        metrics = metric_tracker.compute()
        results[noise_std] = metrics["auroc"]["mean"]

    return results


# ════════════════════════════════════════════════════════════
#  Training + Evaluation per Model
# ════════════════════════════════════════════════════════════

def train_and_evaluate(
    model_name, model, train_loader, val_loader, config,
    feasibility_matrix=None, device="cpu"
):
    """Train a model and compute all 5 metrics."""
    safe_name = model_name.replace(" ", "_").replace("+", "plus").replace("(", "").replace(")", "")
    model_log = f"outputs/experiments/logs/{safe_name}"
    model_ckpt = f"outputs/experiments/checkpoints/{safe_name}"

    config_copy = {**config}
    config_copy["logging"] = {
        **config["logging"],
        "log_dir": model_log,
        "checkpoint_dir": model_ckpt,
    }

    feas_tensor = None
    if feasibility_matrix is not None:
        feas_tensor = feasibility_matrix

    # ── Train ──
    t0 = time.time()
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config_copy,
        feasibility_matrix=feas_tensor,
        target_names=TARGET_NAMES,
    )
    best_auroc = trainer.train()
    train_time = time.time() - t0

    # ── Evaluate on validation set ──
    model.eval()
    device_obj = next(model.parameters()).device

    metric_tracker = MetricTracker(TARGET_NAMES)
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc=f"Evaluating {model_name}", leave=False):
            images = images.to(device_obj)
            targets_dev = targets.to(device_obj)

            if feas_tensor is not None:
                logits = model(images, feasibility_matrix=feas_tensor)
            else:
                logits = model(images)

            metric_tracker.update(targets_dev, logits)
            all_logits.append(torch.sigmoid(logits).cpu().numpy())
            all_targets.append(targets.numpy())

    eval_metrics = metric_tracker.compute()
    predictions = np.concatenate(all_logits, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)

    # ── Metric 1: Accuracy ──
    accuracy = compute_accuracy(targets_np, predictions)

    # ── Metric 2: AUROC (already in eval_metrics) ──
    auroc = eval_metrics["auroc"]

    # ── Metric 3: F1 Score (already in eval_metrics) ──
    f1 = eval_metrics["f1"]

    # ── Metric 4: Attention Entropy ──
    entropy = compute_attention_entropy(model, val_loader, device_obj, feas_tensor)

    # ── Metric 5: Noise Robustness ──
    noise_results = compute_noise_robustness(model, val_loader, device_obj, feas_tensor)

    return {
        "model_name": model_name,
        "train_time_min": train_time / 60,
        "best_train_auroc": best_auroc,
        "accuracy": accuracy,
        "auroc": auroc,
        "f1": f1,
        "attention_entropy": entropy,
        "noise_robustness": noise_results,
    }


# ════════════════════════════════════════════════════════════
#  Plotting
# ════════════════════════════════════════════════════════════

def generate_plots(all_results, output_dir):
    """Generate all comparison plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.facecolor": "white",
        "axes.facecolor": "#f8f9fa",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })

    model_names = [r["model_name"] for r in all_results]
    short_names = [n.replace("ViT + ", "").replace("ViT", "Baseline") for n in model_names]
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0"]

    # ── Plot 1: Main Metrics Bar Chart ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # AUROC
    auroc_vals = [r["auroc"]["mean"] for r in all_results]
    bars = axes[0].bar(short_names, auroc_vals, color=colors, edgecolor="white", linewidth=0.8)
    axes[0].set_title("Mean AUROC", fontweight="bold")
    axes[0].set_ylim(min(auroc_vals) - 0.05, max(auroc_vals) + 0.03)
    axes[0].set_ylabel("AUROC")
    for bar, val in zip(bars, auroc_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[0].tick_params(axis='x', rotation=20)

    # F1 Score
    f1_vals = [r["f1"]["mean"] for r in all_results]
    bars = axes[1].bar(short_names, f1_vals, color=colors, edgecolor="white", linewidth=0.8)
    axes[1].set_title("Mean F1 Score", fontweight="bold")
    axes[1].set_ylim(0, max(f1_vals) + 0.1)
    axes[1].set_ylabel("F1")
    for bar, val in zip(bars, f1_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[1].tick_params(axis='x', rotation=20)

    # Accuracy
    acc_vals = [r["accuracy"]["mean"] for r in all_results]
    bars = axes[2].bar(short_names, acc_vals, color=colors, edgecolor="white", linewidth=0.8)
    axes[2].set_title("Mean Accuracy", fontweight="bold")
    axes[2].set_ylim(min(acc_vals) - 0.05, max(acc_vals) + 0.03)
    axes[2].set_ylabel("Accuracy")
    for bar, val in zip(bars, acc_vals):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[2].tick_params(axis='x', rotation=20)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "main_metrics_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 2: Per-Class AUROC Heatmap ──
    fig, ax = plt.subplots(figsize=(10, 5))
    classes = TARGET_NAMES
    auroc_matrix = np.array([
        [r["auroc"].get(c, 0.0) for c in classes] for r in all_results
    ])

    im = ax.imshow(auroc_matrix, cmap="YlOrRd", aspect="auto", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(short_names)
    ax.set_title("Per-Class AUROC", fontweight="bold")

    for i in range(len(model_names)):
        for j in range(len(classes)):
            ax.text(j, i, f"{auroc_matrix[i, j]:.3f}", ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if auroc_matrix[i, j] > 0.75 else "black")

    plt.colorbar(im, ax=ax, label="AUROC", shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_class_auroc_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 3: Attention Entropy Comparison ──
    fig, ax = plt.subplots(figsize=(9, 5))
    entropy_vals = [r["attention_entropy"]["mean"] for r in all_results]
    entropy_stds = [r["attention_entropy"]["std"] for r in all_results]

    bars = ax.bar(short_names, entropy_vals, color=colors, edgecolor="white",
                  linewidth=0.8, yerr=entropy_stds, capsize=5)
    ax.set_title("Attention Entropy (higher = more spread out)", fontweight="bold")
    ax.set_ylabel("Mean Entropy (nats)")
    for bar, val in zip(bars, entropy_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.tick_params(axis='x', rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "attention_entropy.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 4: Noise Robustness Curves ──
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, r in enumerate(all_results):
        noise = sorted(r["noise_robustness"].keys())
        aurocs_noise = [r["noise_robustness"][n] for n in noise]
        ax.plot(noise, aurocs_noise, marker="o", linewidth=2, label=short_names[i],
                color=colors[i], markersize=6)

    ax.set_xlabel("Noise Level (σ)")
    ax.set_ylabel("AUROC")
    ax.set_title("Robustness to Gaussian Noise", fontweight="bold")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_xlim(-0.01, max(NOISE_LEVELS) + 0.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "noise_robustness.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 5: Radar / Spider Chart ──
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    metric_labels = ["AUROC", "F1", "Accuracy", "Entropy\n(norm)", "Noise\nRobust"]
    num_metrics = len(metric_labels)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]

    # Normalize for radar: 0-1 scale
    max_entropy = max(r["attention_entropy"]["mean"] for r in all_results)
    max_entropy = max(max_entropy, 1e-8)

    for i, r in enumerate(all_results):
        # Noise robustness = AUROC retention at worst noise level
        clean_auroc = r["noise_robustness"][0.0]
        worst_auroc = r["noise_robustness"][max(NOISE_LEVELS)]
        retention = worst_auroc / max(clean_auroc, 1e-8)

        values = [
            r["auroc"]["mean"],
            r["f1"]["mean"],
            r["accuracy"]["mean"],
            r["attention_entropy"]["mean"] / max_entropy,  # normalized
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
    plt.savefig(os.path.join(output_dir, "radar_chart.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  ✓ All plots saved to {output_dir}/")


def generate_tables(all_results, output_dir):
    """Generate result tables as CSV and print to console."""

    # ── Table 1: Main Comparison Table ──
    print(f"\n{'═' * 90}")
    print("  EXPERIMENT RESULTS — 5 Models × 5 Metrics")
    print(f"{'═' * 90}\n")

    header = f"  {'Model':<25} {'AUROC':>8} {'F1':>8} {'Accuracy':>10} {'Entropy':>9} {'Noise↓0.3':>10} {'Time':>8}"
    print(header)
    print(f"  {'─' * 82}")

    csv_lines = ["Model,AUROC,F1,Accuracy,Attention_Entropy,Noise_AUROC_0.3,Train_Time_Min"]

    for r in all_results:
        auroc = r["auroc"]["mean"]
        f1 = r["f1"]["mean"]
        acc = r["accuracy"]["mean"]
        entropy = r["attention_entropy"]["mean"]
        noise_worst = r["noise_robustness"].get(0.3, r["noise_robustness"].get(max(r["noise_robustness"].keys()), 0))
        train_min = r["train_time_min"]

        print(f"  {r['model_name']:<25} {auroc:>8.4f} {f1:>8.4f} {acc:>10.4f} {entropy:>9.3f} {noise_worst:>10.4f} {train_min:>7.1f}m")
        csv_lines.append(f"{r['model_name']},{auroc:.4f},{f1:.4f},{acc:.4f},{entropy:.3f},{noise_worst:.4f},{train_min:.1f}")

    # ── Table 2: Per-Class AUROC ──
    print(f"\n  Per-Class AUROC:")
    print(f"  {'Model':<25}", end="")
    for c in TARGET_NAMES:
        print(f" {c[:12]:>12}", end="")
    print()
    print(f"  {'─' * (25 + 12 * len(TARGET_NAMES))}")

    csv_lines.append("")
    csv_lines.append("Model," + ",".join(TARGET_NAMES))

    for r in all_results:
        print(f"  {r['model_name']:<25}", end="")
        csv_row = [r["model_name"]]
        for c in TARGET_NAMES:
            val = r["auroc"].get(c, 0.0)
            print(f" {val:>12.4f}", end="")
            csv_row.append(f"{val:.4f}")
        print()
        csv_lines.append(",".join(csv_row))

    # ── Table 3: Noise Robustness ──
    print(f"\n  Noise Robustness (AUROC at each σ):")
    print(f"  {'Model':<25}", end="")
    for nl in NOISE_LEVELS:
        print(f" {'σ=' + str(nl):>8}", end="")
    print()
    print(f"  {'─' * (25 + 8 * len(NOISE_LEVELS))}")

    csv_lines.append("")
    csv_lines.append("Model," + ",".join([f"noise_{nl}" for nl in NOISE_LEVELS]))

    for r in all_results:
        print(f"  {r['model_name']:<25}", end="")
        csv_row = [r["model_name"]]
        for nl in NOISE_LEVELS:
            val = r["noise_robustness"].get(nl, 0.0)
            print(f" {val:>8.4f}", end="")
            csv_row.append(f"{val:.4f}")
        print()
        csv_lines.append(",".join(csv_row))

    print(f"\n{'═' * 90}")

    # Save CSV
    csv_path = os.path.join(output_dir, "experiment_results.csv")
    with open(csv_path, "w") as f:
        f.write("\n".join(csv_lines))
    print(f"  ✓ Results saved to {csv_path}")


# ════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Run Comparative Experiments")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--subset", type=int, default=5000)
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip training, load existing checkpoints")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config["training"]["epochs"] = args.epochs
    config["training"]["batch_size"] = args.batch_size
    config["training"]["learning_rate"] = args.lr

    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    print("=" * 70)
    print("  Comparative Experiment: 5 Models × 5 Metrics")
    print("=" * 70)
    print(f"  Epochs:       {args.epochs}")
    print(f"  Subset:       {args.subset}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  LR:           {args.lr}")
    print()

    # ── Load data ──
    print("[1/4] Loading CheXpert dataset...")
    train_loader, val_loader = create_dataloaders(config, subset_size=args.subset)
    print(f"  Train: {len(train_loader.dataset):,} images")
    print(f"  Val:   {len(val_loader.dataset):,} images")

    # ── Load feasibility matrices ──
    print("\n[2/4] Loading feasibility matrices...")
    feasibility_matrices = {}
    for name, cfg in MODELS.items():
        if cfg["feasibility"] is not None:
            fpath = cfg["feasibility"]
            if os.path.exists(fpath):
                data = torch.load(fpath, map_location="cpu")
                fm = data["feasibility_matrix"]
                sparsity = (fm == 0).sum().item() / fm.numel()
                feasibility_matrices[name] = fm
                print(f"  ✓ {name}: {list(fm.shape)}, sparsity={sparsity:.1%}")
            else:
                print(f"  ✗ {name}: {fpath} not found — skipping")
                feasibility_matrices[name] = None
        else:
            feasibility_matrices[name] = None
            print(f"  ○ {name}: No feasibility matrix (baseline)")

    # ── Train and evaluate each model ──
    print(f"\n[3/4] Training and evaluating all models...")
    all_results = []

    for model_idx, (model_name, model_cfg) in enumerate(MODELS.items(), 1):
        print(f"\n{'═' * 70}")
        print(f"  [{model_idx}/5] {model_name}")
        print(f"  {model_cfg['description']}")
        print(f"{'═' * 70}")

        feas = feasibility_matrices.get(model_name)

        if feas is not None:
            model = ConstrainedVisionTransformer.from_pretrained(
                model_name="vit_base_patch16_224", num_classes=5
            )
        else:
            model = VisionTransformer.from_pretrained(
                model_name="vit_base_patch16_224", num_classes=5
            )

        results = train_and_evaluate(
            model_name=model_name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            feasibility_matrix=feas,
        )

        all_results.append(results)

        # Save intermediate results
        safe_name = model_name.replace(" ", "_").replace("+", "plus")
        torch.save(results, os.path.join(EXPERIMENT_DIR, f"{safe_name}_results.pt"))
        print(f"  ✓ {model_name} complete: AUROC={results['auroc']['mean']:.4f}")

    # ── Generate tables and plots ──
    print(f"\n[4/4] Generating result tables and plots...")
    generate_tables(all_results, EXPERIMENT_DIR)
    generate_plots(all_results, EXPERIMENT_DIR)

    # Save all results
    json_results = []
    for r in all_results:
        jr = {
            "model_name": r["model_name"],
            "train_time_min": r["train_time_min"],
            "auroc_mean": r["auroc"]["mean"],
            "f1_mean": r["f1"]["mean"],
            "accuracy_mean": r["accuracy"]["mean"],
            "attention_entropy": r["attention_entropy"]["mean"],
            "noise_robustness": {str(k): v for k, v in r["noise_robustness"].items()},
        }
        json_results.append(jr)

    with open(os.path.join(EXPERIMENT_DIR, "all_results.json"), "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\n{'═' * 70}")
    print(f"  Experiment complete! All results in {EXPERIMENT_DIR}/")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
