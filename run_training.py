#!/usr/bin/env python3
"""
Run the full Structure-Constrained ViT training pipeline.

Trains BOTH baseline and constrained ViTs, then compares them.

This script uses a configurable subset of CheXpert for faster iteration:
  --subset 5000  → train on 5000 images (~5 min/epoch on MPS)
  --subset 0     → train on full dataset (223K images)

Full pipeline:
  1. Load CheXpert dataset (with optional subset)
  2. Load precomputed feasibility matrix
  3. Train baseline ViT (standard attention)
  4. Train constrained ViT (Attention = softmax(QK^T/√d) * A)
  5. Compare & evaluate both models

Usage:
    python run_training.py --epochs 5 --subset 5000 --batch-size 16
    python run_training.py --epochs 10 --subset 10000
    python run_training.py --only constrained --feasibility outputs/feasibility/spectral_feasibility.pt
"""

import os
import argparse
import time
import yaml
import torch
import numpy as np

from torch.utils.data import DataLoader, Subset

from src.data.chexpert_dataset import CheXpertDataset, get_dataset_path, DEFAULT_TARGETS
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.vit import VisionTransformer
from src.models.constrained_vit import ConstrainedVisionTransformer
from src.training.trainer import Trainer
from src.training.metrics import MetricTracker


def parse_args():
    parser = argparse.ArgumentParser(description="Run Full Training Pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--subset", type=int, default=5000,
                       help="Subset size (0 = full dataset)")
    parser.add_argument("--feasibility", type=str,
                       default="outputs/feasibility/spectral_feasibility.pt")
    parser.add_argument("--only", type=str, default="both",
                       choices=["both", "baseline", "constrained"])
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def create_dataloaders(config, subset_size=0):
    """Create train/val dataloaders with optional data subset."""
    data_cfg = config["data"]
    train_cfg = config["training"]

    dataset_dir = data_cfg["dataset_dir"]
    if not os.path.exists(dataset_dir):
        dataset_dir = get_dataset_path()

    train_transform = get_train_transforms(data_cfg["image_size"])
    val_transform = get_val_transforms(data_cfg["image_size"])

    target_columns = data_cfg.get("target_columns", DEFAULT_TARGETS)
    uncertainty = data_cfg.get("uncertainty_policy", "ones")

    train_dataset = CheXpertDataset(
        csv_path=os.path.join(dataset_dir, data_cfg["train_csv"]),
        dataset_dir=dataset_dir,
        target_columns=target_columns,
        uncertainty_policy=uncertainty,
        transform=train_transform,
    )

    val_dataset = CheXpertDataset(
        csv_path=os.path.join(dataset_dir, data_cfg["valid_csv"]),
        dataset_dir=dataset_dir,
        target_columns=target_columns,
        uncertainty_policy=uncertainty,
        transform=val_transform,
    )

    # Subset for faster training
    if subset_size > 0 and subset_size < len(train_dataset):
        np.random.seed(42)
        indices = np.random.choice(len(train_dataset), subset_size, replace=False)
        train_dataset = Subset(train_dataset, indices)
        print(f"  Using subset: {subset_size:,} / {len(indices):,} images")

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=False,  # Disabled for MPS compatibility
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


def train_model(model, train_loader, val_loader, config, feasibility=None, model_name="model"):
    """Train a single model and return results."""
    config_copy = {**config}
    config_copy["logging"] = {
        **config["logging"],
        "log_dir": f"outputs/logs/{model_name}",
        "checkpoint_dir": f"outputs/checkpoints/{model_name}",
    }

    target_names = config["data"].get("target_columns", DEFAULT_TARGETS)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config_copy,
        feasibility_matrix=feasibility,
        target_names=target_names,
    )

    best_auroc = trainer.train()
    return best_auroc, trainer


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides
    config["training"]["epochs"] = args.epochs
    config["training"]["batch_size"] = args.batch_size
    config["training"]["learning_rate"] = args.lr

    model_cfg = config["model"]
    data_cfg = config["data"]

    print("=" * 70)
    print("  Structure-Constrained ViT — Full Training Pipeline")
    print("=" * 70)

    # ── Step 1: Load dataset ──
    print("\n[Step 1] Loading CheXpert dataset...")
    train_loader, val_loader = create_dataloaders(config, subset_size=args.subset)
    print(f"  Train: {len(train_loader.dataset):,} images")
    print(f"  Val:   {len(val_loader.dataset):,} images")
    print(f"  Batches/epoch: {len(train_loader)}")

    # ── Step 2: Load feasibility matrix ──
    feasibility_matrix = None
    if args.only in ("both", "constrained"):
        print(f"\n[Step 2] Loading feasibility matrix: {args.feasibility}")
        feas_data = torch.load(args.feasibility, map_location="cpu")
        feasibility_matrix = feas_data["feasibility_matrix"]
        sparsity = (feasibility_matrix == 0).sum().item() / feasibility_matrix.numel()
        print(f"  Shape: {list(feasibility_matrix.shape)}, Sparsity: {sparsity:.1%}")

    results = {}
    use_pretrained = not args.no_pretrained

    # ── Step 3: Train baseline ViT ──
    if args.only in ("both", "baseline"):
        print(f"\n{'=' * 70}")
        print("  [BASELINE] Standard ViT Training")
        print(f"  Attention: softmax(QK^T / √d)")
        print(f"{'=' * 70}")

        if use_pretrained:
            baseline = VisionTransformer.from_pretrained(
                model_name=model_cfg.get("pretrained_model", "vit_base_patch16_224"),
                num_classes=data_cfg["num_classes"],
            )
        else:
            baseline = VisionTransformer(
                num_classes=data_cfg["num_classes"],
                embed_dim=model_cfg["embed_dim"],
                depth=model_cfg["depth"],
                num_heads=model_cfg["num_heads"],
            )

        t0 = time.time()
        baseline_auroc, baseline_trainer = train_model(
            baseline, train_loader, val_loader, config,
            feasibility=None, model_name="baseline"
        )
        baseline_time = time.time() - t0
        results["baseline"] = {"auroc": baseline_auroc, "time": baseline_time}

    # ── Step 4: Train constrained ViT ──
    if args.only in ("both", "constrained"):
        print(f"\n{'=' * 70}")
        print("  [CONSTRAINED] Structure-Constrained ViT Training")
        print(f"  Attention: softmax(QK^T / √d) * A")
        print(f"{'=' * 70}")

        if use_pretrained:
            constrained = ConstrainedVisionTransformer.from_pretrained(
                model_name=model_cfg.get("pretrained_model", "vit_base_patch16_224"),
                num_classes=data_cfg["num_classes"],
            )
        else:
            constrained = ConstrainedVisionTransformer(
                num_classes=data_cfg["num_classes"],
                embed_dim=model_cfg["embed_dim"],
                depth=model_cfg["depth"],
                num_heads=model_cfg["num_heads"],
            )

        t0 = time.time()
        constrained_auroc, constrained_trainer = train_model(
            constrained, train_loader, val_loader, config,
            feasibility=feasibility_matrix, model_name="constrained"
        )
        constrained_time = time.time() - t0
        results["constrained"] = {"auroc": constrained_auroc, "time": constrained_time}

    # ── Step 5: Comparison ──
    print(f"\n{'=' * 70}")
    print("  RESULTS COMPARISON")
    print(f"{'=' * 70}")
    print(f"\n  {'Model':<20} {'Best AUROC':>12} {'Train Time':>12}")
    print(f"  {'─' * 44}")

    for name, r in results.items():
        auroc = r["auroc"]
        mins = r["time"] / 60
        print(f"  {name:<20} {auroc:>12.4f} {f'{mins:.1f} min':>12}")

    if "baseline" in results and "constrained" in results:
        diff = results["constrained"]["auroc"] - results["baseline"]["auroc"]
        sign = "+" if diff >= 0 else ""
        print(f"\n  Δ AUROC (constrained - baseline): {sign}{diff:.4f}")

    print(f"\n  Checkpoints: outputs/checkpoints/{{baseline,constrained}}/best_model.pt")
    print(f"  TensorBoard: tensorboard --logdir outputs/logs/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
