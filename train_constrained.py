#!/usr/bin/env python3
"""
Train Structure-Constrained Vision Transformer on CheXpert.

Full pipeline:
  1. Load CheXpert dataset
  2. Load precomputed feasibility matrix A
  3. Train ViT with: Attention = softmax(QK^T/√d) * A
  4. Evaluate with AUROC/AUPRC/F1

Usage:
    python train_constrained.py --config configs/default.yaml
    python train_constrained.py --feasibility outputs/feasibility/spectral_feasibility.pt
    python train_constrained.py --dry-run
"""

import argparse
import yaml
import torch

from src.data.chexpert_dataset import get_chexpert_dataloaders
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.constrained_vit import ConstrainedVisionTransformer
from src.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Constrained ViT on CheXpert")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--feasibility", type=str, default="outputs/feasibility/spectral_feasibility.pt",
        help="Path to precomputed feasibility matrix (.pt file)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Run one batch to verify")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Apply overrides
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size

    data_cfg = config["data"]
    model_cfg = config["model"]

    print("=" * 60)
    print("Structure-Constrained ViT — CheXpert Training")
    print("=" * 60)

    # ── Step 1: Load feasibility matrix ──
    print("\n[Step 1] Loading structural feasibility matrix...")
    feas_data = torch.load(args.feasibility, map_location="cpu")
    feasibility_matrix = feas_data["feasibility_matrix"]
    cluster_labels = feas_data.get("cluster_labels", None)
    metadata = feas_data.get("metadata", {})

    sparsity = (feasibility_matrix == 0).sum().item() / feasibility_matrix.numel()
    print(f"  Path:      {args.feasibility}")
    print(f"  Shape:     {list(feasibility_matrix.shape)}")
    print(f"  Sparsity:  {sparsity:.1%} blocked")
    print(f"  Mean P:    {feasibility_matrix.mean():.4f}")
    if metadata:
        print(f"  Algorithm: {metadata.get('algorithm', 'unknown')}")
        print(f"  Mode:      {metadata.get('mode', 'unknown')}")

    # ── Step 2: Load CheXpert dataset ──
    print("\n[Step 2] Loading CheXpert dataset...")
    train_transform = get_train_transforms(data_cfg["image_size"])
    val_transform = get_val_transforms(data_cfg["image_size"])

    train_loader, val_loader = get_chexpert_dataloaders(
        config, train_transform, val_transform
    )
    print(f"  Train samples: {len(train_loader.dataset):,}")
    print(f"  Val samples:   {len(val_loader.dataset):,}")
    print(f"  Batch size:    {config['training']['batch_size']}")

    # ── Step 3: Initialize constrained ViT ──
    print("\n[Step 3] Initializing constrained ViT...")
    print(f"  Attention: softmax(QK^T/√d) * A")

    if args.pretrained and model_cfg.get("use_pretrained", True):
        model = ConstrainedVisionTransformer.from_pretrained(
            model_name=model_cfg.get("pretrained_model", "vit_base_patch16_224"),
            num_classes=data_cfg["num_classes"],
            img_size=data_cfg["image_size"],
            patch_size=model_cfg["patch_size"],
            embed_dim=model_cfg["embed_dim"],
            depth=model_cfg["depth"],
            num_heads=model_cfg["num_heads"],
            mlp_ratio=model_cfg["mlp_ratio"],
            dropout=model_cfg["dropout"],
            attention_dropout=model_cfg["attention_dropout"],
        )
    else:
        model = ConstrainedVisionTransformer(
            img_size=data_cfg["image_size"],
            patch_size=model_cfg["patch_size"],
            num_classes=data_cfg["num_classes"],
            embed_dim=model_cfg["embed_dim"],
            depth=model_cfg["depth"],
            num_heads=model_cfg["num_heads"],
            mlp_ratio=model_cfg["mlp_ratio"],
            dropout=model_cfg["dropout"],
            attention_dropout=model_cfg["attention_dropout"],
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # ── Dry run ──
    if args.dry_run:
        print("\n--- DRY RUN ---")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        images, labels = next(iter(train_loader))
        images = images.to(device)
        F = feasibility_matrix.to(device)

        with torch.no_grad():
            logits = model(images, feasibility_matrix=F)

        print(f"  Input shape:  {images.shape}")
        print(f"  Matrix shape: {F.shape}")
        print(f"  Output shape: {logits.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Pred probs:   {torch.sigmoid(logits[0]).cpu().numpy().round(3)}")
        print("  ✓ Constrained pipeline works!")
        return

    # ── Step 4: Train ──
    print(f"\n[Step 4] Starting training...")
    config["logging"]["log_dir"] = "outputs/logs/constrained"
    config["logging"]["checkpoint_dir"] = "outputs/checkpoints/constrained"

    target_names = data_cfg.get("target_columns", [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"
    ])

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        feasibility_matrix=feasibility_matrix,
        target_names=target_names,
    )

    best_auroc = trainer.train()

    print(f"\n✓ Training complete!")
    print(f"  Best AUROC: {best_auroc:.4f}")
    print(f"  Model saved: outputs/checkpoints/constrained/best_model.pt")


if __name__ == "__main__":
    main()
