#!/usr/bin/env python3
"""
Evaluate and compare baseline vs constrained ViT models.

Usage:
    python evaluate.py --config configs/default.yaml
    python evaluate.py --baseline outputs/checkpoints/baseline/best_model.pt \
                       --constrained outputs/checkpoints/constrained/best_model.pt
"""

import argparse
import yaml
import torch

from src.data.chexpert_dataset import get_chexpert_dataloaders
from src.data.transforms import get_val_transforms
from src.models.vit import VisionTransformer
from src.models.constrained_vit import ConstrainedVisionTransformer
from src.training.evaluate import evaluate_model, compare_models, visualize_attention


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate and Compare Models")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--baseline", type=str,
        default="outputs/checkpoints/baseline/best_model.pt",
        help="Path to baseline model checkpoint"
    )
    parser.add_argument(
        "--constrained", type=str,
        default="outputs/checkpoints/constrained/best_model.pt",
        help="Path to constrained model checkpoint"
    )
    parser.add_argument(
        "--feasibility", type=str,
        default="outputs/feasibility_matrix.pt",
        help="Path to feasibility matrix"
    )
    parser.add_argument(
        "--visualize-attention", action="store_true",
        help="Generate attention visualizations"
    )
    parser.add_argument(
        "--num-attention-samples", type=int, default=5,
        help="Number of samples for attention visualization"
    )
    return parser.parse_args()


def load_model(checkpoint_path, model_class, config):
    """Load a model from a checkpoint."""
    data_cfg = config["data"]
    model_cfg = config["model"]

    model = model_class(
        img_size=data_cfg["image_size"],
        patch_size=model_cfg["patch_size"],
        num_classes=data_cfg["num_classes"],
        embed_dim=model_cfg["embed_dim"],
        depth=model_cfg["depth"],
        num_heads=model_cfg["num_heads"],
        mlp_ratio=model_cfg["mlp_ratio"],
        dropout=0.0,  # No dropout during evaluation
        attention_dropout=0.0,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded {model_class.__name__} from {checkpoint_path} (epoch {checkpoint['epoch']})")

    return model


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    val_transform = get_val_transforms(data_cfg["image_size"])
    _, val_loader = get_chexpert_dataloaders(config, val_transform, val_transform)

    target_names = data_cfg.get("target_columns", [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"
    ])

    # Load feasibility matrix
    feasibility_matrix = None
    try:
        feas_data = torch.load(args.feasibility, map_location="cpu")
        feasibility_matrix = feas_data["feasibility_matrix"]
        print(f"Loaded feasibility matrix from {args.feasibility}")
    except FileNotFoundError:
        print(f"Warning: Feasibility matrix not found at {args.feasibility}")

    results = {}

    # ---- Evaluate Baseline ----
    print("\n" + "=" * 60)
    print("Evaluating Baseline ViT")
    print("=" * 60)

    try:
        baseline_model = load_model(args.baseline, VisionTransformer, config)
        results["baseline"] = evaluate_model(
            baseline_model, val_loader, device=str(device),
            target_names=target_names,
        )
    except FileNotFoundError:
        print(f"Warning: Baseline checkpoint not found at {args.baseline}")
        results["baseline"] = None

    # ---- Evaluate Constrained ----
    print("\n" + "=" * 60)
    print("Evaluating Constrained ViT")
    print("=" * 60)

    try:
        constrained_model = load_model(
            args.constrained, ConstrainedVisionTransformer, config
        )
        results["constrained"] = evaluate_model(
            constrained_model, val_loader, device=str(device),
            feasibility_matrix=feasibility_matrix,
            target_names=target_names,
        )
    except FileNotFoundError:
        print(f"Warning: Constrained checkpoint not found at {args.constrained}")
        results["constrained"] = None

    # ---- Compare ----
    if results["baseline"] and results["constrained"]:
        compare_models(
            results["baseline"],
            results["constrained"],
            target_names=target_names,
            save_path="outputs/comparison_chart.png",
        )

    # ---- Attention Visualization ----
    if args.visualize_attention:
        import os
        viz_dir = config["evaluation"].get("visualization_dir", "outputs/attention_maps")
        os.makedirs(viz_dir, exist_ok=True)

        # Get sample images
        sample_iter = iter(val_loader)
        images, labels = next(sample_iter)

        for i in range(min(args.num_attention_samples, images.shape[0])):
            img = images[i:i+1]

            # Baseline attention
            if results["baseline"] is not None:
                visualize_attention(
                    baseline_model, img,
                    device=str(device),
                    save_path=os.path.join(viz_dir, f"baseline_sample_{i}.png"),
                )

            # Constrained attention
            if results["constrained"] is not None and feasibility_matrix is not None:
                visualize_attention(
                    constrained_model, img,
                    feasibility_matrix=feasibility_matrix,
                    device=str(device),
                    save_path=os.path.join(viz_dir, f"constrained_sample_{i}.png"),
                )

        print(f"\nAttention visualizations saved to {viz_dir}/")


if __name__ == "__main__":
    main()
