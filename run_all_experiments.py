#!/usr/bin/env python3
"""
run_all_experiments.py — 4-Model Comparative Experiment Runner.

Trains and evaluates:
  1. Baseline ViT           (no constraints)
  2. Feature-Constrained    (F_feature only, alpha=0.0)
  3. Anatomy-Constrained    (F_anatomy only, alpha=1.0)
  4. Hybrid-Constrained     (F_hybrid, alpha=0.7) ← primary contribution

Usage:
    python run_all_experiments.py
    python run_all_experiments.py --epochs 5 --skip-training
"""

import os, sys, json, time, argparse
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.chexpert_dataset import CheXpertDataset, get_dataset_path, DEFAULT_TARGETS
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.vit import VisionTransformer
from src.models.constrained_vit import ConstrainedVisionTransformer
from src.models.hybrid_relational_vit import HybridAnatomyGuidedRelationalViT
from src.training.trainer import Trainer
from src.training.metrics import MetricTracker
from src.training.attention_metrics import AttentionMetricsComputer
from src.visualization.visualizer import ExperimentVisualizer
from src.structure.anatomy_prior import AnatomyPriorBuilder

RESULTS_DIR = "outputs/results"
TARGET_NAMES = DEFAULT_TARGETS


def get_device(config):
    sys_cfg = config.get("system", {})
    if sys_cfg.get("use_mps", True) and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_cache(config):
    """Load precomputed feasibility matrices from cache."""
    cache_dir = config.get("structure", {}).get("cache_dir", "outputs/cache")
    matrices = {}
    for key in ["F_anatomy", "F_feature", "F_hybrid"]:
        path = os.path.join(cache_dir, f"{key}.pt")
        if os.path.exists(path):
            data = torch.load(path, map_location="cpu")
            matrices[key] = data["feasibility_matrix"]
            print(f"  ✓ {key}: {list(matrices[key].shape)}")
        else:
            print(f"  ✗ {key}: not found at {path}")
            matrices[key] = None
    return matrices


def create_loaders(config):
    data_cfg  = config["data"]
    train_cfg = config["training"]
    sys_cfg   = config.get("system", {})

    dataset_dir = data_cfg["dataset_dir"]
    if not os.path.exists(dataset_dir):
        dataset_dir = get_dataset_path()

    img_size = data_cfg.get("image_size", 192)
    train_ds = CheXpertDataset(
        csv_path=os.path.join(dataset_dir, data_cfg["train_csv"]),
        dataset_dir=dataset_dir,
        target_columns=TARGET_NAMES,
        uncertainty_policy="ones",
        transform=get_train_transforms(img_size),
        subset_size=data_cfg.get("subset_train", 12000),
    )
    val_ds = CheXpertDataset(
        csv_path=os.path.join(dataset_dir, data_cfg["valid_csv"]),
        dataset_dir=dataset_dir,
        target_columns=TARGET_NAMES,
        uncertainty_policy="ones",
        transform=get_val_transforms(img_size),
        subset_size=data_cfg.get("subset_val", 1500),
    )

    bs = train_cfg["batch_size"]
    nw = train_cfg.get("num_workers", 0)
    pm = sys_cfg.get("pin_memory", False)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=pm, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=pm)
    return train_loader, val_loader


def evaluate_model(model, val_loader, device, F, target_names, model_tag,
                   graph_bias=None, anatomy_prior=None):
    """Full evaluation: AUROC/F1/Acc + attention metrics."""
    model.eval()
    tracker = MetricTracker(target_names)
    all_logits, all_targets = [], []

    is_hybrid = isinstance(model, HybridAnatomyGuidedRelationalViT)

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets_dev = targets.to(device)
            if is_hybrid:
                logits = model(images, graph_bias=graph_bias, anatomy_prior=anatomy_prior)
            elif F is not None:
                logits = model(images, feasibility_matrix=F)
            else:
                logits = model(images)
            tracker.update(targets_dev, logits)
            all_logits.append(torch.sigmoid(logits).cpu().numpy())
            all_targets.append(targets.numpy())

    metrics = tracker.compute()
    preds   = np.concatenate(all_logits)
    tgts    = np.concatenate(all_targets)

    # Per-class accuracy
    binary = (preds >= 0.5).astype(np.float32)
    acc = {}
    for i, name in enumerate(target_names):
        acc[name] = float((binary[:, i] == tgts[:, i]).mean())
    acc["mean"] = float(np.mean(list(acc.values())))

    return {
        "auroc":    metrics["auroc"],
        "f1":       metrics["f1"],
        "accuracy": acc,
    }


def run_experiment(model_name, model, train_loader, val_loader, config,
                   F, device, anatomy_labels, F_anatomy):
    """Train + evaluate one model variant."""
    tag = model_name.lower().replace(" ", "_").replace("-", "_")
    cfg = dict(config)
    cfg["logging"] = {
        "log_dir":        f"outputs/logs/{tag}",
        "checkpoint_dir": "outputs/checkpoints",
        "save_every":     config.get("logging", {}).get("save_every", 5),
        "log_interval":   config.get("logging", {}).get("log_interval", 50),
    }

    t0 = time.time()
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader,
                      config=cfg, feasibility_matrix=F,
                      target_names=TARGET_NAMES, model_tag=tag)
    best_auroc = trainer.train()
    train_min  = (time.time() - t0) / 60

    # Detect hybrid relational model
    is_hybrid = isinstance(model, HybridAnatomyGuidedRelationalViT)

    print(f"\n  Evaluating {model_name}...")
    F_dev = F.to(device) if F is not None else None
    F_anat_dev = F_anatomy.to(device) if F_anatomy is not None else None

    # Load separate biases for hybrid model
    G_dev = None
    A_dev = F_anat_dev
    if is_hybrid:
        cache_dir = config.get("structure", {}).get("cache_dir", "outputs/cache")
        feat_path = os.path.join(cache_dir, "F_feature.pt")
        if os.path.exists(feat_path):
            G_dev = torch.load(feat_path, map_location=device, weights_only=False)["feasibility_matrix"].to(device)
        else:
            G_dev = F_dev  # fallback

    eval_metrics = evaluate_model(
        model, val_loader, device, F_dev, TARGET_NAMES, tag,
        graph_bias=G_dev if is_hybrid else None,
        anatomy_prior=A_dev if is_hybrid else None,
    )

    # Attention metrics
    print(f"  Computing attention metrics...")
    attn_computer = AttentionMetricsComputer(
        model=model, val_loader=val_loader, device=device,
        F_hybrid=F_dev, F_anatomy=F_anat_dev,
        anatomy_labels=anatomy_labels, target_names=TARGET_NAMES,
        max_batches=10,
        graph_bias=G_dev if is_hybrid else None,
        anatomy_prior=A_dev if is_hybrid else None,
    )
    attn_metrics = attn_computer.compute_all(compute_noise=True)

    return {
        "model_name":          model_name,
        "train_time_min":      train_min,
        "best_train_auroc":    best_auroc,
        "auroc":               eval_metrics["auroc"],
        "f1":                  eval_metrics["f1"],
        "accuracy":            eval_metrics["accuracy"],
        "attention_entropy":   attn_metrics["entropy"],
        "attention_sparsity":  attn_metrics["sparsity"],
        "anatomical_attention_ratio": attn_metrics["anatomical_attention_ratio"],
        "region_consistency":  attn_metrics["region_consistency"],
        "noise_robustness":    attn_metrics["noise_robustness"],
    }


def generate_result_tables(all_results, output_dir):
    """Generate CSV and Markdown result tables."""
    os.makedirs(output_dir, exist_ok=True)

    # ── Markdown table ──
    header = "| Model | AUROC | F1 | Accuracy | Entropy | AAR | Noise(σ=0.3) | Time |"
    sep    = "|-------|-------|----|---------|---------|----|-------------|------|"
    rows   = [header, sep]
    csv    = ["Model,AUROC,F1,Accuracy,Entropy,AAR,Noise_03,Time_min"]

    for r in all_results:
        auroc = r["auroc"].get("mean", 0.0)
        f1    = r["f1"].get("mean", 0.0)
        acc   = r["accuracy"].get("mean", 0.0)
        ent   = r["attention_entropy"].get("mean", 0.0)
        aar   = r["anatomical_attention_ratio"].get("mean", 0.0)
        n03   = r["noise_robustness"].get(0.3, r["noise_robustness"].get(0.30, 0.0))
        tm    = r["train_time_min"]

        rows.append(f"| {r['model_name']} | {auroc:.4f} | {f1:.4f} | {acc:.4f} | "
                    f"{ent:.3f} | {aar:.3f} | {n03:.4f} | {tm:.1f}m |")
        csv.append(f"{r['model_name']},{auroc:.4f},{f1:.4f},{acc:.4f},"
                   f"{ent:.3f},{aar:.3f},{n03:.4f},{tm:.1f}")

    # Per-class AUROC table
    rows.append("\n## Per-Class AUROC\n")
    rows.append("| Model | " + " | ".join(TARGET_NAMES) + " |")
    rows.append("|-------|" + "-------|" * len(TARGET_NAMES))
    for r in all_results:
        vals = " | ".join(f"{r['auroc'].get(c, 0.0):.4f}" for c in TARGET_NAMES)
        rows.append(f"| {r['model_name']} | {vals} |")

    md_path  = os.path.join(output_dir, "results_table.md")
    csv_path = os.path.join(output_dir, "experiment_results.csv")

    with open(md_path, "w")  as f: f.write("\n".join(rows))
    with open(csv_path, "w") as f: f.write("\n".join(csv))

    print(f"\n  ✓ Markdown table: {md_path}")
    print(f"  ✓ CSV results:    {csv_path}")

    # Print to console
    print(f"\n{'═'*80}")
    print("  EXPERIMENT RESULTS")
    print(f"{'═'*80}")
    print(f"  {'Model':<28} {'AUROC':>7} {'F1':>7} {'Acc':>7} {'Entropy':>8} "
          f"{'AAR':>6} {'Noise':>7} {'Time':>7}")
    print(f"  {'─'*74}")
    for r in all_results:
        print(f"  {r['model_name']:<28} "
              f"{r['auroc'].get('mean',0):.4f} "
              f"{r['f1'].get('mean',0):.4f} "
              f"{r['accuracy'].get('mean',0):.4f} "
              f"{r['attention_entropy'].get('mean',0):8.3f} "
              f"{r['anatomical_attention_ratio'].get('mean',0):6.3f} "
              f"{r['noise_robustness'].get(0.3,0):7.4f} "
              f"{r['train_time_min']:6.1f}m")
    print(f"{'═'*80}")


def main():
    parser = argparse.ArgumentParser(description="4-Model Comparative Experiments")
    parser.add_argument("--config",        default="configs/default.yaml")
    parser.add_argument("--epochs",        type=int,   default=None)
    parser.add_argument("--skip-training", action="store_true",
                        help="Load cached checkpoints, skip training")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.epochs:
        config["training"]["epochs"] = args.epochs

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs("outputs/visualizations", exist_ok=True)
    device = get_device(config)

    print("=" * 70)
    print("  Hybrid Anatomy-Guided Relational ViT — 6-Model Comparative Experiments")
    print("=" * 70)
    print(f"  Device:  {device}")
    print(f"  Epochs:  {config['training']['epochs']}")
    print(f"  Batch:   {config['training']['batch_size']}")

    # ── Load precomputed matrices ──
    print("\n[1/4] Loading precomputed feasibility matrices...")
    matrices = load_cache(config)
    F_anatomy = matrices.get("F_anatomy")
    F_feature = matrices.get("F_feature")
    F_hybrid  = matrices.get("F_hybrid")

    if F_anatomy is None:
        print("  ⚠ F_anatomy not found. Run: python scripts/precompute_structure.py --dry-run")
        print("  → Generating anatomy prior on-the-fly...")
        img_size  = config["data"].get("image_size", 192)
        grid_size = img_size // 16
        anat_cfg  = config.get("anatomy", {})
        builder   = AnatomyPriorBuilder(**{
            k: anat_cfg.get(k, v) for k, v in [
                ("same_region_weight", 1.0), ("adjacent_weight", 0.6),
                ("distant_weight", 0.15), ("background_weight", 0.05),
            ]
        })
        F_anatomy = builder.build(grid_size=grid_size)

    # Get anatomy labels for attention metrics
    img_size   = config["data"].get("image_size", 192)
    grid_size  = img_size // 16
    anat_cfg   = config.get("anatomy", {})
    anat_build = AnatomyPriorBuilder(
        same_region_weight=anat_cfg.get("same_region_weight", 1.0),
        adjacent_weight=anat_cfg.get("adjacent_weight", 0.6),
        distant_weight=anat_cfg.get("distant_weight", 0.15),
        background_weight=anat_cfg.get("background_weight", 0.05),
    )
    anatomy_labels = anat_build.get_region_map(grid_size)

    # ── Data ──
    print("\n[2/4] Loading data...")
    train_loader, val_loader = create_loaders(config)
    print(f"  Train: {len(train_loader.dataset):,}  Val: {len(val_loader.dataset):,}")

    model_cfg  = config.get("model", {})
    model_name_timm = model_cfg.get("name", "vit_small_patch16_224")
    alpha = config.get("structure", {}).get("alpha", 0.7)

    # Use F_hybrid if available, else build from anatomy + feature
    if F_hybrid is None and F_anatomy is not None and F_feature is not None:
        from src.structure.feasibility import build_hybrid_feasibility
        F_hybrid = build_hybrid_feasibility(F_anatomy, F_feature, alpha=alpha)
    elif F_hybrid is None and F_anatomy is not None:
        # Fallback: use anatomy only as hybrid
        print("  ⚠ F_feature not found — using F_anatomy as F_hybrid")
        F_hybrid = F_anatomy

    # 6 experiment configs: (name, F_matrix, description, model_type)
    # model_type: 'baseline', 'constrained', or 'hybrid_relational'
    EXPERIMENTS = [
        ("Baseline ViT",              None,      "Standard ViT, no structural constraints",           "baseline"),
        ("Feature-Constrained ViT",   F_feature, "Feature prior only (α=0.0)",                       "constrained"),
        ("Anatomy-Constrained ViT",   F_anatomy, "Anatomy prior only (α=1.0)",                       "constrained"),
        ("Hybrid-Constrained ViT",    F_hybrid,  f"Hybrid prior (α={alpha:.1f})",                    "constrained"),
        ("Hybrid Relational ViT",     None,      "Dual soft bias: graph + anatomy ← PRIMARY MODEL",  "hybrid_relational"),
    ]

    # ── Train & Evaluate ──
    print(f"\n[3/4] Training and evaluating {len(EXPERIMENTS)} models...")
    all_results = []

    for exp_idx, (exp_name, F_mat, description, model_type) in enumerate(EXPERIMENTS, 1):
        print(f"\n{'═'*70}")
        print(f"  [{exp_idx}/{len(EXPERIMENTS)}] {exp_name}")
        print(f"  {description}")
        print(f"{'═'*70}")

        # Skip constrained models if no feasibility matrix
        if model_type == "constrained" and F_mat is None:
            print(f"  ⚠ Skipping {exp_name}: feasibility matrix not available")
            print(f"     Run: python scripts/precompute_structure.py")
            continue

        # Build model based on type
        if model_type == "baseline":
            model = VisionTransformer.from_pretrained(
                model_name=model_name_timm, num_classes=5, img_size=img_size
            )
        elif model_type == "hybrid_relational":
            lambda_g = config.get("structure", {}).get("lambda_graph_init", 0.1)
            lambda_a = config.get("structure", {}).get("lambda_anatomy_init", 0.1)
            model = HybridAnatomyGuidedRelationalViT.from_pretrained(
                model_name=model_name_timm, num_classes=5,
                img_size=img_size,
                lambda_graph_init=lambda_g,
                lambda_anatomy_init=lambda_a,
            )
            # For hybrid relational: F_mat for trainer is F_hybrid (compatibility)
            # but actual forward uses graph_bias + anatomy_prior separately
            F_mat = F_hybrid
        else:  # constrained
            lambda_init = config.get("structure", {}).get("lambda_init", 0.1)
            model = ConstrainedVisionTransformer.from_pretrained(
                model_name=model_name_timm, num_classes=5,
                img_size=img_size, lambda_init=lambda_init,
            )

        if not args.skip_training:
            results = run_experiment(
                exp_name, model, train_loader, val_loader,
                config, F_mat, device, anatomy_labels, F_anatomy,
            )
        else:
            tag = exp_name.lower().replace(" ", "_").replace("-", "_")
            ckpt_path = f"outputs/checkpoints/{tag}_best_model.pt"
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model_state_dict"], strict=False)
                print(f"  Loaded checkpoint: {ckpt_path}")
            model = model.to(device)
            F_dev = F_mat.to(device) if F_mat is not None else None
            G_dev = F_feature.to(device) if F_feature is not None else None
            A_dev = F_anatomy.to(device) if F_anatomy is not None else None
            eval_m = evaluate_model(
                model, val_loader, device, F_dev, TARGET_NAMES, tag,
                graph_bias=G_dev if model_type == 'hybrid_relational' else None,
                anatomy_prior=A_dev if model_type == 'hybrid_relational' else None,
            )
            attn_c = AttentionMetricsComputer(
                model=model, val_loader=val_loader, device=device,
                F_hybrid=F_dev, F_anatomy=A_dev,
                anatomy_labels=anatomy_labels, target_names=TARGET_NAMES, max_batches=10,
            )
            attn_m = attn_c.compute_all(compute_noise=True)
            results = {
                "model_name": exp_name,
                "train_time_min": 0.0,
                "auroc": eval_m["auroc"], "f1": eval_m["f1"],
                "accuracy": eval_m["accuracy"],
                "attention_entropy": attn_m["entropy"],
                "attention_sparsity": attn_m["sparsity"],
                "anatomical_attention_ratio": attn_m["anatomical_attention_ratio"],
                "region_consistency": attn_m["region_consistency"],
                "noise_robustness": attn_m["noise_robustness"],
            }

        all_results.append(results)

        # Save intermediate
        tag_safe = exp_name.replace(" ", "_").replace("-", "_")
        out_path = os.path.join(RESULTS_DIR, f"{tag_safe}_results.pt")
        torch.save(results, out_path)
        print(f"  ✓ {exp_name}: AUROC={results['auroc'].get('mean', 0):.4f}")
        if model_type == 'hybrid_relational' and hasattr(model, 'get_lambda_values'):
            lv = model.get_lambda_values()
            print(f"    Learned λ_graph={lv['lambda_graph']:.4f}, λ_anatomy={lv['lambda_anatomy']:.4f}")

    # ── Tables & Plots ──
    print(f"\n[4/4] Generating tables and publication figures...")
    generate_result_tables(all_results, RESULTS_DIR)

    viz = ExperimentVisualizer(output_dir="outputs/visualizations", dpi=150)

    # Anatomy region map
    viz.plot_anatomy_regions(anatomy_labels, grid_size=grid_size,
                             save_name="anatomy_regions.png")

    # Feasibility heatmaps
    if F_anatomy is not None:
        viz.plot_feasibility_heatmap(F_anatomy, "Anatomy Prior (F_anatomy)",
                                     "feasibility_anatomy.png")
    if F_hybrid is not None:
        viz.plot_feasibility_heatmap(F_hybrid, "Hybrid Prior (F_hybrid)",
                                     "feasibility_hybrid.png")
    if F_anatomy is not None and F_feature is not None and F_hybrid is not None:
        viz.plot_hybrid_structure(F_anatomy, F_feature, F_hybrid, alpha=alpha,
                                  save_name="hybrid_structure_comparison.png")

    # Generate all results figures
    viz.generate_all_results_figures(all_results)

    # Save JSON summary
    json_results = []
    for r in all_results:
        jr = {
            "model_name":    r["model_name"],
            "train_time_min": r["train_time_min"],
            "auroc_mean":    r["auroc"].get("mean", 0),
            "f1_mean":       r["f1"].get("mean", 0),
            "accuracy_mean": r["accuracy"].get("mean", 0),
            "entropy":       r["attention_entropy"].get("mean", 0),
            "sparsity":      r["attention_sparsity"].get("mean", 0),
            "aar":           r["anatomical_attention_ratio"].get("mean", 0),
            "rcs":           r["region_consistency"].get("mean", 0),
            "noise_auroc_03": r["noise_robustness"].get(0.3, 0),
            "per_class_auroc": {
                k: v for k, v in r["auroc"].items() if k != "mean"
            },
        }
        json_results.append(jr)

    json_path = os.path.join(RESULTS_DIR, "experiment_summary.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"  ✓ JSON summary: {json_path}")

    print(f"\n{'═'*70}")
    print(f"  All experiments complete!")
    print(f"  Results:        outputs/results/")
    print(f"  Visualizations: outputs/visualizations/")
    print(f"{'═'*70}")


if __name__ == "__main__":
    main()
