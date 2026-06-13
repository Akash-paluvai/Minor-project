#!/usr/bin/env python3
"""
CheXpert Dataset Analysis and Visualization.

Tasks:
1. Inspect folder structure
2. Count number of images
3. Display sample images
4. Identify and analyze label columns
5. Verify the PyTorch dataset loader works

Usage:
    python analyze_dataset.py
    python analyze_dataset.py --config configs/default.yaml
"""

import os
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image
from collections import Counter

import torch
from src.data.chexpert_dataset import CheXpertDataset, get_dataset_path, DEFAULT_TARGETS, ALL_TARGETS
from src.data.transforms import get_val_transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze CheXpert Dataset")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output-dir", type=str, default="outputs/dataset_analysis")
    return parser.parse_args()


def inspect_folder_structure(dataset_dir):
    """Task 1: Inspect the folder structure."""
    print("\n" + "=" * 60)
    print("TASK 1: FOLDER STRUCTURE")
    print("=" * 60)

    print(f"\nDataset root: {dataset_dir}")

    # List top-level contents
    contents = os.listdir(dataset_dir)
    print(f"\nTop-level contents ({len(contents)} items):")
    for item in sorted(contents):
        full_path = os.path.join(dataset_dir, item)
        if os.path.isdir(full_path):
            # Count subdirectories (patients)
            sub_count = len(os.listdir(full_path))
            print(f"  📁 {item}/ ({sub_count} patients)")
        else:
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            print(f"  📄 {item} ({size_mb:.1f} MB)")

    # Show sample patient structure
    train_dir = os.path.join(dataset_dir, "train")
    if os.path.exists(train_dir):
        patients = sorted(os.listdir(train_dir))[:3]
        print(f"\nSample patient structure:")
        for patient in patients:
            patient_dir = os.path.join(train_dir, patient)
            if os.path.isdir(patient_dir):
                print(f"  📁 {patient}/")
                for study in sorted(os.listdir(patient_dir)):
                    study_dir = os.path.join(patient_dir, study)
                    if os.path.isdir(study_dir):
                        files = os.listdir(study_dir)
                        real_files = [f for f in files if not f.startswith("._")]
                        print(f"    📁 {study}/ → {real_files}")


def count_images(dataset_dir):
    """Task 2: Count number of images."""
    print("\n" + "=" * 60)
    print("TASK 2: IMAGE COUNT")
    print("=" * 60)

    counts = {}
    for split in ["train", "valid"]:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir):
            continue

        n_images = 0
        n_patients = 0
        n_studies = 0

        for patient in os.listdir(split_dir):
            patient_dir = os.path.join(split_dir, patient)
            if not os.path.isdir(patient_dir):
                continue
            n_patients += 1
            for study in os.listdir(patient_dir):
                study_dir = os.path.join(patient_dir, study)
                if not os.path.isdir(study_dir):
                    continue
                n_studies += 1
                for f in os.listdir(study_dir):
                    if f.endswith((".jpg", ".png")) and not f.startswith("._"):
                        n_images += 1

        counts[split] = {
            "images": n_images,
            "patients": n_patients,
            "studies": n_studies,
        }
        print(f"\n{split.upper()}:")
        print(f"  Images:   {n_images:,}")
        print(f"  Patients: {n_patients:,}")
        print(f"  Studies:  {n_studies:,}")

    total = sum(c["images"] for c in counts.values())
    print(f"\nTOTAL IMAGES: {total:,}")

    return counts


def display_sample_images(dataset_dir, output_dir):
    """Task 3: Display sample images."""
    print("\n" + "=" * 60)
    print("TASK 3: SAMPLE IMAGES")
    print("=" * 60)

    train_csv = os.path.join(dataset_dir, "train.csv")
    df = pd.read_csv(train_csv)

    # Get diverse samples: different pathologies
    samples = []

    # Get one frontal and one lateral
    frontal = df[df["Frontal/Lateral"] == "Frontal"].iloc[0]
    lateral = df[df["Frontal/Lateral"] == "Lateral"].iloc[:1]
    samples.append(("Frontal View", frontal))
    if len(lateral) > 0:
        samples.append(("Lateral View", lateral.iloc[0]))

    # Get samples with specific findings
    for condition in ["Cardiomegaly", "Pleural Effusion", "Edema", "Atelectasis",
                      "Consolidation", "Pneumothorax"]:
        if condition in df.columns:
            pos = df[df[condition] == 1.0]
            if len(pos) > 0:
                samples.append((condition, pos.iloc[0]))

    # Limit to 8 samples
    samples = samples[:8]
    n_samples = len(samples)
    cols = 4
    rows = (n_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    if rows == 1:
        axes = [axes]
    axes = np.array(axes).flatten()

    for idx, (title, row) in enumerate(samples):
        csv_path = row["Path"]
        # Resolve path
        for prefix in ["CheXpert-v1.0-small/", "CheXpert-v1.0/"]:
            if csv_path.startswith(prefix):
                csv_path = csv_path[len(prefix):]
                break
        img_path = os.path.join(dataset_dir, csv_path)

        try:
            img = Image.open(img_path).convert("RGB")
            axes[idx].imshow(img, cmap="gray")
            axes[idx].set_title(f"{title}\n{row.get('Sex', 'N/A')}, Age {row.get('Age', 'N/A')}",
                              fontsize=10, fontweight="bold")
        except Exception as e:
            axes[idx].text(0.5, 0.5, f"Error: {e}", ha="center", va="center")
            axes[idx].set_title(title)

        axes[idx].axis("off")

    # Hide empty subplots
    for idx in range(n_samples, len(axes)):
        axes[idx].axis("off")

    plt.suptitle("CheXpert Sample Chest X-rays", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "sample_images.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ✓ Saved sample images to {save_path}")


def analyze_labels(dataset_dir, output_dir):
    """Task 4: Identify and analyze label columns."""
    print("\n" + "=" * 60)
    print("TASK 4: LABEL ANALYSIS")
    print("=" * 60)

    train_csv = os.path.join(dataset_dir, "train.csv")
    valid_csv = os.path.join(dataset_dir, "valid.csv")

    train_df = pd.read_csv(train_csv)
    valid_df = pd.read_csv(valid_csv)

    print(f"\nTrain CSV shape: {train_df.shape}")
    print(f"Valid CSV shape: {valid_df.shape}")

    # Identify all columns
    metadata_cols = ["Path", "Sex", "Age", "Frontal/Lateral", "AP/PA"]
    label_cols = [c for c in train_df.columns if c not in metadata_cols]

    print(f"\nMetadata columns ({len(metadata_cols)}): {metadata_cols}")
    print(f"Label columns ({len(label_cols)}): {label_cols}")

    # Label distribution analysis
    print(f"\n{'='*70}")
    print(f"{'Pathology':<30} {'Positive':>10} {'Negative':>10} {'Uncertain':>10} {'Blank':>10}")
    print(f"{'='*70}")

    label_stats = {}
    for col in label_cols:
        vals = train_df[col]
        pos = (vals == 1.0).sum()
        neg = (vals == 0.0).sum()
        unc = (vals == -1.0).sum()
        blank = vals.isna().sum()
        label_stats[col] = {"positive": pos, "negative": neg, "uncertain": unc, "blank": blank}
        print(f"{col:<30} {pos:>10,} {neg:>10,} {unc:>10,} {blank:>10,}")

    # View distribution analysis
    print(f"\n--- View Distribution ---")
    if "Frontal/Lateral" in train_df.columns:
        print(train_df["Frontal/Lateral"].value_counts().to_string())

    print(f"\n--- Sex Distribution ---")
    if "Sex" in train_df.columns:
        print(train_df["Sex"].value_counts().to_string())

    print(f"\n--- Age Statistics ---")
    if "Age" in train_df.columns:
        print(f"  Mean: {train_df['Age'].mean():.1f}")
        print(f"  Std:  {train_df['Age'].std():.1f}")
        print(f"  Min:  {train_df['Age'].min()}")
        print(f"  Max:  {train_df['Age'].max()}")

    # ---- Visualization: Label Distribution Bar Chart ----
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: All labels stacked bar
    pathologies = list(label_stats.keys())
    pos_counts = [label_stats[p]["positive"] for p in pathologies]
    neg_counts = [label_stats[p]["negative"] for p in pathologies]
    unc_counts = [label_stats[p]["uncertain"] for p in pathologies]
    blank_counts = [label_stats[p]["blank"] for p in pathologies]

    x = np.arange(len(pathologies))
    width = 0.2

    axes[0].bar(x - 1.5*width, pos_counts, width, label="Positive (1)", color="#4CAF50", alpha=0.85)
    axes[0].bar(x - 0.5*width, neg_counts, width, label="Negative (0)", color="#F44336", alpha=0.85)
    axes[0].bar(x + 0.5*width, unc_counts, width, label="Uncertain (-1)", color="#FF9800", alpha=0.85)
    axes[0].bar(x + 1.5*width, blank_counts, width, label="Blank (NaN)", color="#9E9E9E", alpha=0.85)

    axes[0].set_xlabel("Pathology", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_title("CheXpert Label Distribution (Train Set)", fontsize=14, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(pathologies, rotation=45, ha="right", fontsize=9)
    axes[0].legend(fontsize=10)
    axes[0].grid(axis="y", alpha=0.3)

    # Plot 2: 5-class competition focus (prevalence after U-Ones)
    comp_pathologies = DEFAULT_TARGETS
    comp_stats = {}
    for col in comp_pathologies:
        vals = train_df[col].copy()
        vals = vals.replace(-1.0, 1.0)  # U-Ones
        vals = vals.fillna(0.0)
        comp_stats[col] = vals.mean()

    bars = axes[1].barh(comp_pathologies, [comp_stats[p] for p in comp_pathologies],
                       color=["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"], alpha=0.85)
    axes[1].set_xlabel("Prevalence (U-Ones policy)", fontsize=12)
    axes[1].set_title("5-Class Competition Pathology Prevalence", fontsize=14, fontweight="bold")
    axes[1].set_xlim(0, 1.0)
    axes[1].grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, [comp_stats[p] for p in comp_pathologies]):
        axes[1].text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                    f"{val:.1%}", va="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "label_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ✓ Saved label distribution chart to {save_path}")

    return label_stats


def test_dataloader(dataset_dir, config, output_dir):
    """Task 5: Build and verify the PyTorch dataset loader."""
    print("\n" + "=" * 60)
    print("TASK 5: PYTORCH DATASET LOADER TEST")
    print("=" * 60)

    data_cfg = config["data"]
    transform = get_val_transforms(data_cfg.get("image_size", 224))

    train_csv = os.path.join(dataset_dir, "train.csv")
    valid_csv = os.path.join(dataset_dir, "valid.csv")

    target_cols = data_cfg.get("target_columns", DEFAULT_TARGETS)

    # Build datasets
    print("\nBuilding train dataset...")
    train_ds = CheXpertDataset(
        csv_path=train_csv,
        dataset_dir=dataset_dir,
        target_columns=target_cols,
        uncertainty_policy=data_cfg.get("uncertainty_policy", "ones"),
        transform=transform,
    )
    print(f"  Train samples: {len(train_ds):,}")

    print("Building valid dataset...")
    val_ds = CheXpertDataset(
        csv_path=valid_csv,
        dataset_dir=dataset_dir,
        target_columns=target_cols,
        uncertainty_policy=data_cfg.get("uncertainty_policy", "ones"),
        transform=transform,
    )
    print(f"  Valid samples: {len(val_ds):,}")

    # Test loading a single sample
    print("\nLoading single sample...")
    img, label = train_ds[0]
    print(f"  Image tensor shape: {img.shape}")
    print(f"  Image dtype:        {img.dtype}")
    print(f"  Image value range:  [{img.min():.3f}, {img.max():.3f}]")
    print(f"  Label shape:        {label.shape}")
    print(f"  Label values:       {label.numpy()}")
    print(f"  Label names:        {target_cols}")

    # Test DataLoader
    print("\nTesting DataLoader...")
    from torch.utils.data import DataLoader
    loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    batch_imgs, batch_labels = next(iter(loader))
    print(f"  Batch images shape:  {batch_imgs.shape}")
    print(f"  Batch labels shape:  {batch_labels.shape}")
    print(f"  Batch label means:   {batch_labels.mean(dim=0).numpy().round(3)}")

    # Label distribution
    print("\nLabel distribution (train):")
    stats = train_ds.get_label_distribution()
    for col, s in stats.items():
        print(f"  {col}: {s['positive']:,} positive ({s['prevalence']:.1%} prevalence)")

    # Visualize a batch
    fig, axes = plt.subplots(2, 8, figsize=(20, 5))
    for i in range(min(16, batch_imgs.shape[0])):
        row, col = i // 8, i % 8
        img = batch_imgs[i]
        # Denormalize for visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()

        axes[row, col].imshow(img)
        # Find positive labels
        pos_labels = [target_cols[j][:4] for j in range(len(target_cols))
                      if batch_labels[i, j] == 1.0]
        title = ", ".join(pos_labels) if pos_labels else "Normal"
        axes[row, col].set_title(title, fontsize=7)
        axes[row, col].axis("off")

    plt.suptitle("Sample Training Batch (16 images)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "sample_batch.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ✓ Saved sample batch to {save_path}")

    print("\n  ✓ DataLoader test PASSED")
    return train_ds, val_ds


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    dataset_dir = config["data"]["dataset_dir"]

    # Auto-download if needed
    if not os.path.exists(dataset_dir):
        print("Dataset not found locally, downloading via kagglehub...")
        dataset_dir = get_dataset_path()

    print("=" * 60)
    print("CheXpert Dataset Analysis")
    print("=" * 60)
    print(f"Dataset: {dataset_dir}")

    # Task 1: Folder structure
    inspect_folder_structure(dataset_dir)

    # Task 2: Count images
    counts = count_images(dataset_dir)

    # Task 3: Sample images
    display_sample_images(dataset_dir, args.output_dir)

    # Task 4: Label analysis
    label_stats = analyze_labels(dataset_dir, args.output_dir)

    # Task 5: DataLoader test
    train_ds, val_ds = test_dataloader(dataset_dir, config, args.output_dir)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nOutputs saved to: {args.output_dir}/")
    print("  - sample_images.png")
    print("  - label_distribution.png")
    print("  - sample_batch.png")


if __name__ == "__main__":
    main()
