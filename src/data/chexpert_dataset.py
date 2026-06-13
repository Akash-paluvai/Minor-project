"""
CheXpert Chest X-ray Dataset Loader — v2 (Hybrid Anatomy-Aware SC-ViT).

Key additions vs v1:
  - subset_size param for lightweight M3 Pro training (subset_train/subset_val from config)
  - MPS-optimized DataLoader: pin_memory=False, num_workers from config
  - Error-resilient image loading (returns zero image on failure)
  - Fixed-seed reproducible subsampling (np.random.default_rng(42))

Tensor shapes:
  image:  (3, H, W)  — H=W=192 for 12×12 patch grid
  labels: (5,)       — binary multi-label float32
"""

import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

# Default 5 competition pathologies
DEFAULT_TARGETS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]

ALL_TARGETS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices",
]


def get_dataset_path():
    """Get CheXpert dataset path, downloading via kagglehub if needed."""
    import kagglehub
    path = kagglehub.dataset_download("ashery/chexpert")
    print(f"  Dataset path: {path}")
    return path


class CheXpertDataset(Dataset):
    """
    CheXpert Chest X-ray dataset with subset support.

    Args:
        csv_path: Path to train.csv or valid.csv
        dataset_dir: Root directory containing images
        target_columns: List of pathology column names
        uncertainty_policy: 'ones' | 'zeros' | 'ignore'
        transform: torchvision transforms
        frontal_only: Only include frontal views
        subset_size: If > 0, randomly subsample this many images (seed=42)
    """

    def __init__(
        self,
        csv_path: str,
        dataset_dir: str = None,
        target_columns: list = None,
        uncertainty_policy: str = "ones",
        transform=None,
        frontal_only: bool = False,
        subset_size: int = 0,
    ):
        super().__init__()
        self.transform = transform
        self.uncertainty_policy = uncertainty_policy
        self.target_columns = target_columns or DEFAULT_TARGETS

        self.df = pd.read_csv(csv_path)

        if dataset_dir is None:
            dataset_dir = os.path.dirname(csv_path)
        self.dataset_dir = dataset_dir

        if frontal_only and "Frontal/Lateral" in self.df.columns:
            self.df = self.df[self.df["Frontal/Lateral"] == "Frontal"]

        self._apply_uncertainty_policy()
        self.df = self.df.dropna(subset=self.target_columns).reset_index(drop=True)

        # Subset sampling — fixed seed for reproducibility
        if subset_size > 0 and subset_size < len(self.df):
            rng = np.random.default_rng(42)
            indices = sorted(rng.choice(len(self.df), size=subset_size, replace=False))
            self.df = self.df.iloc[indices].reset_index(drop=True)

        print(f"  Dataset: {len(self.df)} images, {len(self.target_columns)} classes")

    def _apply_uncertainty_policy(self):
        """Handle uncertain labels (-1)."""
        for col in self.target_columns:
            if col not in self.df.columns:
                self.df[col] = 0.0
                continue
            if self.uncertainty_policy == "ones":
                self.df[col] = self.df[col].replace(-1.0, 1.0)
            elif self.uncertainty_policy == "zeros":
                self.df[col] = self.df[col].replace(-1.0, 0.0)
            elif self.uncertainty_policy == "ignore":
                self.df[col] = self.df[col].replace(-1.0, float("nan"))
            else:
                raise ValueError(f"Unknown uncertainty policy: {self.uncertainty_policy}")
            if self.uncertainty_policy != "ignore":
                self.df[col] = self.df[col].fillna(0.0)

    def _resolve_image_path(self, csv_path: str) -> str:
        """Resolve image path from CSV entry to actual filesystem path."""
        clean_path = csv_path
        for prefix in ["CheXpert-v1.0-small/", "CheXpert-v1.0/"]:
            if clean_path.startswith(prefix):
                clean_path = clean_path[len(prefix):]
                break

        for candidate in [
            os.path.join(self.dataset_dir, clean_path),
            os.path.join(self.dataset_dir, csv_path),
            os.path.join(os.path.dirname(self.dataset_dir), csv_path),
        ]:
            if os.path.exists(candidate):
                return candidate

        return os.path.join(self.dataset_dir, clean_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns:
            image:  (3, H, W) float tensor
            labels: (5,) float tensor
        """
        row = self.df.iloc[idx]
        img_path = self._resolve_image_path(row["Path"])

        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError) as e:
            print(f"  ⚠ Image load error: {img_path} — using blank")
            image = Image.fromarray(np.zeros((192, 192, 3), dtype=np.uint8))

        if self.transform is not None:
            image = self.transform(image)

        labels = torch.tensor(
            row[self.target_columns].values.astype(np.float32),
            dtype=torch.float32,
        )
        return image, labels

    def get_label_distribution(self):
        """Get per-class label distribution statistics."""
        return {
            col: {
                "positive": int(self.df[col].sum()),
                "negative": int((self.df[col] == 0).sum()),
                "total": len(self.df),
                "prevalence": float(self.df[col].mean()),
            }
            for col in self.target_columns
        }


def get_chexpert_dataloaders(config: dict, train_transform=None, val_transform=None):
    """
    Create train and validation DataLoaders from config.

    MPS-safe settings applied automatically:
      - pin_memory=False (MPS requirement)
      - num_workers from config (default 0)
      - subset_train / subset_val honoured

    Returns:
        train_loader, val_loader
    """
    data_cfg  = config["data"]
    train_cfg = config["training"]
    sys_cfg   = config.get("system", {})

    dataset_dir = data_cfg["dataset_dir"]
    if not os.path.exists(dataset_dir):
        print("  Dataset not found locally, downloading via kagglehub...")
        dataset_dir = get_dataset_path()

    train_csv = os.path.join(dataset_dir, data_cfg["train_csv"])
    valid_csv = os.path.join(dataset_dir, data_cfg["valid_csv"])

    target_columns  = data_cfg.get("target_columns", DEFAULT_TARGETS)
    uncertainty     = data_cfg.get("uncertainty_policy", "ones")
    frontal_only    = data_cfg.get("frontal_only", False)
    subset_train    = data_cfg.get("subset_train", 0)
    subset_val      = data_cfg.get("subset_val", 0)

    train_dataset = CheXpertDataset(
        csv_path=train_csv,
        dataset_dir=dataset_dir,
        target_columns=target_columns,
        uncertainty_policy=uncertainty,
        transform=train_transform,
        frontal_only=frontal_only,
        subset_size=subset_train,
    )

    val_dataset = CheXpertDataset(
        csv_path=valid_csv,
        dataset_dir=dataset_dir,
        target_columns=target_columns,
        uncertainty_policy=uncertainty,
        transform=val_transform,
        frontal_only=frontal_only,
        subset_size=subset_val,
    )

    pin_memory  = sys_cfg.get("pin_memory", False)   # Must be False for MPS
    num_workers = train_cfg.get("num_workers", 0)     # 0 is safest on MPS

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader
