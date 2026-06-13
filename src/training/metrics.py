"""
Training metrics for multi-label classification.

Computes AUROC, AUPRC, and F1 score per class and averaged.
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def compute_auroc(targets: np.ndarray, predictions: np.ndarray, target_names: list = None):
    """
    Compute per-class and mean AUROC.

    Args:
        targets: (N, C) binary ground truth
        predictions: (N, C) predicted probabilities
        target_names: optional list of class names

    Returns:
        dict with per-class and mean AUROC
    """
    n_classes = targets.shape[1]
    aurocs = {}

    for i in range(n_classes):
        # Skip classes where only one label is present
        if len(np.unique(targets[:, i])) < 2:
            continue

        name = target_names[i] if target_names else f"class_{i}"
        aurocs[name] = roc_auc_score(targets[:, i], predictions[:, i])

    if aurocs:
        aurocs["mean"] = np.mean(list(aurocs.values()))
    else:
        aurocs["mean"] = 0.0

    return aurocs


def compute_auprc(targets: np.ndarray, predictions: np.ndarray, target_names: list = None):
    """
    Compute per-class and mean AUPRC (Average Precision).

    Args:
        targets: (N, C) binary ground truth
        predictions: (N, C) predicted probabilities
        target_names: optional list of class names

    Returns:
        dict with per-class and mean AUPRC
    """
    n_classes = targets.shape[1]
    auprcs = {}

    for i in range(n_classes):
        if len(np.unique(targets[:, i])) < 2:
            continue

        name = target_names[i] if target_names else f"class_{i}"
        auprcs[name] = average_precision_score(targets[:, i], predictions[:, i])

    if auprcs:
        auprcs["mean"] = np.mean(list(auprcs.values()))
    else:
        auprcs["mean"] = 0.0

    return auprcs


def compute_f1(targets: np.ndarray, predictions: np.ndarray, threshold: float = 0.5,
               target_names: list = None):
    """
    Compute per-class and mean F1 score.

    Args:
        targets: (N, C) binary ground truth
        predictions: (N, C) predicted probabilities
        threshold: threshold for converting probabilities to binary
        target_names: optional list of class names

    Returns:
        dict with per-class and mean F1
    """
    binary_preds = (predictions >= threshold).astype(np.float32)
    n_classes = targets.shape[1]
    f1s = {}

    for i in range(n_classes):
        name = target_names[i] if target_names else f"class_{i}"
        f1s[name] = f1_score(targets[:, i], binary_preds[:, i], zero_division=0)

    f1s["mean"] = np.mean([v for k, v in f1s.items() if k != "mean"])

    return f1s


class MetricTracker:
    """Track and aggregate metrics across batches."""

    def __init__(self, target_names: list = None):
        self.target_names = target_names
        self.reset()

    def reset(self):
        self.all_targets = []
        self.all_predictions = []

    def update(self, targets: torch.Tensor, predictions: torch.Tensor):
        """
        Accumulate batch predictions.

        Args:
            targets: (B, C) ground truth
            predictions: (B, C) predicted logits or probabilities
        """
        # Apply sigmoid if logits (values outside [0,1])
        if predictions.min() < 0 or predictions.max() > 1:
            predictions = torch.sigmoid(predictions)

        self.all_targets.append(targets.cpu().numpy())
        self.all_predictions.append(predictions.cpu().detach().numpy())

    def compute(self):
        """Compute all metrics on accumulated predictions."""
        targets = np.concatenate(self.all_targets, axis=0)
        predictions = np.concatenate(self.all_predictions, axis=0)

        results = {
            "auroc": compute_auroc(targets, predictions, self.target_names),
            "auprc": compute_auprc(targets, predictions, self.target_names),
            "f1": compute_f1(targets, predictions, target_names=self.target_names),
        }

        return results

    def summary_string(self, results: dict = None):
        """Format metrics as a readable string."""
        if results is None:
            results = self.compute()

        lines = []
        for metric_name, values in results.items():
            mean_val = values.get("mean", 0.0)
            lines.append(f"  {metric_name.upper()}: {mean_val:.4f}")
            for k, v in values.items():
                if k != "mean":
                    lines.append(f"    {k}: {v:.4f}")

        return "\n".join(lines)
