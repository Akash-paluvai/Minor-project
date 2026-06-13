"""
Attention Analysis Metrics for Hybrid Anatomy-Aware SC-ViT.

Implements 5 attention analysis metrics:

  1. Attention Entropy        — how concentrated/spread attention is
  2. Attention Sparsity       — fraction of near-zero attention weights
  3. Anatomical Attention Ratio — fraction of attention to plausible anatomy
  4. Region Consistency Score  — consistency of attention within a region
  5. Noise Robustness          — AUROC retention under Gaussian noise

Usage:
    metrics = AttentionMetricsComputer(model, val_loader, device, F_hybrid, anatomy_map)
    results = metrics.compute_all()

All metrics return scalars in a dict for easy logging/comparison.
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


# ════════════════════════════════════════════════════════════
#  Individual Metric Functions
# ════════════════════════════════════════════════════════════

def compute_attention_entropy(attn_weights_list: list) -> dict:
    """
    Compute mean attention entropy across all layers and heads.

    Entropy H = -Σ p_i · log(p_i) measures how spread the attention is:
      High H → uniform (unfocused) attention — typical of baseline ViT
      Low  H → concentrated (focused) attention — goal of constrained ViT

    Args:
        attn_weights_list: List of (B, heads, N, N) attention tensors per layer

    Returns:
        dict with 'mean', 'std', 'per_layer' keys
    """
    all_entropies = []

    for attn_w in attn_weights_list:
        # attn_w: (B, heads, N, N) — already post-softmax probabilities
        # Clamp for numerical stability
        attn_w = attn_w.clamp(min=1e-9)

        # H = -Σ p · log(p) over the N key dimension
        # Shape: (B, heads, N)
        entropy = -(attn_w * attn_w.log()).sum(dim=-1)

        # Average over tokens, heads, batch
        all_entropies.append(entropy.mean().item())

    if not all_entropies:
        return {"mean": 0.0, "std": 0.0, "per_layer": []}

    return {
        "mean":      float(np.mean(all_entropies)),
        "std":       float(np.std(all_entropies)),
        "per_layer": all_entropies,
    }


def compute_attention_sparsity(attn_weights_list: list, threshold: float = 0.01) -> dict:
    """
    Compute attention sparsity — fraction of weights below threshold.

    High sparsity → model focuses on few key patches (more specific).
    Structure constraints should increase sparsity vs baseline.

    Args:
        attn_weights_list: List of (B, heads, N, N) per layer
        threshold: Weights below this counted as "near-zero" (default 0.01)

    Returns:
        dict with 'mean', 'std', 'per_layer'
    """
    all_sparsities = []

    for attn_w in attn_weights_list:
        sparsity = (attn_w < threshold).float().mean().item()
        all_sparsities.append(sparsity)

    if not all_sparsities:
        return {"mean": 0.0, "std": 0.0, "per_layer": []}

    return {
        "mean":      float(np.mean(all_sparsities)),
        "std":       float(np.std(all_sparsities)),
        "per_layer": all_sparsities,
    }


def compute_anatomical_attention_ratio(
    attn_weights_list: list,
    anatomy_labels: np.ndarray,
    F_anatomy: torch.Tensor,
    anatomy_threshold: float = 0.3,
) -> dict:
    """
    Compute Anatomical Attention Ratio (AAR).

    Definition:
        AAR = fraction of total attention weight directed toward
              anatomically plausible patch pairs (F_anatomy > threshold)

    High AAR → model attends to anatomically sensible regions.
    Constrained models should have higher AAR than baseline.

    Args:
        attn_weights_list: List of (B, heads, N+1, N+1) per layer
                           N+1 includes CLS token at index 0
        anatomy_labels:    (N,) int array — region per patch (0..6)
        F_anatomy:         (N+1, N+1) anatomy feasibility matrix
        anatomy_threshold: Min F_anatomy value to count as "plausible" (default 0.3)

    Returns:
        dict with 'mean', 'std', 'per_layer'
    """
    # Build binary plausibility mask from F_anatomy
    # Shape: (N+1, N+1) → True where anatomically plausible
    plausible_mask = (F_anatomy >= anatomy_threshold).float()  # (N+1, N+1)

    all_ratios = []

    for attn_w in attn_weights_list:
        # attn_w: (B, heads, N+1, N+1)
        B, heads, N_total, _ = attn_w.shape
        mask = plausible_mask.to(attn_w.device)  # (N+1, N+1)

        # Fraction of attention going to plausible pairs
        # attn_w * mask sums attention only over plausible (i,j) pairs
        plausible_attn = (attn_w * mask.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # (B, heads, N+1)
        total_attn     = attn_w.sum(dim=-1)                                      # (B, heads, N+1)

        ratio = (plausible_attn / (total_attn + 1e-8)).mean().item()
        all_ratios.append(ratio)

    if not all_ratios:
        return {"mean": 0.0, "std": 0.0, "per_layer": []}

    return {
        "mean":      float(np.mean(all_ratios)),
        "std":       float(np.std(all_ratios)),
        "per_layer": all_ratios,
    }


def compute_region_consistency_score(
    attn_weights_list: list,
    anatomy_labels: np.ndarray,
) -> dict:
    """
    Compute Region Consistency Score (RCS).

    Definition:
        For each anatomical region, measure the variance of attention
        weights received by patches within that region.
        Lower variance → more consistent region-level attention.

    RCS = 1 - mean(intra-region attention variance)
    Values closer to 1 → more consistent (better for constrained models).

    Args:
        attn_weights_list: List of (B, heads, N+1, N+1) per layer
        anatomy_labels:    (N,) int labels for patches (excludes CLS at idx 0)

    Returns:
        dict with 'mean', 'std', 'per_region', 'per_layer'
    """
    unique_regions = np.unique(anatomy_labels)
    all_layer_scores = []

    for attn_w in attn_weights_list:
        B, heads, N_total, _ = attn_w.shape
        # Patch attention columns: indices 1..N (skip CLS at 0)
        patch_attn = attn_w[:, :, :, 1:]  # (B, heads, N+1, N)

        region_variances = []
        for r in unique_regions:
            region_patch_indices = np.where(anatomy_labels == r)[0]
            if len(region_patch_indices) < 2:
                continue
            # Attention received by patches in this region
            region_attn = patch_attn[:, :, :, region_patch_indices]   # (B, heads, N+1, k)
            # Variance across patch positions within region
            var = region_attn.var(dim=-1).mean().item()
            region_variances.append(var)

        # RCS = 1 - mean_variance (normalized, higher = more consistent)
        if region_variances:
            score = 1.0 - float(np.mean(region_variances))
            score = max(0.0, score)  # clamp to [0, 1]
        else:
            score = 0.0

        all_layer_scores.append(score)

    if not all_layer_scores:
        return {"mean": 0.0, "std": 0.0, "per_layer": []}

    return {
        "mean":      float(np.mean(all_layer_scores)),
        "std":       float(np.std(all_layer_scores)),
        "per_layer": all_layer_scores,
    }


def compute_noise_robustness(
    model: nn.Module,
    val_loader,
    device: torch.device,
    feasibility_matrix: torch.Tensor = None,
    noise_levels: list = None,
    target_names: list = None,
    graph_bias: torch.Tensor = None,
    anatomy_prior: torch.Tensor = None,
) -> dict:
    """
    Compute AUROC retention at different Gaussian noise levels.

    Measures how well structural constraints improve robustness:
        - Baseline ViT: attention can shift to noise artifacts
        - Constrained ViT: anatomy priors anchor attention, more robust

    Args:
        model:              ViT model
        val_loader:         Validation DataLoader
        device:             Torch device
        feasibility_matrix: Optional (N+1, N+1) for constrained model
        noise_levels:       List of σ values (default [0.0, 0.1, 0.2, 0.3])
        target_names:       Pathology names
        graph_bias:         Optional graph bias for hybrid relational model
        anatomy_prior:      Optional anatomy prior for hybrid relational model

    Returns:
        dict mapping σ → mean AUROC
    """
    from .metrics import MetricTracker

    if noise_levels is None:
        noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3]
    if target_names is None:
        target_names = ["Atelectasis", "Cardiomegaly", "Consolidation",
                        "Edema", "Pleural Effusion"]

    # Detect model type
    from src.models.hybrid_relational_vit import HybridAnatomyGuidedRelationalViT
    is_hybrid = isinstance(model, HybridAnatomyGuidedRelationalViT)

    model.eval()
    results = {}

    for sigma in noise_levels:
        tracker = MetricTracker(target_names)

        with torch.no_grad():
            for images, targets in tqdm(
                val_loader, desc=f"Noise σ={sigma:.2f}", leave=False
            ):
                images  = images.to(device)
                targets = targets.to(device)

                if sigma > 0:
                    noise  = torch.randn_like(images) * sigma
                    images = images + noise

                if is_hybrid:
                    logits = model(images, graph_bias=graph_bias, anatomy_prior=anatomy_prior)
                elif feasibility_matrix is not None:
                    logits = model(images, feasibility_matrix=feasibility_matrix)
                else:
                    logits = model(images)

                tracker.update(targets, logits)

        metrics = tracker.compute()
        results[sigma] = metrics["auroc"].get("mean", 0.0)

    return results


# ════════════════════════════════════════════════════════════
#  All-in-One Metrics Computer
# ════════════════════════════════════════════════════════════

class AttentionMetricsComputer:
    """
    Compute all 5 attention analysis metrics for a given model.

    Collects attention weights from the validation set and computes:
      1. Attention Entropy
      2. Attention Sparsity
      3. Anatomical Attention Ratio (AAR)
      4. Region Consistency Score (RCS)
      5. Noise Robustness

    Args:
        model:              ViT or ConstrainedVisionTransformer
        val_loader:         Validation DataLoader
        device:             Torch device (mps/cpu/cuda)
        F_hybrid:           (N+1, N+1) hybrid feasibility matrix (or None)
        F_anatomy:          (N+1, N+1) anatomy matrix for AAR computation
        anatomy_labels:     (N,) region labels for patches (from AnatomyPriorBuilder)
        target_names:       Pathology class names
        max_batches:        Max batches for attention collection (default 20)
    """

    def __init__(
        self,
        model:          nn.Module,
        val_loader,
        device:         torch.device,
        F_hybrid:       torch.Tensor = None,
        F_anatomy:      torch.Tensor = None,
        anatomy_labels: np.ndarray   = None,
        target_names:   list         = None,
        max_batches:    int          = 20,
        graph_bias:     torch.Tensor = None,
        anatomy_prior:  torch.Tensor = None,
    ):
        self.model          = model
        self.val_loader     = val_loader
        self.device         = device
        self.F_hybrid       = F_hybrid
        self.F_anatomy      = F_anatomy
        self.anatomy_labels = anatomy_labels
        self.target_names   = target_names or [
            "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"
        ]
        self.max_batches = max_batches
        self.graph_bias = graph_bias
        self.anatomy_prior = anatomy_prior

        # Detect model type
        from src.models.hybrid_relational_vit import HybridAnatomyGuidedRelationalViT
        self._is_hybrid = isinstance(model, HybridAnatomyGuidedRelationalViT)

    def _collect_attention(self) -> list:
        """
        Collect attention weights from first max_batches validation batches.

        Returns:
            Aggregated per-layer attention: list of (B_total, heads, N, N) tensors
        """
        self.model.eval()
        layer_attns = None

        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(self.val_loader):
                if batch_idx >= self.max_batches:
                    break
                images = images.to(self.device)

                if self._is_hybrid:
                    _, attn_list = self.model(
                        images,
                        graph_bias=self.graph_bias,
                        anatomy_prior=self.anatomy_prior,
                        return_attention=True,
                    )
                elif self.F_hybrid is not None:
                    _, attn_list = self.model(
                        images,
                        feasibility_matrix=self.F_hybrid,
                        return_attention=True,
                    )
                else:
                    _, attn_list = self.model(images, return_attention=True)

                if layer_attns is None:
                    layer_attns = [[aw.cpu()] for aw in attn_list]
                else:
                    for i, aw in enumerate(attn_list):
                        layer_attns[i].append(aw.cpu())

        if layer_attns is None:
            return []

        return [torch.cat(batches, dim=0) for batches in layer_attns]

    def compute_all(self, compute_noise: bool = True) -> dict:
        """
        Compute all 5 attention metrics.

        Args:
            compute_noise: Whether to run noise robustness (slower)

        Returns:
            dict with keys: entropy, sparsity, anatomical_attention_ratio,
                            region_consistency, noise_robustness
        """
        print(f"  Computing attention metrics (max {self.max_batches} batches)...")

        # Collect attention weights
        attn_list = self._collect_attention()

        results = {}

        # 1. Entropy
        print("  [1/5] Attention Entropy...")
        results["entropy"] = compute_attention_entropy(attn_list)

        # 2. Sparsity
        print("  [2/5] Attention Sparsity...")
        results["sparsity"] = compute_attention_sparsity(attn_list)

        # 3. Anatomical Attention Ratio
        if self.F_anatomy is not None and self.anatomy_labels is not None:
            print("  [3/5] Anatomical Attention Ratio...")
            results["anatomical_attention_ratio"] = compute_anatomical_attention_ratio(
                attn_list, self.anatomy_labels, self.F_anatomy
            )
        else:
            print("  [3/5] Anatomical Attention Ratio — skipped (no anatomy info)")
            results["anatomical_attention_ratio"] = {"mean": 0.0, "std": 0.0}

        # 4. Region Consistency Score
        if self.anatomy_labels is not None:
            print("  [4/5] Region Consistency Score...")
            results["region_consistency"] = compute_region_consistency_score(
                attn_list, self.anatomy_labels
            )
        else:
            print("  [4/5] Region Consistency — skipped (no anatomy labels)")
            results["region_consistency"] = {"mean": 0.0, "std": 0.0}

        # 5. Noise Robustness
        if compute_noise:
            print("  [5/5] Noise Robustness...")
            results["noise_robustness"] = compute_noise_robustness(
                self.model, self.val_loader, self.device,
                self.F_hybrid, target_names=self.target_names,
                graph_bias=self.graph_bias, anatomy_prior=self.anatomy_prior,
            )
        else:
            results["noise_robustness"] = {0.0: 0.0, 0.3: 0.0}

        # Summary
        print(f"\n  Attention Metrics Summary:")
        print(f"    Entropy:       {results['entropy']['mean']:.4f}")
        print(f"    Sparsity:      {results['sparsity']['mean']:.4f}")
        print(f"    Anatomy Ratio: {results['anatomical_attention_ratio']['mean']:.4f}")
        print(f"    Region Cons.:  {results['region_consistency']['mean']:.4f}")
        noise_03 = results["noise_robustness"].get(0.3, 0.0)
        print(f"    Noise AUROC:   {noise_03:.4f} (σ=0.3)")

        return results
