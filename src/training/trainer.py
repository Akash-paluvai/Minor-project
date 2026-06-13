"""
MPS-Optimized Trainer for Hybrid Anatomy-Aware SC-ViT.

Key changes from v1:
  - MPS-first device selection (no CUDA requirement)
  - AMP disabled on MPS (use_amp=False when device=mps)
  - Gradient accumulation support (grad_accum_steps)
  - No GradScaler on MPS
  - pin_memory=False enforced
  - Backbone freezing integrated
  - Logs lambda_structure value each epoch
  - torch.compile() disabled

Supports both:
  - VisionTransformer (baseline, no feasibility matrix)
  - ConstrainedVisionTransformer (constrained, with feasibility matrix)
"""

import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm

from .metrics import MetricTracker


class Trainer:
    """
    Training loop for multi-label CheXpert classification.

    Supports both standard ViT and Hybrid Anatomy-Aware Constrained ViT.

    MPS optimizations:
        - AMP disabled (MPS AMP is unstable)
        - No GradScaler
        - gradient accumulation for effective larger batch sizes
        - Backbone freezing reduces memory/compute

    Args:
        model:              VisionTransformer or ConstrainedVisionTransformer
        train_loader:       Training DataLoader (pin_memory=False)
        val_loader:         Validation DataLoader
        config:             Parsed YAML config dict
        feasibility_matrix: Optional (N+1, N+1) matrix for constrained ViT
        target_names:       List of pathology names
        model_tag:          Short name for logging (e.g., 'hybrid', 'baseline')
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config: dict,
        feasibility_matrix: torch.Tensor = None,
        target_names: list = None,
        model_tag: str = "model",
    ):
        self.model              = model
        self.train_loader       = train_loader
        self.val_loader         = val_loader
        self.config             = config
        self.feasibility_matrix = feasibility_matrix
        self.model_tag          = model_tag
        self.target_names       = target_names or [
            "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"
        ]

        # Detect model type for forward pass routing
        from src.models.hybrid_relational_vit import HybridAnatomyGuidedRelationalViT
        self._is_hybrid_relational = isinstance(model, HybridAnatomyGuidedRelationalViT)

        # For hybrid relational: split feasibility into graph_bias + anatomy_prior
        self._graph_bias = None
        self._anatomy_prior = None
        if self._is_hybrid_relational and feasibility_matrix is not None:
            # The trainer receives F_hybrid; we also need separate F_anatomy
            # Load from cache if available
            import os as _os
            cache_dir = config.get("structure", {}).get("cache_dir", "outputs/cache")
            for key, attr in [("F_feature", "_graph_bias"), ("F_anatomy", "_anatomy_prior")]:
                fpath = _os.path.join(cache_dir, f"{key}.pt")
                if _os.path.exists(fpath):
                    data = torch.load(fpath, map_location="cpu", weights_only=False)
                    setattr(self, attr, data["feasibility_matrix"])
            # Fallback: use feasibility_matrix as graph_bias
            if self._graph_bias is None:
                self._graph_bias = feasibility_matrix
            if self._anatomy_prior is None:
                from src.structure.anatomy_prior import AnatomyPriorBuilder
                img_size = config["data"].get("image_size", 192)
                self._anatomy_prior = AnatomyPriorBuilder().build(img_size // 16)

        train_cfg = config["training"]
        log_cfg   = config.get("logging", {})
        sys_cfg   = config.get("system", {})

        # ── Device: MPS-first, then CPU ──
        # IMPORTANT: Never use CUDA-only paths on M3 Pro
        if torch.backends.mps.is_available() and sys_cfg.get("use_mps", True):
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print(f"  Device: {self.device}")
        self.model = self.model.to(self.device)

        # Move feasibility matrix to device (stays there for entire training)
        if self.feasibility_matrix is not None:
            self.feasibility_matrix = self.feasibility_matrix.to(self.device)

        # ── Backbone freezing ──
        if train_cfg.get("freeze_backbone", False):
            freeze_ratio = train_cfg.get("freeze_ratio", 0.7)
            if hasattr(self.model, "freeze_backbone"):
                self.model.freeze_backbone(freeze_ratio)

        # ── Loss: Weighted BCEWithLogitsLoss for multi-label ──
        # CheXpert is heavily imbalanced (most labels are negative).
        # pos_weight = N_negative / N_positive per class compensates for this.
        pos_weight = self._compute_pos_weight(train_loader)
        if pos_weight is not None:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device))
            print(f"  Using weighted BCE: pos_weight={[f'{w:.2f}' for w in pos_weight.tolist()]}")
        else:
            self.criterion = nn.BCEWithLogitsLoss()
            print(f"  Using unweighted BCE (could not compute pos_weight)")

        # ── Optimizer: AdamW on trainable params only ──
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=train_cfg.get("learning_rate", 1e-4),
            weight_decay=train_cfg.get("weight_decay", 1e-4),
        )

        # ── LR Scheduler ──
        self.epochs       = train_cfg.get("epochs", 10)
        self.warmup_epochs = train_cfg.get("warmup_epochs", 2)

        if train_cfg.get("scheduler", "cosine") == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(self.epochs - self.warmup_epochs, 1),
                eta_min=1e-6,
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=5, gamma=0.5
            )

        # ── AMP: DISABLED on MPS (unstable), optional on CUDA only ──
        self.use_amp = (
            train_cfg.get("use_amp", False) and
            self.device.type == "cuda"
        )
        # GradScaler only for CUDA AMP
        if self.use_amp:
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None

        self.gradient_clip = train_cfg.get("gradient_clip", 1.0)

        # ── Gradient accumulation ──
        # Effective batch = batch_size * grad_accum_steps
        self.grad_accum_steps = train_cfg.get("grad_accum_steps", 2)

        # ── Logging dirs ──
        self.log_dir       = log_cfg.get("log_dir", f"outputs/logs/{model_tag}")
        self.checkpoint_dir = log_cfg.get("checkpoint_dir", "outputs/checkpoints")
        self.save_every    = log_cfg.get("save_every", 5)
        self.log_interval  = log_cfg.get("log_interval", 50)

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # TensorBoard (optional, skipped if not available)
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.log_dir)
            self._use_tb = True
        except ImportError:
            self.writer  = None
            self._use_tb = False

        # ── Metric trackers ──
        self.train_metrics = MetricTracker(self.target_names)
        self.val_metrics   = MetricTracker(self.target_names)

        self.best_auroc  = 0.0
        self.global_step = 0

    # ─────────────────────────────────────────────────────────
    #  Pos-Weight Computation for Imbalanced BCE
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _compute_pos_weight(train_loader, max_batches=100):
        """
        Compute per-class pos_weight = N_negative / N_positive from training data.

        This is critical for CheXpert: most labels are negative, so without
        pos_weight the model learns to always predict low probabilities.

        Returns:
            torch.Tensor of shape (num_classes,) or None if computation fails
        """
        try:
            pos_counts = None
            total = 0

            for batch_idx, (_, targets) in enumerate(train_loader):
                if batch_idx >= max_batches:
                    break
                # targets: (B, C) with values 0/1
                if pos_counts is None:
                    pos_counts = torch.zeros(targets.shape[1])
                pos_counts += (targets > 0.5).float().sum(dim=0)
                total += targets.shape[0]

            if pos_counts is None or total == 0:
                return None

            neg_counts = total - pos_counts
            # pos_weight = N_neg / N_pos, clamped to avoid extreme values
            pos_weight = neg_counts / (pos_counts + 1e-6)
            pos_weight = pos_weight.clamp(min=1.0, max=50.0)

            print(f"  Class balance (from {total} samples):")
            for i, (p, n) in enumerate(zip(pos_counts, neg_counts)):
                print(f"    Class {i}: {int(p)} pos / {int(n)} neg → weight={pos_weight[i]:.2f}")

            return pos_weight
        except Exception as e:
            print(f"  ⚠ Could not compute pos_weight: {e}")
            return None

    # ─────────────────────────────────────────────────────────
    #  Public Training Interface
    # ─────────────────────────────────────────────────────────

    def train(self) -> float:
        """
        Run the full training loop.

        Returns:
            best_auroc: Best validation AUROC achieved over all epochs
        """
        base_lr = self.config["training"].get("learning_rate", 1e-4)
        print(f"\n{'=' * 60}")
        print(f"  Training: {self.model_tag}")
        print(f"{'=' * 60}")
        print(f"  Device:         {self.device}")
        print(f"  Epochs:         {self.epochs}")
        print(f"  Batch size:     {self.config['training']['batch_size']}")
        print(f"  Grad accum:     {self.grad_accum_steps} "
              f"(eff. batch: {self.config['training']['batch_size'] * self.grad_accum_steps})")
        print(f"  LR:             {base_lr:.2e}")
        print(f"  AMP:            {self.use_amp}")
        print(f"  Constrained:    {self.feasibility_matrix is not None}")
        print(f"  Model type:     {'hybrid_relational' if self._is_hybrid_relational else 'standard'}")
        has_lambda = hasattr(self.model, "lambda_structure")
        has_dual_lambda = hasattr(self.model, "lambda_graph")
        if has_dual_lambda:
            print(f"  λ_graph:        {self.model.lambda_graph.item():.4f} (learnable)")
            print(f"  λ_anatomy:      {self.model.lambda_anatomy.item():.4f} (learnable)")
        elif has_lambda:
            print(f"  λ_structure:    {self.model.lambda_structure.item():.4f} (learnable)")
        print(f"{'=' * 60}\n")

        t_start = time.time()

        for epoch in range(1, self.epochs + 1):
            t_epoch = time.time()

            # Warmup: linear LR ramp
            if epoch <= self.warmup_epochs:
                warmup_factor = epoch / max(self.warmup_epochs, 1)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = base_lr * warmup_factor

            # Train one epoch
            train_loss = self._train_epoch(epoch)

            # Validate
            val_loss, val_results = self._validate_epoch(epoch)

            # Step scheduler (after warmup)
            if epoch > self.warmup_epochs:
                self.scheduler.step()

            mean_auroc = val_results.get("auroc", {}).get("mean", 0.0)
            lr_now     = self.optimizer.param_groups[0]["lr"]
            epoch_sec  = time.time() - t_epoch

            # TensorBoard
            if self._use_tb:
                self.writer.add_scalar(f"{self.model_tag}/train_loss", train_loss, epoch)
                self.writer.add_scalar(f"{self.model_tag}/val_loss",   val_loss,   epoch)
                self.writer.add_scalar(f"{self.model_tag}/val_auroc",  mean_auroc, epoch)
                self.writer.add_scalar(f"{self.model_tag}/lr",         lr_now,     epoch)
                if has_dual_lambda:
                    self.writer.add_scalar(
                        f"{self.model_tag}/lambda_graph",
                        self.model.lambda_graph.item(), epoch
                    )
                    self.writer.add_scalar(
                        f"{self.model_tag}/lambda_anatomy",
                        self.model.lambda_anatomy.item(), epoch
                    )
                elif has_lambda:
                    self.writer.add_scalar(
                        f"{self.model_tag}/lambda_structure",
                        self.model.lambda_structure.item(), epoch
                    )

            # Console log
            lambda_str = ""
            if has_dual_lambda:
                lambda_str = (f"  λ_graph: {self.model.lambda_graph.item():.4f}  "
                              f"λ_anatomy: {self.model.lambda_anatomy.item():.4f}\n")
            elif has_lambda:
                lambda_str = f"  λ_structure: {self.model.lambda_structure.item():.4f}\n"
            print(
                f"\nEpoch {epoch}/{self.epochs} ({epoch_sec:.0f}s)\n"
                f"  Train Loss: {train_loss:.4f}\n"
                f"  Val Loss:   {val_loss:.4f}\n"
                f"  Val AUROC:  {mean_auroc:.4f}\n"
                f"  LR:         {lr_now:.2e}\n"
                f"{lambda_str}"
            )

            # Per-class AUROC
            for name in self.target_names:
                score = val_results.get("auroc", {}).get(name, 0.0)
                print(f"    {name}: {score:.4f}")

            # Checkpoint
            if mean_auroc > self.best_auroc:
                self.best_auroc = mean_auroc
                self._save_checkpoint(epoch, is_best=True)
                print(f"  ★ New best AUROC: {mean_auroc:.4f}")

            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch, is_best=False)

        total_min = (time.time() - t_start) / 60
        print(f"\n{'=' * 60}")
        print(f"  Training complete: {total_min:.1f} min")
        print(f"  Best Val AUROC: {self.best_auroc:.4f}")
        print(f"{'=' * 60}\n")

        if self._use_tb:
            self.writer.close()

        return self.best_auroc

    # ─────────────────────────────────────────────────────────
    #  Private: Train / Validate Epochs
    # ─────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch with gradient accumulation."""
        self.model.train()
        self.train_metrics.reset()

        total_loss = 0.0
        num_steps  = 0

        self.optimizer.zero_grad()  # zero at start of epoch

        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {epoch} [Train]",
            dynamic_ncols=True,
        )

        for batch_idx, (images, targets) in pbar:
            images  = images.to(self.device)
            targets = targets.to(self.device)

            # ── Forward pass (AMP on CUDA only) ──
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    logits = self._forward(images)
                    loss   = self.criterion(logits, targets) / self.grad_accum_steps
                self.scaler.scale(loss).backward()
            else:
                logits = self._forward(images)
                loss   = self.criterion(logits, targets) / self.grad_accum_steps
                loss.backward()

            # ── Gradient accumulation step ──
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.gradient_clip > 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

            # Track (use unscaled loss for display)
            actual_loss = loss.item() * self.grad_accum_steps
            total_loss += actual_loss
            num_steps  += 1
            self.train_metrics.update(targets, logits)
            self.global_step += 1

            if batch_idx % self.log_interval == 0:
                if self._use_tb:
                    self.writer.add_scalar("step/train_loss", actual_loss, self.global_step)
                pbar.set_postfix(loss=f"{actual_loss:.4f}")

        # Handle remaining gradients if total steps not divisible by accum_steps
        if len(self.train_loader) % self.grad_accum_steps != 0:
            if self.gradient_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            if self.use_amp and self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()

        return total_loss / max(num_steps, 1)

    @torch.no_grad()
    def _validate_epoch(self, epoch: int):
        """Validate for one epoch."""
        self.model.eval()
        self.val_metrics.reset()

        total_loss = 0.0
        num_steps  = 0

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]", dynamic_ncols=True)

        for images, targets in pbar:
            images  = images.to(self.device)
            targets = targets.to(self.device)

            logits = self._forward(images)
            loss   = self.criterion(logits, targets)

            total_loss += loss.item()
            num_steps  += 1
            self.val_metrics.update(targets, logits)

        val_loss = total_loss / max(num_steps, 1)
        results  = self.val_metrics.compute()

        if self._use_tb:
            for metric_name, values in results.items():
                for cls_name, score in values.items():
                    self.writer.add_scalar(
                        f"{self.model_tag}/val/{metric_name}/{cls_name}", score, epoch
                    )

        return val_loss, results

    def _forward(self, images: torch.Tensor) -> torch.Tensor:
        """Run forward pass, routing to correct model API."""
        if self._is_hybrid_relational:
            return self.model(
                images,
                graph_bias=self._graph_bias,
                anatomy_prior=self._anatomy_prior,
            )
        elif self.feasibility_matrix is not None:
            return self.model(images, feasibility_matrix=self.feasibility_matrix)
        return self.model(images)

    # ─────────────────────────────────────────────────────────
    #  Checkpointing
    # ─────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch":               epoch,
            "model_tag":           self.model_tag,
            "model_state_dict":    self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_auroc":          self.best_auroc,
            "config":              self.config,
        }

        tag = f"{self.model_tag}_"
        fname = f"{tag}best_model.pt" if is_best else f"{tag}checkpoint_epoch_{epoch}.pt"
        path  = os.path.join(self.checkpoint_dir, fname)

        torch.save(checkpoint, path)
        print(f"  Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load a checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_auroc = checkpoint.get("best_auroc", 0.0)
        print(f"  ✓ Loaded checkpoint: {path} (epoch {checkpoint['epoch']})")
        return checkpoint["epoch"]
