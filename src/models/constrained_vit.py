"""
Hybrid Anatomy-Aware Structure-Constrained Vision Transformer.

KEY CHANGE from v1 — Attention formulation:

    v1 (post-softmax multiplicative mask — hard/rigid):
        Attention = softmax(QK^T / √d) ⊙ F

    v2 (pre-softmax additive log-bias — soft/learnable):
        Attention = softmax(QK^T / √d + λ · log(F_hybrid + ε))

    Where:
        F_hybrid = α · F_anatomy + (1-α) · F_feature   ∈ R^(N+1 × N+1)
        λ = self.lambda_structure  (learnable nn.Parameter, init=0.1)
        ε = 1e-6  (numerical stability)

    Why this is better:
        - Log-bias nudges attention rather than zeroing it (soft constraint)
        - λ is learnable: model decides how much structure to enforce
        - Long-range interactions suppressed but not eliminated
        - Prevents attention collapse that multiplicative masking can cause

Tensor shape reference:
    Input:            (B, 3, 192, 192)
    After patch embed: (B, 144, embed_dim)     # 12×12 patches for 192px
    With CLS token:   (B, 145, embed_dim)      # token 0 = CLS
    Q, K, V:          (B, num_heads, 145, head_dim)
    Attention scores:  (B, num_heads, 145, 145)
    structural_bias:   (145, 145) broadcast to all batch/heads
    F_hybrid:          (145, 145)
    Logits:            (B, 5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F_functional

from .vit import PatchEmbedding, MLP


# ════════════════════════════════════════════════════════════
#  Soft Structure-Constrained Multi-Head Attention
# ════════════════════════════════════════════════════════════

class SoftConstrainedMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with soft structural bias via additive log-prior.

    Implements the NEW constrained attention formula:

        Attention = softmax(QK^T / √d  +  λ · log(F_hybrid + ε))

    Where:
        Q, K, V      = query, key, value projections   (B, heads, N, head_dim)
        d            = head dimension
        F_hybrid     = hybrid feasibility matrix        (N, N)  values in [0,1]
        λ            = learnable scalar (self.lambda_structure in parent model)
        ε = 1e-6     = numerical stability constant

    Properties:
        - When F_hybrid[i,j] = 1.0  → log-bias = 0    → no change to attention
        - When F_hybrid[i,j] = 0.5  → log-bias ≈ -0.7λ → mild suppression
        - When F_hybrid[i,j] = 0.05 → log-bias ≈ -3.0λ → strong suppression
        - F_hybrid[0,:] = 1.0       → CLS token unconstrained

    Args:
        embed_dim:         Total embedding dimension
        num_heads:         Number of attention heads
        attention_dropout: Dropout on attention weights
        proj_dropout:      Dropout on output projection
    """

    def __init__(
        self,
        embed_dim: int = 384,
        num_heads: int = 6,
        attention_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        # Fused QKV projection: (B, N, D) → (B, N, 3D)
        self.qkv          = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj         = nn.Linear(embed_dim, embed_dim, bias=True)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(
        self,
        x: torch.Tensor,
        feasibility_matrix: torch.Tensor = None,
        lambda_structure:   torch.Tensor = None,
    ):
        """
        Forward pass with optional soft structural bias.

        Args:
            x:                  (B, N, D) token embeddings, N = 1 + num_patches
            feasibility_matrix: (N, N) float32, values in [0,1]
                                If None → standard unconstrained attention
            lambda_structure:   scalar tensor — learnable bias strength
                                If None → bias strength = 0

        Returns:
            output:       (B, N, D) attended embeddings
            attn_weights: (B, num_heads, N, N) attention weights (post-softmax)
        """
        B, N, D = x.shape

        # ── Step 1: QKV projections ──
        # (B, N, 3D) → reshape → (3, B, heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)      # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)               # each (B, heads, N, head_dim)

        # ── Step 2: Scaled dot-product scores ──
        # scores: (B, heads, N, N)
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale

        # ── Step 3: Structural bias (pre-softmax additive) ──
        if feasibility_matrix is not None and lambda_structure is not None:
            # F: (N, N) → ensure float32 on same device
            F = feasibility_matrix.to(x.device).float()

            # Compute log-bias: λ · log(F + ε)
            # Shape: (N, N) → broadcast to (1, 1, N, N) for all batches/heads
            epsilon = 1e-6
            log_bias = lambda_structure * torch.log(F + epsilon)   # (N, N)
            log_bias = log_bias.unsqueeze(0).unsqueeze(0)           # (1, 1, N, N)

            attn_scores = attn_scores + log_bias   # (B, heads, N, N)

        # ── Step 4: Softmax + dropout ──
        attn_weights = F_functional.softmax(attn_scores, dim=-1)  # (B, heads, N, N)
        attn_weights = self.attn_dropout(attn_weights)

        # ── Step 5: Weighted value aggregation ──
        # (B, heads, N, N) @ (B, heads, N, head_dim) → (B, heads, N, head_dim)
        out = (attn_weights @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_dropout(out)

        return out, attn_weights


# ════════════════════════════════════════════════════════════
#  Constrained Transformer Block
# ════════════════════════════════════════════════════════════

class ConstrainedTransformerBlock(nn.Module):
    """
    Transformer block with soft structure-constrained attention.

    Architecture:
        x → LayerNorm → SoftConstrainedMHA → Residual
          → LayerNorm → MLP → Residual
    """

    def __init__(
        self,
        embed_dim:         int   = 384,
        num_heads:         int   = 6,
        mlp_ratio:         float = 4.0,
        dropout:           float = 0.0,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = SoftConstrainedMultiHeadAttention(
            embed_dim, num_heads, attention_dropout, proj_dropout=dropout
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp   = MLP(embed_dim, mlp_ratio, dropout)

    def forward(
        self,
        x:                  torch.Tensor,
        feasibility_matrix: torch.Tensor = None,
        lambda_structure:   torch.Tensor = None,
    ):
        """
        Args:
            x:                  (B, N, D)
            feasibility_matrix: (N, N) or None
            lambda_structure:   scalar tensor or None

        Returns:
            x:            (B, N, D) updated embeddings
            attn_weights: (B, heads, N, N)
        """
        # Pre-norm → constrained attention → residual
        normed   = self.norm1(x)
        attn_out, attn_weights = self.attn(normed, feasibility_matrix, lambda_structure)
        x = x + attn_out

        # Pre-norm → MLP → residual
        x = x + self.mlp(self.norm2(x))

        return x, attn_weights


# ════════════════════════════════════════════════════════════
#  Hybrid Anatomy-Aware Structure-Constrained ViT
# ════════════════════════════════════════════════════════════

class ConstrainedVisionTransformer(nn.Module):
    """
    Vision Transformer with hybrid anatomy-aware soft structural attention bias.

    Core attention formula:
        Attention = softmax(QK^T / √d + λ · log(F_hybrid + ε))

    Where F_hybrid = α · F_anatomy + (1-α) · F_feature

    Key features vs v1:
        1. Soft log-bias (additive, pre-softmax) instead of hard masking
        2. Learnable λ_structure parameter (starts weak, adapts)
        3. Backbone freezing support (freeze first freeze_ratio% of blocks)
        4. Compatible with ViT-Tiny / ViT-Small / ViT-Base from timm
        5. Correct positional embedding for 192px (144 patches) and 224px (196 patches)

    Args:
        img_size:         Input image size (192 for M3 Pro, 224 for full)
        patch_size:       Patch size — must be 16 for timm pretrained models
        in_channels:      Input channels (3 for RGB)
        num_classes:      Output classes (5 for CheXpert)
        embed_dim:        Embedding dimension (384 small, 192 tiny, 768 base)
        depth:            Number of transformer blocks (12 standard)
        num_heads:        Attention heads (6 small, 3 tiny, 12 base)
        mlp_ratio:        MLP expansion ratio (4.0)
        dropout:          Dropout rate
        attention_dropout: Attention dropout rate
        lambda_init:      Initial value of learnable λ_structure (default 0.1)
    """

    def __init__(
        self,
        img_size:          int   = 192,
        patch_size:        int   = 16,
        in_channels:       int   = 3,
        num_classes:       int   = 5,
        embed_dim:         int   = 384,
        depth:             int   = 12,
        num_heads:         int   = 6,
        mlp_ratio:         float = 4.0,
        dropout:           float = 0.1,
        attention_dropout: float = 0.0,
        lambda_init:       float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim   = embed_dim
        self.depth       = depth
        self.img_size    = img_size

        # Patch embedding: (B,C,H,W) → (B, num_patches, embed_dim)
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        # For 192px/16px: num_patches = (192//16)^2 = 12^2 = 144
        # For 224px/16px: num_patches = (224//16)^2 = 14^2 = 196

        # Learnable CLS token: prepended before patches
        # Shape during forward: (B, 1, embed_dim) → expanded to batch
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embeddings: (1, num_patches+1, embed_dim)
        # +1 for CLS token at index 0
        self.pos_embed   = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)

        # ── Learnable structural constraint strength ──
        # λ_structure: scalar nn.Parameter initialized to lambda_init (e.g. 0.1)
        # Training adjusts this: stronger → more anatomy guidance,
        #                        weaker  → freer attention
        self.lambda_structure = nn.Parameter(torch.tensor(float(lambda_init)))

        # Stack of constrained transformer blocks
        self.blocks = nn.ModuleList([
            ConstrainedTransformerBlock(
                embed_dim, num_heads, mlp_ratio, dropout, attention_dropout
            )
            for _ in range(depth)
        ])

        # Classification head: CLS token → logits
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following ViT best practices."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module_weights)

    @staticmethod
    def _init_module_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def freeze_backbone(self, freeze_ratio: float = 0.7):
        """
        Freeze patch embedding and the first (freeze_ratio * depth) blocks.

        This dramatically reduces:
          - Memory usage (no gradients for frozen params)
          - Training instability
          - Compute time per step

        Args:
            freeze_ratio: Fraction of transformer blocks to freeze (default 0.7 = 70%)
                          e.g., depth=12, ratio=0.7 → freeze blocks 0-7, train 8-11

        What stays trainable:
          - Last (1-freeze_ratio) transformer blocks
          - Classification head (self.head + self.norm)
          - self.lambda_structure (learnable constraint strength)
          - Positional embeddings (self.pos_embed, self.cls_token)
        """
        n_freeze = int(self.depth * freeze_ratio)

        # Freeze patch embedding
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        print(f"  Frozen: patch_embed")

        # Freeze first n_freeze transformer blocks
        for i, block in enumerate(self.blocks):
            if i < n_freeze:
                for param in block.parameters():
                    param.requires_grad = False

        n_train = self.depth - n_freeze
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(
            f"  Frozen: blocks 0-{n_freeze-1} / {self.depth}"
            f"  Trainable: blocks {n_freeze}-{self.depth-1}, head, λ"
        )
        print(f"  Trainable params: {trainable:,} / {total:,} ({trainable/total:.1%})")

    def forward(
        self,
        x:                  torch.Tensor,
        feasibility_matrix: torch.Tensor = None,
        return_attention:   bool = False,
    ):
        """
        Forward pass with optional soft structural bias.

        Args:
            x:                  (B, C, H, W) input images
            feasibility_matrix: (N+1, N+1) hybrid feasibility matrix in [0, 1]
                                If None → unconstrained attention (baseline mode)
            return_attention:   If True, return per-layer attention weights

        Returns:
            logits:       (B, num_classes) classification logits
            attn_weights: list of (B, heads, N+1, N+1) — only if return_attention=True
        """
        B = x.shape[0]

        # ── Patch embedding: (B,C,H,W) → (B, num_patches, embed_dim) ──
        x = self.patch_embed(x)   # (B, 144, embed_dim) for 192px

        # ── Prepend CLS token: (B, 1, D) cat → (B, 145, D) ──
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # ── Add positional embeddings ──
        x = x + self.pos_embed    # (B, 145, embed_dim)
        x = self.pos_dropout(x)

        # ── Transformer blocks with structural bias ──
        all_attn_weights = []
        for block in self.blocks:
            x, attn_w = block(x, feasibility_matrix, self.lambda_structure)
            if return_attention:
                all_attn_weights.append(attn_w)

        # ── Classification from CLS token (index 0) ──
        x         = self.norm(x)
        cls_output = x[:, 0]           # (B, embed_dim)
        logits     = self.head(cls_output)  # (B, num_classes)

        if return_attention:
            return logits, all_attn_weights
        return logits

    def get_num_patches(self) -> int:
        """Return number of patches (excluding CLS token)."""
        return self.patch_embed.num_patches

    def get_lambda(self) -> float:
        """Return current value of learnable structural constraint strength."""
        return self.lambda_structure.item()

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "vit_small_patch16_224",
        num_classes: int = 5,
        img_size: int = 192,
        lambda_init: float = 0.1,
        **kwargs,
    ):
        """
        Create a ConstrainedVisionTransformer with pretrained weights from timm.

        Supported models:
            vit_tiny_patch16_224    embed_dim=192, num_heads=3,  depth=12
            vit_small_patch16_224   embed_dim=384, num_heads=6,  depth=12
            vit_base_patch16_224    embed_dim=768, num_heads=12, depth=12

        Positional embeddings are interpolated if img_size differs from training size.

        Args:
            model_name:  timm model identifier
            num_classes: Number of output classes
            img_size:    Target image size (192 for M3 Pro optimized)
            lambda_init: Initial structural bias strength

        Returns:
            model: ConstrainedVisionTransformer with pretrained weights loaded
        """
        import timm

        # Map timm model configs
        MODEL_CONFIGS = {
            "vit_tiny_patch16_224":  {"embed_dim": 192, "num_heads": 3,  "depth": 12},
            "vit_small_patch16_224": {"embed_dim": 384, "num_heads": 6,  "depth": 12},
            "vit_base_patch16_224":  {"embed_dim": 768, "num_heads": 12, "depth": 12},
        }

        cfg = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["vit_small_patch16_224"])
        cfg.update(kwargs)

        print(f"  Creating ConstrainedViT from '{model_name}'")
        print(f"    img_size={img_size}, embed_dim={cfg['embed_dim']}, "
              f"num_heads={cfg['num_heads']}, depth={cfg['depth']}")

        # Create our model with correct arch params
        model = cls(
            img_size=img_size,
            num_classes=num_classes,
            lambda_init=lambda_init,
            **cfg,
        )

        # Load pretrained timm model (trained on 224px)
        try:
            pretrained = timm.create_model(
                model_name, pretrained=True, img_size=224,
            )
        except Exception as e:
            print(f"  ⚠ timm pretrained load failed: {e}")
            print(f"  → Continuing with random initialization")
            return model

        # ── Copy matching weights ──
        pretrained_dict = pretrained.state_dict()
        model_dict      = model.state_dict()

        mapped = {}
        skipped = []
        for k, v in pretrained_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                mapped[k] = v
            elif k in model_dict:
                skipped.append(f"{k}: pretrained={v.shape} vs model={model_dict[k].shape}")

        # ── Handle positional embedding size mismatch (224→192) ──
        # timm pos_embed: (1, 197, D) for 224px → need (1, 145, D) for 192px
        if "pos_embed" in pretrained_dict and "pos_embed" in model_dict:
            pretrained_pos = pretrained_dict["pos_embed"]   # (1, 197, D)
            model_pos_size = model_dict["pos_embed"].shape   # (1, 145, D)

            if pretrained_pos.shape != model_pos_size:
                print(f"  Interpolating pos_embed: {list(pretrained_pos.shape)} → {list(model_pos_size)}")
                # Separate CLS and patch embeddings
                cls_pos    = pretrained_pos[:, :1, :]           # (1, 1, D)
                patch_pos  = pretrained_pos[:, 1:, :]           # (1, 196, D)

                # Compute target grid sizes
                src_grid = int(patch_pos.shape[1] ** 0.5)       # 14
                tgt_grid = int((model_pos_size[1] - 1) ** 0.5)  # 12

                # Reshape and interpolate: (1, D, 14, 14) → (1, D, 12, 12)
                patch_pos = patch_pos.reshape(1, src_grid, src_grid, -1)
                patch_pos = patch_pos.permute(0, 3, 1, 2)        # (1, D, 14, 14)
                patch_pos = torch.nn.functional.interpolate(
                    patch_pos, size=(tgt_grid, tgt_grid), mode="bicubic", align_corners=False
                )                                                  # (1, D, 12, 12)
                patch_pos = patch_pos.permute(0, 2, 3, 1)         # (1, 12, 12, D)
                patch_pos = patch_pos.reshape(1, tgt_grid * tgt_grid, -1)  # (1, 144, D)

                # Concatenate CLS back
                interp_pos_embed = torch.cat([cls_pos, patch_pos], dim=1)  # (1, 145, D)
                mapped["pos_embed"] = interp_pos_embed

        model_dict.update(mapped)
        model.load_state_dict(model_dict, strict=False)

        n_total = len(model_dict)
        n_loaded = len(mapped)
        print(f"  ✓ Loaded {n_loaded}/{n_total} weights from {model_name}")
        if skipped:
            print(f"  ⚠ Skipped {len(skipped)} mismatched keys")

        return model
