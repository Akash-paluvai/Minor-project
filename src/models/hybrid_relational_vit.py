"""
Hybrid Anatomy-Guided Relational Vision Transformer.

PRIMARY CONTRIBUTION — the headline model of this project.

Attention formulation:
    Attention = softmax(QK^T / √d  +  λ_graph · G  +  λ_anatomy · A)

Where:
    QK^T / √d    = standard scaled dot-product attention logits
    G            = graph relational bias (learned/precomputed from DINO features)
    A            = anatomy-aware spatial prior (7-region chest X-ray layout)
    λ_graph      = learnable scalar (nn.Parameter, init=0.1)
    λ_anatomy    = learnable scalar (nn.Parameter, init=0.1)

Key design principles:
    - SOFT additive biases, NOT hard masks
    - No binary blocking, no -inf constraints
    - Both λ terms are trainable — model learns how much structure to enforce
    - Initialized small to avoid early overconstraint
    - Preserves long-range interactions and attention entropy
    - CLS token is unconstrained (fully connected)

Research narrative:
    Previous experiments showed:
    - Hard constraints → entropy collapse, reduced AUROC
    - GNN relational approach → best performance (adaptive structure)
    - Soft guidance > rigid restriction

    This model combines the best of both:
    - Adaptive graph relations (from GNN insight)
    - Anatomy-aware spatial priors (domain knowledge)
    - Entropy-preserving soft attention (flexibility)

Tensor shapes (192×192 input, 16×16 patches):
    Input:         (B, 3, 192, 192)
    Patches:       (B, 144, D)        12×12 grid
    With CLS:      (B, 145, D)
    G, A:          (145, 145)          including CLS token
    Logits:        (B, 5)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit import PatchEmbedding, MLP


# ════════════════════════════════════════════════════════════
#  Hybrid Relational Multi-Head Attention
# ════════════════════════════════════════════════════════════

class HybridRelationalAttention(nn.Module):
    """
    Multi-Head Attention with dual soft relational biases.

    Implements:
        scores = QK^T / √d  +  λ_graph · G  +  λ_anatomy · A
        Attention = softmax(scores)

    Both G (graph) and A (anatomy) are pre-softmax additive biases that
    NUDGE attention rather than block it. This preserves long-range
    interactions while encouraging anatomically and relationally
    meaningful attention patterns.

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

        self.qkv          = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj         = nn.Linear(embed_dim, embed_dim, bias=True)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(
        self,
        x: torch.Tensor,
        graph_bias: torch.Tensor = None,
        anatomy_prior: torch.Tensor = None,
        lambda_graph: torch.Tensor = None,
        lambda_anatomy: torch.Tensor = None,
    ):
        """
        Forward pass with dual soft relational biases.

        Args:
            x:              (B, N, D) token embeddings
            graph_bias:     (N, N) graph relational bias — soft, values in [0,1]
            anatomy_prior:  (N, N) anatomy spatial prior — soft, values in [0,1]
            lambda_graph:   learnable scalar — graph bias strength
            lambda_anatomy: learnable scalar — anatomy bias strength

        Returns:
            output:       (B, N, D) attended embeddings
            attn_weights: (B, num_heads, N, N) post-softmax attention weights
        """
        B, N, D = x.shape

        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        # Scaled dot-product scores
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)

        # ── Soft Graph Relational Bias ──
        # G is NOT a hard mask. It's an additive bias that gently encourages
        # relationally meaningful attention patterns.
        if graph_bias is not None and lambda_graph is not None:
            G = graph_bias.to(x.device).float()
            # log-transform: maps [0,1] → (-inf, 0], preserving relative ordering
            # λ_graph scales the influence — small init means gentle guidance
            eps = 1e-6
            g_bias = lambda_graph * torch.log(G + eps)  # (N, N)
            attn_scores = attn_scores + g_bias.unsqueeze(0).unsqueeze(0)

        # ── Soft Anatomy Spatial Prior ──
        # A encodes spatial neighbourhood knowledge from chest X-ray anatomy.
        # Same-region patches get boost, distant regions get mild suppression.
        if anatomy_prior is not None and lambda_anatomy is not None:
            A = anatomy_prior.to(x.device).float()
            eps = 1e-6
            a_bias = lambda_anatomy * torch.log(A + eps)  # (N, N)
            attn_scores = attn_scores + a_bias.unsqueeze(0).unsqueeze(0)

        # Softmax + dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted value aggregation
        out = (attn_weights @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_dropout(out)

        return out, attn_weights


# ════════════════════════════════════════════════════════════
#  Hybrid Relational Transformer Block
# ════════════════════════════════════════════════════════════

class HybridRelationalTransformerBlock(nn.Module):
    """
    Transformer block with dual soft relational attention.

    Architecture:
        x → LayerNorm → HybridRelationalAttention(G, A) → Residual
          → LayerNorm → MLP → Residual
    """

    def __init__(
        self,
        embed_dim: int = 384,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = HybridRelationalAttention(
            embed_dim, num_heads, attention_dropout, proj_dropout=dropout
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp   = MLP(embed_dim, mlp_ratio, dropout)

    def forward(
        self,
        x: torch.Tensor,
        graph_bias: torch.Tensor = None,
        anatomy_prior: torch.Tensor = None,
        lambda_graph: torch.Tensor = None,
        lambda_anatomy: torch.Tensor = None,
    ):
        """
        Args:
            x:              (B, N, D)
            graph_bias:     (N, N) or None
            anatomy_prior:  (N, N) or None
            lambda_graph:   scalar tensor or None
            lambda_anatomy: scalar tensor or None

        Returns:
            x:            (B, N, D) updated embeddings
            attn_weights: (B, heads, N, N)
        """
        normed = self.norm1(x)
        attn_out, attn_weights = self.attn(
            normed, graph_bias, anatomy_prior, lambda_graph, lambda_anatomy
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights


# ════════════════════════════════════════════════════════════
#  Hybrid Anatomy-Guided Relational ViT (Main Model)
# ════════════════════════════════════════════════════════════

class HybridAnatomyGuidedRelationalViT(nn.Module):
    """
    Hybrid Anatomy-Guided Relational Vision Transformer.

    The PRIMARY PROPOSED MODEL — central contribution of this project.

    Combines:
        - Adaptive graph relational bias (G) from learned feature similarity
        - Anatomy-aware spatial prior (A) from domain knowledge
        - Entropy-preserving soft attention (no hard masking)
        - Learnable bias strengths (λ_graph, λ_anatomy)

    This model achieves the best balance between:
        - Performance (AUROC, F1)
        - Interpretability (anatomy-guided attention)
        - Flexibility (entropy preservation)
        - Anatomical plausibility

    Args:
        img_size:          Input image size (192 for M3 Pro optimization)
        patch_size:        Patch size (16)
        in_channels:       Input channels (3)
        num_classes:       Output classes (5 for CheXpert)
        embed_dim:         Embedding dimension
        depth:             Number of transformer blocks
        num_heads:         Number of attention heads
        mlp_ratio:         MLP hidden dimension ratio
        dropout:           Dropout rate
        attention_dropout: Attention dropout rate
        lambda_graph_init: Initial value for λ_graph (default 0.1)
        lambda_anatomy_init: Initial value for λ_anatomy (default 0.1)
    """

    def __init__(
        self,
        img_size: int = 192,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 5,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        lambda_graph_init: float = 0.1,
        lambda_anatomy_init: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim   = embed_dim
        self.depth       = depth
        self.img_size    = img_size

        # ── Patch embedding ──
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # ── CLS token + positional embeddings ──
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed   = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)

        # ── Learnable structure strengths ──
        # Initialized SMALL to avoid early overconstraint.
        # The model learns how much graph/anatomy guidance to use.
        self.lambda_graph   = nn.Parameter(torch.tensor(lambda_graph_init))
        self.lambda_anatomy = nn.Parameter(torch.tensor(lambda_anatomy_init))

        # ── Transformer blocks ──
        self.blocks = nn.ModuleList([
            HybridRelationalTransformerBlock(
                embed_dim, num_heads, mlp_ratio, dropout, attention_dropout
            )
            for _ in range(depth)
        ])

        # ── Classification head ──
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # ── Weight initialization ──
        self._init_weights()

    def _init_weights(self):
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

    def freeze_backbone(self, ratio: float = 0.7):
        """
        Freeze the first `ratio` fraction of transformer blocks.

        This is critical for M3 Pro efficiency: only fine-tune
        the last few layers + classification head.
        """
        n_freeze = int(self.depth * ratio)
        # Freeze patch embedding
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        # Freeze first n_freeze blocks
        for i in range(n_freeze):
            for param in self.blocks[i].parameters():
                param.requires_grad = False

        # λ parameters are ALWAYS trainable
        self.lambda_graph.requires_grad = True
        self.lambda_anatomy.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"  Frozen {n_freeze}/{self.depth} blocks, "
              f"trainable: {trainable:,}/{total:,} ({trainable/total:.1%})")

    def forward(
        self,
        x: torch.Tensor,
        graph_bias: torch.Tensor = None,
        anatomy_prior: torch.Tensor = None,
        return_attention: bool = False,
    ):
        """
        Forward pass with dual soft relational guidance.

        Args:
            x:                (B, C, H, W) input images
            graph_bias:       (N, N) graph relational bias matrix
            anatomy_prior:    (N, N) anatomy spatial prior matrix
            return_attention: if True, return attention weights from all layers

        Returns:
            logits:       (B, num_classes)
            attn_weights: list of (B, heads, N, N) if return_attention=True
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, D)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N, D) where N = num_patches + 1

        # Positional embedding
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        # Transformer blocks with dual soft biases
        all_attn_weights = []
        for block in self.blocks:
            x, attn_w = block(
                x,
                graph_bias=graph_bias,
                anatomy_prior=anatomy_prior,
                lambda_graph=self.lambda_graph,
                lambda_anatomy=self.lambda_anatomy,
            )
            if return_attention:
                all_attn_weights.append(attn_w)

        # Classification from CLS token
        x = self.norm(x)
        cls_output = x[:, 0]  # (B, D)
        logits = self.head(cls_output)  # (B, num_classes)

        if return_attention:
            return logits, all_attn_weights
        return logits

    def get_num_patches(self):
        """Return number of patches (excluding CLS)."""
        return self.patch_embed.num_patches

    def get_lambda_values(self):
        """Return current learned λ values for logging."""
        return {
            "lambda_graph": self.lambda_graph.item(),
            "lambda_anatomy": self.lambda_anatomy.item(),
        }

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "vit_small_patch16_224",
        num_classes: int = 5,
        img_size: int = 192,
        lambda_graph_init: float = 0.1,
        lambda_anatomy_init: float = 0.1,
        **kwargs,
    ):
        """
        Create model with pretrained weights from timm.

        Loads ViT backbone weights and interpolates positional embeddings
        for non-224 image sizes. λ parameters are initialized fresh.
        """
        import timm

        MODEL_CONFIGS = {
            "vit_tiny_patch16_224":  {"embed_dim": 192, "num_heads": 3,  "depth": 12},
            "vit_small_patch16_224": {"embed_dim": 384, "num_heads": 6,  "depth": 12},
            "vit_base_patch16_224":  {"embed_dim": 768, "num_heads": 12, "depth": 12},
        }

        cfg = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["vit_small_patch16_224"])
        cfg.update(kwargs)

        print(f"  Creating HybridAnatomyGuidedRelationalViT from '{model_name}', img_size={img_size}")
        model = cls(
            img_size=img_size, num_classes=num_classes,
            lambda_graph_init=lambda_graph_init,
            lambda_anatomy_init=lambda_anatomy_init,
            **cfg,
        )

        # Load pretrained weights
        try:
            pretrained = timm.create_model(model_name, pretrained=True, img_size=224)
        except Exception as e:
            print(f"  ⚠ timm pretrained load failed: {e}")
            return model

        pretrained_dict = pretrained.state_dict()
        model_dict      = model.state_dict()

        # Map matching weights
        mapped = {}
        for k, v in pretrained_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                mapped[k] = v

        # Interpolate positional embeddings for non-224 sizes
        if "pos_embed" in pretrained_dict and "pos_embed" in model_dict:
            pretrained_pos = pretrained_dict["pos_embed"]
            model_pos_shape = model_dict["pos_embed"].shape
            if pretrained_pos.shape != model_pos_shape:
                cls_pos   = pretrained_pos[:, :1, :]
                patch_pos = pretrained_pos[:, 1:, :]
                src_grid  = int(patch_pos.shape[1] ** 0.5)
                tgt_grid  = int((model_pos_shape[1] - 1) ** 0.5)
                patch_pos = patch_pos.reshape(1, src_grid, src_grid, -1).permute(0, 3, 1, 2)
                patch_pos = torch.nn.functional.interpolate(
                    patch_pos, size=(tgt_grid, tgt_grid), mode="bicubic", align_corners=False
                ).permute(0, 2, 3, 1).reshape(1, tgt_grid * tgt_grid, -1)
                mapped["pos_embed"] = torch.cat([cls_pos, patch_pos], dim=1)
                print(f"  Interpolated pos_embed: {src_grid}×{src_grid} → {tgt_grid}×{tgt_grid}")

        model_dict.update(mapped)
        model.load_state_dict(model_dict, strict=False)
        print(f"  ✓ Loaded {len(mapped)}/{len(model_dict)} pretrained weights")
        print(f"  λ_graph={lambda_graph_init}, λ_anatomy={lambda_anatomy_init} (learnable)")

        return model
