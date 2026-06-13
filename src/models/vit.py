"""
Vision Transformer (ViT) — Baseline Implementation.

A clean ViT-B/16 implementation for multi-label classification,
with optional loading of pretrained weights from timm.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """
    Convert image into a sequence of patch embeddings using a Conv2d projection.

    For a 192×192 image with 16×16 patches → 12×12 = 144 patches (M3 Pro optimized).
    For a 224×224 image with 16×16 patches → 14×14 = 196 patches (standard).
    """

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W) → (B, embed_dim, H/P, W/P)
        x = self.projection(x)
        # Flatten spatial dims → (B, embed_dim, num_patches) → (B, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention with scaled dot-product.

    Supports optional attention mask for constrained variants.
    """

    def __init__(self, embed_dim=768, num_heads=12, attention_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, x, attention_mask=None):
        """
        Args:
            x: (B, N, D) input token embeddings
            attention_mask: optional (N, N) additive mask (0 = allow, -inf = block)

        Returns:
            output: (B, N, D)
            attn_weights: (B, num_heads, N, N)
        """
        B, N, D = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)

        # Apply attention mask if provided
        if attention_mask is not None:
            attn = attn + attention_mask.unsqueeze(0).unsqueeze(0)

        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum
        out = (attn_weights @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_dropout(out)

        return out, attn_weights


class MLP(nn.Module):
    """Feed-forward network in transformer block."""

    def __init__(self, embed_dim=768, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single transformer encoder block: LayerNorm → MHA → LayerNorm → MLP.
    """

    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0,
                 dropout=0.0, attention_dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(
            embed_dim, num_heads, attention_dropout, proj_dropout=dropout
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x, attention_mask=None):
        """
        Returns:
            x: (B, N, D) output embeddings
            attn_weights: (B, heads, N, N) attention weights
        """
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, attn_weights = self.attn(normed, attention_mask)
        x = x + attn_out

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x, attn_weights


class VisionTransformer(nn.Module):
    """
    Vision Transformer for multi-label classification.

    Args:
        img_size: Input image size (default 224)
        patch_size: Patch size (default 16)
        in_channels: Number of input channels
        num_classes: Number of output classes
        embed_dim: Embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim ratio
        dropout: Dropout rate
        attention_dropout: Attention dropout rate
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=5,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
        attention_dropout=0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS token + positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, attention_dropout)
            for _ in range(depth)
        ])

        # Final norm + classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
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

    def forward(self, x, return_attention=False):
        """
        Args:
            x: (B, C, H, W) input images
            return_attention: if True, also return attention weights from all layers

        Returns:
            logits: (B, num_classes)
            attn_weights: list of (B, heads, N, N) if return_attention=True
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, embed_dim)

        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        # Transformer blocks
        all_attn_weights = []
        for block in self.blocks:
            x, attn_w = block(x)
            if return_attention:
                all_attn_weights.append(attn_w)

        # Classification from CLS token
        x = self.norm(x)
        cls_output = x[:, 0]  # (B, embed_dim)
        logits = self.head(cls_output)  # (B, num_classes)

        if return_attention:
            return logits, all_attn_weights
        return logits

    def get_num_patches(self):
        """Return number of patches (excluding CLS)."""
        return self.patch_embed.num_patches

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "vit_small_patch16_224",
        num_classes: int = 5,
        img_size: int = 192,
        **kwargs,
    ):
        """
        Create a VisionTransformer with pretrained weights from timm.

        Supported models:
            vit_tiny_patch16_224    embed_dim=192, num_heads=3
            vit_small_patch16_224   embed_dim=384, num_heads=6
            vit_base_patch16_224    embed_dim=768, num_heads=12

        Positional embeddings are bicubic-interpolated for img_size != 224.
        """
        import timm

        MODEL_CONFIGS = {
            "vit_tiny_patch16_224":  {"embed_dim": 192, "num_heads": 3,  "depth": 12},
            "vit_small_patch16_224": {"embed_dim": 384, "num_heads": 6,  "depth": 12},
            "vit_base_patch16_224":  {"embed_dim": 768, "num_heads": 12, "depth": 12},
        }

        cfg = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["vit_small_patch16_224"])
        cfg.update(kwargs)

        print(f"  Creating VisionTransformer from '{model_name}', img_size={img_size}")
        model = cls(img_size=img_size, num_classes=num_classes, **cfg)

        try:
            pretrained = timm.create_model(model_name, pretrained=True, img_size=224)
        except Exception as e:
            print(f"  ⚠ timm pretrained load failed: {e}")
            return model

        pretrained_dict = pretrained.state_dict()
        model_dict      = model.state_dict()

        mapped = {}
        for k, v in pretrained_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                mapped[k] = v

        # Interpolate positional embeddings if sizes differ
        if "pos_embed" in pretrained_dict and "pos_embed" in model_dict:
            pretrained_pos = pretrained_dict["pos_embed"]   # (1, 197, D)
            model_pos_size = model_dict["pos_embed"].shape   # (1, 145, D)
            if pretrained_pos.shape != model_pos_size:
                cls_pos   = pretrained_pos[:, :1, :]
                patch_pos = pretrained_pos[:, 1:, :]
                src_grid  = int(patch_pos.shape[1] ** 0.5)
                tgt_grid  = int((model_pos_size[1] - 1) ** 0.5)
                patch_pos = patch_pos.reshape(1, src_grid, src_grid, -1).permute(0, 3, 1, 2)
                patch_pos = torch.nn.functional.interpolate(
                    patch_pos, size=(tgt_grid, tgt_grid), mode="bicubic", align_corners=False
                ).permute(0, 2, 3, 1).reshape(1, tgt_grid * tgt_grid, -1)
                mapped["pos_embed"] = torch.cat([cls_pos, patch_pos], dim=1)
                print(f"  Interpolated pos_embed: 224px→12x12 grid ({tgt_grid}²+1={tgt_grid**2+1} tokens)")

        model_dict.update(mapped)
        model.load_state_dict(model_dict, strict=False)
        print(f"  ✓ Loaded {len(mapped)}/{len(model_dict)} weights from {model_name}")
        return model
