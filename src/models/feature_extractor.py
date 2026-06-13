"""
Self-Supervised Feature Extractor for Patch-Level Representations.

Supports three pretrained Vision Transformer backends:
  1. DINO ViT  — Self-distillation with no labels (facebook/dino-vitb16)
  2. MAE ViT   — Masked autoencoder (facebook/vit-mae-base)
  3. HuggingFace ViT — Standard pretrained ViT (google/vit-base-patch16-224)

Each backend extracts patch tokens φ_i ∈ R^d where:
  i = patch index (0..195 for 14×14 grid)
  d = embedding dimension (768 for ViT-B)

All models are frozen — no gradients flow through them.
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


# ────────────────────────────────────────────────────────────
#  Factory function
# ────────────────────────────────────────────────────────────

def create_feature_extractor(
    model_type: str = "dino",
    model_name: str = None,
    device: str = "cpu",
) -> "PatchFeatureExtractor":
    """
    Factory to create a patch feature extractor.

    Args:
        model_type: 'dino', 'mae', or 'huggingface'
        model_name: Specific model name (optional override)
        device: 'cpu' or 'cuda'

    Returns:
        PatchFeatureExtractor subclass instance
    """
    model_type = model_type.lower()

    if model_type == "dino":
        extractor = DINOFeatureExtractor(model_name=model_name)
    elif model_type == "mae":
        extractor = MAEFeatureExtractor(model_name=model_name)
    elif model_type in ("huggingface", "hf", "vit"):
        extractor = HuggingFaceViTExtractor(model_name=model_name)
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Choose from: 'dino', 'mae', 'huggingface'"
        )

    return extractor.to(device)


# ────────────────────────────────────────────────────────────
#  Base class
# ────────────────────────────────────────────────────────────

class PatchFeatureExtractor(nn.Module):
    """
    Base class for all patch-level feature extractors.

    Every subclass must implement:
        _load_model()      → load and freeze the backbone
        _extract_patches() → return (B, N_patches, D) patch embeddings

    Output invariant:
        For a 224×224 image with 16×16 patches:
        output shape = (B, 196, embed_dim)
    """

    def __init__(self, model_name: str, embed_dim: int):
        super().__init__()
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.backbone = None
        self._load_model()
        self._freeze()

    def _load_model(self):
        raise NotImplementedError

    def _extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _freeze(self):
        """Freeze all parameters — no gradients."""
        if self.backbone is not None:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

    def train(self, mode=True):
        """Override to keep backbone always in eval mode."""
        super().train(mode)
        if self.backbone is not None:
            self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract patch-level embeddings.

        Args:
            x: (B, 3, 224, 224) input images (normalized)

        Returns:
            patch_embeddings: (B, N_patches, embed_dim)
                             → (B, 196, 768) for ViT-B/16 on 224×224
        """
        return self._extract_patches(x)

    def get_output_dim(self) -> int:
        return self.embed_dim

    def get_num_patches(self, img_size: int = 224, patch_size: int = 16) -> int:
        return (img_size // patch_size) ** 2

    @torch.no_grad()
    def extract_from_dataloader(
        self,
        dataloader,
        num_samples: int = None,
        device: str = None,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Extract patch embeddings from an entire dataloader.

        Args:
            dataloader: yields (images, labels)
            num_samples: max images to process (None = all)
            device: override device
            show_progress: show tqdm bar

        Returns:
            all_features: (total_images, N_patches, embed_dim)
        """
        if device is None:
            device = next(self.parameters()).device if len(list(self.parameters())) > 0 else "cpu"
            # For models that might not have parameters registered on self
            if hasattr(self, 'backbone'):
                try:
                    device = next(self.backbone.parameters()).device
                except StopIteration:
                    device = torch.device("cpu")

        self.eval()
        all_features = []
        count = 0

        iterator = tqdm(dataloader, desc="Extracting features") if show_progress else dataloader

        for images, _ in iterator:
            if num_samples and count >= num_samples:
                break

            images = images.to(device)
            features = self.forward(images)  # (B, N, D)
            all_features.append(features.cpu())

            count += images.shape[0]
            if show_progress:
                iterator.set_postfix(extracted=count)

        all_features = torch.cat(all_features, dim=0)

        if num_samples:
            all_features = all_features[:num_samples]

        return all_features


# ────────────────────────────────────────────────────────────
#  1. DINO ViT Feature Extractor
# ────────────────────────────────────────────────────────────

class DINOFeatureExtractor(PatchFeatureExtractor):
    """
    DINO (Self-Distillation with No Labels) ViT extractor.

    DINO produces especially rich patch-level features because the
    self-supervised objective encourages meaningful local representations.
    The attention maps naturally segment objects.

    Default model: facebook/dino-vitb16 (ViT-B/16, 768-dim)
    """

    DEFAULTS = {
        "dino_vitb16": {"hub": "facebookresearch/dino:main", "fn": "dino_vitb16", "dim": 768},
        "dino_vits16": {"hub": "facebookresearch/dino:main", "fn": "dino_vits16", "dim": 384},
        "dino_vitb8":  {"hub": "facebookresearch/dino:main", "fn": "dino_vitb8",  "dim": 768},
        "dino_vits8":  {"hub": "facebookresearch/dino:main", "fn": "dino_vits8",  "dim": 384},
    }

    def __init__(self, model_name: str = None):
        self._user_model_name = model_name or "dino_vitb16"
        info = self.DEFAULTS.get(self._user_model_name, self.DEFAULTS["dino_vitb16"])
        super().__init__(model_name=self._user_model_name, embed_dim=info["dim"])

    def _load_model(self):
        info = self.DEFAULTS.get(self._user_model_name, self.DEFAULTS["dino_vitb16"])

        try:
            # Primary: load from torch.hub (Facebook DINO repo)
            self.backbone = torch.hub.load(
                info["hub"], info["fn"], pretrained=True,
            )
            print(f"  ✓ Loaded DINO model: {info['fn']} from torch.hub")
        except Exception as e:
            print(f"  ⚠ torch.hub failed: {e}")
            print(f"  → Falling back to timm...")
            import timm
            timm_name = "vit_base_patch16_224.dino" if "vitb16" in self._user_model_name else "vit_small_patch16_224.dino"
            self.backbone = timm.create_model(timm_name, pretrained=True, num_classes=0)
            print(f"  ✓ Loaded DINO model: {timm_name} from timm")

    def _extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        DINO models expose get_intermediate_layers() for clean patch extraction.
        Fallback: forward_features + strip CLS token.
        """
        if hasattr(self.backbone, "get_intermediate_layers"):
            # DINO native method — returns list of layer outputs
            outputs = self.backbone.get_intermediate_layers(x, n=1)
            patch_tokens = outputs[0]  # (B, N+1, D) — includes CLS token
            # Strip CLS token at index 0
            if patch_tokens.shape[1] > (x.shape[-1] // 16) ** 2:
                patch_tokens = patch_tokens[:, 1:, :]
            return patch_tokens

        if hasattr(self.backbone, "forward_features"):
            features = self.backbone.forward_features(x)
            if isinstance(features, torch.Tensor) and features.dim() == 3:
                return features[:, 1:, :]  # strip CLS token

        # Fallback: hook-based
        return self._hook_extract(x)

    def _hook_extract(self, x):
        features = {}
        def hook_fn(module, inp, out):
            features["out"] = out
        handle = self.backbone.blocks[-1].register_forward_hook(hook_fn)
        _ = self.backbone(x)
        handle.remove()
        out = features["out"]
        if isinstance(out, tuple):
            out = out[0]
        if out.dim() == 3 and out.shape[1] > 196:
            out = out[:, 1:, :]
        return out


# ────────────────────────────────────────────────────────────
#  2. MAE ViT Feature Extractor
# ────────────────────────────────────────────────────────────

class MAEFeatureExtractor(PatchFeatureExtractor):
    """
    MAE (Masked Autoencoder) ViT extractor.

    MAE learns to reconstruct masked patches, producing encoders that
    understand spatial structure. We use the encoder (not decoder) to
    extract patch-level features.

    Default model: facebook/vit-mae-base (ViT-B/16, 768-dim)
    """

    def __init__(self, model_name: str = None):
        model_name = model_name or "facebook/vit-mae-base"
        super().__init__(model_name=model_name, embed_dim=768)

    def _load_model(self):
        try:
            from transformers import ViTMAEModel
            self.backbone = ViTMAEModel.from_pretrained(self.model_name)
            self._backend = "transformers"
            print(f"  ✓ Loaded MAE model: {self.model_name} from HuggingFace")
        except ImportError:
            print("  ⚠ transformers not installed, falling back to timm...")
            import timm
            self.backbone = timm.create_model(
                "vit_base_patch16_224.mae", pretrained=True, num_classes=0
            )
            self._backend = "timm"
            print("  ✓ Loaded MAE model from timm")

    def _extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        if self._backend == "transformers":
            # HuggingFace MAE: pass pixel_values, get hidden states
            # MAE randomly masks patches during forward; we disable masking
            # by setting noise to zeros (deterministic, no masking)
            outputs = self.backbone(
                pixel_values=x,
                noise=torch.zeros(x.shape[0], 196, device=x.device),
            )
            # last_hidden_state: (B, N+1, D) — includes CLS token at index 0
            hidden = outputs.last_hidden_state
            patch_tokens = hidden[:, 1:, :]  # strip CLS
            return patch_tokens
        else:
            # timm backend
            features = self.backbone.forward_features(x)
            if isinstance(features, torch.Tensor) and features.dim() == 3:
                return features[:, 1:, :]
            return features


# ────────────────────────────────────────────────────────────
#  3. HuggingFace ViT Feature Extractor
# ────────────────────────────────────────────────────────────

class HuggingFaceViTExtractor(PatchFeatureExtractor):
    """
    Standard HuggingFace Vision Transformer extractor.

    Uses pretrained ViT models from HuggingFace (supervised or self-supervised).
    Extracts patch tokens from the last hidden layer.

    Default model: google/vit-base-patch16-224 (ViT-B/16, 768-dim)
    """

    def __init__(self, model_name: str = None):
        model_name = model_name or "google/vit-base-patch16-224"
        super().__init__(model_name=model_name, embed_dim=768)

    def _load_model(self):
        try:
            from transformers import ViTModel
            self.backbone = ViTModel.from_pretrained(self.model_name)
            self._backend = "transformers"
            print(f"  ✓ Loaded HuggingFace ViT: {self.model_name}")
        except ImportError:
            print("  ⚠ transformers not installed, falling back to timm...")
            import timm
            self.backbone = timm.create_model(
                "vit_base_patch16_224", pretrained=True, num_classes=0
            )
            self._backend = "timm"
            print("  ✓ Loaded ViT from timm (fallback)")

    def _extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        if self._backend == "transformers":
            outputs = self.backbone(pixel_values=x)
            # last_hidden_state: (B, N+1, D) with CLS at index 0
            hidden = outputs.last_hidden_state
            patch_tokens = hidden[:, 1:, :]
            return patch_tokens
        else:
            features = self.backbone.forward_features(x)
            if isinstance(features, torch.Tensor) and features.dim() == 3:
                return features[:, 1:, :]
            return features


# ────────────────────────────────────────────────────────────
#  Embedding storage utilities
# ────────────────────────────────────────────────────────────

def save_embeddings(
    embeddings: torch.Tensor,
    save_path: str,
    metadata: dict = None,
):
    """
    Save extracted patch embeddings to disk.

    Args:
        embeddings: (N_images, N_patches, embed_dim) tensor
        save_path: path to .pt file
        metadata: optional metadata dict (model_name, num_images, etc.)
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    data = {
        "embeddings": embeddings,
        "shape": list(embeddings.shape),
        "num_images": embeddings.shape[0],
        "num_patches": embeddings.shape[1],
        "embed_dim": embeddings.shape[2],
    }

    if metadata:
        data["metadata"] = metadata

    torch.save(data, save_path)
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"  ✓ Saved embeddings: {save_path} ({size_mb:.1f} MB)")
    print(f"    Shape: {list(embeddings.shape)}")


def load_embeddings(load_path: str) -> dict:
    """
    Load saved patch embeddings.

    Returns:
        dict with 'embeddings' tensor and metadata
    """
    data = torch.load(load_path, map_location="cpu")
    print(f"  ✓ Loaded embeddings: {load_path}")
    print(f"    Shape: {data['shape']}")
    return data
