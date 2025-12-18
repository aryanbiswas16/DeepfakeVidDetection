from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch
import torch.nn as nn

@dataclass
class EncoderInfo:
    name: str
    embed_dim: int
    image_size: int

def _try_build_dinov2() -> tuple[nn.Module, EncoderInfo] | None:
    """
    Attempts to build a DINOv2 ViT encoder via timm.
    If timm or the model name isn't available, returns None.
    """
    try:
        import timm  # type: ignore
    except Exception:
        return None

    # Common timm names vary by install; try a few.
    # Prefer DINOv3 (newer) if available, then DINOv2.
    candidates = [
        # DINOv3 (timm)
        "vit_small_patch16_dinov3",
        "vit_base_patch16_dinov3",
        "vit_large_patch16_dinov3",
        # DINOv2 (timm)
        "vit_large_patch14_dinov2",
        "vit_large_patch14_dinov2.lvd142m",
        "vit_base_patch14_dinov2",
        "vit_base_patch14_dinov2.lvd142m",
    ]

    for name in candidates:
        try:
            model = timm.create_model(name, pretrained=True, num_classes=0)
            embed_dim = getattr(model, "num_features", None) or getattr(model, "embed_dim", None)
            if embed_dim is None:
                # fallback guess
                embed_dim = 1024
            info = EncoderInfo(name=f"timm:{name}", embed_dim=int(embed_dim), image_size=224)
            return model, info
        except Exception:
            continue

    return None

def _build_resnet50_fallback() -> tuple[nn.Module, EncoderInfo]:
    from torchvision.models import resnet50, ResNet50_Weights  # type: ignore

    backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
    # Remove classifier: output will be pooled features (2048)
    backbone.fc = nn.Identity()
    return backbone, EncoderInfo(name="torchvision:resnet50", embed_dim=2048, image_size=224)

class FrameEncoder(nn.Module):
    """
    Frame encoder with a DINOv2-first, ResNet50-fallback policy.

    - If DINOv2 is available (via timm), freeze everything, then unfreeze only LayerNorm params.
    - If using ResNet50 fallback, freeze everything by default (train head only).
    """

    def __init__(self, device: str = "cpu", layernorm_tuning: bool = True):
        super().__init__()
        self.device = device

        built = _try_build_dinov2()
        if built is None:
            warnings.warn("DINOv2 not available; using ResNet50 fallback.")
            model, info = _build_resnet50_fallback()
            self.is_dinov2 = False
        else:
            model, info = built
            self.is_dinov2 = True

        self.model = model
        self.info = info
        self.image_size = info.image_size
        self.embed_dim = info.embed_dim

        # Freeze everything
        for p in self.model.parameters():
            p.requires_grad = False

        # LayerNorm tuning for ViT-style models
        if self.is_dinov2 and layernorm_tuning:
            for m in self.model.modules():
                if isinstance(m, nn.LayerNorm):
                    for p in m.parameters():
                        p.requires_grad = True

        self.to(device)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes images: x is [B, 3, H, W] -> embeddings [B, D]
        """
        self.eval()
        emb = self.model(x)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.model(x)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


