"""
MICCAI 2024 ViT-B model for AD classification.

Uses the Vision_Transformer3D from the MICCAI reference implementation
with MAE-pretrained weights. No MNI registration required.

Reference:
    Kunanbayev et al., "Training ViT with Limited Data for Alzheimer's
    Disease Classification: an Empirical Study", MICCAI 2024.
"""

import logging
import math
import copy
from functools import partial
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ── Building blocks (from MICCAI reference) ─────────────────────────────


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Truncated normal initialization."""
    with torch.no_grad():
        l = (1. + math.erf((a - mean) / std / math.sqrt(2.))) / 2.
        u = (1. + math.erf((b - mean) / std / math.sqrt(2.))) / 2.
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
    return tensor


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed3D(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.Conv3d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        sample = torch.rand((1, 1, *img_size))
        self.n_patches = self.proj(sample).flatten(2).shape[2]

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).flatten(2)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True,
                 drop_path_rate=0., p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, n_heads=n_heads, qkv_bias=qkv_bias,
                              attn_p=attn_p, proj_p=p)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim, p=p)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ── Main model ───────────────────────────────────────────────────────────


class MICCAIViTClassifier(nn.Module):
    """
    3D ViT-B classifier matching the MICCAI 2024 architecture exactly.

    Input:  (B, 1, 128, 128, 128)
    Output: dict with 'logits' (B, n_classes) and 'features' (B, embed_dim)
    """

    def __init__(
        self,
        img_size=(128, 128, 128),
        patch_size=16,
        n_classes=2,
        embed_dim=768,
        depth=12,
        n_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.1,
        p=0.,
        attn_p=0.1,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.embed_dim = embed_dim
        self.depth = depth

        # Patch embedding
        self.patch_embed = PatchEmbed3D(img_size, patch_size, embed_dim)
        n_patches = self.patch_embed.n_patches

        # CLS token + positional embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 1e-6)
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + n_patches, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=p)

        # Transformer blocks with stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(embed_dim, n_heads, mlp_ratio, qkv_bias, dpr[i], p, attn_p)
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # Classification head
        self.head = nn.Linear(embed_dim, n_classes)

        # Initialize
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def load_pretrained(self, checkpoint_path: str):
        """Load MAE-pretrained weights (encoder only, skip decoder + head)."""
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["net"] if "net" in ckpt else ckpt

        # Filter out decoder and head keys
        encoder_dict = {}
        skipped = []
        for k, v in state_dict.items():
            if any(skip in k for skip in ["decoder_", "mask_token", "head."]):
                skipped.append(k)
                continue
            encoder_dict[k] = v

        msg = self.load_state_dict(encoder_dict, strict=False)
        n_loaded = len(encoder_dict) - len(msg.unexpected_keys)
        n_missing = len(msg.missing_keys)
        log.info(f"Loaded MICCAI MAE checkpoint: {n_loaded} matched, "
                 f"{n_missing} missing, {len(skipped)} decoder/head keys skipped")
        if msg.missing_keys:
            log.info(f"  Missing: {msg.missing_keys}")
        return msg

    def freeze_backbone(self):
        """Freeze everything except classification head."""
        for name, p in self.named_parameters():
            if not name.startswith("head."):
                p.requires_grad = False
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        log.info(f"Backbone frozen: {trainable:,} / {total:,} params trainable")

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for p in self.parameters():
            p.requires_grad = True
        total = sum(p.numel() for p in self.parameters())
        log.info(f"All params unfrozen: {total:,} trainable")

    def forward(self, x, return_hidden=False):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        hidden_states = []
        for blk in self.blocks:
            x = blk(x)
            if return_hidden:
                hidden_states.append(x)

        x = self.norm(x)

        features = x[:, 0]  # CLS token
        logits = self.head(features)

        out = {"logits": logits, "features": features}
        if return_hidden:
            out["hidden_states"] = hidden_states
        return out


def build_model(checkpoint_path=None, n_classes=2, device="cuda"):
    """Build and optionally load pretrained MICCAI ViT-B model."""
    model = MICCAIViTClassifier(n_classes=n_classes)

    if checkpoint_path:
        model.load_pretrained(checkpoint_path)

    model = model.to(device)
    return model
