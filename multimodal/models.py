"""Multimodal fusion models for AD classification."""

import torch
import torch.nn as nn


class MLPFusion(nn.Module):
    """Late fusion: concatenate vision + text CLS, classify with small MLP.

    ~197K trainable params with default settings.
    """

    def __init__(self, d_vision=768, d_text=768, d_hidden=128, n_classes=2, dropout=0.3):
        super().__init__()
        self.d_vision = d_vision
        self.d_text = d_text
        self.mlp = nn.Sequential(
            nn.Linear(d_vision + d_text, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, n_classes),
        )

    def forward(self, v_cls, t_cls):
        fused = torch.cat([v_cls, t_cls], dim=-1)  # (B, 1536)
        logits = self.mlp(fused)  # (B, 2)
        return {"logits": logits, "features": fused}
