"""Shared bottleneck + heads (identical for every arm; only the input width differs)."""
from __future__ import annotations

import torch
from torch import nn


class ImageHead(nn.Module):
    """standardize(train stats) -> Linear(D,1024) -> GELU -> Dropout(0.2) -> Linear(1024,768) -> LayerNorm = v_imgfor.

    Heads read v_imgfor directly: binary Linear(768,1) and auxiliary Linear(768,3). Both back-propagate into the bottleneck.
    """

    def __init__(self, in_dim, hidden=1024, out_dim=768, dropout=0.2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(in_dim))
        self.register_buffer("std", torch.ones(in_dim))
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, out_dim), nn.LayerNorm(out_dim))
        self.binary = nn.Linear(out_dim, 1)
        self.aux = nn.Linear(out_dim, 3)

    def set_statistics(self, features):
        """Per-dimension mean/std from TRAIN rows only."""
        self.mean.copy_(features.mean(0))
        self.std.copy_(features.std(0).clamp_min(1e-6))

    def embed(self, x):
        return self.net((x - self.mean) / self.std)

    def forward(self, x):
        v = self.embed(x)
        return dict(v=v, logit=self.binary(v).squeeze(-1), aux=self.aux(v))
