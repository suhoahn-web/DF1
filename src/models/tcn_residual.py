"""Causal TCN residual corrector (spec §4)."""
from __future__ import annotations

import torch
import torch.nn as nn


class CausalConv1d(nn.Conv1d):
    """Conv1d with left-only padding so output[t] never sees input[>t]."""

    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        self._left_pad = (kernel_size - 1) * dilation
        super().__init__(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x):
        x = nn.functional.pad(x, (self._left_pad, 0))
        return super().forward(x)


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.norm1 = nn.GroupNorm(1, out_ch)
        self.norm2 = nn.GroupNorm(1, out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = self.drop(self.act(self.norm1(self.conv1(x))))
        h = self.drop(self.act(self.norm2(self.conv2(h))))
        return h + self.skip(x)


class ResidualTCN(nn.Module):
    """[B, C, L] -> residual correction vector [B, H] (in SCALED units)."""

    def __init__(self, in_channels: int, horizon: int = 7,
                 hidden: int = 64, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            CausalConv1d(in_channels, 32, kernel_size, dilation=1),
            nn.GroupNorm(1, 32), nn.GELU(), nn.Dropout(dropout),
        )
        self.block1 = TCNBlock(32, hidden, kernel_size, dilation=2, dropout=dropout)
        self.block2 = TCNBlock(hidden, hidden, kernel_size, dilation=4, dropout=dropout)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.block2(self.block1(self.stem(x)))
        return self.head(z[:, :, -1])
