"""Horizon-wise context gate (spec §6) and gated assembly."""
from __future__ import annotations

import torch
import torch.nn as nn


class HorizonGate(nn.Module):
    """context [B, H, D] -> gate values [B, H] in (0, 1). Shared MLP across horizons."""

    def __init__(self, d_in: int, hidden: int = 64, hidden2: int = 32, dropout: float = 0.1,
                 init_bias: float = -2.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden2), nn.GELU(),
            nn.Linear(hidden2, 1),
        )
        # start mostly CLOSED (sigmoid(-2) ~= 0.12): correction must earn its
        # activation; with a harmful-on-average correction, an open start
        # leaves the optimizer on an always-on plateau
        nn.init.constant_(self.mlp[-1].bias, init_bias)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.mlp(context).squeeze(-1))


class LinearGate(nn.Module):
    """Logistic-regression gate: single linear layer + sigmoid. Complexity
    baseline for the MLP gate (review round 1, P9)."""

    def __init__(self, d_in: int, init_bias: float = -2.0, **_ignored):
        super().__init__()
        self.lin = nn.Linear(d_in, 1)
        nn.init.constant_(self.lin.bias, init_bias)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.lin(context).squeeze(-1))


def gated_forecast(base: torch.Tensor, r_hat: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """final = clamp(base + g * r_hat, 0). All [B, H]."""
    return torch.clamp(base + gate * r_hat, min=0.0)
