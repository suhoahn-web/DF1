"""Normalized asymmetric operational loss (newsvendor-inspired; CLAUDE.md §11).

Framing (frozen): this is a point-forecast, single-period cost PROXY, not an
inventory simulation. The accuracy->cost link is a hypothesis to verify
(Theodorou et al. 2025), so costs are reported as a separate axis with ratio
sensitivity, never assumed from accuracy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RATIOS = [(1, 1), (2, 1), (3, 1), (5, 1)]


def asymmetric_cost(y: np.ndarray, yhat: np.ndarray, c_under: float, c_over: float) -> np.ndarray:
    under = np.maximum(y - yhat, 0)
    over = np.maximum(yhat - y, 0)
    return c_under * under + c_over * over


def cost_table(df: pd.DataFrame, model_cols: list[str], normalizer_col: str = "lightgbm") -> pd.DataFrame:
    """Mean cost per model x ratio, plus % improvement vs the base model."""
    y = df["actual"].to_numpy()
    rows = []
    for cu, co in RATIOS:
        base_cost = asymmetric_cost(y, df[normalizer_col].to_numpy(), cu, co).mean()
        for m in model_cols:
            c = asymmetric_cost(y, df[m].to_numpy(), cu, co).mean()
            rows.append({
                "ratio": f"{cu}:{co}", "model": m, "mean_cost": c,
                "pct_vs_base": 100 * (c - base_cost) / base_cost,
            })
    return pd.DataFrame(rows)


def per_series_cost(df: pd.DataFrame, model_col: str, cu: float, co: float) -> pd.Series:
    return df.groupby(["store_nbr", "item_nbr"], observed=True).apply(
        lambda g: asymmetric_cost(g["actual"].to_numpy(), g[model_col].to_numpy(), cu, co).mean(),
        include_groups=False,
    )
