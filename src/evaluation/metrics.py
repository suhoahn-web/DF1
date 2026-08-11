"""Accuracy metrics. WAPE is primary; MAPE deliberately absent (zero demand)."""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-9


def wape(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.abs(y - yhat).sum() / (np.abs(y).sum() + EPS))


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.abs(y - yhat).mean())


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(((y - yhat) ** 2).mean()))


def rmsle(y: np.ndarray, yhat: np.ndarray) -> float:
    ly = np.log1p(np.clip(y, 0, None))
    lyhat = np.log1p(np.clip(yhat, 0, None))
    return float(np.sqrt(((ly - lyhat) ** 2).mean()))


def mase(y: np.ndarray, yhat: np.ndarray, naive: np.ndarray) -> float:
    """Scaled by the Seasonal-Naive error on the SAME evaluation rows.

    Out-of-sample MASE variant: denominator = MAE of seasonal naive forecasts
    for the identical (series, origin, horizon) rows, so models and scale are
    compared on exactly the same targets.
    """
    denom = np.abs(y - naive).mean()
    return float(np.abs(y - yhat).mean() / (denom + EPS))


ALL = {"wape": wape, "mae": mae, "rmse": rmse, "rmsle": rmsle}


def evaluate(df: pd.DataFrame, model_col: str, by: list[str] | None = None) -> pd.DataFrame:
    """Metrics for one prediction column, optionally grouped (e.g., regime, fold)."""

    def _one(g: pd.DataFrame) -> pd.Series:
        y, yhat = g["actual"].to_numpy(), g[model_col].to_numpy()
        out = {k: f(y, yhat) for k, f in ALL.items()}
        if "seasonal_naive" in g.columns:
            out["mase"] = mase(y, yhat, g["seasonal_naive"].to_numpy())
        out["n"] = len(g)
        return pd.Series(out)

    if by:
        return df.groupby(by, observed=True).apply(_one, include_groups=False).reset_index()
    return _one(df).to_frame().T


def per_series_metric(df: pd.DataFrame, model_col: str, metric: str = "wape") -> pd.Series:
    """One metric value per series (input to Wilcoxon / win-rate)."""
    f = ALL[metric]
    return df.groupby(["store_nbr", "item_nbr"], observed=True).apply(
        lambda g: f(g["actual"].to_numpy(), g[model_col].to_numpy()), include_groups=False
    )
