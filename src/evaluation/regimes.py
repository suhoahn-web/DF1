"""Regime slice labels for evaluation rows (multi-label; regimes may overlap).

Regimes are EVALUATION slices only -- never training labels (spec §10).
The high-volatility threshold comes from TRAINING data only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import load_config


def high_vol_threshold(train_rows: pd.DataFrame, cfg=None) -> float:
    """Upper-quantile CV_28 threshold computed on training-period rows only."""
    cfg = cfg or load_config()
    q = cfg["regimes"]["high_volatility_quantile"]
    return float(train_rows["cv_28"].quantile(q))


def add_regime_flags(df: pd.DataFrame, hv_threshold: float) -> pd.DataFrame:
    df = df.copy()
    df["regime_promotion"] = (df["onpromotion"] == 1).astype("int8")
    df["regime_post_promo"] = df["post_promo_1_3"].astype("int8")
    df["regime_holiday"] = df["is_holiday"].astype("int8")
    df["regime_high_vol"] = (df["cv_28"] >= hv_threshold).astype("int8")
    df["regime_low_vol"] = (df["cv_28"] < hv_threshold).astype("int8")
    df["regime_normal"] = (
        (df["regime_promotion"] == 0)
        & (df["regime_post_promo"] == 0)
        & (df["regime_holiday"] == 0)
        & (df["regime_high_vol"] == 0)
    ).astype("int8")
    return df


REGIME_COLS = [
    "regime_normal", "regime_promotion", "regime_post_promo",
    "regime_holiday", "regime_high_vol", "regime_low_vol",
]


def regime_metrics(df: pd.DataFrame, model_cols: list[str]) -> pd.DataFrame:
    """Long table: metric x model x regime slice."""
    from src.evaluation.metrics import evaluate

    rows = []
    for regime in REGIME_COLS:
        sub = df[df[regime] == 1]
        if len(sub) == 0:
            continue
        for m in model_cols:
            r = evaluate(sub, m)
            r["regime"], r["model"] = regime.replace("regime_", ""), m
            rows.append(r)
    return pd.concat(rows, ignore_index=True)
