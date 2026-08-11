"""Series-level paired tests: Wilcoxon (+ Holm), bootstrap CI, win rate."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.evaluation.metrics import per_series_metric


def paired_comparison(
    df: pd.DataFrame, model_a: str, model_b: str, metric: str = "wape",
    n_boot: int = 2000, seed: int = 20260810,
) -> dict:
    """Compare A vs B on per-series metric values. Negative diff = A better."""
    # exclude series whose actuals sum to ~0 in the window: their per-series
    # WAPE is denominator-degenerate (blows up any mean-based statistic)
    totals = df.groupby(["store_nbr", "item_nbr"], observed=True)["actual"].sum()
    valid = totals[totals >= 1.0].index
    dfv = df.set_index(["store_nbr", "item_nbr"]).loc[valid].reset_index()

    a = per_series_metric(dfv, model_a, metric)
    b = per_series_metric(dfv, model_b, metric)
    diff = (a - b).dropna()

    stat, p = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
    rng = np.random.default_rng(seed)
    boots = np.array([
        diff.sample(frac=1, replace=True, random_state=rng.integers(2**31)).mean()
        for _ in range(n_boot)
    ])
    return {
        "model_a": model_a, "model_b": model_b, "metric": metric,
        "n_series": len(diff),
        "mean_diff": float(diff.mean()),
        "median_diff": float(diff.median()),
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
        "win_rate_a": float((diff < 0).mean()),
        "tie_rate": float((diff == 0).mean()),
        "loss_rate_a": float((diff > 0).mean()),
        "wilcoxon_p": float(p),
    }


def holm_correction(results: list[dict]) -> pd.DataFrame:
    """Holm step-down over the family of comparisons (Demšar 2006 convention)."""
    out = pd.DataFrame(results).sort_values("wilcoxon_p").reset_index(drop=True)
    m = len(out)
    adj = (out["wilcoxon_p"] * (m - out.index)).cummax().clip(upper=1.0)
    out["holm_p"] = adj
    out["significant_5pct"] = out["holm_p"] < 0.05
    return out
