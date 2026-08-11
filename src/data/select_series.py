"""Series eligibility report and stratified sampling.

Eligibility criteria are frozen in configs/data.yaml (series_eligibility).
All metrics are computed on the ANALYSIS PERIOD only, after cleaning.
Stratification: promo-frequency terciles x volatility (CV) terciles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import load_config, resolve_path


def compute_eligibility(cfg=None) -> pd.DataFrame:
    cfg = cfg or load_config()
    interim = resolve_path(cfg["paths"]["interim_dir"])
    start = pd.Timestamp(cfg["analysis_period"]["start"])
    end = pd.Timestamp(cfg["analysis_period"]["end"])

    sales = pd.read_parquet(
        interim / "sales_clean.parquet",
        columns=["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"],
    )
    sales = sales[(sales["date"] >= start) & (sales["date"] <= end)]
    full_stats = pd.read_parquet(interim / "series_stats_full.parquet")

    pos = sales[sales["unit_sales"] > 0]
    last365 = pos[pos["date"] > end - pd.Timedelta(days=365)]

    key = ["store_nbr", "item_nbr"]
    m = pos.groupby(key, observed=True).agg(
        nonzero_days_total=("date", "nunique"),
        mean_sales=("unit_sales", "mean"),
        std_sales=("unit_sales", "std"),
    )
    m["nonzero_days_last_365"] = last365.groupby(key, observed=True)["date"].nunique()
    m["promo_days_total"] = (
        sales[sales["onpromotion"] == 1].groupby(key, observed=True)["date"].nunique()
    )
    m = m.fillna({"nonzero_days_last_365": 0, "promo_days_total": 0, "std_sales": 0.0})
    m = m.reset_index().merge(full_stats, on=key, how="left")

    m["history_days"] = (end - m["first_sale"]).dt.days
    m["cv"] = m["std_sales"] / (m["mean_sales"] + cfg["features"]["eps"])
    m["promo_share"] = m["promo_days_total"] / m["history_days"].clip(lower=1)

    e = cfg["series_eligibility"]
    m["eligible"] = (
        (m["first_sale"] <= pd.Timestamp(e["max_first_sale_date"]))
        & (m["history_days"] >= e["min_history_days"])
        & (m["nonzero_days_total"] >= e["min_nonzero_days_total"])
        & (m["nonzero_days_last_365"] >= e["min_nonzero_days_last_365"])
        & (m["promo_days_total"] >= e["min_promo_days_total"])
    )

    out = resolve_path(cfg["paths"]["processed_dir"]) / "series_eligibility.parquet"
    m.to_parquet(out, index=False)
    print(
        f"eligibility: {len(m):,} series scanned, {int(m['eligible'].sum()):,} eligible"
        f" -> {out.name}"
    )
    return m


def stratified_sample(cfg=None, n: int | None = None, out_name: str = "series_dev.parquet") -> pd.DataFrame:
    """Sample n series from the eligible pool, stratified by promo share x CV terciles."""
    cfg = cfg or load_config()
    processed = resolve_path(cfg["paths"]["processed_dir"])
    n = n or cfg["sampling"]["dev_n_series"]

    m = pd.read_parquet(processed / "series_eligibility.parquet")
    pool = m[m["eligible"]].copy()
    if len(pool) < n:
        raise ValueError(f"eligible pool ({len(pool)}) smaller than requested n={n}")

    qp = cfg["sampling"]["stratify_by"]["promo_frequency_bins"]
    qv = cfg["sampling"]["stratify_by"]["volatility_bins"]
    pool["promo_bin"] = pd.qcut(pool["promo_share"].rank(method="first"), qp, labels=False)
    pool["vol_bin"] = pd.qcut(pool["cv"].rank(method="first"), qv, labels=False)

    rng = np.random.default_rng(cfg["sampling"]["random_seed"])
    picks = []
    for (_, _), g in pool.groupby(["promo_bin", "vol_bin"], observed=True):
        k = max(1, round(n * len(g) / len(pool)))
        picks.append(g.sample(n=min(k, len(g)), random_state=rng.integers(2**31)))
    sample = pd.concat(picks).head(n)

    sample.to_parquet(processed / out_name, index=False)
    print(f"sampled {len(sample)} series (stratified {qp}x{qv}) -> {out_name}")
    return sample


if __name__ == "__main__":
    compute_eligibility()
    stratified_sample()
