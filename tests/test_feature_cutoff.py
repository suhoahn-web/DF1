"""Gold-standard leakage test: perturbing sales AFTER the origin must not
change any feature computed as of that origin."""
from __future__ import annotations

import pandas as pd

from src.data import make_features
from src.utils.config import load_config, resolve_path

ORIGIN = pd.Timestamp("2017-05-16")


def _supervised(cfg):
    panel = make_features.build_daily_panel(cfg)
    sup = make_features.make_supervised(panel, [ORIGIN], horizon=cfg["forecast"]["horizon"])
    return sup.sort_values(["store_nbr", "item_nbr", "horizon"]).reset_index(drop=True)


def test_future_sales_do_not_affect_features(synth_root):
    cfg = load_config()
    interim = resolve_path(cfg["paths"]["interim_dir"])
    series = pd.read_parquet(interim / "series_stats_full.parquet")[["store_nbr", "item_nbr"]]
    series.to_parquet(resolve_path(cfg["paths"]["processed_dir"]) / "series_dev.parquet")

    before = _supervised(cfg)

    # corrupt every sale strictly after the origin
    p = interim / "sales_clean.parquet"
    sales = pd.read_parquet(p)
    future = sales["date"] > ORIGIN
    assert future.any()
    sales.loc[future, "unit_sales"] = 9999.0
    sales.to_parquet(p, index=False)

    after = _supervised(cfg)

    feature_cols = [
        c for c in before.columns
        if c not in ("target", "unit_sales") and before[c].dtype.kind in "ifb"
    ]
    pd.testing.assert_frame_equal(before[feature_cols], after[feature_cols])
    # sanity: the corruption did reach the targets
    assert (after["target"] == 9999.0).any()


def test_target_anchored_lags_never_pass_origin(synth_root):
    cfg = load_config()
    horizon = cfg["forecast"]["horizon"]
    for lag in make_features.TARGET_LAGS:
        for h in range(1, horizon + 1):
            assert h - lag <= 0, f"lag_{lag} at h={h} would read past the origin"
