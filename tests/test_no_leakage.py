"""Split invariants, zero-fill discipline, and OOF cutoff guarantees."""
from __future__ import annotations

import pandas as pd

from src.data import make_features, splits
from src.utils.config import load_config, resolve_path


def test_split_invariants():
    splits.validate_splits()


def test_oof_cutoffs_strictly_before_targets():
    df = splits.oof_base_cutoffs()
    # every OOF prediction target (origin+1 .. origin+7) is after the base cutoff
    assert (df["base_train_cutoff"] < df["origin_date"]).all()
    first_target = df["origin_date"] + pd.Timedelta(days=1)
    assert (df["base_train_cutoff"] < first_target).all()


def test_tcn_gate_separation():
    df = splits.oof_base_cutoffs()
    tcn = df[df["stage"] == "tcn_train"]["origin_date"]
    gate = df[df["stage"] == "gate_train"]["origin_date"]
    ev = df[df["stage"] == "eval_input"]["origin_date"]
    assert len(tcn) > 0 and len(gate) > 0 and len(ev) > 0
    assert tcn.max() < gate.min() < ev.min()
    share = len(tcn) / (len(tcn) + len(gate))
    assert 0.6 <= share <= 0.8, f"TCN share {share:.2f} outside intended ~70%"
    # eval_input origins must never be usable as training targets
    from src.utils.config import load_config
    cfg = load_config()
    assert ev.min() > pd.Timestamp(cfg["splits"]["oof"]["train_last_origin"])


def test_zero_fill_starts_at_first_sale(synth_root):
    cfg = load_config()
    interim = resolve_path(cfg["paths"]["interim_dir"])

    # delay one series' first sale so the grid must start later for it
    p = interim / "sales_clean.parquet"
    sales = pd.read_parquet(p)
    key = (sales["store_nbr"] == 2) & (sales["item_nbr"] == 100)
    late_start = pd.Timestamp("2016-01-01")
    sales = sales[~(key & (sales["date"] < late_start))]
    sales.to_parquet(p, index=False)
    stats = pd.read_parquet(interim / "series_stats_full.parquet")
    m = (stats["store_nbr"] == 2) & (stats["item_nbr"] == 100)
    stats.loc[m, "first_sale"] = sales.loc[key & (sales["unit_sales"] > 0), "date"].min()
    stats.to_parquet(interim / "series_stats_full.parquet", index=False)

    series = stats[["store_nbr", "item_nbr"]]
    panel = make_features.build_daily_panel(cfg, series)

    s = panel[(panel["store_nbr"] == 2) & (panel["item_nbr"] == 100)]
    assert s["date"].min() >= late_start, "grid must not start before first sale"
    other = panel[(panel["store_nbr"] == 1) & (panel["item_nbr"] == 100)]
    assert other["date"].min() < late_start


def test_panel_has_no_missing_dates(synth_root):
    cfg = load_config()
    interim = resolve_path(cfg["paths"]["interim_dir"])
    series = pd.read_parquet(interim / "series_stats_full.parquet")[["store_nbr", "item_nbr"]]
    panel = make_features.build_daily_panel(cfg, series)
    for _, g in panel.groupby(["store_nbr", "item_nbr"], observed=True):
        d = g["date"].sort_values()
        assert (d.diff().dropna() == pd.Timedelta(days=1)).all(), "gap in daily grid"
