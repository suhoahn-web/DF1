"""Leakage-safe feature engineering.

Conventions (documented in the audit report and the paper):

  Origin framing. A forecast is made at the end of origin day t for targets
  t+1 .. t+7. Every *historical* feature uses sales on days <= t only.

  Origin-anchored features (computed on the daily panel, valid "as of" day t):
    recent_1..recent_3        sales(t), sales(t-1), sales(t-2)
    roll_{mean,std}_{7,14,28} windows ending at t
    roll_median_{7,28}, roll_max_{7,28}, roll_min_{7,28}
    zero_ratio_28, cv_28

  Target-anchored observable lags (depend on horizon h, still <= t for h <= 7):
    lag7  = sales(t+h-7)   same weekday as the target
    lag14, lag21, lag28, lag35, lag56 analogously

  Future-known covariates at the target date t+h (whitelisted: calendar,
  holiday schedule, planned promotions -- assumption stated in the paper):
    calendar (dow/week/month/weekend/payday), is_holiday, earthquake dummy,
    onpromotion and derived promo features, store x family promo share.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import load_config, resolve_path

EPS = 1e-6


# ---------------------------------------------------------------- daily panel
def build_daily_panel(cfg=None, series: pd.DataFrame | None = None) -> pd.DataFrame:
    """Zero-filled daily panel for the selected series, with as-of-day features."""
    cfg = cfg or load_config()
    interim = resolve_path(cfg["paths"]["interim_dir"])
    processed = resolve_path(cfg["paths"]["processed_dir"])
    if series is None:
        series = pd.read_parquet(processed / "series_dev.parquet")

    start_buf = pd.Timestamp(cfg["analysis_period"]["start"]) - pd.Timedelta(
        days=cfg["analysis_period"]["feature_buffer_days"]
    )
    end = pd.Timestamp(cfg["analysis_period"]["end"])
    key = ["store_nbr", "item_nbr"]

    sales = pd.read_parquet(interim / "sales_clean.parquet")
    sales = sales.merge(series[key], on=key, how="inner")

    # promo intensity: share of promoted items per store x family x day,
    # computed over ALL recorded rows (plan-based covariate, future-known)
    items = pd.read_parquet(interim / "items.parquet")
    all_sales = pd.read_parquet(
        interim / "sales_clean.parquet", columns=["date", "store_nbr", "item_nbr", "onpromotion"]
    ).merge(items[["item_nbr", "family"]], on="item_nbr", how="left")
    promo_share = (
        all_sales.assign(p=(all_sales["onpromotion"] == 1).astype("float32"))
        .groupby(["store_nbr", "family", "date"], observed=True)["p"]
        .mean()
        .rename("promo_share_store_family")
        .reset_index()
    )
    del all_sales

    # zero-filled grid from max(first_sale, start_buf) per series
    stats = pd.read_parquet(interim / "series_stats_full.parquet")
    series = series[key].merge(stats[key + ["first_sale"]], on=key, how="left")
    frames = []
    for (s, it), row in series.set_index(key).iterrows():
        d0 = max(row["first_sale"], start_buf)
        idx = pd.date_range(d0, end, freq="D")
        frames.append(pd.DataFrame({"store_nbr": s, "item_nbr": it, "date": idx}))
    grid = pd.concat(frames, ignore_index=True)

    panel = grid.merge(sales, on=key + ["date"], how="left")
    panel["unit_sales"] = panel["unit_sales"].fillna(0.0).astype("float32")
    # grid-filled rows have no promo record; assumption: promoted items sell >= 1
    panel["onpromotion"] = panel["onpromotion"].fillna(0).replace(-1, 0).astype("int8")

    panel = panel.sort_values(key + ["date"]).reset_index(drop=True)
    g = panel.groupby(key, observed=True)["unit_sales"]

    # origin-anchored historical features (windows END at the row's date)
    for k in (1, 2, 3):
        panel[f"recent_{k}"] = g.shift(k - 1)
    for w in cfg["features"]["rolling_windows_mean_std"]:
        panel[f"roll_mean_{w}"] = g.transform(lambda s, w=w: s.rolling(w, min_periods=1).mean())
        panel[f"roll_std_{w}"] = g.transform(lambda s, w=w: s.rolling(w, min_periods=2).std())
    for w in cfg["features"]["rolling_windows_median"]:
        panel[f"roll_median_{w}"] = g.transform(lambda s, w=w: s.rolling(w, min_periods=1).median())
    for w in cfg["features"]["rolling_windows_maxmin"]:
        panel[f"roll_max_{w}"] = g.transform(lambda s, w=w: s.rolling(w, min_periods=1).max())
        panel[f"roll_min_{w}"] = g.transform(lambda s, w=w: s.rolling(w, min_periods=1).min())
    zw = cfg["features"]["zero_ratio_window"]
    panel["zero_ratio_28"] = g.transform(
        lambda s, w=zw: s.eq(0).rolling(w, min_periods=1).mean()
    )
    panel["cv_28"] = panel["roll_std_28"] / (panel["roll_mean_28"] + EPS)

    # future-known covariates on the panel date itself
    panel = _add_calendar(panel, cfg)
    holidays = pd.read_parquet(interim / "holidays_per_store.parquet")
    panel = panel.merge(holidays, on=["store_nbr", "date"], how="left")
    panel["is_holiday"] = panel["is_holiday"].fillna(0).astype("int8")
    panel = panel.merge(items[["item_nbr", "family"]], on="item_nbr", how="left")
    panel = panel.merge(promo_share, on=["store_nbr", "family", "date"], how="left")
    panel["promo_share_store_family"] = panel["promo_share_store_family"].fillna(0.0)
    panel = _add_promo_dynamics(panel, key)
    return panel


def _add_calendar(df: pd.DataFrame, cfg) -> pd.DataFrame:
    d = df["date"]
    df["day_of_week"] = d.dt.dayofweek.astype("int8")
    df["week_of_year"] = d.dt.isocalendar().week.astype("int8")
    df["month"] = d.dt.month.astype("int8")
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
    df["is_payday"] = ((d.dt.day == 15) | (d.dt.day == d.dt.days_in_month)).astype("int8")
    eq0 = pd.Timestamp(cfg["earthquake"]["date"])
    eq1 = eq0 + pd.Timedelta(days=cfg["earthquake"]["window_days"] - 1)
    df["is_earthquake_window"] = ((d >= eq0) & (d <= eq1)).astype("int8")
    return df


def _add_promo_dynamics(panel: pd.DataFrame, key: list[str]) -> pd.DataFrame:
    """Promo start/end/streak/days-since features from the (plan-based) promo flag."""
    p = panel.groupby(key, observed=True)["onpromotion"]
    prev = p.shift(1).fillna(0).astype("int8")
    panel["promo_start"] = ((panel["onpromotion"] == 1) & (prev == 0)).astype("int8")
    panel["promo_end_prev"] = ((panel["onpromotion"] == 0) & (prev == 1)).astype("int8")

    # streak length while on promo; days since last promo ended while off promo
    def _streak(s: pd.Series) -> pd.Series:
        grp = (s != s.shift()).cumsum()
        return s.groupby(grp).cumcount() + 1

    streak = panel.groupby(key, observed=True)["onpromotion"].transform(_streak)
    panel["promo_streak"] = np.where(panel["onpromotion"] == 1, streak, 0).astype("int16")
    off_streak = np.where(panel["onpromotion"] == 0, streak, 0).astype("int32")
    # days_since_promo_end == off-streak, but only meaningful if a promo ever happened
    ever = panel.groupby(key, observed=True)["onpromotion"].transform("cummax")
    panel["days_since_promo_end"] = np.where(ever == 1, off_streak, -1).astype("int32")
    pp = panel["days_since_promo_end"]
    panel["post_promo_1_3"] = ((pp >= 1) & (pp <= 3)).astype("int8")
    panel["post_promo_1_7"] = ((pp >= 1) & (pp <= 7)).astype("int8")
    return panel


# ------------------------------------------------------------ supervised rows
ORIGIN_FEATURES = [
    "recent_1", "recent_2", "recent_3",
    "roll_mean_7", "roll_mean_14", "roll_mean_28",
    "roll_std_7", "roll_std_14", "roll_std_28",
    "roll_median_7", "roll_median_28",
    "roll_max_7", "roll_max_28", "roll_min_7", "roll_min_28",
    "zero_ratio_28", "cv_28",
]
TARGET_FEATURES = [
    "day_of_week", "week_of_year", "month", "is_weekend", "is_payday",
    "is_earthquake_window", "is_holiday", "onpromotion",
    "promo_start", "promo_streak", "days_since_promo_end",
    "post_promo_1_3", "post_promo_1_7", "promo_share_store_family",
]
TARGET_LAGS = [7, 14, 21, 28, 35, 56]


def make_supervised(panel: pd.DataFrame, origins: list[pd.Timestamp], horizon: int = 7) -> pd.DataFrame:
    """Long-format supervised rows: one row per (series, origin, horizon)."""
    key = ["store_nbr", "item_nbr"]
    panel = panel.set_index(key + ["date"]).sort_index()
    sales_lookup = panel["unit_sales"]

    rows = []
    for origin in origins:
        origin = pd.Timestamp(origin)
        at_origin = panel.xs(origin, level="date", drop_level=True)[ORIGIN_FEATURES]
        for h in range(1, horizon + 1):
            tdate = origin + pd.Timedelta(days=h)
            try:
                at_target = panel.xs(tdate, level="date", drop_level=True)
            except KeyError:
                continue
            r = at_origin.join(at_target[TARGET_FEATURES], how="inner")
            r["target"] = at_target["unit_sales"]
            for lag in TARGET_LAGS:
                ldate = tdate - pd.Timedelta(days=lag)
                assert ldate <= origin, f"target lag {lag} would leak (h={h})"
                lag_vals = sales_lookup.xs(ldate, level="date", drop_level=True)
                r[f"lag_{lag}"] = lag_vals
            r["origin_date"] = origin
            r["target_date"] = tdate
            r["horizon"] = np.int8(h)
            rows.append(r.reset_index())
    out = pd.concat(rows, ignore_index=True)

    # static metadata as categorical codes
    cfg = load_config()
    interim = resolve_path(cfg["paths"]["interim_dir"])
    items = pd.read_parquet(interim / "items.parquet")
    stores = pd.read_parquet(interim / "stores.parquet")
    out = out.merge(items, on="item_nbr", how="left").merge(stores, on="store_nbr", how="left")
    for c in ["family", "class", "city", "state", "type", "cluster"]:
        out[c] = out[c].astype("category")
    return out


if __name__ == "__main__":
    cfg = load_config()
    panel = build_daily_panel(cfg)
    print(f"panel: {len(panel):,} rows, {panel.shape[1]} cols")
