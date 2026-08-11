"""Deterministic preprocessing of the raw Favorita CSVs.

Two passes over train.csv (125M rows, too large to load at once):

  Pass 1  chunked scan of the FULL file -> per-series first/last sale date and
          row counts. First-sale dates must come from the full history so that
          the zero-filling rule ("carried but zero" vs "not yet carried") is
          not distorted by the analysis-period restriction.

  Pass 2  chunked filter (date >= analysis start - buffer) -> cleaned parquet.

Cleaning rules are frozen in configs/data.yaml (see CLAUDE.md §4).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import load_config, resolve_path

TRAIN_DTYPES = {
    "id": "int64",
    "store_nbr": "int16",
    "item_nbr": "int32",
    "unit_sales": "float32",
}
CHUNKSIZE = 5_000_000


def _raw(cfg) -> Path:
    return resolve_path(cfg["paths"]["raw_dir"])


def _interim(cfg) -> Path:
    return resolve_path(cfg["paths"]["interim_dir"])


def scan_series_stats(cfg) -> pd.DataFrame:
    """Pass 1: full-file per-series stats (first/last sale, counts)."""
    out = _interim(cfg) / "series_stats_full.parquet"
    if out.exists():
        return pd.read_parquet(out)

    parts = []
    reader = pd.read_csv(
        _raw(cfg) / "train.csv",
        usecols=["date", "store_nbr", "item_nbr", "unit_sales"],
        dtype={k: v for k, v in TRAIN_DTYPES.items() if k != "id"},
        parse_dates=["date"],
        chunksize=CHUNKSIZE,
    )
    for i, chunk in enumerate(reader):
        pos = chunk[chunk["unit_sales"] > 0]
        g = pos.groupby(["store_nbr", "item_nbr"], observed=True).agg(
            first_sale=("date", "min"),
            last_sale=("date", "max"),
            n_pos_rows=("date", "size"),
        )
        parts.append(g.reset_index())
        print(f"  pass1 chunk {i}: {len(chunk):,} rows")

    stats = (
        pd.concat(parts)
        .groupby(["store_nbr", "item_nbr"], observed=True)
        .agg(
            first_sale=("first_sale", "min"),
            last_sale=("last_sale", "max"),
            n_pos_rows=("n_pos_rows", "sum"),
        )
        .reset_index()
    )
    stats.to_parquet(out, index=False)
    print(f"pass1 done: {len(stats):,} series -> {out.name}")
    return stats


def build_clean_sales(cfg) -> Path:
    """Pass 2: filtered, cleaned long sales table (sparse; zeros NOT filled here)."""
    out = _interim(cfg) / "sales_clean.parquet"
    if out.exists():
        return out

    start = pd.Timestamp(cfg["analysis_period"]["start"]) - pd.Timedelta(
        days=cfg["analysis_period"]["feature_buffer_days"]
    )
    end = pd.Timestamp(cfg["analysis_period"]["end"])

    parts = []
    reader = pd.read_csv(
        _raw(cfg) / "train.csv",
        usecols=["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"],
        dtype={k: v for k, v in TRAIN_DTYPES.items() if k != "id"},
        parse_dates=["date"],
        chunksize=CHUNKSIZE,
    )
    for i, chunk in enumerate(reader):
        m = (chunk["date"] >= start) & (chunk["date"] <= end)
        if m.any():
            sub = chunk.loc[m].copy()
            if cfg["cleaning"]["clip_negative_sales_to_zero"]:
                sub["unit_sales"] = sub["unit_sales"].clip(lower=0)
            # tri-state -> int8: 0 False, 1 True, -1 missing (resolved at grid fill)
            sub["onpromotion"] = (
                sub["onpromotion"]
                .map({True: 1, False: 0, "True": 1, "False": 0})
                .fillna(-1)
                .astype("int8")
            )
            parts.append(sub)
        print(f"  pass2 chunk {i}: kept {int(m.sum()):,}")

    sales = pd.concat(parts, ignore_index=True)
    sales.to_parquet(out, index=False)
    print(f"pass2 done: {len(sales):,} rows -> {out.name}")
    return out


def build_holidays(cfg) -> pd.DataFrame:
    """National/regional/local holiday flags per (date, store), transfer-aware."""
    raw = _raw(cfg)
    hol = pd.read_csv(raw / "holidays_events.csv", parse_dates=["date"])
    stores = pd.read_csv(raw / "stores.csv")

    # A transferred holiday is celebrated on the row with type == "Transfer";
    # the original row (transferred == True) is a normal day.
    hol = hol[~((hol["type"] == "Holiday") & (hol["transferred"] == True))]  # noqa: E712
    hol = hol[hol["type"] != "Work Day"]  # Work Day = compensating workday, not a holiday

    nat = hol[hol["locale"] == "National"][["date"]].drop_duplicates()
    nat["is_holiday_national"] = np.int8(1)

    reg = hol[hol["locale"] == "Regional"][["date", "locale_name"]].drop_duplicates()
    loc = hol[hol["locale"] == "Local"][["date", "locale_name"]].drop_duplicates()

    grid = stores[["store_nbr", "city", "state"]].merge(nat, how="cross")
    grid = grid.rename(columns={"date": "hol_date"})  # cross join then explode is heavy;
    # simpler: build per-store holiday table by concatenation
    rows = []
    for _, s in stores.iterrows():
        d_nat = nat["date"]
        d_reg = reg.loc[reg["locale_name"] == s["state"], "date"]
        d_loc = loc.loc[loc["locale_name"] == s["city"], "date"]
        dates = pd.concat([d_nat, d_reg, d_loc]).drop_duplicates()
        rows.append(pd.DataFrame({"store_nbr": s["store_nbr"], "date": dates}))
    per_store = pd.concat(rows, ignore_index=True)
    per_store["is_holiday"] = np.int8(1)
    return per_store


def run(cfg=None) -> dict:
    cfg = cfg or load_config()
    stats = scan_series_stats(cfg)
    sales_path = build_clean_sales(cfg)
    holidays = build_holidays(cfg)
    holidays.to_parquet(_interim(cfg) / "holidays_per_store.parquet", index=False)

    for name in ["stores", "items"]:
        pd.read_csv(_raw(cfg) / f"{name}.csv").to_parquet(
            _interim(cfg) / f"{name}.parquet", index=False
        )
    print("preprocess complete")
    return {"series_stats": stats, "sales_path": sales_path}


if __name__ == "__main__":
    run()
