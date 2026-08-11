"""Synthetic fixtures: a tiny fake project (configs + interim data) in tmp_path."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REAL_ROOT))

from src.utils import config as config_mod  # noqa: E402


@pytest.fixture()
def synth_root(tmp_path, monkeypatch):
    """Fake PROJECT_ROOT with real configs and synthetic interim parquet files."""
    shutil.copytree(REAL_ROOT / "configs", tmp_path / "configs")
    monkeypatch.setattr(config_mod, "PROJECT_ROOT", tmp_path)

    cfg = config_mod.load_config()
    interim = tmp_path / cfg["paths"]["interim_dir"]
    interim.mkdir(parents=True)
    (tmp_path / cfg["paths"]["processed_dir"]).mkdir(parents=True)

    rng = np.random.default_rng(0)
    start, end = pd.Timestamp("2015-01-01"), pd.Timestamp("2017-08-15")
    dates = pd.date_range(start, end, freq="D")
    frames = []
    for store, item in [(1, 100), (1, 200), (2, 100)]:
        sales = rng.poisson(5, len(dates)).astype("float32")
        promo = (rng.random(len(dates)) < 0.1).astype("int8")
        sales[promo == 1] *= 3
        df = pd.DataFrame(
            {
                "date": dates,
                "store_nbr": np.int16(store),
                "item_nbr": np.int32(item),
                "unit_sales": sales,
                "onpromotion": promo,
            }
        )
        # sparsify: drop zero-sale rows like the real train.csv
        frames.append(df[df["unit_sales"] > 0])
    sales_clean = pd.concat(frames, ignore_index=True)
    sales_clean.to_parquet(interim / "sales_clean.parquet", index=False)

    stats = (
        sales_clean.groupby(["store_nbr", "item_nbr"], observed=True)
        .agg(first_sale=("date", "min"), last_sale=("date", "max"), n_pos_rows=("date", "size"))
        .reset_index()
    )
    stats.to_parquet(interim / "series_stats_full.parquet", index=False)

    pd.DataFrame(
        {"item_nbr": np.int32([100, 200]), "family": ["GROCERY I", "BEVERAGES"],
         "class": [1010, 2020], "perishable": [0, 1]}
    ).to_parquet(interim / "items.parquet", index=False)
    pd.DataFrame(
        {"store_nbr": np.int16([1, 2]), "city": ["Quito", "Guayaquil"],
         "state": ["Pichincha", "Guayas"], "type": ["A", "B"], "cluster": [1, 2]}
    ).to_parquet(interim / "stores.parquet", index=False)
    pd.DataFrame(
        {"store_nbr": np.int16([1, 2]), "date": [pd.Timestamp("2016-12-25")] * 2,
         "is_holiday": np.int8([1, 1])}
    ).to_parquet(interim / "holidays_per_store.parquet", index=False)

    return tmp_path
