"""TFT-base variant, step 2 (local): assemble the TFT residual dataset under
tag 'tftbase' so the existing Phase 3/4 scripts run unchanged.

Inputs: results/tft/tft_oof.parquet (stale-TFT OOF predictions, from lab06)
        results/tft/tft_fold{1,2,3}.parquet + tft_holdout.parquet (per-fold TFTs)
Outputs: data/processed/oof_residuals_tftbase.parquet (+ tag-aliased copies of
         panel/supervised/baselines so --tag tftbase resolves).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import splits  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402

KEY = ["store_nbr", "item_nbr", "origin_date", "horizon"]


def main() -> None:
    cfg = load_config()
    processed = resolve_path(cfg["paths"]["processed_dir"])
    results = resolve_path(cfg["paths"]["results_dir"])

    parts = [pd.read_parquet(results / "tft" / "tft_oof.parquet")]
    for f in ["fold1", "fold2", "fold3", "holdout"]:
        parts.append(pd.read_parquet(results / "tft" / f"tft_{f}.parquet"))
    tft = pd.concat(parts, ignore_index=True)
    tft["origin_date"] = pd.to_datetime(tft["origin_date"])
    tft["target_date"] = pd.to_datetime(tft["target_date"])
    assert not tft.duplicated(KEY).any()

    sup = pd.read_parquet(processed / "supervised_final.parquet",
                          columns=KEY + ["target_date", "target"])
    oof = sup.merge(tft.rename(columns={"tft": "base_oof"})[KEY + ["base_oof"]],
                    on=KEY, how="inner")
    oof["residual"] = oof["target"] - oof["base_oof"]

    stages = splits.oof_base_cutoffs(cfg)[["origin_date", "stage"]]
    oof = oof.merge(stages, on="origin_date", how="left")
    oof = oof.dropna(subset=["stage"])
    print("stage coverage:\n", oof.drop_duplicates(["origin_date"])["stage"].value_counts())

    oof.to_parquet(processed / "oof_residuals_tftbase.parquet", index=False)
    print(f"oof_residuals_tftbase: {len(oof):,} rows")

    # tag aliases so 04/05 --tag tftbase find their inputs unchanged
    for src, dst in [
        (processed / "panel_final.parquet", processed / "panel_tftbase.parquet"),
        (processed / "supervised_final.parquet", processed / "supervised_tftbase.parquet"),
        (results / "predictions" / "baselines_final.parquet",
         results / "predictions" / "baselines_tftbase.parquet"),
    ]:
        if not dst.exists():
            shutil.copyfile(src, dst)
            print(f"aliased {src.name} -> {dst.name}")


if __name__ == "__main__":
    main()
