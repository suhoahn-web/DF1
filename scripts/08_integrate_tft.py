"""Merge lab06 TFT predictions into the final prediction table and regenerate
the main comparison table with the TFT row (folds 1-3 only; holdout untouched).

Usage:
    python scripts/08_integrate_tft.py
Expects results/tft/tft_fold{1,2,3}.parquet + tft_holdout.parquet locally
(retrieved from lab06).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import operational_cost as oc  # noqa: E402
from src.evaluation import regimes  # noqa: E402
from src.evaluation.metrics import evaluate  # noqa: E402
from src.evaluation.statistical_tests import holm_correction, paired_comparison  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402

MODELS = ["seasonal_naive", "lightgbm", "tft", "always_on", "constant_gate", "gated"]


def main() -> None:
    cfg = load_config()
    results = resolve_path(cfg["paths"]["results_dir"])
    tft_dir = results / "tft"

    parts = [pd.read_parquet(p) for p in sorted(tft_dir.glob("tft_*.parquet"))]
    tft = pd.concat(parts, ignore_index=True)
    tft["origin_date"] = pd.to_datetime(tft["origin_date"])
    print(f"TFT predictions: {len(tft):,} rows from {len(parts)} folds")

    ev = pd.read_parquet(results / "predictions" / "phase4_final.parquet")
    key = ["store_nbr", "item_nbr", "origin_date", "horizon"]
    merged = ev.merge(tft[key + ["tft"]], on=key, how="left")
    n_missing = merged["tft"].isna().sum()
    if n_missing:
        print(f"WARNING: {n_missing:,} rows without TFT prediction "
              f"({100 * n_missing / len(merged):.2f}%) — check series coverage")
    merged.to_parquet(results / "predictions" / "phase4_final_with_tft.parquet", index=False)

    scope = merged[(merged["fold"] != "holdout") & merged["tft"].notna()]
    rows = []
    for m in MODELS:
        r = evaluate(scope, m).iloc[0]
        promo = evaluate(scope[scope["regime_promotion"] == 1], m).iloc[0]["wape"]
        pp = evaluate(scope[scope["regime_post_promo"] == 1], m).iloc[0]["wape"]
        hv = evaluate(scope[scope["regime_high_vol"] == 1], m).iloc[0]["wape"]
        cost31 = oc.asymmetric_cost(scope["actual"].to_numpy(), scope[m].to_numpy(), 3, 1).mean()
        rows.append({"model": m, "wape": r["wape"], "mae": r["mae"], "rmse": r["rmse"],
                     "mase": r.get("mase", np.nan), "promo_wape": promo,
                     "post_promo_wape": pp, "high_vol_wape": hv, "cost_3_1": cost31})
    main_table = pd.DataFrame(rows)
    main_table.to_csv(results / "metrics" / "main_table_final_with_tft.csv", index=False)
    print("\nmain table (folds 1-3, with TFT):")
    print(main_table.round(4).to_string(index=False))

    tests = holm_correction([
        paired_comparison(scope, "gated", "tft"),
        paired_comparison(scope, "lightgbm", "tft"),
    ])
    tests.to_csv(results / "statistical_tests" / "wilcoxon_tft_final.csv", index=False)
    print("\nTFT statistical comparisons (Holm):")
    print(tests[["model_a", "model_b", "mean_diff", "ci_low", "ci_high",
                 "win_rate_a", "holm_p"]].round(5).to_string(index=False))


if __name__ == "__main__":
    main()
