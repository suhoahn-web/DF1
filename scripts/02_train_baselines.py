"""Phase 2: Seasonal Naive + global LightGBM on the rolling folds, plus
temporal OOF residual generation. Produces the long-format prediction table
that drives all later paper tables (spec §16).

Usage:
    python scripts/02_train_baselines.py [--tag dev|final]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import splits  # noqa: E402
from src.evaluation import regimes  # noqa: E402
from src.evaluation.metrics import evaluate  # noqa: E402
from src.models import lightgbm_global as lgbm  # noqa: E402
from src.training import make_oof_residuals  # noqa: E402
from src.utils.config import load_config, resolve_path, set_seed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dev", choices=["dev", "final"])
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["sampling"]["random_seed"])
    processed = resolve_path(cfg["paths"]["processed_dir"])
    results = resolve_path(cfg["paths"]["results_dir"])
    (results / "metrics").mkdir(exist_ok=True)
    (results / "predictions").mkdir(exist_ok=True)

    sup = pd.read_parquet(processed / f"supervised_{args.tag}.parquet")
    folds = splits.fold_origins(cfg)

    preds = []
    for fold_name, origins in folds.items():
        fold_rows = sup[sup["origin_date"].isin(origins)].copy()
        cutoff = min(origins) - pd.Timedelta(days=1)
        train_rows = sup[sup["target_date"] <= cutoff]
        assert train_rows["target_date"].max() < fold_rows["target_date"].min()

        model = lgbm.train(train_rows)
        fold_rows["lightgbm"] = lgbm.predict(model, fold_rows)
        fold_rows["seasonal_naive"] = fold_rows["lag_7"]
        fold_rows["fold"] = fold_name

        # high-vol threshold from this fold's TRAINING rows only
        hv = regimes.high_vol_threshold(train_rows, cfg)
        fold_rows = regimes.add_regime_flags(fold_rows, hv)
        fold_rows["hv_threshold"] = hv
        preds.append(fold_rows)
        print(f"{fold_name}: train {len(train_rows):,} rows -> predict {len(fold_rows):,} rows")

    table = pd.concat(preds, ignore_index=True).rename(columns={"target": "actual"})
    keep = (
        ["store_nbr", "item_nbr", "origin_date", "target_date", "horizon", "fold", "actual",
         "seasonal_naive", "lightgbm", "onpromotion", "cv_28", "hv_threshold"]
        + regimes.REGIME_COLS
    )
    table[keep].to_parquet(results / "predictions" / f"baselines_{args.tag}.parquet", index=False)

    # overall + regime metrics (exclude holdout from all Phase-2 reporting)
    dev_folds = table[table["fold"] != "holdout"]
    overall = pd.concat(
        [evaluate(dev_folds, m).assign(model=m) for m in ["seasonal_naive", "lightgbm"]],
        ignore_index=True,
    )
    by_regime = regimes.regime_metrics(dev_folds, ["seasonal_naive", "lightgbm"])
    overall.to_csv(results / "metrics" / f"baseline_overall_{args.tag}.csv", index=False)
    by_regime.to_csv(results / "metrics" / f"baseline_regimes_{args.tag}.csv", index=False)
    print("\noverall (folds 1-3, holdout excluded):")
    print(overall[["model", "wape", "mae", "rmse", "rmsle", "mase"]].to_string(index=False))

    make_oof_residuals.run(args.tag, cfg)


if __name__ == "__main__":
    main()
