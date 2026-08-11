"""Phase 3 runner: TCN residual corrector -> A0 vs A1 (always-on) vs A1.5
(constant gate) on folds 1-3. Holdout rows are written but never scored here.

Usage:
    python scripts/04_train_residual.py [--tag dev|final]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import splits  # noqa: E402
from src.evaluation import regimes  # noqa: E402
from src.evaluation.metrics import evaluate  # noqa: E402
from src.training import train_residual as tr  # noqa: E402
from src.utils.config import load_config, resolve_path, set_seed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dev", choices=["dev", "final", "tftbase"])
    args = ap.parse_args()

    dcfg = load_config()
    set_seed(load_config("tcn_gate.yaml")["seed"])
    processed = resolve_path(dcfg["paths"]["processed_dir"])
    results = resolve_path(dcfg["paths"]["results_dir"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    data = tr.build_sequence_dataset(args.tag)
    model = tr.train_tcn(data, device)
    torch.save(model.state_dict(), processed / f"tcn_{args.tag}.pt")

    rhat = tr.predict_residuals(model, data, device)
    rhat.to_parquet(processed / f"rhat_{args.tag}.parquet", index=False)
    c_star = tr.tune_constant_gate(rhat)
    (results / "metrics").mkdir(exist_ok=True)
    pd.DataFrame([{"c_star": c_star}]).to_csv(results / "metrics" / f"constant_gate_{args.tag}.csv", index=False)

    # ---- fold evaluation (A0 / A1 / A1.5), holdout excluded from scoring ----
    folds = splits.fold_origins(dcfg)
    origin_to_fold = {o: name for name, os_ in folds.items() for o in os_}
    ev = rhat[rhat["stage"] == "eval_input"].copy()
    ev["fold"] = ev["origin_date"].map(origin_to_fold)
    ev = ev.dropna(subset=["fold"])

    ev["lightgbm"] = ev["base_oof"]
    ev["always_on"] = np.clip(ev["base_oof"] + ev["r_hat"], 0, None)
    ev["constant_gate"] = np.clip(ev["base_oof"] + c_star * ev["r_hat"], 0, None)

    # regime flags + seasonal naive come from the baseline prediction table
    base_table = pd.read_parquet(results / "predictions" / f"baselines_{args.tag}.parquet")
    keep = ["store_nbr", "item_nbr", "origin_date", "horizon", "seasonal_naive",
            "onpromotion", "cv_28", "hv_threshold"] + regimes.REGIME_COLS
    ev = ev.merge(base_table[keep], on=["store_nbr", "item_nbr", "origin_date", "horizon"], how="left")
    ev.to_parquet(results / "predictions" / f"phase3_{args.tag}.parquet", index=False)

    models = ["lightgbm", "always_on", "constant_gate"]
    dev_rows = ev[ev["fold"] != "holdout"]
    overall = pd.concat([evaluate(dev_rows, m).assign(model=m) for m in models], ignore_index=True)
    by_regime = regimes.regime_metrics(dev_rows, models)
    overall.to_csv(results / "metrics" / f"phase3_overall_{args.tag}.csv", index=False)
    by_regime.to_csv(results / "metrics" / f"phase3_regimes_{args.tag}.csv", index=False)

    print("\nfolds 1-3 (holdout excluded):")
    print(overall[["model", "wape", "mae", "rmse", "mase"]].to_string(index=False))
    print("\nWAPE by regime:")
    print(by_regime.pivot_table(index="regime", columns="model", values="wape").round(4).to_string())


if __name__ == "__main__":
    main()
