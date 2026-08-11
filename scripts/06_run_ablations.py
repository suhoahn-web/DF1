"""Phase 5 ablations: retrain the gate with feature groups removed.

  A2 = drop recent_error context
  A3 = drop promotion context
  A5 = drop demand_state (volatility) context   [optional in spec]

Everything else (TCN, r_hat, base, splits, seeds) identical to A4.

Usage:
    python scripts/06_run_ablations.py [--tag dev|final]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import regimes  # noqa: E402
from src.evaluation.metrics import evaluate  # noqa: E402
from src.training import train_gate as tg  # noqa: E402
from src.utils.config import load_config, resolve_path, set_seed  # noqa: E402

ABLATIONS = {
    "A2_no_recent_error": ["recent_error"],
    "A3_no_promotion": ["promotion"],
    "A5_no_demand_state": ["demand_state"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dev", choices=["dev", "final", "tftbase"])
    args = ap.parse_args()

    dcfg = load_config()
    set_seed(load_config("tcn_gate.yaml")["seed"])
    results = resolve_path(dcfg["paths"]["results_dir"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = ["store_nbr", "item_nbr", "origin_date", "horizon"]

    ctx = tg.build_context_table(args.tag)
    ev = pd.read_parquet(results / "predictions" / f"phase4_{args.tag}.parquet")

    for name, drop in ABLATIONS.items():
        _, preds, _ = tg.train_gate(args.tag, drop_groups=drop, device=device, ctx=ctx)
        col = f"gated_{name}"
        sub = preds[preds["stage"] == "eval_input"][key + ["gated"]].rename(columns={"gated": col})
        ev = ev.merge(sub, on=key, how="left")

    ev.to_parquet(results / "predictions" / f"ablations_{args.tag}.parquet", index=False)

    models = ["lightgbm", "always_on", "constant_gate", "gated"] + [
        f"gated_{n}" for n in ABLATIONS
    ]
    dev_rows = ev[ev["fold"] != "holdout"]
    overall = pd.concat([evaluate(dev_rows, m).assign(model=m) for m in models], ignore_index=True)
    by_regime = regimes.regime_metrics(dev_rows, models)
    overall.to_csv(results / "ablations" / f"ablation_overall_{args.tag}.csv", index=False)
    by_regime.to_csv(results / "ablations" / f"ablation_regimes_{args.tag}.csv", index=False)
    print(overall[["model", "wape", "mae", "mase"]].to_string(index=False))


if __name__ == "__main__":
    (resolve_path(load_config()["paths"]["results_dir"]) / "ablations").mkdir(exist_ok=True)
    main()
