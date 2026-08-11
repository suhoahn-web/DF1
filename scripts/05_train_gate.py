"""Phase 4 runner: full context gate (A4) -> folds 1-3 comparison vs
A0/A1/A1.5, plus gate-behavior diagnostics (spec §9, §19).

Usage:
    python scripts/05_train_gate.py [--tag dev|final]
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
from src.training import train_gate as tg  # noqa: E402
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

    gate, preds, stats = tg.train_gate(args.tag, drop_groups=None, device=device)
    torch.save(gate.state_dict(), processed / f"gate_full_{args.tag}.pt")
    stats.to_parquet(processed / f"gate_norm_stats_{args.tag}.parquet")

    # ---- merge with phase3 fold table ----
    p3 = pd.read_parquet(results / "predictions" / f"phase3_{args.tag}.parquet")
    key = ["store_nbr", "item_nbr", "origin_date", "horizon"]
    ev = p3.merge(
        preds[preds["stage"] == "eval_input"][key + ["gate_value", "gated"]],
        on=key, how="inner",
    )
    ev.to_parquet(results / "predictions" / f"phase4_{args.tag}.parquet", index=False)

    models = ["lightgbm", "always_on", "constant_gate", "gated"]
    dev_rows = ev[ev["fold"] != "holdout"]
    overall = pd.concat([evaluate(dev_rows, m).assign(model=m) for m in models], ignore_index=True)
    by_regime = regimes.regime_metrics(dev_rows, models)
    overall.to_csv(results / "metrics" / f"phase4_overall_{args.tag}.csv", index=False)
    by_regime.to_csv(results / "metrics" / f"phase4_regimes_{args.tag}.csv", index=False)

    print("\nfolds 1-3 (holdout excluded):")
    print(overall[["model", "wape", "mae", "rmse", "mase"]].to_string(index=False))
    print("\nWAPE by regime:")
    print(by_regime.pivot_table(index="regime", columns="model", values="wape").round(4).to_string())

    # ---- gate diagnostics (spec §9) ----
    d = dev_rows
    diag = {
        "gate_mean": d["gate_value"].mean(),
        "gate_median": d["gate_value"].median(),
        "gate_std": d["gate_value"].std(),
        "gate_p10": d["gate_value"].quantile(0.1),
        "gate_p90": d["gate_value"].quantile(0.9),
        "gate_mean_promo": d.loc[d["regime_promotion"] == 1, "gate_value"].mean(),
        "gate_mean_nonpromo": d.loc[d["regime_promotion"] == 0, "gate_value"].mean(),
        "gate_mean_holiday": d.loc[d["regime_holiday"] == 1, "gate_value"].mean(),
        "gate_mean_highvol": d.loc[d["regime_high_vol"] == 1, "gate_value"].mean(),
        "gate_mean_normal": d.loc[d["regime_normal"] == 1, "gate_value"].mean(),
    }
    pd.DataFrame([diag]).to_csv(results / "metrics" / f"gate_diagnostics_{args.tag}.csv", index=False)
    print("\ngate diagnostics:")
    for k, v in diag.items():
        print(f"  {k}: {v:.4f}")

    # mean gate by recent-error decile (uses gate_train-period context stats)
    ctx_err = tg.build_context_table(args.tag)
    ctx_err = ctx_err[ctx_err["stage"] == "eval_input"].merge(
        d[key], on=key, how="inner"
    )
    if len(ctx_err):
        merged = d.merge(ctx_err[key + ["res_mae_28"]], on=key, how="left")
        merged["err_decile"] = pd.qcut(merged["res_mae_28"].rank(method="first"), 10, labels=False)
        by_decile = merged.groupby("err_decile", observed=True)["gate_value"].mean()
        by_decile.to_csv(results / "metrics" / f"gate_by_error_decile_{args.tag}.csv")
        print("\nmean gate by recent-error decile (0=low err .. 9=high):")
        print(by_decile.round(4).to_string())


if __name__ == "__main__":
    main()
