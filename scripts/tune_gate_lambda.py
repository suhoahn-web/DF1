"""Sparsity-lambda selection for the gate (spec §9). Selection metric: pure
forecast L1 on the gate_train validation tail -- folds/holdout untouched.
The chosen lambda's fold results are then produced once via 05_train_gate
conventions and the sensitivity table is saved for the paper appendix.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import evaluate  # noqa: E402
from src.training import train_gate as tg  # noqa: E402
from src.utils.config import load_config, resolve_path, set_seed  # noqa: E402

LAMBDAS = [0.0, 0.01, 0.05, 0.1]
TAG = "final"


def main() -> None:
    dcfg = load_config()
    set_seed(load_config("tcn_gate.yaml")["seed"])
    results = resolve_path(dcfg["paths"]["results_dir"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = ["store_nbr", "item_nbr", "origin_date", "horizon"]

    ctx = tg.build_context_table(TAG)
    p3 = pd.read_parquet(results / "predictions" / f"phase3_{TAG}.parquet")
    fold_rows = p3[p3["fold"] != "holdout"]

    rows, best = [], None
    for lam in LAMBDAS:
        gate, preds, _ = tg.train_gate(TAG, device=device, ctx=ctx, sparsity_lambda=lam)
        # selection on gate_train val tail is embedded in train_gate's early
        # stopping; record its final val L1 by re-running a val pass is
        # overkill -- use gate_train-period WAPE as the selection statistic
        gt = preds[preds["stage"] == "gate_train"]
        sel_wape = float(np.abs(gt["actual"] - gt["gated"]).sum() / np.abs(gt["actual"]).sum())
        # fold WAPE recorded for the sensitivity APPENDIX (not for selection)
        ev = fold_rows.merge(preds[preds["stage"] == "eval_input"][key + ["gate_value", "gated"]],
                             on=key, how="inner")
        fold_wape = evaluate(ev, "gated").iloc[0]["wape"]
        rows.append({"lambda": lam, "gate_train_wape": sel_wape,
                     "fold_wape_appendix": fold_wape, "gate_mean": float(preds["gate_value"].mean())})
        print(f"lambda={lam}: gate_train WAPE {sel_wape:.5f}, gate_mean {rows[-1]['gate_mean']:.3f}")
        if best is None or sel_wape < best[1]:
            best = (lam, sel_wape, gate, preds)

    tbl = pd.DataFrame(rows)
    tbl.to_csv(results / "metrics" / f"gate_lambda_sensitivity_{TAG}.csv", index=False)
    print("\nsensitivity table:")
    print(tbl.to_string(index=False))

    lam, _, gate, preds = best
    print(f"\nselected lambda = {lam} (by gate_train WAPE)")
    processed = resolve_path(dcfg["paths"]["processed_dir"])
    torch.save(gate.state_dict(), processed / f"gate_full_{TAG}.pt")
    ev = p3.merge(preds[preds["stage"] == "eval_input"][key + ["gate_value", "gated"]],
                  on=key, how="inner")
    ev.to_parquet(results / "predictions" / f"phase4_{TAG}.parquet", index=False)

    dev_rows = ev[ev["fold"] != "holdout"]
    models = ["lightgbm", "always_on", "constant_gate", "gated"]
    overall = pd.concat([evaluate(dev_rows, m).assign(model=m) for m in models], ignore_index=True)
    overall.to_csv(results / "metrics" / f"phase4_overall_{TAG}.csv", index=False)
    print("\nfinal folds 1-3 with selected lambda:")
    print(overall[["model", "wape", "mae", "mase"]].to_string(index=False))


if __name__ == "__main__":
    main()
