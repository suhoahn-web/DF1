"""Phase 5: statistical tests, operational cost, oracle-gate analysis, and
auto-generated paper tables (spec §17-§20). All computed from the saved
long-format prediction tables -- never recomputes models.

Usage:
    python scripts/07_make_paper_tables.py [--tag dev|final] [--include-holdout]

--include-holdout is the ONE-SHOT final evaluation switch. Run it exactly once,
after every model choice is frozen, and record the run date below.
Holdout evaluation runs: (none yet)
"""
from __future__ import annotations

import argparse
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

MODELS = ["seasonal_naive", "lightgbm", "always_on", "constant_gate", "gated"]


def oracle_gate_analysis(df: pd.DataFrame) -> dict:
    """Headroom: oracle g*=1 iff applying the correction reduces the error."""
    y = df["actual"].to_numpy()
    base = df["lightgbm"].to_numpy()
    ao = df["always_on"].to_numpy()
    gated = df["gated"].to_numpy()

    better = np.abs(y - ao) < np.abs(y - base)
    oracle = np.where(better, ao, base)

    def _wape(yhat):
        return np.abs(y - yhat).sum() / (np.abs(y).sum() + 1e-9)

    w_base, w_ao, w_gated, w_oracle = map(_wape, [base, ao, gated, oracle])
    headroom = w_base - w_oracle
    achieved = (w_base - w_gated) / headroom if headroom > 0 else np.nan
    # gate-benefit alignment: was the gate high when correction helped?
    g = df["gate_value"].to_numpy()
    return {
        "wape_base": w_base, "wape_always_on": w_ao, "wape_gated": w_gated,
        "wape_oracle": w_oracle, "headroom_wape": headroom,
        "achieved_fraction": achieved,
        "correction_helps_rate": float(better.mean()),
        "gate_mean_when_helps": float(g[better].mean()),
        "gate_mean_when_hurts": float(g[~better].mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dev", choices=["dev", "final"])
    ap.add_argument("--include-holdout", action="store_true")
    args = ap.parse_args()

    dcfg = load_config()
    results = resolve_path(dcfg["paths"]["results_dir"])
    for sub in ["metrics", "statistical_tests", "figures"]:
        (results / sub).mkdir(exist_ok=True)

    ev = pd.read_parquet(results / "predictions" / f"phase4_{args.tag}.parquet")
    scope = ev if args.include_holdout else ev[ev["fold"] != "holdout"]
    label = "with_holdout" if args.include_holdout else "folds"

    # ---- main table (spec §20) ----
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
    main_table.to_csv(results / "metrics" / f"main_table_{args.tag}_{label}.csv", index=False)

    # ---- per-fold robustness ----
    fold_tbl = pd.concat(
        [evaluate(scope[scope["fold"] == f], m).assign(model=m, fold=f)
         for f in scope["fold"].unique() for m in MODELS],
        ignore_index=True,
    )
    fold_tbl.to_csv(results / "metrics" / f"fold_robustness_{args.tag}_{label}.csv", index=False)

    # ---- statistical tests (family: key comparisons, Holm-corrected) ----
    comparisons = [
        ("gated", "lightgbm"), ("gated", "always_on"), ("gated", "constant_gate"),
        ("always_on", "lightgbm"), ("constant_gate", "lightgbm"),
    ]
    tests = holm_correction([paired_comparison(scope, a, b) for a, b in comparisons])
    tests.to_csv(results / "statistical_tests" / f"wilcoxon_{args.tag}_{label}.csv", index=False)

    # ---- operational cost ----
    cost = oc.cost_table(scope, MODELS)
    cost.to_csv(results / "metrics" / f"operational_cost_{args.tag}_{label}.csv", index=False)

    # ---- oracle gate ----
    oracle = pd.DataFrame([oracle_gate_analysis(scope)])
    oracle.to_csv(results / "metrics" / f"oracle_gate_{args.tag}_{label}.csv", index=False)

    print("main table:")
    print(main_table.round(4).to_string(index=False))
    print("\nstatistical tests (Holm):")
    print(tests[["model_a", "model_b", "mean_diff", "ci_low", "ci_high",
                 "win_rate_a", "wilcoxon_p", "holm_p"]].round(5).to_string(index=False))
    print("\noracle gate:")
    print(oracle.round(4).to_string(index=False))
    print("\noperational cost (% vs LightGBM):")
    print(cost.pivot_table(index="ratio", columns="model", values="pct_vs_base").round(3).to_string())


if __name__ == "__main__":
    main()
