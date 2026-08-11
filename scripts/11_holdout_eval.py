"""THE one-shot holdout evaluation (FREEZE_DECLARATION §4).

Evaluates the frozen models on the holdout window only (origins 2017-07-18 ..
2017-08-08). Run exactly once after the design freeze; record the run date in
FREEZE_DECLARATION.md. No retuning of any kind may follow.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import operational_cost as oc  # noqa: E402
from src.evaluation import regimes  # noqa: E402
from src.evaluation.metrics import evaluate  # noqa: E402
from src.evaluation.statistical_tests import holm_correction, paired_comparison  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402


def block(name: str, df: pd.DataFrame, models: list[str], base: str) -> None:
    results = resolve_path(load_config()["paths"]["results_dir"])
    hold = df[df["fold"] == "holdout"]
    print(f"\n===== {name}: holdout ({hold.groupby(['store_nbr','item_nbr']).ngroups:,} series) =====")

    overall = pd.concat([evaluate(hold, m).assign(model=m) for m in models], ignore_index=True)
    print(overall[["model", "wape", "mae", "rmse", "mase"]].round(4).to_string(index=False))

    by_regime = regimes.regime_metrics(hold, models)
    print("\nWAPE by regime:")
    print(by_regime.pivot_table(index="regime", columns="model", values="wape").round(4).to_string())

    cost = oc.cost_table(hold, models, normalizer_col=base)
    print("\ncost (% vs base):")
    print(cost.pivot_table(index="ratio", columns="model", values="pct_vs_base").round(3).to_string())

    tests = holm_correction([
        paired_comparison(hold, "gated", base),
        paired_comparison(hold, "gated", "always_on"),
        paired_comparison(hold, "gated", "constant_gate"),
    ])
    print("\nper-series tests (Holm):")
    print(tests[["model_a", "model_b", "mean_diff", "ci_low", "ci_high",
                 "win_rate_a", "holm_p"]].round(5).to_string(index=False))

    tag = name.lower().replace(" ", "_")
    overall.to_csv(results / "metrics" / f"HOLDOUT_overall_{tag}.csv", index=False)
    by_regime.to_csv(results / "metrics" / f"HOLDOUT_regimes_{tag}.csv", index=False)
    cost.to_csv(results / "metrics" / f"HOLDOUT_cost_{tag}.csv", index=False)
    tests.to_csv(results / "statistical_tests" / f"HOLDOUT_wilcoxon_{tag}.csv", index=False)


def main() -> None:
    results = resolve_path(load_config()["paths"]["results_dir"])

    lgb = pd.read_parquet(results / "predictions" / "phase4_final_with_tft.parquet")
    block("LightGBM base", lgb.dropna(subset=["tft"]),
          ["seasonal_naive", "lightgbm", "tft", "always_on", "constant_gate", "gated"],
          base="lightgbm")

    tft = pd.read_parquet(results / "predictions" / "phase4_tftbase.parquet")
    tft = tft.rename(columns={"lightgbm": "tft_base"})
    block("TFT base", tft, ["tft_base", "always_on", "constant_gate", "gated"], base="tft_base")


if __name__ == "__main__":
    main()
