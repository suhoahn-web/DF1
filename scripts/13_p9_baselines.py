"""Review round-1 P9 baselines. Development folds ONLY (holdout untouched);
both are disclosed in the manuscript as post-freeze exploratory additions.

Part 1  A0+: LightGBM base augmented with the gate's recent-error features.
        Controls a confound: the error features exist only where the OOF
        residual stream exists (origins >= 2016-04), so we train BOTH an
        augmented and an unaugmented model on the identical restricted rows
        and compare that pair.
Part 2  Linear gate: logistic-regression gate on the same 26 context features,
        same training protocol as the MLP gate, both bases.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import splits  # noqa: E402
from src.evaluation.metrics import evaluate  # noqa: E402
from src.models import lightgbm_global as lgbm  # noqa: E402
from src.training import train_gate as tg  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402

RESULTS = resolve_path(load_config()["paths"]["results_dir"])
PROCESSED = resolve_path(load_config()["paths"]["processed_dir"])
KEY = ["store_nbr", "item_nbr", "origin_date"]
ERR_COLS = ["res_mae_7", "res_mae_28", "res_bias_7", "res_bias_28",
            "underforecast_rate_28", "last_abs_res"]


def build_err_features() -> pd.DataFrame:
    oof = pd.read_parquet(PROCESSED / "oof_residuals_final.parquet",
                          columns=["store_nbr", "item_nbr", "target_date", "residual"])
    oof = oof.rename(columns={"target_date": "date"}).sort_values(["store_nbr", "item_nbr", "date"])
    g = oof.groupby(["store_nbr", "item_nbr"], observed=True)["residual"]
    oof["res_mae_7"] = g.transform(lambda s: s.abs().rolling(7, min_periods=4).mean())
    oof["res_mae_28"] = g.transform(lambda s: s.abs().rolling(28, min_periods=14).mean())
    oof["res_bias_7"] = g.transform(lambda s: s.rolling(7, min_periods=4).mean())
    oof["res_bias_28"] = g.transform(lambda s: s.rolling(28, min_periods=14).mean())
    oof["underforecast_rate_28"] = g.transform(
        lambda s: s.gt(0).astype(float).rolling(28, min_periods=14).mean())
    oof["last_abs_res"] = oof["residual"].abs()
    return oof.drop(columns=["residual"]).rename(columns={"date": "origin_date"})


def part1_a0plus() -> pd.DataFrame:
    sup = pd.read_parquet(PROCESSED / "supervised_final.parquet")
    err = build_err_features()
    sup = sup.merge(err, on=KEY, how="left")
    restricted = sup[sup["origin_date"] >= pd.Timestamp("2016-04-01")]

    folds = splits.fold_origins(load_config())
    rows = []
    for fold_name in ["fold1", "fold2", "fold3"]:
        origins = folds[fold_name]
        cutoff = min(origins) - pd.Timedelta(days=1)
        train_rows = restricted[restricted["target_date"] <= cutoff]
        fold_rows = restricted[restricted["origin_date"].isin(origins)].copy()
        assert train_rows["target_date"].max() < fold_rows["target_date"].min()

        m_r = lgbm.train(train_rows)                                    # A0r
        m_p = lgbm.train(train_rows, features=lgbm.FEATURES + ERR_COLS)  # A0+
        fold_rows["a0_restricted"] = lgbm.predict(m_r, fold_rows)
        fold_rows["a0_plus"] = lgbm.predict(m_p, fold_rows)
        fold_rows["fold"] = fold_name
        rows.append(fold_rows)
        print(f"{fold_name}: A0r/A0+ trained on {len(train_rows):,} restricted rows")

    ev = pd.concat(rows, ignore_index=True).rename(columns={"target": "actual"})
    # attach gated predictions for the same rows
    p4 = pd.read_parquet(RESULTS / "predictions" / "phase4_final.parquet")
    ev = ev.merge(p4[KEY + ["horizon", "gated", "lightgbm"]],
                  on=KEY + ["horizon"], how="inner")
    out = pd.concat([
        evaluate(ev, m).assign(model=m)
        for m in ["lightgbm", "a0_restricted", "a0_plus", "gated"]
    ], ignore_index=True)
    out.to_csv(RESULTS / "metrics" / "p9_a0plus.csv", index=False)
    print("\nP9 part 1 (folds 1-3):")
    print(out[["model", "wape", "mae"]].round(4).to_string(index=False))
    return out


def part2_linear_gate() -> pd.DataFrame:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for tag, basecol in [("final", "lightgbm"), ("tftbase", "tft_base")]:
        _, preds, _ = tg.train_gate(tag, device=device, gate_arch="linear")
        ev = preds[preds["stage"] == "eval_input"].copy()
        folds = splits.fold_origins(load_config())
        holdout = set(folds["holdout"])
        ev = ev[~ev["origin_date"].isin(holdout)]
        rows.append({"base": tag,
                     "linear_gate_wape": float(np.abs(ev["actual"] - ev["gated"]).sum()
                                               / np.abs(ev["actual"]).sum()),
                     "gate_mean": float(ev["gate_value"].mean())})
        print(f"{tag}: linear gate fold WAPE {rows[-1]['linear_gate_wape']:.4f}")
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "metrics" / "p9_linear_gate.csv", index=False)
    return out


if __name__ == "__main__":
    part1_a0plus()
    part2_linear_gate()
    print("\nP9 DONE")
