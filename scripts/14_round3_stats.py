"""Round-3 statistics and two small experiments (review round 2).

1  Effect sizes for the confirmatory family: rank-biserial, median diff,
   P10/P25/P50/P75/P90 of per-series WAPE differences.
2  Ablation pairwise tests: A4 vs A2/A3/A5, overall + promotion slice.
3  Slice-level store-clustered bootstrap CIs (pooled slice claims need
   pooled uncertainty).
4  DGT-style single-signal switching gate: g in {0,1} from one error signal,
   threshold tuned on the gate-training period (folds evaluation only).
5  Pinball reference: LightGBM trained at the 0.75 fractile, cost(3:1)
   reference line (folds only).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import splits  # noqa: E402
from src.evaluation import operational_cost as oc  # noqa: E402
from src.evaluation.statistical_tests import holm_correction, paired_comparison  # noqa: E402
from src.evaluation.metrics import per_series_metric  # noqa: E402
from src.models import lightgbm_global as lgbm  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402

RESULTS = resolve_path(load_config()["paths"]["results_dir"])
PROCESSED = resolve_path(load_config()["paths"]["processed_dir"])

BASES = {"lgb": ("phase4_final.parquet", "lightgbm"),
         "tft": ("phase4_tftbase.parquet", "tft_base")}


def load(base):
    fname, basecol = BASES[base]
    d = pd.read_parquet(RESULTS / "predictions" / fname)
    if basecol != "lightgbm":
        d = d.rename(columns={"lightgbm": basecol})
    return d, basecol


def wape(y, yhat):
    return float(np.abs(y - yhat).sum() / (np.abs(y).sum() + 1e-9))


def diffs_for(d, a, b):
    totals = d.groupby(["store_nbr", "item_nbr"], observed=True)["actual"].sum()
    valid = totals[totals >= 1.0].index
    dv = d.set_index(["store_nbr", "item_nbr"]).loc[valid].reset_index()
    return (per_series_metric(dv, a) - per_series_metric(dv, b)).dropna()


# ---------------------------------------------------------------- 1 effect sizes
def effect_sizes():
    rows = []
    for base, (_, basecol) in BASES.items():
        d, basecol = load(base)
        for scope in ["folds", "holdout"]:
            s = d[d["fold"] == "holdout"] if scope == "holdout" else d[d["fold"] != "holdout"]
            for other in [basecol, "always_on", "constant_gate"]:
                diff = diffs_for(s, "gated", other)
                n_pos, n_neg = int((diff > 0).sum()), int((diff < 0).sum())
                rb = (n_neg - n_pos) / max(n_pos + n_neg, 1)  # >0 = gated better
                q = np.percentile(diff, [10, 25, 50, 75, 90])
                rows.append({"base": base, "scope": scope, "other": other,
                             "rank_biserial": rb, "median_diff": float(np.median(diff)),
                             "p10": q[0], "p25": q[1], "p50": q[2], "p75": q[3], "p90": q[4]})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "statistical_tests" / "effect_sizes.csv", index=False)
    print("1) effect sizes saved")
    return out


# ---------------------------------------------------------------- 2 ablation tests
def ablation_pairwise():
    rows = []
    for tag, base in [("final", "lgb"), ("tftbase", "tft")]:
        ev = pd.read_parquet(RESULTS / "predictions" / f"ablations_{tag}.parquet")
        ev = ev[ev["fold"] != "holdout"]
        for abl in ["gated_A2_no_recent_error", "gated_A3_no_promotion", "gated_A5_no_demand_state"]:
            if abl not in ev.columns:
                continue
            for slice_name, mask in [("overall", ev.index),
                                     ("promotion", ev.index[ev["regime_promotion"] == 1])]:
                sub = ev.loc[mask].dropna(subset=[abl])
                r = paired_comparison(sub, "gated", abl, n_boot=500)
                r.update({"base": base, "ablation": abl, "slice": slice_name})
                rows.append(r)
    out = holm_correction(rows)
    out.to_csv(RESULTS / "statistical_tests" / "ablation_pairwise.csv", index=False)
    sig = out[out["significant_5pct"]][["base", "ablation", "slice", "mean_diff", "holm_p"]]
    print("2) ablation pairwise saved; significant rows:")
    print(sig.round(5).to_string(index=False) if len(sig) else "  (none)")
    return out


# ---------------------------------------------------------------- 3 slice cluster bootstrap
def slice_cluster_bootstrap(n_boot=2000, seed=20260810):
    rng = np.random.default_rng(seed)
    rows = []
    for base, (_, basecol) in BASES.items():
        d, basecol = load(base)
        for scope in ["folds", "holdout"]:
            s = d[d["fold"] == "holdout"] if scope == "holdout" else d[d["fold"] != "holdout"]
            for regime in ["regime_promotion", "regime_high_vol", "regime_holiday"]:
                sub = s[s[regime] == 1]
                agg = pd.DataFrame({
                    "y": sub.groupby("store_nbr", observed=True)["actual"].apply(lambda x: np.abs(x).sum()),
                    "g": sub.assign(e=(sub["gated"] - sub["actual"]).abs()).groupby("store_nbr", observed=True)["e"].sum(),
                    "b": sub.assign(e=(sub[basecol] - sub["actual"]).abs()).groupby("store_nbr", observed=True)["e"].sum(),
                }).dropna()
                A = agg.to_numpy(); n = len(A)
                idx = rng.integers(0, n, size=(n_boot, n))
                sums = A[idx].sum(axis=1)
                diffs = sums[:, 1] / (sums[:, 0] + 1e-9) - sums[:, 2] / (sums[:, 0] + 1e-9)
                rows.append({"base": base, "scope": scope, "regime": regime.replace("regime_", ""),
                             "mean_diff": float(diffs.mean()),
                             "ci_low": float(np.percentile(diffs, 2.5)),
                             "ci_high": float(np.percentile(diffs, 97.5))})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "statistical_tests" / "slice_cluster_bootstrap.csv", index=False)
    print("3) slice cluster bootstrap saved:")
    print(out.round(5).to_string(index=False))
    return out


# ---------------------------------------------------------------- 4 DGT-style gate
def dgt_gate():
    rows = []
    for base, (_, basecol) in BASES.items():
        d, basecol = load(base)
        # single signal: recent 7-day residual bias magnitude at the origin
        ctx = pd.read_parquet(PROCESSED / f"oof_residuals_{'final' if base == 'lgb' else 'tftbase'}.parquet",
                              columns=["store_nbr", "item_nbr", "target_date", "residual"])
        ctx = ctx.rename(columns={"target_date": "origin_date"}).sort_values(
            ["store_nbr", "item_nbr", "origin_date"])
        g = ctx.groupby(["store_nbr", "item_nbr"], observed=True)["residual"]
        ctx["sig"] = g.transform(lambda s: s.rolling(7, min_periods=4).mean().abs())
        d = d.merge(ctx[["store_nbr", "item_nbr", "origin_date", "sig"]].drop_duplicates(
            ["store_nbr", "item_nbr", "origin_date"]), on=["store_nbr", "item_nbr", "origin_date"], how="left")
        gt = d[d["stage"] == "gate_train"].dropna(subset=["sig"]) if "stage" in d.columns else None
        ev = d[d["fold"].isin(["fold1", "fold2", "fold3"])].dropna(subset=["sig"])
        # threshold grid on gate_train (quantiles of the signal)
        best = None
        pool = gt if gt is not None and len(gt) else ev
        for q in np.arange(0.1, 1.0, 0.1):
            thr = pool["sig"].quantile(q)
            yhat = np.where(pool["sig"] > thr,
                            np.clip(pool[basecol] + pool["r_hat"], 0, None), pool[basecol]) \
                if "r_hat" in pool.columns else None
            if yhat is None:
                break
            w = wape(pool["actual"].to_numpy(), yhat)
            if best is None or w < best[1]:
                best = (float(thr), w, float(q))
        thr = best[0]
        yhat = np.where(ev["sig"] > thr, np.clip(ev[basecol] + ev["r_hat"], 0, None), ev[basecol])
        rows.append({"base": base, "threshold_q": best[2],
                     "dgt_gate_wape": wape(ev["actual"].to_numpy(), yhat),
                     "gated_wape": wape(ev["actual"].to_numpy(), ev["gated"].to_numpy()),
                     "base_wape": wape(ev["actual"].to_numpy(), ev[basecol].to_numpy())})
        print(f"4) {base}: DGT gate {rows[-1]['dgt_gate_wape']:.4f} vs gated {rows[-1]['gated_wape']:.4f}")
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "metrics" / "dgt_gate.csv", index=False)
    return out


# ---------------------------------------------------------------- 5 pinball reference
def pinball_reference():
    sup = pd.read_parquet(PROCESSED / "supervised_final.parquet")
    folds = splits.fold_origins(load_config())
    parts = []
    for fold_name in ["fold1", "fold2", "fold3"]:
        origins = folds[fold_name]
        cutoff = min(origins) - pd.Timedelta(days=1)
        train_rows = sup[sup["target_date"] <= cutoff]
        fold_rows = sup[sup["origin_date"].isin(origins)].copy()
        m = lgbm.train(train_rows, params={"objective": "quantile", "alpha": 0.75,
                                           "metric": "quantile"})
        fold_rows["q75"] = lgbm.predict(m, fold_rows)
        parts.append(fold_rows)
        print(f"5) {fold_name}: quantile model trained")
    ev = pd.concat(parts).rename(columns={"target": "actual"})
    p4 = pd.read_parquet(RESULTS / "predictions" / "phase4_final.parquet")
    ev = ev.merge(p4[["store_nbr", "item_nbr", "origin_date", "horizon",
                      "lightgbm", "always_on", "gated"]],
                  on=["store_nbr", "item_nbr", "origin_date", "horizon"], how="inner")
    rows = []
    y = ev["actual"].to_numpy()
    for m in ["lightgbm", "always_on", "gated", "q75"]:
        rows.append({"model": m,
                     "wape": wape(y, ev[m].to_numpy()),
                     "cost_3_1": float(oc.asymmetric_cost(y, ev[m].to_numpy(), 3, 1).mean())})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "metrics" / "pinball_reference.csv", index=False)
    print(out.round(4).to_string(index=False))
    return out


if __name__ == "__main__":
    effect_sizes()
    ablation_pairwise()
    slice_cluster_bootstrap()
    dgt_gate()
    pinball_reference()
    print("\nROUND3 STATS DONE")
