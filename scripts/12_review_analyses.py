"""Review round-1 analyses (P3, P4, Appendix B). Post-hoc computations on the
frozen prediction tables only — no model retraining, holdout predictions reused
as saved (analysis of frozen outputs, permitted under the freeze).

P4a  regime-sliced paired Wilcoxon (Holm within each family)
P4b  store-cluster bootstrap CIs for pooled WAPE differences
P4c  aggregate Diebold-Mariano test with HAC variance (h-1 lags)
P3   dual-estimand table: pooled volume-weighted vs series-level WAPE
B    post-promotion window sensitivity (1-3 vs 1-7 days)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import evaluate  # noqa: E402
from src.evaluation.statistical_tests import holm_correction, paired_comparison  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402

RESULTS = resolve_path(load_config()["paths"]["results_dir"])
KEY = ["store_nbr", "item_nbr", "origin_date", "horizon"]

BASES = {
    "lgb": ("phase4_final.parquet", "lightgbm"),
    "tft": ("phase4_tftbase.parquet", "tft_base"),
}


def load(base):
    fname, basecol = BASES[base]
    d = pd.read_parquet(RESULTS / "predictions" / fname)
    if basecol != "lightgbm":
        d = d.rename(columns={"lightgbm": basecol})
    return d, basecol


def wape(y, yhat):
    return float(np.abs(y - yhat).sum() / (np.abs(y).sum() + 1e-9))


# ------------------------------------------------------------------ P4a
def regime_sliced_tests(scope_name):
    rows = []
    for base, (_, basecol) in BASES.items():
        d, basecol = load(base)
        d = d[d["fold"] == "holdout"] if scope_name == "holdout" else d[d["fold"] != "holdout"]
        for regime in ["regime_promotion", "regime_post_promo", "regime_holiday",
                       "regime_high_vol", "regime_normal"]:
            sub = d[d[regime] == 1]
            for other in [basecol, "always_on"]:
                r = paired_comparison(sub, "gated", other, n_boot=1000)
                r.update({"base": base, "regime": regime.replace("regime_", ""), "scope": scope_name})
                rows.append(r)
    out = holm_correction(rows)
    out.to_csv(RESULTS / "statistical_tests" / f"regime_sliced_wilcoxon_{scope_name}.csv", index=False)
    sig = out[out["significant_5pct"]]
    print(f"P4a {scope_name}: {len(sig)}/{len(out)} comparisons significant after Holm")
    return out


# ------------------------------------------------------------------ P4b
def cluster_bootstrap(scope_name, n_boot=2000, seed=20260810):
    """WAPE = sum|err| / sum|y| decomposes over stores, so aggregate per store
    once and bootstrap the aggregates (vectorized; no row-level resampling)."""
    rng = np.random.default_rng(seed)
    rows = []
    for base, (_, basecol) in BASES.items():
        d, basecol = load(base)
        d = d[d["fold"] == "holdout"] if scope_name == "holdout" else d[d["fold"] != "holdout"]
        models = ["gated", basecol, "always_on", "constant_gate"]
        agg = pd.DataFrame({"y": d.groupby("store_nbr", observed=True)["actual"]
                           .apply(lambda s: np.abs(s).sum())})
        for m in models:
            agg[m] = d.assign(e=(d[m] - d["actual"]).abs()).groupby(
                "store_nbr", observed=True)["e"].sum()
        A = agg.to_numpy()                      # [n_stores, 1+len(models)]
        n = len(A)
        idx = rng.integers(0, n, size=(n_boot, n))
        sums = A[idx].sum(axis=1)               # [n_boot, cols]
        y_sum = sums[:, 0]
        w = {m: sums[:, j + 1] / (y_sum + 1e-9) for j, m in enumerate(models)}
        for other in [basecol, "always_on", "constant_gate"]:
            diffs = w["gated"] - w[other]
            rows.append({"base": base, "other": other, "scope": scope_name,
                         "mean_diff": float(diffs.mean()),
                         "ci_low": float(np.percentile(diffs, 2.5)),
                         "ci_high": float(np.percentile(diffs, 97.5))})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "statistical_tests" / f"cluster_bootstrap_{scope_name}.csv", index=False)
    print(f"P4b {scope_name}:")
    print(out.round(5).to_string(index=False))
    return out


# ------------------------------------------------------------------ P4c
def dm_test(e1, e2, h=7):
    """Aggregate DM with HAC (uniform kernel, h-1 lags) on |e| loss differential."""
    d = np.abs(e1) - np.abs(e2)
    n = len(d)
    dbar = d.mean()
    gamma = [np.sum((d[k:] - dbar) * (d[:n - k] - dbar)) / n for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        var = gamma[0] / n
    stat = dbar / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(stat)))
    return float(stat), float(p)


def aggregate_dm(scope_name):
    rows = []
    for base, (_, basecol) in BASES.items():
        d, basecol = load(base)
        d = d[d["fold"] == "holdout"] if scope_name == "holdout" else d[d["fold"] != "holdout"]
        d["target_date"] = d["origin_date"] + pd.to_timedelta(d["horizon"], unit="D")
        agg = d.groupby("target_date", observed=True)[["actual", basecol, "always_on", "gated"]].sum()
        for other in [basecol, "always_on"]:
            e_g = (agg["actual"] - agg["gated"]).to_numpy()
            e_o = (agg["actual"] - agg[other]).to_numpy()
            stat, p = dm_test(e_g, e_o)
            rows.append({"base": base, "other": other, "scope": scope_name,
                         "n_days": len(agg), "dm_stat": stat, "p": p})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "statistical_tests" / f"aggregate_dm_{scope_name}.csv", index=False)
    print(f"P4c {scope_name}:")
    print(out.round(4).to_string(index=False))
    return out


# ------------------------------------------------------------------ P3
def dual_estimand_table():
    rows = []
    for base, (_, basecol) in BASES.items():
        d, basecol = load(base)
        for scope_name in ["folds", "holdout"]:
            s = d[d["fold"] == "holdout"] if scope_name == "holdout" else d[d["fold"] != "holdout"]
            totals = s.groupby(["store_nbr", "item_nbr"], observed=True)["actual"].sum()
            valid = totals[totals >= 1.0].index
            sv = s.set_index(["store_nbr", "item_nbr"]).loc[valid].reset_index()
            for m in [basecol, "always_on", "constant_gate", "gated"]:
                per_series = sv.groupby(["store_nbr", "item_nbr"], observed=True).apply(
                    lambda g, m=m: wape(g["actual"].to_numpy(), g[m].to_numpy()),
                    include_groups=False)
                rows.append({"base": base, "scope": scope_name, "model": m,
                             "pooled_wape": wape(s["actual"].to_numpy(), s[m].to_numpy()),
                             "series_mean_wape": float(per_series.mean()),
                             "series_median_wape": float(per_series.median())})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "metrics" / "dual_estimand_table.csv", index=False)
    print("P3 dual-estimand table saved")
    return out


# ------------------------------------------------------------------ B
def postpromo_sensitivity():
    sup = pd.read_parquet(resolve_path(load_config()["paths"]["processed_dir"]) / "supervised_final.parquet",
                          columns=KEY + ["post_promo_1_7"])
    rows = []
    for base, (_, basecol) in BASES.items():
        d, basecol = load(base)
        d = d.merge(sup, on=KEY, how="left", suffixes=("", "_sup"))
        pp7 = "post_promo_1_7" if "post_promo_1_7" in d.columns else "post_promo_1_7_sup"
        for scope_name in ["folds", "holdout"]:
            s = d[d["fold"] == "holdout"] if scope_name == "holdout" else d[d["fold"] != "holdout"]
            for window, col in [("1-3", "regime_post_promo"), ("1-7", pp7)]:
                sub = s[s[col] == 1]
                for m in [basecol, "always_on", "gated"]:
                    rows.append({"base": base, "scope": scope_name, "window": window, "model": m,
                                 "wape": wape(sub["actual"].to_numpy(), sub[m].to_numpy()),
                                 "n": len(sub)})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "metrics" / "postpromo_window_sensitivity.csv", index=False)
    print("Appendix B saved:")
    print(out.pivot_table(index=["base", "scope", "window"], columns="model", values="wape").round(4).to_string())
    return out


if __name__ == "__main__":
    for scope in ["folds", "holdout"]:
        regime_sliced_tests(scope)
        cluster_bootstrap(scope)
        aggregate_dm(scope)
    dual_estimand_table()
    postpromo_sensitivity()
    print("\nALL REVIEW ANALYSES DONE")
