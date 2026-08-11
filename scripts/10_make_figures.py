"""Paper figures (matplotlib, 300 dpi, PDF+PNG). Light surface, validated
reference palette from the dataviz skill (fixed slot order, <=3 series).

Fig 1  design timeline (OOF / TCN / gate / folds / holdout)
Fig 2  gate behavior: (a) mean gate by context, (b) by recent-error decile
Fig 3  regime WAPE change (gated - base), both bases, folds 1-3
Fig 4  case study around a promotion (TFT base)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import load_config, resolve_path  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e3e2df"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})

RESULTS = resolve_path(load_config()["paths"]["results_dir"])
FIG = RESULTS / "figures"
FIG.mkdir(exist_ok=True)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


# ------------------------------------------------------------------ figure 1
def fig1_timeline():
    fig, ax = plt.subplots(figsize=(7.0, 2.1))
    rows = [
        ("Base training data", "2015-01-01", "2016-02-29", GRID),
        ("OOF: TCN training", "2016-03-01", "2016-12-27", BLUE),
        ("OOF: gate training", "2017-01-03", "2017-04-18", AQUA),
        ("Validation folds 1-3", "2017-04-25", "2017-07-18", ORANGE),
        ("Holdout (one shot)", "2017-07-19", "2017-08-15", "#e34948"),
    ]
    for i, (label, s, e, c) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, (pd.Timestamp(e) - pd.Timestamp(s)).days, left=pd.Timestamp(s),
                height=0.62, color=c, edgecolor=SURFACE, linewidth=2)
        ax.text(pd.Timestamp(s) - pd.Timedelta(days=12), y, label,
                ha="right", va="center", fontsize=8.5, color=INK)
    ax.set_yticks([])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_xlim(pd.Timestamp("2014-06-01"), pd.Timestamp("2017-09-15"))
    ax.grid(axis="y", visible=False)
    ax.set_title("Expanding base retraining (monthly) with strictly ordered "
                 "TCN / gate / evaluation periods", fontsize=9.5, color=INK2, loc="left")
    save(fig, "fig1_timeline")


# ------------------------------------------------------------------ figure 2
def _gate_by_context(tag):
    p = RESULTS / "predictions" / ("phase4_final.parquet" if tag == "lgb" else "phase4_tftbase.parquet")
    d = pd.read_parquet(p)
    d = d[d["fold"] != "holdout"]
    return {
        "Promotion": d.loc[d["regime_promotion"] == 1, "gate_value"].mean(),
        "Holiday": d.loc[d["regime_holiday"] == 1, "gate_value"].mean(),
        "High volatility": d.loc[d["regime_high_vol"] == 1, "gate_value"].mean(),
        "Normal": d.loc[d["regime_normal"] == 1, "gate_value"].mean(),
    }, d


def _gate_by_decile(tag, d):
    from src.training.train_gate import build_context_table
    ctx = build_context_table("final" if tag == "lgb" else "tftbase")
    key = ["store_nbr", "item_nbr", "origin_date", "horizon"]
    m = d.merge(ctx[key + ["res_mae_28"]].drop_duplicates(key), on=key, how="left")
    m["decile"] = pd.qcut(m["res_mae_28"].rank(method="first"), 10, labels=False)
    return m.groupby("decile", observed=True)["gate_value"].mean()


def fig2_gate_behavior():
    ctx_l, d_l = _gate_by_context("lgb")
    ctx_t, d_t = _gate_by_context("tft")
    dec_l = _gate_by_decile("lgb", d_l)
    dec_t = _gate_by_decile("tft", d_t)

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    labels = list(ctx_l)
    y = np.arange(len(labels))[::-1]
    a.scatter(list(ctx_l.values()), y + 0.13, s=42, color=BLUE, zorder=3, label="LightGBM base")
    a.scatter(list(ctx_t.values()), y - 0.13, s=42, color=ORANGE, zorder=3, label="TFT base")
    for yi, (vl, vt) in zip(y, zip(ctx_l.values(), ctx_t.values())):
        a.plot([0, max(vl, vt)], [yi, yi], color=GRID, lw=0.8, zorder=1)
    a.set_yticks(y, labels)
    a.set_xlim(0, 1)
    a.set_xlabel("Mean gate value")
    a.set_title("(a) Gate opens on promotions,\ncloses under volatility", fontsize=9, loc="left")
    a.grid(axis="y", visible=False)

    b.plot(dec_l.index + 1, dec_l.values, color=BLUE, lw=2, marker="o", ms=4.5)
    b.plot(dec_t.index + 1, dec_t.values, color=ORANGE, lw=2, marker="o", ms=4.5)
    b.text(10.15, dec_l.values[-1], "LightGBM\nbase", color=BLUE, fontsize=8, va="center")
    b.text(10.15, dec_t.values[-1], "TFT base", color=ORANGE, fontsize=8, va="center")
    b.set_xlabel("Recent base-error decile (1 = lowest)")
    b.set_ylabel("Mean gate value")
    b.set_xlim(0.5, 12.6)
    b.set_xticks(range(1, 11))
    b.set_title("(b) Gate rises with recent base error", fontsize=9, loc="left")
    a.legend(loc="lower right", fontsize=7.5, frameon=False)
    save(fig, "fig2_gate_behavior")


# ------------------------------------------------------------------ figure 3
def fig3_regime_delta():
    rows = []
    for tag, basecol, label in [("final", "lightgbm", "LightGBM base"),
                                ("tftbase", "lightgbm", "TFT base")]:
        r = pd.read_csv(RESULTS / "metrics" / f"phase4_regimes_{tag}.csv") if \
            (RESULTS / "metrics" / f"phase4_regimes_{tag}.csv").exists() else None
        d = pd.read_parquet(RESULTS / "predictions" / f"phase4_{tag}.parquet")
        d = d[d["fold"] != "holdout"]
        from src.evaluation.metrics import evaluate
        for regime in ["regime_promotion", "regime_post_promo", "regime_holiday",
                       "regime_high_vol", "regime_normal"]:
            sub = d[d[regime] == 1]
            w_g = evaluate(sub, "gated").iloc[0]["wape"]
            w_b = evaluate(sub, basecol).iloc[0]["wape"]
            rows.append({"base": label, "regime": regime.replace("regime_", ""),
                         "delta": 100 * (w_g - w_b) / w_b})
    df = pd.DataFrame(rows)
    order = ["promotion", "holiday", "high_vol", "post_promo", "normal"]
    names = {"promotion": "Promotion", "holiday": "Holiday", "high_vol": "High volatility",
             "post_promo": "Post-promotion", "normal": "Normal"}

    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    y = np.arange(len(order))[::-1]
    for off, (label, color) in zip((0.18, -0.18), [("LightGBM base", BLUE), ("TFT base", ORANGE)]):
        vals = [df[(df["base"] == label) & (df["regime"] == r)]["delta"].iloc[0] for r in order]
        ax.barh(y + off, vals, height=0.32, color=color, label=label,
                edgecolor=SURFACE, linewidth=1.5)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, [names[r] for r in order])
    ax.set_xlabel("WAPE change vs. base (%)   ← better")
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    ax.grid(axis="y", visible=False)
    save(fig, "fig3_regime_delta")


# ------------------------------------------------------------------ figure 4
def fig4_case_study():
    d = pd.read_parquet(RESULTS / "predictions" / "phase4_tftbase.parquet")
    d = d[d["fold"] != "holdout"].copy()
    d["gain"] = (d["lightgbm"] - d["actual"]).abs() - (d["gated"] - d["actual"]).abs()
    key = ["store_nbr", "item_nbr"]
    pick = d.groupby(key, observed=True).agg(
        gain=("gain", "sum"), promo_share=("onpromotion", "mean"),
        gate_std=("gate_value", "std"), n=("gain", "size"),
    )
    # a case must SHOW selectivity: mixed promo/non-promo days, varying gate
    ok = pick[(pick["n"] >= 30) & pick["promo_share"].between(0.15, 0.5)
              & (pick["gate_std"] > 0.2)]
    s, it = ok.sort_values("gain").index[-1]

    full = pd.read_parquet(RESULTS / "predictions" / "phase4_tftbase.parquet")
    ts = full[(full["store_nbr"] == s) & (full["item_nbr"] == it)
              & (full["fold"] != "holdout")].sort_values("horizon")
    ts["target_date"] = ts["origin_date"] + pd.to_timedelta(ts["horizon"], unit="D")
    ts = ts.drop_duplicates("target_date").sort_values("target_date")
    w = ts[(ts["target_date"] >= ts.loc[ts["onpromotion"] == 1, "target_date"].min()
            - pd.Timedelta(days=10))].head(42)

    fig, (a, b) = plt.subplots(2, 1, figsize=(7.0, 3.4), sharex=True,
                               height_ratios=[3, 1], gridspec_kw={"hspace": 0.12})
    promo = w[w["onpromotion"] == 1]["target_date"]
    for pdt in promo:
        a.axvspan(pdt - pd.Timedelta(hours=12), pdt + pd.Timedelta(hours=12),
                  color=AQUA, alpha=0.12, lw=0)
        b.axvspan(pdt - pd.Timedelta(hours=12), pdt + pd.Timedelta(hours=12),
                  color=AQUA, alpha=0.12, lw=0)
    a.plot(w["target_date"], w["actual"], color=INK, lw=1.4, label="Actual")
    a.plot(w["target_date"], w["lightgbm"], color=BLUE, lw=2, label="TFT base")
    a.plot(w["target_date"], w["gated"], color=ORANGE, lw=2, label="Gated (CGRC)")
    a.legend(fontsize=7.5, frameon=False, ncols=3, loc="upper left")
    a.set_ylabel("Unit sales")
    a.set_title(f"Store {s}, item {it} — shaded: promotion days", fontsize=9,
                loc="left", color=INK2)
    b.fill_between(w["target_date"], w["gate_value"], color=ORANGE, alpha=0.65, lw=0)
    b.set_ylabel("Gate")
    b.set_ylim(0, 1)
    b.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    save(fig, "fig4_case_study")


if __name__ == "__main__":
    fig1_timeline()
    fig2_gate_behavior()
    fig3_regime_delta()
    fig4_case_study()
    print("all figures done")
