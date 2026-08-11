"""Phase 4: gate context features and Stage-2 gate training (spec §6-§7).

Context feature groups are defined centrally so ablations (A2/A3/A5) can drop
a group by name and retrain the gate with everything else identical.

Gate training data: gate_train stage only -- the TCN never saw these origins,
so its r_hat there is out-of-sample (spec §3, TCN -> gate separation).
Loss is scale-normalized Huber on the final forecast (spec §7 Stage 2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.gated_residual import HorizonGate, gated_forecast
from src.utils.config import load_config, resolve_path

FEATURE_GROUPS = {
    "promotion": [
        "onpromotion", "promo_start", "promo_streak", "days_since_promo_end",
        "post_promo_1_3", "post_promo_1_7", "promo_share_store_family",
    ],
    "recent_error": [
        "res_mae_7", "res_mae_28", "res_bias_7", "res_bias_28",
        "underforecast_rate_28", "last_abs_res",
    ],
    "demand_state": [
        "roll_mean_7", "roll_mean_28", "roll_std_7", "roll_std_28",
        "cv_28", "zero_ratio_28",
    ],
    "calendar": ["is_holiday", "is_weekend", "is_payday", "day_of_week", "month"],
    "base_level": ["base_rel_mean", "horizon_frac"],
}
ALL_GROUPS = list(FEATURE_GROUPS)


def context_columns(drop_groups: list[str] | None = None) -> list[str]:
    drop = set(drop_groups or [])
    return [c for g in ALL_GROUPS if g not in drop for c in FEATURE_GROUPS[g]]


# ------------------------------------------------------------ context table
def build_context_table(tag: str = "dev") -> pd.DataFrame:
    """One row per (series, origin, horizon) with all gate context features."""
    dcfg = load_config()
    processed = resolve_path(dcfg["paths"]["processed_dir"])
    key = ["store_nbr", "item_nbr"]
    eps = dcfg["features"]["eps"]

    rhat = pd.read_parquet(processed / f"rhat_{tag}.parquet")
    sup = pd.read_parquet(
        processed / f"supervised_{tag}.parquet",
        columns=key + ["origin_date", "horizon",
                       "onpromotion", "promo_start", "promo_streak", "days_since_promo_end",
                       "post_promo_1_3", "post_promo_1_7", "promo_share_store_family",
                       "is_holiday", "is_weekend", "is_payday", "day_of_week", "month",
                       "roll_mean_7", "roll_mean_28", "roll_std_7", "roll_std_28",
                       "cv_28", "zero_ratio_28"],
    )
    ctx = rhat.merge(sup, on=key + ["origin_date", "horizon"], how="left")

    # base-reliability features from the daily residual stream, as of the origin
    oof = pd.read_parquet(
        processed / f"oof_residuals_{tag}.parquet",
        columns=key + ["target_date", "residual"],
    ).rename(columns={"target_date": "date"}).sort_values(key + ["date"])
    g = oof.groupby(key, observed=True)["residual"]
    oof["res_mae_7"] = g.transform(lambda s: s.abs().rolling(7, min_periods=4).mean())
    oof["res_mae_28"] = g.transform(lambda s: s.abs().rolling(28, min_periods=14).mean())
    oof["res_bias_7"] = g.transform(lambda s: s.rolling(7, min_periods=4).mean())
    oof["res_bias_28"] = g.transform(lambda s: s.rolling(28, min_periods=14).mean())
    oof["underforecast_rate_28"] = g.transform(
        lambda s: s.gt(0).astype(float).rolling(28, min_periods=14).mean()
    )
    oof["last_abs_res"] = oof["residual"].abs()
    rel = oof.drop(columns=["residual"]).rename(columns={"date": "origin_date"})
    ctx = ctx.merge(rel, on=key + ["origin_date"], how="left")

    ctx["base_rel_mean"] = ctx["base_oof"] / (ctx["roll_mean_28"] + eps)
    ctx["horizon_frac"] = ctx["horizon"] / dcfg["forecast"]["horizon"]
    return ctx


def _normalize(ctx: pd.DataFrame, cols: list[str], stats: pd.DataFrame | None = None):
    """Z-score using gate_train statistics only."""
    if stats is None:
        base = ctx[ctx["stage"] == "gate_train"]
        stats = pd.DataFrame({"mean": base[cols].mean(), "std": base[cols].std().replace(0, 1)})
    X = (ctx[cols] - stats["mean"]) / stats["std"]
    return X.fillna(0.0).to_numpy(dtype="float32"), stats


# ---------------------------------------------------------------- training
def _to_wide(ctx: pd.DataFrame, X: np.ndarray, H: int):
    """Long (N*H) rows -> wide [N, H, ...] tensors aligned by (series, origin)."""
    ctx = ctx.reset_index(drop=True)
    ctx["_row"] = np.arange(len(ctx))
    wide = ctx.pivot_table(
        index=["store_nbr", "item_nbr", "origin_date", "stage", "scale"],
        columns="horizon", values="_row", aggfunc="first",
    )
    full = wide.dropna()
    rows = full.to_numpy(dtype=int)                       # [N, H]
    meta = full.index.to_frame(index=False)
    ctx_np = {c: ctx[c].to_numpy() for c in ["base_oof", "r_hat", "actual"]}
    return {
        "context": X[rows],                               # [N, H, D]
        "base": ctx_np["base_oof"][rows].astype("float32"),
        "r_hat": ctx_np["r_hat"][rows].astype("float32"),
        "actual": ctx_np["actual"][rows].astype("float32"),
        "meta": meta,
    }


def train_gate(tag: str = "dev", drop_groups: list[str] | None = None,
               device: str = "cpu", ctx: pd.DataFrame | None = None,
               sparsity_lambda: float | None = None, gate_arch: str = "mlp"):
    if sparsity_lambda is None:
        sparsity_lambda = load_config("tcn_gate.yaml")["gate"].get("sparsity_lambda", 0.0)
    dcfg, mcfg = load_config(), load_config("tcn_gate.yaml")
    gcfg = mcfg["gate"]
    torch.manual_seed(mcfg["seed"])
    H = dcfg["forecast"]["horizon"]

    if ctx is None:
        ctx = build_context_table(tag)
    cols = context_columns(drop_groups)
    X, stats = _normalize(ctx, cols)
    wide = _to_wide(ctx, X, H)
    meta = wide["meta"]

    gt = np.where(meta["stage"] == "gate_train")[0]
    cut = np.quantile(meta.loc[gt, "origin_date"].astype("int64"), 1 - gcfg["val_fraction"])
    tr_ix = gt[meta.loc[gt, "origin_date"].astype("int64") < cut]
    va_ix = gt[meta.loc[gt, "origin_date"].astype("int64") >= cut]

    scale = meta["scale"].to_numpy(dtype="float32")[:, None]

    def loader(ix, shuffle):
        ds = TensorDataset(
            torch.from_numpy(wide["context"][ix]),
            torch.from_numpy(wide["base"][ix]),
            torch.from_numpy(wide["r_hat"][ix]),
            torch.from_numpy(wide["actual"][ix]),
            torch.from_numpy(scale[ix]),
        )
        return DataLoader(ds, batch_size=gcfg["batch_size"], shuffle=shuffle)

    from src.models.gated_residual import LinearGate
    if gate_arch == "linear":
        gate = LinearGate(len(cols)).to(device)
    else:
        gate = HorizonGate(len(cols), gcfg["hidden"], gcfg["hidden2"], gcfg["dropout"]).to(device)
    opt = torch.optim.AdamW(gate.parameters(), lr=gcfg["lr"], weight_decay=gcfg["weight_decay"])

    def step(batch, train: bool):
        c, b, r, y, s = [t.to(device) for t in batch]
        g = gate(c)
        final = gated_forecast(b, r, g)
        # evaluation-aligned loss: raw-unit L1 (minimizing sum|err| = WAPE
        # numerator). Scale-normalized Huber upweighted low-volume series and
        # taught the gate a policy that hurt WAPE (observed 2026-08-10).
        l1 = (final - y).abs().mean()
        loss = l1 + sparsity_lambda * g.mean() if train else l1
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        return float(l1.item())      # val tracking uses the pure forecast term

    best, best_state, patience = np.inf, None, 0
    for epoch in range(gcfg["max_epochs"]):
        gate.train()
        for batch in loader(tr_ix, True):
            step(batch, True)
        gate.eval()
        with torch.no_grad():
            vl = float(np.mean([step(b, False) for b in loader(va_ix, False)]))
        if vl < best - 1e-6:
            best, best_state, patience = vl, {k: v.clone() for k, v in gate.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= gcfg["early_stop_patience"]:
                break
    gate.load_state_dict(best_state)
    print(f"gate trained ({len(cols)} features, drop={drop_groups}): best val {best:.5f}")

    # predictions for every sample
    gate.eval()
    with torch.no_grad():
        gv = []
        for i in range(0, len(wide["context"]), 8192):
            gv.append(gate(torch.from_numpy(wide["context"][i:i + 8192]).to(device)).cpu().numpy())
    g_all = np.concatenate(gv)                            # [N, H]

    out = []
    for h in range(1, H + 1):
        r = meta.copy()
        r["horizon"] = np.int8(h)
        r["gate_value"] = g_all[:, h - 1]
        r["base_oof"] = wide["base"][:, h - 1]
        r["r_hat"] = wide["r_hat"][:, h - 1]
        r["actual"] = wide["actual"][:, h - 1]
        r["gated"] = np.clip(r["base_oof"] + r["gate_value"] * r["r_hat"], 0, None)
        out.append(r)
    preds = pd.concat(out, ignore_index=True)
    return gate, preds, stats
