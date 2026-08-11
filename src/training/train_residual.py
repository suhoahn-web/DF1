"""Phase 3: residual sequence dataset, TCN training, A1/A1.5 corrections.

Base-model convention (documented in the paper): the operational base is the
monthly-retrained global LightGBM (the OOF procedure). Its predictions
`base_oof` exist for every weekly origin from 2016-03 through the holdout, are
always out-of-sample, and define both (a) the residual stream the TCN reads
and (b) the base forecast that A0/A1/A1.5/A4 all correct -- "same base
predictions" held constant (spec §11).

Scaling: one scale per sample = rolling_mean_28 at the origin (+eps), applied
to residual/sales/base channels and the residual target; predictions are
un-scaled by the same factor (spec §5).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.tcn_residual import ResidualTCN
from src.utils.config import load_config, resolve_path


def _cfgs():
    return load_config("data.yaml"), load_config("tcn_gate.yaml")


# ------------------------------------------------------------------- dataset
def build_sequence_dataset(tag: str = "dev", lookback: int | None = None) -> dict:
    """Tensors for every OOF origin with a full lookback of residual history."""
    dcfg, mcfg = _cfgs()
    processed = resolve_path(dcfg["paths"]["processed_dir"])
    L = lookback or mcfg["tcn"]["lookback"]
    H = dcfg["forecast"]["horizon"]
    eps = mcfg["scaling"]["eps"]
    clip = mcfg["scaling"]["clip_scaled"]
    key = ["store_nbr", "item_nbr"]

    oof = pd.read_parquet(processed / f"oof_residuals_{tag}.parquet")
    panel = pd.read_parquet(
        processed / f"panel_{tag}.parquet",
        columns=key + ["date", "unit_sales", "onpromotion", "is_holiday",
                       "cv_28", "zero_ratio_28", "roll_mean_28"],
    )

    # daily residual stream: weekly origins x h=1..7 tile the calendar exactly
    stream = oof[key + ["target_date", "base_oof", "target", "residual"]].rename(
        columns={"target_date": "date"}
    )
    assert not stream.duplicated(key + ["date"]).any(), "residual stream must be daily-unique"
    panel = panel.merge(stream, on=key + ["date"], how="left")
    panel = panel.sort_values(key + ["date"]).reset_index(drop=True)

    origins = oof[key + ["origin_date", "stage"]].drop_duplicates()
    per_origin = oof.pivot_table(
        index=key + ["origin_date"], columns="horizon",
        values=["base_oof", "target"], aggfunc="first",
    )

    chans = mcfg["tcn"]["channels"]
    seqs, tgts, metas = [], [], []
    for (s, it), g in panel.groupby(key, observed=True):
        g = g.reset_index(drop=True)
        date_pos = {d: i for i, d in enumerate(g["date"])}
        res = g["residual"].to_numpy(dtype="float32")
        sales = g["unit_sales"].to_numpy(dtype="float32")
        base = g["base_oof"].to_numpy(dtype="float32")
        promo = g["onpromotion"].to_numpy(dtype="float32")
        hol = g["is_holiday"].to_numpy(dtype="float32")
        cv = g["cv_28"].to_numpy(dtype="float32")
        zf = (sales == 0).astype("float32")
        scale_arr = g["roll_mean_28"].to_numpy(dtype="float32")

        for _, orow in origins[(origins["store_nbr"] == s) & (origins["item_nbr"] == it)].iterrows():
            o = orow["origin_date"]
            i = date_pos.get(o)
            if i is None or i - L + 1 < 0:
                continue
            sl = slice(i - L + 1, i + 1)
            if np.isnan(res[sl]).any():
                continue  # residual stream not yet available for full lookback
            # +1.0 floor (not eps): near-zero-demand series would otherwise
            # explode the scaled values and dominate every scaled loss
            scale = scale_arr[i] + 1.0
            ch_data = {
                "residual_scaled": np.clip(res[sl] / scale, -clip, clip),
                "sales_scaled": np.clip(sales[sl] / scale, 0, clip),
                "base_oof_scaled": np.clip(base[sl] / scale, 0, clip),
                "onpromotion": promo[sl],
                "is_holiday": hol[sl],
                "cv_28": cv[sl],
                "zero_flag": zf[sl],
            }
            seqs.append(np.stack([ch_data[c] for c in chans]))
            try:
                bo = per_origin.loc[(s, it, o)]
            except KeyError:
                continue
            base_vec = bo["base_oof"].to_numpy(dtype="float32")
            act_vec = bo["target"].to_numpy(dtype="float32")
            tgts.append(np.clip((act_vec - base_vec) / scale, -clip, clip))
            metas.append((s, it, o, orow["stage"], scale, *base_vec, *act_vec))

    seq = np.stack(seqs).astype("float32")
    tgt = np.stack(tgts).astype("float32")
    meta = pd.DataFrame(
        metas,
        columns=key + ["origin_date", "stage", "scale"]
        + [f"base_h{h}" for h in range(1, H + 1)]
        + [f"actual_h{h}" for h in range(1, H + 1)],
    )
    print(f"sequence dataset: {len(seq):,} samples, seq {seq.shape}, stages:")
    print(meta["stage"].value_counts().to_string())
    return {"seq": seq, "target": tgt, "meta": meta}


# ------------------------------------------------------------------ training
def train_tcn(data: dict, device: str = "cpu", overrides: dict | None = None) -> ResidualTCN:
    dcfg, mcfg = _cfgs()
    t = {**mcfg["tcn"], **(overrides or {})}
    torch.manual_seed(mcfg["seed"])

    m = data["meta"]
    idx = np.where(m["stage"] == "tcn_train")[0]
    cut = np.quantile(m.loc[idx, "origin_date"].astype("int64"), 1 - t["val_fraction"])
    tr = idx[m.loc[idx, "origin_date"].astype("int64") < cut]
    va = idx[m.loc[idx, "origin_date"].astype("int64") >= cut]

    def loader(ix, shuffle):
        ds = TensorDataset(torch.from_numpy(data["seq"][ix]), torch.from_numpy(data["target"][ix]))
        return DataLoader(ds, batch_size=t["batch_size"], shuffle=shuffle)

    model = ResidualTCN(
        in_channels=len(t["channels"]), horizon=dcfg["forecast"]["horizon"],
        hidden=t["hidden"], kernel_size=t["kernel_size"], dropout=t["dropout"],
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    loss_fn = torch.nn.HuberLoss(delta=t["huber_delta"])

    best, best_state, patience = np.inf, None, 0
    for epoch in range(t["max_epochs"]):
        model.train()
        for xb, yb in loader(tr, True):
            opt.zero_grad()
            loss = loss_fn(model(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([
                loss_fn(model(xb.to(device)), yb.to(device)).item() for xb, yb in loader(va, False)
            ]))
        if vl < best - 1e-5:
            best, best_state, patience = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= t["early_stop_patience"]:
                break
        if epoch % 5 == 0:
            print(f"  epoch {epoch}: val huber {vl:.5f} (best {best:.5f})")
    model.load_state_dict(best_state)
    print(f"TCN trained: best val huber {best:.5f} at epoch {epoch - patience}")
    return model


def predict_residuals(model: ResidualTCN, data: dict, device: str = "cpu") -> pd.DataFrame:
    """Un-scaled residual corrections r_hat for every sample, long format."""
    dcfg, _ = _cfgs()
    H = dcfg["forecast"]["horizon"]
    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(data["seq"]), 4096):
            xb = torch.from_numpy(data["seq"][i:i + 4096]).to(device)
            preds.append(model(xb).cpu().numpy())
    r_scaled = np.concatenate(preds)

    m = data["meta"].copy()
    rows = []
    for h in range(1, H + 1):
        r = m[["store_nbr", "item_nbr", "origin_date", "stage", "scale"]].copy()
        r["horizon"] = np.int8(h)
        r["base_oof"] = m[f"base_h{h}"]
        r["actual"] = m[f"actual_h{h}"]
        r["r_hat"] = r_scaled[:, h - 1] * m["scale"].to_numpy()
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


def tune_constant_gate(rhat: pd.DataFrame) -> float:
    """A1.5: scalar c minimizing WAPE on the gate_train stage (same data the
    learned gate will train on -- fair parity)."""
    g = rhat[rhat["stage"] == "gate_train"]
    y = g["actual"].to_numpy()
    grid = np.arange(0.0, 1.01, 0.05)
    scores = []
    for c in grid:
        yhat = np.clip(g["base_oof"].to_numpy() + c * g["r_hat"].to_numpy(), 0, None)
        scores.append(np.abs(y - yhat).sum() / (np.abs(y).sum() + 1e-9))
    c_star = float(grid[int(np.argmin(scores))])
    print(f"constant gate c* = {c_star:.2f} (gate_train WAPE {min(scores):.4f}, "
          f"c=0 -> {scores[0]:.4f}, c=1 -> {scores[-1]:.4f})")
    return c_star
