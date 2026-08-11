"""TFT-base variant, step 1 (lab06): one TFT trained on data <= 2016-02-29
(pre-OOF cutoff), then predictions for every weekly OOF origin 2016-03-01 ..
2017-04-18. Combined with the existing per-fold TFT predictions, this gives a
daily TFT residual stream for training the TCN/gate on a TFT base.

Design note (documented in the paper): unlike the monthly-retrained LightGBM
base, the OOF-period TFT base is trained once (GPU cost); the fold/holdout
predictions come from the per-fold TFTs already trained. Both stream segments
are honest out-of-sample predictions.

Usage (on lab06):
    ~/miniconda3/envs/tft/bin/python scripts/03b_tft_oof.py \
        --panel data/processed/panel_final.parquet --out results/tft
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

import importlib.util
spec = importlib.util.spec_from_file_location("tft_common", Path(__file__).parent / "03_train_tft.py")
tft_common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tft_common)

CUTOFF = pd.Timestamp("2016-02-29")
OOF_ORIGINS = pd.date_range("2016-03-01", "2017-04-18", freq="W-TUE")
H, SEED = 7, 20260810


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--out", default="results/tft")
    args = ap.parse_args()

    pl.seed_everything(SEED)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    df = tft_common.load_panel(args.panel)
    # keep only series with enough history BEFORE the cutoff: series born later
    # are unseen categories for the encoder (KeyError) and lack encoder input
    first = df.groupby("series_id", observed=True)["date"].transform("min")
    ok = first <= CUTOFF - pd.Timedelta(days=63)
    n_drop = df.loc[~ok, "series_id"].nunique()
    df = df[ok].copy()
    df["series_id"] = df["series_id"].astype(str)
    print(f"dropped {n_drop} late-start series; {df['series_id'].nunique()} remain", flush=True)
    train_df = df[df["date"] <= CUTOFF]
    training = tft_common.make_dataset(train_df[train_df["date"] <= CUTOFF - pd.Timedelta(days=28)])
    validation = TimeSeriesDataSet.from_dataset(
        training, train_df, min_prediction_idx=int(
            train_df.loc[train_df["date"] > CUTOFF - pd.Timedelta(days=28), "time_idx"].min()
        ), stop_randomization=True,
    )
    train_dl = training.to_dataloader(train=True, batch_size=1024, num_workers=8,
                                      persistent_workers=True)
    val_dl = validation.to_dataloader(train=False, batch_size=2048, num_workers=8,
                                      persistent_workers=True)

    tft = TemporalFusionTransformer.from_dataset(
        training, hidden_size=64, attention_head_size=4, dropout=0.1,
        hidden_continuous_size=32, learning_rate=1e-3, loss=QuantileLoss(),
        log_interval=-1, reduce_on_plateau_patience=3,
    )
    trainer = pl.Trainer(
        max_epochs=20, accelerator="gpu", devices=1, precision="bf16-mixed",
        gradient_clip_val=0.1, enable_progress_bar=False, logger=False,
        callbacks=[EarlyStopping(monitor="val_loss", patience=3)],
        enable_checkpointing=False,
    )
    trainer.fit(tft, train_dl, val_dl)

    rows = []
    for origin in OOF_ORIGINS:
        sl = df[df["date"] <= origin + pd.Timedelta(days=H)]
        pred_ds = TimeSeriesDataSet.from_dataset(training, sl, predict=True, stop_randomization=True)
        dl = pred_ds.to_dataloader(train=False, batch_size=2048, num_workers=8)
        raw = tft.predict(dl, mode="prediction", return_index=True,
                          trainer_kwargs={"accelerator": "gpu", "logger": False,
                                          "enable_progress_bar": False})
        pred = raw.output.cpu().numpy()
        idx = raw.index
        for i, sid in enumerate(idx["series_id"]):
            s, it = sid.split("_")
            for h in range(1, H + 1):
                rows.append({"store_nbr": int(s), "item_nbr": int(it),
                             "origin_date": origin, "horizon": h,
                             "target_date": origin + pd.Timedelta(days=h),
                             "tft": max(0.0, float(pred[i, h - 1]))})
        print(f"oof origin {origin.date()}: {len(idx):,} series", flush=True)

    pd.DataFrame(rows).to_parquet(out / "tft_oof.parquet", index=False)
    print(f"saved {out / 'tft_oof.parquet'}")


if __name__ == "__main__":
    main()
