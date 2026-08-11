"""TFT baseline (Lim et al. 2021) via pytorch-forecasting. Runs on lab06 A100.

Protocol: one TFT per fold, trained only on panel rows with date <= cutoff
(= fold's first origin - 1 day), mirroring the fold LightGBM in
02_train_baselines.py. Predictions for each origin use encoder data <= origin
and decoder rows origin+1..origin+7 (future-known covariates only).

Usage (on lab06):
    ~/miniconda3/envs/tft/bin/python scripts/03_train_tft.py \
        --panel data/processed/panel_final.parquet --fold fold1 --out results/tft
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

FOLDS = {
    "fold1": ["2017-04-25", "2017-05-02", "2017-05-09", "2017-05-16"],
    "fold2": ["2017-05-23", "2017-05-30", "2017-06-06", "2017-06-13"],
    "fold3": ["2017-06-20", "2017-06-27", "2017-07-04", "2017-07-11"],
    "holdout": ["2017-07-18", "2017-07-25", "2017-08-01", "2017-08-08"],
}
H = 7
ENCODER_LEN = 56
SEED = 20260810

KNOWN_REALS = ["onpromotion", "is_holiday", "is_payday", "promo_share_store_family",
               "is_earthquake_window", "time_idx"]
KNOWN_CATS = ["day_of_week", "month"]
STATIC_CATS = ["store_nbr", "item_nbr", "family"]


def load_panel(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["series_id"] = df["store_nbr"].astype(str) + "_" + df["item_nbr"].astype(str)
    t0 = df["date"].min()
    df["time_idx"] = (df["date"] - t0).dt.days
    for c in STATIC_CATS + KNOWN_CATS:
        df[c] = df[c].astype(str).astype("category")
    for c in ["onpromotion", "is_holiday", "is_payday", "is_earthquake_window"]:
        df[c] = df[c].astype("float32")
    df["unit_sales"] = df["unit_sales"].astype("float32")
    # TFT cannot handle NaN reals; early-history rolling NaNs -> 0
    df["promo_share_store_family"] = df["promo_share_store_family"].fillna(0.0)
    return df


def make_dataset(df: pd.DataFrame) -> TimeSeriesDataSet:
    return TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="unit_sales",
        group_ids=["series_id"],
        max_encoder_length=ENCODER_LEN,
        min_encoder_length=ENCODER_LEN // 2,
        max_prediction_length=H,
        min_prediction_length=H,
        static_categoricals=STATIC_CATS,
        time_varying_known_reals=KNOWN_REALS,
        time_varying_known_categoricals=KNOWN_CATS,
        time_varying_unknown_reals=["unit_sales"],
        target_normalizer=GroupNormalizer(groups=["series_id"], transformation="softplus"),
        allow_missing_timesteps=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--fold", required=True, choices=list(FOLDS))
    ap.add_argument("--out", default="results/tft")
    ap.add_argument("--max-epochs", type=int, default=20)
    args = ap.parse_args()

    pl.seed_everything(SEED)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    df = load_panel(args.panel)
    origins = [pd.Timestamp(o) for o in FOLDS[args.fold]]
    cutoff = min(origins) - pd.Timedelta(days=1)

    train_df = df[df["date"] <= cutoff]
    training = make_dataset(train_df[train_df["date"] <= cutoff - pd.Timedelta(days=28)])
    validation = TimeSeriesDataSet.from_dataset(
        training, train_df, min_prediction_idx=int(
            train_df.loc[train_df["date"] > cutoff - pd.Timedelta(days=28), "time_idx"].min()
        ), stop_randomization=True,
    )
    # A100-sized batches; 256/4 workers left the GPU ~50% idle (observed fold1)
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
        # bf16 (not fp16): TFT's attention mask bias overflows c10::Half
        max_epochs=args.max_epochs, accelerator="gpu", devices=1, precision="bf16-mixed",
        gradient_clip_val=0.1, enable_progress_bar=False, logger=False,
        callbacks=[EarlyStopping(monitor="val_loss", patience=3)],
        enable_checkpointing=False,
    )
    trainer.fit(tft, train_dl, val_dl)

    # ---- predict each origin: encoder <= origin, decoder = origin+1..origin+H ----
    rows = []
    for origin in origins:
        sl = df[df["date"] <= origin + pd.Timedelta(days=H)]
        pred_ds = TimeSeriesDataSet.from_dataset(training, sl, predict=True, stop_randomization=True)
        dl = pred_ds.to_dataloader(train=False, batch_size=2048, num_workers=8)
        raw = tft.predict(dl, mode="prediction", return_index=True,
                          trainer_kwargs={"accelerator": "gpu", "logger": False,
                                          "enable_progress_bar": False})
        pred = raw.output.cpu().numpy()          # [N, H] median quantile
        idx = raw.index                          # series_id per row
        for i, sid in enumerate(idx["series_id"]):
            s, it = sid.split("_")
            for h in range(1, H + 1):
                rows.append({"store_nbr": int(s), "item_nbr": int(it),
                             "origin_date": origin, "horizon": h,
                             "target_date": origin + pd.Timedelta(days=h),
                             "tft": max(0.0, float(pred[i, h - 1]))})
        print(f"{args.fold} origin {origin.date()}: {len(idx):,} series predicted")

    pd.DataFrame(rows).to_parquet(out / f"tft_{args.fold}.parquet", index=False)
    print(f"saved {out / f'tft_{args.fold}.parquet'}")


if __name__ == "__main__":
    main()
