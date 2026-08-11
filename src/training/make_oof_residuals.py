"""Temporal OOF base predictions and residuals (spec §3).

For each monthly retrain cutoff, a base LightGBM is trained ONLY on supervised
rows with target_date <= cutoff; it then predicts the OOF origins of that
month. Residual = actual - oof_prediction. TCN/gate later train only on these.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import splits
from src.models import lightgbm_global as lgbm
from src.utils.config import load_config, resolve_path


def run(tag: str = "dev", cfg=None) -> pd.DataFrame:
    cfg = cfg or load_config()
    processed = resolve_path(cfg["paths"]["processed_dir"])
    out_path = processed / f"oof_residuals_{tag}.parquet"

    sup = pd.read_parquet(processed / f"supervised_{tag}.parquet")
    cutmap = splits.oof_base_cutoffs(cfg)

    parts = []
    for cutoff, grp in cutmap.groupby("base_train_cutoff"):
        train_rows = sup[sup["target_date"] <= cutoff]
        origins = grp["origin_date"].to_numpy()
        pred_rows = sup[sup["origin_date"].isin(origins)].copy()
        if len(train_rows) == 0 or len(pred_rows) == 0:
            continue
        model = lgbm.train(train_rows)
        pred_rows["base_oof"] = lgbm.predict(model, pred_rows)
        # leakage guarantee, asserted at generation time
        assert train_rows["target_date"].max() < pred_rows["target_date"].min()
        parts.append(pred_rows)
        print(
            f"  cutoff {cutoff.date()}: trained on {len(train_rows):,} rows, "
            f"predicted {len(pred_rows):,} rows ({len(origins)} origins)"
        )

    oof = pd.concat(parts, ignore_index=True)
    oof["residual"] = oof["target"] - oof["base_oof"]
    oof = oof.merge(cutmap[["origin_date", "stage"]], on="origin_date", how="left")
    oof.to_parquet(out_path, index=False)
    print(f"OOF residuals: {len(oof):,} rows -> {out_path.name}")
    return oof


if __name__ == "__main__":
    run()
