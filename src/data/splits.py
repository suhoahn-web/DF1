"""Rolling-origin split definitions (frozen in configs/data.yaml)."""
from __future__ import annotations

import pandas as pd

from src.utils.config import load_config


def fold_origins(cfg=None) -> dict[str, list[pd.Timestamp]]:
    cfg = cfg or load_config()
    s = cfg["splits"]
    out = {name: [pd.Timestamp(o) for o in f["origins"]] for name, f in s["folds"].items()}
    out["holdout"] = [pd.Timestamp(o) for o in s["holdout"]["origins"]]
    return out


def oof_origins(cfg=None) -> pd.DatetimeIndex:
    cfg = cfg or load_config()
    o = cfg["splits"]["oof"]
    return pd.date_range(o["first_origin"], o["last_origin"], freq=o["origin_frequency"])


def oof_base_cutoffs(cfg=None) -> pd.DataFrame:
    """Map each OOF origin to its base-model training cutoff.

    The base model serving an origin is trained only on rows whose TARGET date
    is <= cutoff, where cutoff = (start of the origin's retrain period) - 1 day.
    This guarantees max_training_target_date < origin + 1 (first forecast day).
    """
    cfg = cfg or load_config()
    origins = oof_origins(cfg)
    freq = cfg["splits"]["oof"]["base_retrain_frequency"]
    period_start = origins.to_period(freq[:1] if freq == "MS" else freq).to_timestamp()
    cutoffs = period_start - pd.Timedelta(days=1)
    df = pd.DataFrame({"origin_date": origins, "base_train_cutoff": cutoffs})

    o = cfg["splits"]["oof"]
    tcn_end = pd.Timestamp(o["tcn_train_end"])
    gate_start = pd.Timestamp(o["gate_train_start"])
    train_last = pd.Timestamp(o["train_last_origin"])
    assert tcn_end < gate_start <= train_last, "stage boundaries out of order"
    df["stage"] = "tcn_train"
    df.loc[df["origin_date"] >= gate_start, "stage"] = "gate_train"
    # origins past the training boundary only feed the rolling residual INPUT
    # stream for fold/holdout evaluation; never training targets
    df.loc[df["origin_date"] > train_last, "stage"] = "eval_input"
    return df


def validate_splits(cfg=None) -> None:
    """Frozen-design invariants; called by unit tests."""
    cfg = cfg or load_config()
    folds = fold_origins(cfg)
    horizon = cfg["forecast"]["horizon"]
    end = pd.Timestamp(cfg["analysis_period"]["end"])

    all_fold_origins = [o for f in folds.values() for o in f]
    assert max(all_fold_origins) + pd.Timedelta(days=horizon) <= end

    oof = oof_base_cutoffs(cfg)
    assert (oof["base_train_cutoff"] < oof["origin_date"]).all()
    train_stages = oof[oof["stage"].isin(["tcn_train", "gate_train"])]
    assert train_stages["origin_date"].max() < min(folds["fold1"])

    # folds ordered and their TARGET windows non-overlapping, holdout last
    prev_last_target = pd.Timestamp.min
    for name in ["fold1", "fold2", "fold3", "holdout"]:
        first_target = min(folds[name]) + pd.Timedelta(days=1)
        assert first_target > prev_last_target, f"{name} target window overlaps previous"
        prev_last_target = max(folds[name]) + pd.Timedelta(days=horizon)
    print("split invariants OK")


if __name__ == "__main__":
    validate_splits()
