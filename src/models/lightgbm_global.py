"""Global LightGBM base forecaster (horizon-as-feature, spec §2)."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.data.make_features import ORIGIN_FEATURES, TARGET_FEATURES, TARGET_LAGS

CATEGORICALS = ["family", "class", "city", "state", "type", "cluster"]
FEATURES = (
    ORIGIN_FEATURES
    + TARGET_FEATURES
    + [f"lag_{lag}" for lag in TARGET_LAGS]
    + ["horizon", "perishable", "store_nbr", "item_nbr"]
    + CATEGORICALS
)

DEFAULT_PARAMS = {
    "objective": "regression_l1",
    "metric": "l1",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "min_data_in_leaf": 50,
    "verbosity": -1,
    "seed": 20260810,
}


def _prep(df: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    X = df[features or FEATURES].copy()
    for c in ("store_nbr", "item_nbr"):
        X[c] = X[c].astype("category")
    return X


def train(
    train_rows: pd.DataFrame,
    valid_rows: pd.DataFrame | None = None,
    params: dict | None = None,
    num_boost_round: int = 2000,
    features: list[str] | None = None,
) -> lgb.Booster:
    p = {**DEFAULT_PARAMS, **(params or {})}
    dtrain = lgb.Dataset(_prep(train_rows, features), label=train_rows["target"])
    callbacks, valid_sets = [], []
    if valid_rows is not None and len(valid_rows):
        valid_sets = [lgb.Dataset(_prep(valid_rows, features), label=valid_rows["target"],
                                  reference=dtrain)]
        callbacks = [lgb.early_stopping(100, verbose=False)]
    booster = lgb.train(p, dtrain, num_boost_round=num_boost_round,
                        valid_sets=valid_sets, callbacks=callbacks)
    booster._feature_list = features or FEATURES
    return booster


def predict(model: lgb.Booster, rows: pd.DataFrame) -> np.ndarray:
    features = getattr(model, "_feature_list", None)
    pred = model.predict(_prep(rows, features),
                         num_iteration=getattr(model, "best_iteration", None))
    return np.clip(pred, 0, None)
