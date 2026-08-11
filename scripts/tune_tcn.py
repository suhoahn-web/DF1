"""Small TCN hyperparameter search (spec §14). Selection metric: always-on
WAPE on the gate_train stage (OOF validation territory -- folds/holdout never
touched). Winner is frozen into configs/tcn_gate.yaml by hand afterwards.

Usage:
    python scripts/tune_tcn.py [--tag dev|final]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training import train_residual as tr  # noqa: E402
from src.utils.config import load_config, resolve_path, set_seed  # noqa: E402

GRID = [
    {"lookback": 28, "hidden": 64, "kernel_size": 3, "lr": 1e-3, "dropout": 0.1},   # current
    {"lookback": 28, "hidden": 64, "kernel_size": 5, "lr": 1e-3, "dropout": 0.1},
    {"lookback": 28, "hidden": 32, "kernel_size": 3, "lr": 1e-3, "dropout": 0.2},
    {"lookback": 28, "hidden": 64, "kernel_size": 3, "lr": 3e-4, "dropout": 0.1},
    {"lookback": 56, "hidden": 64, "kernel_size": 3, "lr": 1e-3, "dropout": 0.1},
    {"lookback": 56, "hidden": 64, "kernel_size": 5, "lr": 1e-3, "dropout": 0.2},
    {"lookback": 56, "hidden": 32, "kernel_size": 3, "lr": 3e-4, "dropout": 0.1},
    {"lookback": 56, "hidden": 64, "kernel_size": 5, "lr": 3e-4, "dropout": 0.1},
]


def gate_train_wape(rhat: pd.DataFrame, c: float = 1.0) -> float:
    g = rhat[rhat["stage"] == "gate_train"]
    y = g["actual"].to_numpy()
    yhat = np.clip(g["base_oof"].to_numpy() + c * g["r_hat"].to_numpy(), 0, None)
    return float(np.abs(y - yhat).sum() / (np.abs(y).sum() + 1e-9))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dev", choices=["dev", "final"])
    args = ap.parse_args()
    set_seed(load_config("tcn_gate.yaml")["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    datasets: dict[int, dict] = {}
    rows = []
    for i, hp in enumerate(GRID):
        L = hp["lookback"]
        if L not in datasets:
            datasets[L] = tr.build_sequence_dataset(args.tag, lookback=L)
        data = datasets[L]
        model = tr.train_tcn(data, device, overrides=hp)
        rhat = tr.predict_residuals(model, data, device)
        base_wape = gate_train_wape(rhat, c=0.0)
        ao_wape = gate_train_wape(rhat, c=1.0)
        rows.append({**hp, "gate_train_wape_base": base_wape, "gate_train_wape_ao": ao_wape,
                     "delta": ao_wape - base_wape})
        print(f"[{i+1}/{len(GRID)}] {hp} -> AO {ao_wape:.4f} (base {base_wape:.4f})")

    res = pd.DataFrame(rows).sort_values("gate_train_wape_ao")
    out = resolve_path(load_config()["paths"]["results_dir"]) / "metrics"
    res.to_csv(out / f"tcn_search_{args.tag}.csv", index=False)
    print("\nranked results:")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
