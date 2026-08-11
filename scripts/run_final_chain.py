"""Detached runner for the final-scale chain (Phase 2 -> 3 -> 4).

Runs each step as a subprocess, appends everything to results/final_chain.log,
and writes CHAIN DONE / CHAIN FAILED markers that an external monitor can watch.
"""
from __future__ import annotations

import datetime
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "final_chain.log"

STEPS = [
    [sys.executable, "scripts/02_train_baselines.py", "--tag", "final"],
    [sys.executable, "scripts/04_train_residual.py", "--tag", "final"],
    [sys.executable, "scripts/05_train_gate.py", "--tag", "final"],
]


def main() -> None:
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a", buffering=1, encoding="utf-8") as log:
        for step in STEPS:
            log.write(f"\n=== {datetime.datetime.now():%Y-%m-%d %H:%M:%S} :: {' '.join(step[1:])}\n")
            r = subprocess.run(step, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if r.returncode != 0:
                log.write(f"\nCHAIN FAILED (exit {r.returncode}) at {step[1]}\n")
                sys.exit(r.returncode)
        log.write(f"\nCHAIN DONE {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")


if __name__ == "__main__":
    main()
