"""Detached runner: TFT-base variant pipeline (09 -> 04 -> 05, tag tftbase).
Gate uses the frozen lambda=0.1 from configs/tcn_gate.yaml (no retuning)."""
from __future__ import annotations

import datetime
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "tftbase.log"

STEPS = [
    [sys.executable, "scripts/09_build_tftbase.py"],
    [sys.executable, "scripts/04_train_residual.py", "--tag", "tftbase"],
    [sys.executable, "scripts/05_train_gate.py", "--tag", "tftbase"],
]

with open(LOG, "a", buffering=1, encoding="utf-8") as log:
    for step in STEPS:
        log.write(f"\n=== {datetime.datetime.now():%Y-%m-%d %H:%M:%S} :: {' '.join(step[1:])}\n")
        r = subprocess.run(step, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            log.write(f"\nTFTBASE FAILED (exit {r.returncode}) at {step[1]}\n")
            sys.exit(r.returncode)
    log.write(f"\nTFTBASE DONE {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
