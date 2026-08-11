"""Detached runner: Phase 3 + 4 only (after the scale-floor fix)."""
from __future__ import annotations

import datetime
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "final_chain.log"

STEPS = [
    [sys.executable, "scripts/04_train_residual.py", "--tag", "final"],
    [sys.executable, "scripts/05_train_gate.py", "--tag", "final"],
]

with open(LOG, "a", buffering=1, encoding="utf-8") as log:
    for step in STEPS:
        log.write(f"\n=== {datetime.datetime.now():%Y-%m-%d %H:%M:%S} :: {' '.join(step[1:])} (scale-fix rerun)\n")
        r = subprocess.run(step, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            log.write(f"\nCHAIN FAILED (exit {r.returncode}) at {step[1]}\n")
            sys.exit(r.returncode)
    log.write(f"\nCHAIN DONE {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
