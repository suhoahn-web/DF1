"""Detached runner: Phase 5 (ablations + paper tables, folds only)."""
from __future__ import annotations

import datetime
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "phase5.log"

STEPS = [
    [sys.executable, "scripts/06_run_ablations.py", "--tag", "final"],
    [sys.executable, "scripts/07_make_paper_tables.py", "--tag", "final"],
]

with open(LOG, "a", buffering=1, encoding="utf-8") as log:
    for step in STEPS:
        log.write(f"\n=== {datetime.datetime.now():%Y-%m-%d %H:%M:%S} :: {' '.join(step[1:])}\n")
        r = subprocess.run(step, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            log.write(f"\nPHASE5 FAILED (exit {r.returncode}) at {step[1]}\n")
            sys.exit(r.returncode)
    log.write(f"\nPHASE5 DONE {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
