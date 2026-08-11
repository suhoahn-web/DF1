"""Download and extract the Corporacion Favorita competition data.

Prerequisites (one-time, manual):
  1. pip install kaggle py7zr
  2. Create an API token at kaggle.com -> Account -> Create New Token,
     save it as  %USERPROFILE%/.kaggle/kaggle.json
  3. Accept the competition rules in a browser:
     https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/rules

Alternatively, download the zip manually from the competition page and place
it (or the extracted csv/csv.7z files) in data/raw/, then re-run this script
to finish extraction.
"""
from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import load_config, resolve_path  # noqa: E402

EXPECTED = [
    "train.csv",
    "test.csv",
    "stores.csv",
    "items.csv",
    "holidays_events.csv",
    "oil.csv",
    "transactions.csv",
]


def extract_7z(archive: Path, out_dir: Path) -> None:
    import py7zr

    print(f"extracting {archive.name} ...")
    with py7zr.SevenZipFile(archive, mode="r") as z:
        z.extractall(path=out_dir)


def main() -> None:
    cfg = load_config()
    raw = resolve_path(cfg["paths"]["raw_dir"])
    comp = cfg["kaggle"]["competition"]

    if not any(raw.iterdir()):
        print(f"downloading {comp} via kaggle CLI ...")
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", comp, "-p", str(raw)],
            check=True,
        )

    for z in raw.glob("*.zip"):
        print(f"extracting {z.name} ...")
        with zipfile.ZipFile(z) as f:
            f.extractall(raw)

    for a in raw.glob("*.7z"):
        target = raw / a.name.replace(".7z", "")
        if not target.exists():
            extract_7z(a, raw)

    missing = [f for f in EXPECTED if not (raw / f).exists()]
    if missing:
        sys.exit(f"missing after extraction: {missing}")
    print("all raw files present:", ", ".join(EXPECTED))


if __name__ == "__main__":
    main()
