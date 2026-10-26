"""Download nflverse play-by-play, participation and FTN charting, one file per season each, and record them.

    python scripts/pull.py                       # 2016-2025 into data/raw
    python scripts/pull.py --seasons 2024 2025

data/manifest.csv (committed) holds each file's size and SHA-256, so build.py can tell when a fresh
download differs from the one the results were made from. nflverse revises old seasons now and then.
"""

import argparse
import csv
import datetime
import hashlib
import time
import urllib.request
from pathlib import Path

URL = "https://github.com/nflverse/nflverse-data/releases/download/{tag}/{name}"
FILES = {"pbp": "play_by_play_{}.parquet", "pbp_participation": "pbp_participation_{}.parquet",
         "ftn_charting": "ftn_charting_{}.parquet"}
FTN_FIRST = 2022  # FTN's play-action charting starts in 2022
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "manifest.csv"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, nargs="+", default=list(range(2016, 2026)))
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    rows = {r["file"]: r for r in csv.DictReader(MANIFEST.open())} if MANIFEST.exists() else {}
    for season in args.seasons:
        for tag, pattern in FILES.items():
            if tag == "ftn_charting" and season < FTN_FIRST:
                continue
            name = pattern.format(season)
            path = RAW / name
            if not path.exists():
                print(f"downloading {name}")
                part = path.with_suffix(".part")
                urllib.request.urlretrieve(URL.format(tag=tag, name=name), part)
                part.rename(path)
                time.sleep(1)
            if name not in rows:
                rows[name] = {"file": name, "bytes": path.stat().st_size, "sha256": sha256(path),
                              "downloaded": datetime.date.today().isoformat()}
    with MANIFEST.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "bytes", "sha256", "downloaded"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows[k] for k in sorted(rows))


if __name__ == "__main__":
    main()
