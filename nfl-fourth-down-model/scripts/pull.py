"""Download nflverse play-by-play, one parquet file per season, and record what was downloaded.

    python scripts/pull.py                       # 1999-2025 into data/raw
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

URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{}.parquet"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "manifest.csv"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, nargs="+", default=list(range(1999, 2026)))
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    rows = {}
    if MANIFEST.exists():
        rows = {int(r["season"]): r for r in csv.DictReader(MANIFEST.open())}
    for season in args.seasons:
        path = RAW / f"play_by_play_{season}.parquet"
        if not path.exists():
            print(f"downloading {path.name}")
            part = path.with_suffix(".part")
            urllib.request.urlretrieve(URL.format(season), part)
            part.rename(path)
            time.sleep(2)
        if season not in rows:
            rows[season] = {"season": season, "bytes": path.stat().st_size, "sha256": sha256(path),
                            "downloaded": datetime.date.today().isoformat()}
    with MANIFEST.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["season", "bytes", "sha256", "downloaded"])
        w.writeheader()
        w.writerows(rows[s] for s in sorted(rows))


if __name__ == "__main__":
    main()
