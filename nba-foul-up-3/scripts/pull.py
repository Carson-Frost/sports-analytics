"""Download the season bundles of NBA.com Stats JSON from sportsdataverse, and record them.

    python scripts/pull.py                 # 1996-97 to 2025-26 into data/raw
    python scripts/pull.py --years 2026    # one bundle, named by the year the season ends

data/manifest.csv (committed) holds each file's size and SHA-256, so build.py can tell when a fresh
download differs from the one the results were made from. sportsdataverse refreshes bundles now
and then.
"""

import argparse
import csv
import datetime
import hashlib
import time
import urllib.request
from pathlib import Path

from foulup3 import FIRST, LAST

URL = (
    "https://github.com/sportsdataverse/hoopR-nba-stats-raw/releases/download/"
    "nba-stats-raw-json/nba_stats_json_{year}.tar.gz"
)
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "manifest.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, nargs="+", default=list(range(FIRST + 1, LAST + 2)))
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    rows = {r["file"]: r for r in csv.DictReader(MANIFEST.open())} if MANIFEST.exists() else {}
    for year in args.years:
        name = f"nba_stats_json_{year}.tar.gz"
        path = RAW / name
        if not path.exists():
            print(f"downloading {name}", flush=True)
            part = path.with_suffix(".part")
            urllib.request.urlretrieve(URL.format(year=year), part)
            part.rename(path)  # only a complete download gets the real name
            time.sleep(2)
        if name not in rows:
            rows[name] = {
                "file": name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "downloaded": datetime.date.today().isoformat(),
            }
    with MANIFEST.open("w", newline="") as f:
        fields = ["file", "bytes", "sha256", "downloaded"]
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows[k] for k in sorted(rows))


if __name__ == "__main__":
    main()
