"""Download the season bundles from the sportsdataverse hoopR-nba-stats-raw GitHub release.

Files already present are skipped, so this only fetches what is missing. Usage:

    python scripts/pull.py                      # 2015 to 2026 into data/raw/bundles
    python scripts/pull.py 2027 --to some/dir   # one bundle somewhere else
"""

import argparse
import time
import urllib.request
from pathlib import Path

URL = (
    "https://github.com/sportsdataverse/hoopR-nba-stats-raw/releases/download/"
    "nba-stats-raw-json/nba_stats_json_{year}.tar.gz"
)
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("years", nargs="*", type=int, default=list(range(2015, 2027)))
    parser.add_argument("--to", type=Path, default=ROOT / "data" / "raw" / "bundles")
    args = parser.parse_args()
    args.to.mkdir(parents=True, exist_ok=True)
    for year in args.years:
        path = args.to / f"nba_stats_json_{year}.tar.gz"
        if path.exists():
            print("have", path.name)
            continue
        print("downloading", path.name)
        part = path.with_suffix(".part")
        urllib.request.urlretrieve(URL.format(year=year), part)
        part.rename(path)  # only a complete download gets the real name
        time.sleep(2)  # be polite to GitHub between files


if __name__ == "__main__":
    main()
