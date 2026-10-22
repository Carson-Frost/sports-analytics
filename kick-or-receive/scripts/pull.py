"""Download the raw files from nflverse, skipping any that are already present.

    python scripts/pull.py                      # everything, into data/raw
    python scripts/pull.py --seasons 2024 --dest scratch/raw
"""

import argparse
import time
import urllib.request
from pathlib import Path

PBP = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{}.parquet"
GAMES = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"


def fetch(url, path):
    if path.exists():
        print(f"have {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "kick-or-receive research"})
    with urllib.request.urlopen(request, timeout=120) as response:
        path.write_bytes(response.read())
    print(f"got {path}")
    time.sleep(1)  # one file at a time, with a pause


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, nargs="*", default=list(range(1999, 2027)))
    parser.add_argument("--dest", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    fetch(GAMES, args.dest / "games.csv")
    for season in args.seasons:
        fetch(PBP.format(season), args.dest / "pbp" / f"play_by_play_{season}.parquet")


if __name__ == "__main__":
    main()
