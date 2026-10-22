"""Download the Basketball Reference draft pages, 1990 to 2016, skipping any already on disk.

Usage: python scripts/pull.py [out_dir]   (default data/raw)

Basketball Reference allows at most 20 requests a minute; this waits 4 seconds between them.
"""

import sys
import time
import urllib.request
from pathlib import Path

from draft_value.data import URL, YEARS

PAUSE_SECONDS = 4
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    fetched = 0
    for year in YEARS:
        path = out_dir / f"NBA_{year}.html"
        if path.exists():
            continue
        if fetched:
            time.sleep(PAUSE_SECONDS)
        request = urllib.request.Request(
            URL.format(year=year), headers={"User-Agent": "nba-draft-pick-value research script"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:  # raises on 4xx/5xx
            path.write_bytes(response.read())
        fetched += 1
        print(f"Fetched {path.name}")
    print(f"{fetched} fetched, {len(YEARS) - fetched} already present in {out_dir}")


if __name__ == "__main__":
    main()
