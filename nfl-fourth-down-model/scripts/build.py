"""Raw play-by-play to one table of regulation, regular-season plays with game states.

Stops if a raw file differs from data/manifest.csv, since the results would no longer match.
"""

import csv

from fourth_down.data import PLAYS, RAW, ROOT, SEASONS, load

MANIFEST = ROOT / "data" / "manifest.csv"


def check_manifest():
    import hashlib
    for row in csv.DictReader(MANIFEST.open(encoding="utf-8")):
        path = RAW / f"play_by_play_{row['season']}.parquet"
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise SystemExit(f"{path.name} differs from data/manifest.csv (downloaded {row['downloaded']}). "
                             "Delete the manifest row to accept the new file, then rerun everything.")


def main():
    check_manifest()
    plays = load(SEASONS)
    plays.to_parquet(PLAYS)
    print(f"{len(plays):,} plays, {plays.game_id.nunique():,} games, "
          f"{plays.season.min()}-{plays.season.max()} -> {PLAYS}")


if __name__ == "__main__":
    main()
