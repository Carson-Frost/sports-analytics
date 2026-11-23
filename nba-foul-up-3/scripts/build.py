"""Parse every regular-season game in the bundles into late 4th-quarter and overtime possessions.

Stops if a bundle differs from data/manifest.csv, since the results would no longer match.
Writes data/derived/games.parquet and data/derived/possessions.parquet. From 2017-18, when the
current timeout rules began, each possession also has the timeouts both teams have left.
"""

import csv
import json
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from foulup3 import SEASONS
from foulup3.pbp import parse_game

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "manifest.csv"
OUT = ROOT / "data" / "derived"
COLUMNS = ["clock", "period", "teamId", "scoreHome", "scoreAway", "description", "actionType",
           "subType", "shotValue"]  # fmt: skip


def check_manifest():
    from pull import sha256  # same folder

    for row in csv.DictReader(MANIFEST.open(encoding="utf-8")):
        path = RAW / row["file"]
        if sha256(path) != row["sha256"]:
            raise SystemExit(
                f"{path.name} differs from data/manifest.csv (downloaded {row['downloaded']}). "
                "Delete the manifest row to accept the new file, then rerun everything."
            )


def season(season: str) -> tuple[list[dict], list[dict]]:
    """Games and possessions for one season, read straight from its bundle."""
    year = int(season[:4]) + 1  # bundles are named by the year the season ends
    log, pbps = None, {}
    with tarfile.open(RAW / f"nba_stats_json_{year}.tar.gz") as tar:
        for m in tar:
            if m.name == f"leaguegamelog/{year}/regular-season.json":
                rs = json.load(tar.extractfile(m))["resultSets"][0]
                log = pd.DataFrame(rs["rowSet"], columns=rs["headers"])
            elif m.name.startswith(f"playbyplayv3/{year}/002") and m.name.endswith(".json"):
                actions = json.load(tar.extractfile(m))["game"]["actions"]
                if actions:  # the April 2013 Celtics-Pacers game was never played
                    pbps[m.name.split("/")[-1][:-5]] = pd.DataFrame(actions)[COLUMNS]
    games, possessions = [], []
    for game_id, two in log.groupby("GAME_ID"):
        if game_id in pbps:
            game, rows = parse_game(pbps[game_id], two, game_id, season)
            games.append(game)
            possessions += rows
    print(f"{season}: {len(games)} games, {len(possessions)} possessions", flush=True)
    return games, possessions


def main():
    check_manifest()
    with ProcessPoolExecutor() as pool:
        parts = list(pool.map(season, SEASONS))
    games = pd.DataFrame([g for p in parts for g in p[0]])
    possessions = pd.DataFrame([s for p in parts for s in p[1]])
    OUT.mkdir(parents=True, exist_ok=True)
    games.to_parquet(OUT / "games.parquet", index=False)
    possessions.to_parquet(OUT / "possessions.parquet", index=False)
    print(f"{len(games):,} games, {len(possessions):,} possessions; final score differs from the "
          f"game log in {(~games.score_matches).sum()} games")  # fmt: skip


if __name__ == "__main__":
    main()
