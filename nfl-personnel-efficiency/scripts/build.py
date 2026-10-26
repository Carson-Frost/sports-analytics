"""Join play-by-play to participation: one row per regular-season run or dropback, with its grouping.

Stops if a raw file differs from data/manifest.csv, since the results would no longer match.
"""

import csv
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from personnel import defensive_backs, grouping, situation

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "manifest.csv"
PLAYS = ROOT / "data" / "plays.parquet"
SEASONS = range(2016, 2026)
PBP = ["game_id", "play_id", "season", "season_type", "week", "posteam", "defteam", "down", "ydstogo", "yardline_100",
       "play_type", "pass", "epa", "wp"]
PARTICIPATION = ["nflverse_game_id", "play_id", "offense_personnel", "defense_personnel", "defenders_in_box"]


def check_manifest():
    for row in csv.DictReader(MANIFEST.open()):
        path = RAW / row["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise SystemExit(f"{path.name} differs from data/manifest.csv (downloaded {row['downloaded']}). "
                             "Delete the manifest row to accept the new file, then rerun everything.")


def season_plays(season):
    pbp = pd.read_parquet(RAW / f"play_by_play_{season}.parquet", columns=PBP)
    pbp = pbp[(pbp.season_type == "REG") & pbp.play_type.isin(["pass", "run"]) & pbp.down.notna()]
    part = pd.read_parquet(RAW / f"pbp_participation_{season}.parquet", columns=PARTICIPATION)
    part = part.rename(columns={"nflverse_game_id": "game_id"})
    plays = pbp.merge(part, on=["game_id", "play_id"], how="left", validate="one_to_one")
    ftn = RAW / f"ftn_charting_{season}.parquet"
    if not ftn.exists():  # FTN charting starts in 2022
        return plays.assign(play_action=np.nan)
    pa = pd.read_parquet(ftn, columns=["nflverse_game_id", "nflverse_play_id", "is_play_action"])
    pa = pa.rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id",
                            "is_play_action": "play_action"})
    pa["play_action"] = pa.play_action.astype(float)
    return plays.merge(pa, on=["game_id", "play_id"], how="left", validate="one_to_one")


def main():
    check_manifest()
    plays = pd.concat([season_plays(s) for s in SEASONS], ignore_index=True)
    plays["grouping"] = plays.offense_personnel.map(grouping)
    missing = plays.grouping.isna()
    fakes = plays.grouping == "Special Teams"  # fake punts and field goals: not an offensive grouping
    plays = plays[~missing & ~fakes].copy()
    plays["situation"] = situation(plays.down, plays.ydstogo, plays.yardline_100)
    plays["defensive_backs"] = plays.defense_personnel.map(defensive_backs)
    plays["team_season"] = plays.posteam + " " + plays.season.astype(str)
    plays.drop(columns=["season_type", "play_type", "offense_personnel", "defense_personnel"]).to_parquet(PLAYS)
    print(f"{len(plays):,} runs and dropbacks, {plays.game_id.nunique():,} games, "
          f"{missing.sum()} without personnel and {fakes.sum()} fake kicks dropped -> {PLAYS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
