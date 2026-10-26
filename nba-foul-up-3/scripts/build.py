"""Parse every regular-season game in the bundles into late 4th-quarter and overtime segments.

Stops if a bundle differs from data/manifest.csv, since the results would no longer match.
Writes data/derived/games.parquet and data/derived/segments.parquet. From 2017-18, when the
current timeout rules began, each segment also has the timeouts both teams have left.
"""

import csv
import json
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from foulup3 import SEASONS
from foulup3.pbp import (
    TIMEOUT_RULES_FROM,
    action,
    clean,
    game_segments,
    home_team,
    label,
    scores,
    timeouts_left,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "manifest.csv"
OUT = ROOT / "data" / "derived"
MAX_SECONDS = 24  # only segments that start with 24 seconds or less left in the period
COLUMNS = ["clock", "period", "teamId", "scoreHome", "scoreAway", "description", "actionType",
           "subType", "shotValue"]  # fmt: skip


def check_manifest():
    from pull import sha256  # same folder

    for row in csv.DictReader(MANIFEST.open()):
        path = RAW / row["file"]
        if sha256(path) != row["sha256"]:
            raise SystemExit(
                f"{path.name} differs from data/manifest.csv (downloaded {row['downloaded']}). "
                "Delete the manifest row to accept the new file, then rerun everything."
            )


def season(season: str) -> tuple[list[dict], list[dict]]:
    """Games and segments for one season, read straight from its bundle."""
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
                    pbps[m.name.split("/")[-1][:-5]] = clean(pd.DataFrame(actions)[COLUMNS])
    games, segs = [], []
    for game_id, two in log.groupby("GAME_ID"):
        if game_id not in pbps:
            continue
        pbp = pbps[game_id]
        home_id = home_team(pbp)
        h, a = two[two.TEAM_ID == home_id].iloc[0], two[two.TEAM_ID != home_id].iloc[0]
        final = tuple(scores(pbp).iloc[-1])
        games.append(dict(game_id=game_id, season=season, date=h.GAME_DATE, home=h.TEAM_ID,
                          away=a.TEAM_ID, home_team=h.TEAM_NAME, away_team=a.TEAM_NAME,
                          home_pts=h.PTS, away_pts=a.PTS,
                          score_matches=final == (h.PTS, a.PTS)))  # fmt: skip
        # team rebounds and turnovers are written with the nickname: "CELTICS Rebound"
        names = {t.TEAM_ID: t.TEAM_NAME.split()[-1].lower() for t in (h, a)}
        other = {h.TEAM_ID: a.TEAM_ID, a.TEAM_ID: h.TEAM_ID}
        left = {team: timeouts_left(pbp, nick) for team, nick in names.items()}
        known = season >= TIMEOUT_RULES_FROM
        for s in game_segments(pbp, h.TEAM_ID, a.TEAM_ID, names):
            if s["t0"] <= MAX_SECONDS:
                won = (h.PTS > a.PTS) == (s["off"] == h.TEAM_ID)
                timeouts = left[s["off"]](s["period"], s["t0"]) if known else None
                opp = left[other[s["off"]]](s["period"], s["t0"]) if known else None
                segs.append(dict(s, game_id=game_id, season=season, action=action(s),
                                 label=label(s["result"], s["made"], s["fta"], s["ftm"], s["next"]),
                                 offense_won=won, timeouts=timeouts,
                                 opp_timeouts=opp))  # fmt: skip
    print(f"{season}: {len(games)} games, {len(segs)} segments", flush=True)
    return games, segs


def main():
    check_manifest()
    with ProcessPoolExecutor() as pool:
        parts = list(pool.map(season, SEASONS))
    games = pd.DataFrame([g for p in parts for g in p[0]])
    segs = pd.DataFrame([s for p in parts for s in p[1]]).drop(columns=["n", "foul_t"])
    segs = segs.astype({"d0": int, "delta": int, "off": int})
    OUT.mkdir(parents=True, exist_ok=True)
    games.to_parquet(OUT / "games.parquet", index=False)
    segs.to_parquet(OUT / "segments.parquet", index=False)
    print(f"{len(games):,} games, {len(segs):,} segments; final score differs from the game log "
          f"in {(~games.score_matches).sum()} games")  # fmt: skip


if __name__ == "__main__":
    main()
