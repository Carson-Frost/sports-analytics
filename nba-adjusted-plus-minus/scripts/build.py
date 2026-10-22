"""Read the season bundles and write every regular season stint to artifacts/stints.csv.gz.

Also writes results/data_checks.csv: per season, how many games were kept, and how well the
reconstructed lineups agree with the games that have official rotation data.
"""

import json
import tarfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

from nbapm.stints import game_stints, roster

ROOT = Path(__file__).resolve().parents[1]
BUNDLES = ROOT / "data" / "raw" / "bundles"
SEASONS = range(2015, 2027)  # bundle years: 2015 is 2014-15, 2026 is 2025-26
WANTED = {"playbyplayv3", "boxscoretraditionalv3", "gamerotation", "leaguegamelog"}


def season_label(year):
    return f"{year - 1}-{str(year)[2:]}"


def read_bundle(year):
    """{game_id: {endpoint: json}} for regular season games, plus {game_id: date}."""
    games = defaultdict(dict)
    dates = {}
    with tarfile.open(BUNDLES / f"nba_stats_json_{year}.tar.gz") as tf:
        for m in tf:
            parts = m.name.split("/")
            if len(parts) != 3 or parts[0] not in WANTED or not m.isfile():
                continue
            if parts[0] == "leaguegamelog":
                if parts[2] == "regular-season.json":
                    rs = json.load(tf.extractfile(m))["resultSets"][0]
                    i, d = rs["headers"].index("GAME_ID"), rs["headers"].index("GAME_DATE")
                    dates.update({row[i]: row[d] for row in rs["rowSet"]})
            elif parts[2].startswith("002"):  # 002 = regular season
                games[parts[2][:-5]][parts[0]] = json.load(tf.extractfile(m))
    return games, dates


def rotation_agreement(stints, rotation):
    """Seconds of stints whose ten players match the official rotation at the stint's middle."""
    rows = [r for rs in rotation["resultSets"] for r in rs["rowSet"]]
    matched = 0.0
    for s in stints:
        middle = (s["start"] + s["seconds"] / 2) * 10  # rotation times are tenths of a second
        on = {r[4] for r in rows if r[7] <= middle < r[8]}
        if on == set(s["home"] + s["away"]):
            matched += s["seconds"]
    return matched


def main():
    stint_rows, player_rows, checks = [], [], []
    for year in SEASONS:
        games, dates = read_bundle(year)
        kept = dropped = 0
        rot_total = rot_matched = 0.0
        for gid, f in sorted(games.items()):
            if "playbyplayv3" not in f or "boxscoretraditionalv3" not in f or gid not in dates:
                dropped += 1
                continue
            box = f["boxscoretraditionalv3"]
            stints = game_stints(f["playbyplayv3"], box)
            b = box["boxScoreTraditional"]
            box_pts = [
                sum(p["statistics"]["points"] for p in b[t]["players"])
                for t in ("homeTeam", "awayTeam")
            ]
            if [sum(s["home_pts"] for s in stints), sum(s["away_pts"] for s in stints)] != box_pts:
                dropped += 1  # play-by-play scoring disagrees with the box score
                continue
            kept += 1
            if "gamerotation" in f:
                rot_total += sum(s["seconds"] for s in stints)
                rot_matched += rotation_agreement(stints, f["gamerotation"])
            home, away = b["homeTeam"]["teamTricode"], b["awayTeam"]["teamTricode"]
            for s in stints:
                stint_rows.append(
                    [
                        season_label(year),
                        gid,
                        dates[gid],
                        home,
                        away,
                        s["seconds"],
                        s["poss"],
                        s["home_pts"],
                        s["away_pts"],
                        *s["home"],
                        *s["away"],
                    ]
                )
            _, _, players = roster(box)
            team_code = {b["homeTeam"]["teamId"]: home, b["awayTeam"]["teamId"]: away}
            for pid, p in players.items():
                player_rows.append(
                    [season_label(year), pid, p["name"], team_code[p["team"]], p["seconds"]]
                )
        checks.append(
            [
                season_label(year),
                kept,
                dropped,
                round(rot_total / 3600, 1),
                round(rot_matched / rot_total, 4),
            ]
        )
        print(
            season_label(year),
            "games kept",
            kept,
            "dropped",
            dropped,
            "rotation match",
            round(rot_matched / rot_total, 4),
        )

    out = ROOT / "artifacts"
    out.mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    cols = [
        "season",
        "game_id",
        "date",
        "home_team",
        "away_team",
        "seconds",
        "poss",
        "home_pts",
        "away_pts",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
    ]
    pd.DataFrame(stint_rows, columns=cols).to_csv(out / "stints.csv.gz", index=False)
    pd.DataFrame(player_rows, columns=["season", "player_id", "name", "team", "seconds"]).to_csv(
        out / "player_games.csv.gz", index=False
    )
    pd.DataFrame(
        checks, columns=["season", "games", "games_dropped", "rotation_hours", "rotation_match"]
    ).to_csv(ROOT / "results" / "data_checks.csv", index=False)


if __name__ == "__main__":
    main()
