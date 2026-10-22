"""Parse every game in data/raw into late 4th-quarter and overtime segments.

Writes data/derived/segments.parquet.
"""

from pathlib import Path

import pandas as pd

from foulup3 import SEASONS
from foulup3.pbp import action, game_segments, label, scores

RAW = Path("data/raw")
OUT = Path("data/derived")
MAX_SECONDS = 24  # only segments that start with 24 seconds or less left in the period


def games() -> pd.DataFrame:
    """One row per game: season, home and away team, final points, and each team's nickname."""
    rows = []
    for season in SEASONS:
        g = pd.read_parquet(RAW / f"games_{season[:4]}{season[-2:]}_RegularSeason.parquet")
        # "BOS vs. ATL" is Boston's row at home; "ATL @ BOS" is Atlanta's row away. Neutral-site
        # games write both rows as "away @ home", so name the home team rather than the row
        words = g.MATCHUP.str.split()
        home_abbr = words.str[-1].where(g.MATCHUP.str.contains("@"), words.str[0])
        g["home"] = g.TEAM_ABBREVIATION == home_abbr
        home = g[g.home].set_index("GAME_ID")
        away = g[~g.home].set_index("GAME_ID")
        both = home.join(away, lsuffix="_h", rsuffix="_a", how="inner")
        rows.append(
            pd.DataFrame(
                {
                    "game_id": both.index,
                    "season": season,
                    "home": both.TEAM_ID_h.values,
                    "away": both.TEAM_ID_a.values,
                    "home_pts": both.PTS_h.values,
                    "away_pts": both.PTS_a.values,
                    # the last word of the team name is how team rebounds and turnovers are written
                    "home_name": both.TEAM_NAME_h.str.split().str[-1].str.lower().values,
                    "away_name": both.TEAM_NAME_a.str.split().str[-1].str.lower().values,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def main():
    g = games()
    segs, mismatched = [], []
    for row in g.itertuples(index=False):
        pbp = pd.read_parquet(RAW / "pbp" / f"{row.game_id}.parquet")
        final = scores(pbp).iloc[-1]  # a check on the score parsing and the home/away join
        if (final.home_pts, final.away_pts) != (row.home_pts, row.away_pts):
            mismatched.append(row.game_id)
        names = {row.home: row.home_name, row.away: row.away_name}
        for s in game_segments(pbp, row.home, row.away, names):
            if s["t0"] <= MAX_SECONDS:
                won = (row.home_pts > row.away_pts) == (s["off"] == row.home)
                segs.append(
                    dict(
                        s,
                        game_id=row.game_id,
                        season=row.season,
                        action=action(s),
                        label=label(s["result"], s["made"], s["fta"], s["ftm"], s["next"]),
                        offense_won=won,
                    )
                )
    segs = pd.DataFrame(segs).drop(columns=["n", "foul_t"])
    segs = segs.astype({"d0": int, "delta": int, "off": int})
    OUT.mkdir(parents=True, exist_ok=True)
    segs.to_parquet(OUT / "segments.parquet", index=False)
    print(
        f"{g.game_id.nunique()} games, {segs.game_id.nunique()} with segments, "
        f"{len(segs)} segments, final score differs from the game log in {mismatched}"
    )


if __name__ == "__main__":
    main()
