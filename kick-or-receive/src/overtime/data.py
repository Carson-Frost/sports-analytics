"""Shape the raw nflverse files into one play table and one overtime game table."""

from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
DERIVED = Path("data/derived")
FIRST_SEASON, LAST_SEASON = 1999, 2026
LAST_WEEK_2026 = 5  # the raw snapshot ends here; later pulls are cut back to it

COLUMNS = [
    "game_id", "play_id", "season", "season_type", "week", "qtr", "game_half",
    "game_seconds_remaining", "half_seconds_remaining", "posteam", "defteam", "play_type",
    "down", "ydstogo", "yardline_100", "score_differential", "touchdown", "td_team",
    "interception", "fumble_lost", "safety", "kick_distance", "field_goal_result",
    "extra_point_result", "two_point_conv_result", "fixed_drive", "fixed_drive_result",
]


def read_season(season, raw=RAW):
    pbp = pd.read_parquet(raw / "pbp" / f"play_by_play_{season}.parquet", columns=COLUMNS)
    if season == 2026:
        pbp = pbp[pbp.week <= LAST_WEEK_2026]
    return pbp


def add_next_play(pbp):
    """Put the next real play's team, spot, down and clock on every row (same game and half)."""
    # timeouts and other team-less rows would break the chain from one snap to the next
    pbp = pbp[pbp.play_type.notna() & pbp.posteam.notna()].sort_values(["game_id", "play_id"])
    pbp = pbp.copy()
    nxt = pbp.groupby("game_id")[["posteam", "yardline_100", "down", "ydstogo",
                                  "game_seconds_remaining", "game_half", "play_type"]].shift(-1)
    same_half = nxt.game_half == pbp.game_half
    for col in ["posteam", "yardline_100", "down", "ydstogo", "play_type"]:
        pbp[f"next_{col}"] = nxt[col].where(same_half)
    pbp["secs"] = (pbp.game_seconds_remaining - nxt.game_seconds_remaining).where(same_half)
    return pbp


def build_plays(raw=RAW):
    return pd.concat([add_next_play(read_season(s, raw))
                      for s in range(FIRST_SEASON, LAST_SEASON + 1)], ignore_index=True)


def rule_for(season, game_type):
    """Overtime rule set and clock (seconds, None = untimed) for one game."""
    playoff = game_type != "REG"
    if playoff:
        rule = "sudden" if season <= 2009 else "modified" if season <= 2021 else "both"
        return rule, None
    if season <= 2011:
        return "sudden", 900
    if season <= 2024:
        return "modified", 900 if season <= 2016 else 600
    return "both", 600


def build_ot_games(plays, raw=RAW):
    """Every game with an overtime kickoff, who received it and who won.

    Overtimes come from the play-by-play, not the schedule's overtime flag, which misses some.
    """
    kicks = plays[(plays.qtr == 5) & (plays.play_type == "kickoff")]
    receiver = kicks.sort_values("play_id").groupby("game_id").posteam.first()
    games = pd.read_csv(raw / "games.csv")
    games = games[games.game_id.isin(receiver.index) & games.result.notna()]
    games = games.assign(receiver=games.game_id.map(receiver))
    winner = np.where(games.result > 0, games.home_team,
                      np.where(games.result < 0, games.away_team, "TIE"))
    rules = [rule_for(s, t) for s, t in zip(games.season, games.game_type, strict=True)]
    return pd.DataFrame({
        "game_id": games.game_id, "season": games.season, "game_type": games.game_type,
        "rule": [r for r, _ in rules],
        "clock": [c if c else 0 for _, c in rules],  # 0 = untimed
        "receiver": games.receiver,
        "outcome": np.where(winner == "TIE", 0, np.where(winner == games.receiver, 1, -1)),
    })
