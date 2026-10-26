"""Load nflverse play-by-play and describe every play as a game state."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PLAYS = ROOT / "data" / "plays.parquet"
SEASONS = range(1999, 2026)

# what the win-probability model sees, always from the side of the team with the ball
FEATURES = [
    "score_differential",
    "game_seconds_remaining",
    "half_seconds_remaining",
    "yardline_100",
    "down",
    "ydstogo",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "receive_2h_ko",
    "posteam_spread",
    "home",
]

COLUMNS = [
    "game_id", "season", "season_type", "week", "home_team", "away_team", "posteam", "defteam",
    "qtr", "time", "down", "ydstogo", "yardline_100", "score_differential",
    "game_seconds_remaining", "half_seconds_remaining", "posteam_timeouts_remaining",
    "defteam_timeouts_remaining", "home_opening_kickoff", "result", "play_type", "qb_kneel",
    "yards_gained", "first_down_rush", "first_down_pass", "touchdown", "td_team",
    "field_goal_result", "kick_distance", "return_yards", "touchback", "punt_blocked",
    "fixed_drive", "drive_start_transition", "spread_line", "vegas_wp", "desc",
]


def load(seasons=SEASONS, raw=RAW):
    """Every regulation play of every regular-season game, with the state columns added."""
    plays = pd.concat(
        [pd.read_parquet(raw / f"play_by_play_{s}.parquet", columns=COLUMNS) for s in seasons],
        ignore_index=True,
    )
    plays = plays[(plays.season_type == "REG") & (plays.qtr <= 4)]
    return add_state(plays)


def time_scaled(score, spread, game_seconds):
    """nflfastR's two time-scaled inputs: the spread fades as the game goes on, a lead grows.

    decay = exp(-4 x share of the game played): 1 at kickoff, 0.14 at the half, 0.02 at the end.
    The spread is multiplied by it and the lead divided by it.
    """
    decay = np.exp(-4 * (3600 - game_seconds) / 3600)
    return spread * decay, score / decay


def add_state(plays):
    """Add who gets the second-half kickoff, the spread, and whether the team with the ball won."""
    at_home = plays.posteam == plays.home_team
    # the team that received the opening kickoff kicks off to start the second half
    receives_second_half = at_home != (plays.home_opening_kickoff == 1)
    first_half = plays.game_seconds_remaining > 1800
    home_margin = np.sign(plays.result)
    won = np.where(at_home, home_margin, -home_margin)
    plays = plays.assign(
        receive_2h_ko=(receives_second_half & first_half).astype(int),
        home=at_home.astype(int),
        # points the team with the ball was favored by before kickoff (spread_line is the home side)
        posteam_spread=np.where(at_home, plays.spread_line, -plays.spread_line),
    )
    spread_time, diff_time_ratio = time_scaled(plays.score_differential, plays.posteam_spread,
                                               plays.game_seconds_remaining)
    return plays.assign(
        spread_time=spread_time,
        diff_time_ratio=diff_time_ratio,
        posteam_won=pd.Series(won, index=plays.index).map({1.0: 1, -1.0: 0}),  # ties -> NaN
    )


def states(plays):
    """Plays the win-probability model can learn from: a real snap and a game that was decided."""
    keep = plays.posteam.notna() & plays.posteam_won.notna()
    return plays[keep].dropna(subset=FEATURES)
