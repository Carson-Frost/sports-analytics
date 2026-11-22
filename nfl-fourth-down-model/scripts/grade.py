"""Grade every fourth down from 2015 to 2025, with 90% intervals from resampling games.

The win-probability model is fixed (nflfastR's). What's uncertain is everything measured from our own
plays: conversion rates, gains, field goals, punts, kickoffs and clock. Each of 200 resamples draws
games with replacement, refits those, grades every fourth down again and weights the go rate by the
same draw.

Writes results/graded_fourth_downs.csv.gz, results/by_season.csv, results/recommendation_map.csv and
models/submodels.json (the fitted sub-models, for scripts/call.py).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fourth_down.data import PLAYS
from fourth_down.decide import FIRST_SEASON, choice, submodels_to_json
from fourth_down.grading import Grader
from fourth_down.wp import WinProbability

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SUBMODELS = ROOT / "models" / "submodels.json"
LAST_SEASON = 2025
STATE = ["score_differential", "yardline_100", "ydstogo", "posteam_spread", "posteam_timeouts_remaining",
         "defteam_timeouts_remaining", "game_seconds_remaining", "half_seconds_remaining", "home"]


def fourth_downs(plays):
    f = plays[(plays.down == 4) & plays.season.between(FIRST_SEASON, LAST_SEASON)
              & plays.play_type.isin(["pass", "run", "punt", "field_goal"]) & (plays.qb_kneel != 1)]
    return f.dropna(subset=STATE)


def map_grid():
    """Two even teams, tie game, start of the second quarter, three timeouts each, home side unknown."""
    rows = [(togo, yl) for togo in range(1, 11) for yl in range(10, 90, 10)]
    return pd.DataFrame(rows, columns=["ydstogo", "yardline_100"]).assign(
        score_differential=0, game_seconds_remaining=2700, half_seconds_remaining=900, down=4,
        posteam_timeouts_remaining=3, defteam_timeouts_remaining=3, receive_2h_ko=0, posteam_spread=0.0,
        season=LAST_SEASON, home=np.nan)


def main():
    plays = pd.read_parquet(PLAYS)
    grader = Grader.fit(WinProbability(), plays[plays.season.between(FIRST_SEASON, LAST_SEASON)])
    SUBMODELS.write_text(json.dumps({"point": submodels_to_json(grader.point),
                                     "resamples": [submodels_to_json(s) for s in grader.resamples]}))

    f = fourth_downs(plays)
    f = f.assign(coach=choice(f.play_type))
    graded, seasons = grader.grade_seasons(f)
    cols = ["game_id", "season", "week", "posteam", "defteam", "qtr", "time", "down", "ydstogo", "yardline_100",
            "score_differential", "coach", "go", "fg", "punt", "go_edge", "edge_low", "edge_high", "cost", "call",
            "gap", "gap_low", "gap_high", "clear"]
    RESULTS.mkdir(exist_ok=True)
    f.join(graded)[cols].round(4).to_csv(RESULTS / "graded_fourth_downs.csv.gz", index=False)
    seasons.round(1).to_csv(RESULTS / "by_season.csv", index=False)

    grid = map_grid()
    m = grid[["ydstogo", "yardline_100"]].join(grader.grade(grid).drop(columns=["clear_go", "clear_kick"]))
    m.round(4).to_csv(RESULTS / "recommendation_map.csv", index=False)
    print(seasons.round(1).to_string())

if __name__ == "__main__":
    main()
