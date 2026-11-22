"""Grade every fourth down from 2015 to 2025, with 90% intervals from resampling games.

The win-probability model is fixed (nflfastR's). What's uncertain is everything measured from our own
plays: conversion rates, gains, field goals, punts, kickoffs and clock. Each resample draws games with
replacement, refits those, and grades every fourth down again.

Writes results/graded_fourth_downs.csv.gz, results/by_season.csv, results/recommendation_map.csv and
models/submodels.json (the fitted sub-models, for scripts/call.py).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fourth_down.data import PLAYS
from fourth_down.decide import FIRST_SEASON, choice, fit_submodels, submodels_to_json
from fourth_down.grading import HIGH, LOW, Grader
from fourth_down.wp import WinProbability

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SUBMODELS = ROOT / "models" / "submodels.json"
LAST_SEASON = 2025
RESAMPLES = 200
STATE = ["score_differential", "yardline_100", "ydstogo", "posteam_spread", "posteam_timeouts_remaining",
         "defteam_timeouts_remaining", "game_seconds_remaining", "half_seconds_remaining", "home"]


def fourth_downs(plays):
    f = plays[(plays.down == 4) & plays.season.between(FIRST_SEASON, LAST_SEASON)
              & plays.play_type.isin(["pass", "run", "punt", "field_goal"]) & (plays.qb_kneel != 1)]
    return f.dropna(subset=STATE)


def resample_games(plays, rng):
    games = plays.game_id.unique()
    counts = pd.Series(rng.choice(games, len(games))).value_counts()
    out = plays.loc[plays.index.repeat(plays.game_id.map(counts).fillna(0).astype(int))]
    copy = out.groupby(level=0).cumcount().astype(str)  # a game drawn twice becomes two games
    return out.assign(game_id=out.game_id + "_" + copy).reset_index(drop=True)


def resampled_submodels(fit_on, rng, n=RESAMPLES):
    for _ in range(n):
        yield fit_submodels(resample_games(fit_on, rng))


def map_grid():
    """Two even teams, tie game, start of the second quarter, three timeouts each, home side unknown."""
    rows = [(togo, yl) for togo in range(1, 11) for yl in range(10, 90, 10)]
    return pd.DataFrame(rows, columns=["ydstogo", "yardline_100"]).assign(
        score_differential=0, game_seconds_remaining=2700, half_seconds_remaining=900, down=4,
        posteam_timeouts_remaining=3, defteam_timeouts_remaining=3, receive_2h_ko=0, posteam_spread=0.0,
        season=LAST_SEASON, home=np.nan)


def main():
    wp = WinProbability()
    plays = pd.read_parquet(PLAYS)
    fit_on = plays[plays.season.between(FIRST_SEASON, LAST_SEASON)]
    f = fourth_downs(plays)
    f = f.assign(coach=choice(f.play_type))

    sub = fit_submodels(fit_on)
    rng = np.random.default_rng(2025)
    resamples = []
    for i, sub_b in enumerate(resampled_submodels(fit_on, rng)):
        resamples.append(sub_b)
        if i % 20 == 0:
            print("resample", i, flush=True)
    SUBMODELS.write_text(json.dumps({"point": submodels_to_json(sub),
                                     "resamples": [submodels_to_json(s) for s in resamples]}))
    grader = Grader(wp, sub, resamples)
    graded, costs = grader.grade(f, resample_costs=True)
    point = f.join(graded)
    point["went"] = point.coach == "go"
    cols = ["game_id", "season", "week", "posteam", "defteam", "qtr", "time", "down", "ydstogo", "yardline_100",
            "score_differential", "coach", "go", "fg", "punt", "go_edge", "edge_low", "edge_high", "cost", "call",
            "gap", "gap_low", "gap_high", "clear"]
    RESULTS.mkdir(exist_ok=True)
    point[cols].round(4).to_csv(RESULTS / "graded_fourth_downs.csv.gz", index=False)

    # by season: go rate in clear-go spots (interval: resampling games of that season),
    # share of fourth downs that are clear calls, and wins lost on clear calls
    rows = []
    season_ids = point.season.to_numpy()
    for season, d in point.groupby("season"):
        cg = d[d.clear_go]
        games = cg.game_id.unique()
        rates = []
        for _ in range(1000):
            pick = pd.Series(rng.choice(games, len(games))).value_counts()
            w = cg.game_id.map(pick).fillna(0)
            rates.append((w * cg.went).sum() / w.sum())
        k = (season_ids == season) & point.clear.to_numpy()
        lost = costs[:, k].sum(axis=1)
        rows.append({
            "season": season,
            "fourth_downs": len(d),
            "clear_calls": int(d.clear.sum()),
            "clear_share": round(100 * d.clear.mean(), 1),
            "clear_go_spots": len(cg),
            "go_rate": round(100 * cg.went.mean(), 1),
            "go_rate_low": round(100 * np.percentile(rates, LOW), 1),
            "go_rate_high": round(100 * np.percentile(rates, HIGH), 1),
            "went_when_clear_kick": round(100 * d.went[d.clear_kick].mean(), 1),
            "wins_lost": round(float(np.median(lost)), 1),
            "wins_lost_low": round(float(np.percentile(lost, LOW)), 1),
            "wins_lost_high": round(float(np.percentile(lost, HIGH)), 1),
        })
    pd.DataFrame(rows).to_csv(RESULTS / "by_season.csv", index=False)

    grid = map_grid()
    m = grid[["ydstogo", "yardline_100"]].join(grader.grade(grid).drop(columns=["clear_go", "clear_kick"]))
    m.round(4).to_csv(RESULTS / "recommendation_map.csv", index=False)
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main()
