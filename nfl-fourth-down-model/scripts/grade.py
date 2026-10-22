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
from fourth_down.decide import FIRST_SEASON, choice, choice_wp, fit_submodels, grade, submodels_to_json
from fourth_down.wp import WinProbability

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SUBMODELS = ROOT / "models" / "submodels.json"
LAST_SEASON = 2025
RESAMPLES = 200
LOW, HIGH = 5, 95  # percentiles of a 90% interval
CLEAR = 0.01  # a clear call wins by at least a point of win probability across the whole interval
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
    """Two even teams, tie game, start of the second quarter, three timeouts each."""
    rows = [(togo, yl, home) for togo in range(1, 11) for yl in range(80, 0, -10) for home in (0, 1)]
    return pd.DataFrame(rows, columns=["ydstogo", "yardline_100", "home"]).assign(
        score_differential=0, game_seconds_remaining=2700, half_seconds_remaining=900, down=4,
        posteam_timeouts_remaining=3, defteam_timeouts_remaining=3, receive_2h_ko=0, posteam_spread=0.0,
        season=LAST_SEASON)


def map_wp(wp, sub, grid):
    """Win probability of each choice on the grid, averaged over the offense being home and away."""
    return choice_wp(wp, sub, grid).groupby([grid.ydstogo, grid.yardline_100]).mean()


def map_edge(wp, sub, grid):
    """Go edge on the grid, averaged over the offense being home and away."""
    g = grade(choice_wp(wp, sub, grid), pd.Series("punt", index=grid.index))
    return g.go_edge.groupby([grid.ydstogo, grid.yardline_100]).mean()


def main():
    wp = WinProbability()
    plays = pd.read_parquet(PLAYS)
    fit_on = plays[plays.season.between(FIRST_SEASON, LAST_SEASON)]
    f = fourth_downs(plays)
    coach = choice(f.play_type)
    grid = map_grid()

    sub = fit_submodels(fit_on)
    point = f.join(choice_wp(wp, sub, f)).join(grade(choice_wp(wp, sub, f), coach))
    point_map = map_edge(wp, sub, grid)

    rng = np.random.default_rng(2025)
    edges, costs, maps, saved = [], [], [], []
    for i, sub_b in enumerate(resampled_submodels(fit_on, rng)):
        saved.append(submodels_to_json(sub_b))
        g = grade(choice_wp(wp, sub_b, f), coach)
        edges.append(g.go_edge.to_numpy())
        costs.append(g.cost.to_numpy())
        maps.append(map_edge(wp, sub_b, grid).to_numpy())
        if i % 20 == 0:
            print("resample", i, flush=True)
    edges, costs, maps = np.array(edges), np.array(costs), np.array(maps)
    SUBMODELS.write_text(json.dumps({"point": submodels_to_json(sub), "resamples": saved}))

    point["edge_low"], point["edge_high"] = np.percentile(edges, [LOW, HIGH], axis=0)
    point["coach"] = coach
    point["went"] = coach == "go"
    point["clear_go"] = point.edge_low >= CLEAR
    point["clear_kick"] = point.edge_high <= -CLEAR
    point["clear"] = point.clear_go | point.clear_kick
    cols = ["game_id", "season", "week", "posteam", "defteam", "qtr", "time", "down", "ydstogo", "yardline_100",
            "score_differential", "coach", "go", "fg", "punt", "go_edge", "edge_low", "edge_high", "cost", "clear"]
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

    m = map_wp(wp, sub, grid).join(point_map.rename("go_edge")).reset_index()
    m["low"], m["high"] = np.percentile(maps, [LOW, HIGH], axis=0)
    m.round(4).to_csv(RESULTS / "recommendation_map.csv", index=False)
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main()
