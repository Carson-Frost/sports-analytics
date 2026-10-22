"""Choose lambda on 2014-15 to 2024-25, test on 2025-26, then rate every 2025-26 player.

Every season is split by date: the first two thirds of its games are for fitting, and the model
is scored on the final margin of every game in the last third, using who actually played.

Writes results/lambda_tuning.csv, model_test.csv, last_season_test.csv and ratings_2025_26.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from nbapm.rapm import (
    AWAY,
    HOME,
    fit,
    game_errors,
    lineup_rating,
    predict,
    raw_plus_minus,
    rmse,
    team_net,
    z_score,
)

ROOT = Path(__file__).resolve().parents[1]
TEST_SEASON = "2025-26"
LAMBDAS = [0, 250, 500, 1000, 1500, 2000, 3000, 4000, 6000, 8000, 12000, 16000, 24000]
BASELINES = ["home court", "team net rating", "raw plus-minus"]
BOOTSTRAPS = 200
MIN_MINUTES = 1000  # players shown in the charts and counted in the README
Z90 = 1.645  # 90% of a normal lies within 1.645 standard deviations


def split(season):
    """First two thirds of a season's games by date, and the last third."""
    games = season.drop_duplicates("game_id").sort_values(["date", "game_id"])
    cut = games["date"].iloc[round(2 * len(games) / 3)]
    return season[season["date"] < cut], season[season["date"] >= cut]


def main():
    stints = pd.read_csv(ROOT / "artifacts" / "stints.csv.gz")
    splits = {name: split(season) for name, season in stints.groupby("season")}
    tuning = [name for name in splits if name != TEST_SEASON]

    # 1. lambda: per-game errors in the last third of every tuning season, for every lambda
    errors = {}
    for name in tuning:
        train, test = splits[name]
        for lam in LAMBDAS:
            errors[name, lam] = game_errors(test, predict(train, test, "ridge", lam))
        for method in BASELINES:
            errors[name, method] = game_errors(test, predict(train, test, method))
        print(name, "done")

    def pooled(key):
        return np.concatenate([errors[name, key] for name in tuning])

    curve = pd.DataFrame(
        {"lambda": LAMBDAS, "rmse": [rmse(pooled(lam)) for lam in LAMBDAS], "games": len(pooled(0))}
    )
    curve.to_csv(ROOT / "results" / "lambda_tuning.csv", index=False)
    best = int(curve.loc[curve["rmse"].idxmin(), "lambda"])
    print(curve, "\nbest lambda", best)

    # 2. compare with simpler methods, on the tuning seasons and on 2025-26
    train, test = splits[TEST_SEASON]
    for lam in (0, best):
        errors[TEST_SEASON, lam] = game_errors(test, predict(train, test, "ridge", lam))
    for method in BASELINES:
        errors[TEST_SEASON, method] = game_errors(test, predict(train, test, method))
    keys = [
        ("home court", "home court"),
        ("team net rating", "team net rating"),
        ("raw plus-minus", "raw plus-minus"),
        ("adjusted plus-minus", 0),
        ("regularized adjusted plus-minus", best),
    ]
    rows = []
    for sample, get in (
        (f"{tuning[0]} to {tuning[-1]}", pooled),
        (TEST_SEASON, lambda k: errors[TEST_SEASON, k]),
    ):
        for method, key in keys:
            z = np.nan if key == best else z_score(get(best), get(key))
            rows.append([sample, method, len(get(key)), rmse(get(key)), z])
    table = pd.DataFrame(rows, columns=["sample", "method", "games", "rmse", "z_rapm_vs_method"])
    table.round(3).to_csv(ROOT / "results" / "model_test.csv", index=False)
    print(table)

    # 2b. does last season help? add its games, or shrink toward its ratings instead of 0
    last = stints[stints["season"] == tuning[-1]]
    team_errors = errors[TEST_SEASON, "team net rating"]
    variants = {
        "with 2024-25 games": fit(pd.concat([last, train]), best),
        "shrunk toward 2024-25 ratings": fit(train, best, prior=fit(last, best)[1]),
    }
    rows = []
    for name, (hca, ratings) in variants.items():
        per100 = (
            hca
            + lineup_rating(test, ratings, HOME, "sum")
            - lineup_rating(test, ratings, AWAY, "sum")
        )
        e = game_errors(test, per100)
        rows.append([name, len(e), rmse(e), rmse(team_errors), z_score(e, team_errors)])
    last_season = pd.DataFrame(
        rows, columns=["variant", "games", "rmse", "team_net_rmse", "z_vs_team_net"]
    )
    last_season.round(3).to_csv(ROOT / "results" / "last_season_test.csv", index=False)
    print(last_season)

    # 3. final ratings: every 2025-26 game, 90% posterior intervals, and how often each player
    # comes out first among 1,000-minute players when the season's games are resampled
    season = stints[stints["season"] == TEST_SEASON]
    hca, rating, sd = fit(season, best, sd=True)
    game_ids = season["game_id"].unique()
    rng = np.random.default_rng(2026)
    draws = []
    for _ in range(BOOTSTRAPS):
        counts = pd.Series(rng.choice(game_ids, len(game_ids))).value_counts()
        draws.append(fit(season, best, season["game_id"].map(counts).fillna(0).to_numpy())[1])
    draws = pd.concat(draws, axis=1)

    pg = pd.read_csv(ROOT / "artifacts" / "player_games.csv.gz")
    pg = pg[pg["season"] == TEST_SEASON]
    team = (
        pg.groupby(["player_id", "team"])["seconds"]
        .sum()
        .reset_index()
        .sort_values("seconds")
        .drop_duplicates("player_id", keep="last")
        .set_index("player_id")["team"]
    )  # the team he played the most minutes for
    players = pg.groupby("player_id").agg(name=("name", "last"), minutes=("seconds", "sum"))
    players["minutes"] = (players["minutes"] / 60).round(0)
    players["team"] = team
    players = players[players.index.isin(rating.index)]
    # shown so that 0 is the minutes-weighted average player
    shift = np.average(rating[players.index], weights=players["minutes"])
    players["rapm"] = rating - shift
    players["low"] = players["rapm"] - Z90 * sd
    players["high"] = players["rapm"] + Z90 * sd
    qualified = players.index[players["minutes"] >= MIN_MINUTES]
    firsts = draws.loc[qualified].idxmax().value_counts()
    players["first_in_resamples"] = firsts.reindex(players.index).fillna(0).astype(int)
    players["raw_plus_minus"] = raw_plus_minus(season)
    players["team_net"] = players["team"].map(team_net(season))
    players = players.sort_values("rapm", ascending=False)
    players.round(2).to_csv(ROOT / "results" / "ratings_2025_26.csv")
    print("home court", round(hca, 2), "per 100 possessions")


if __name__ == "__main__":
    main()
