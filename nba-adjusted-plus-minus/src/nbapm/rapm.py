"""Adjusted plus-minus: the margin Bradley-Terry model, with ten players per row instead of two
teams.

    margin per 100 possessions = home court + sum(home five) - sum(away five) + noise

fit by possession-weighted ridge regression, so every rating is points per 100 possessions
relative to an average player.
"""

import numpy as np
import pandas as pd
from scipy import sparse

HOME = ["h1", "h2", "h3", "h4", "h5"]
AWAY = ["a1", "a2", "a3", "a4", "a5"]


def design(stints, players):
    """Sparse matrix with +1 for each home player and -1 for each away player in every stint."""
    col = {pid: j for j, pid in enumerate(players)}
    rows = np.repeat(np.arange(len(stints)), 10)
    cols = [col[pid] for pid in stints[HOME + AWAY].to_numpy().ravel()]
    vals = np.tile([1.0] * 5 + [-1.0] * 5, len(stints))
    return sparse.csr_matrix((vals, (rows, cols)), shape=(len(stints), len(players)))


def fit(stints, lam, weights=None, prior=None, sd=False):
    """Home court and one net rating per player (no offense/defense split).

    lam = 0 is plain least squares (adjusted plus-minus with no shrinkage); larger lam pulls
    every rating toward `prior` (a Series of ratings, missing players 0), or toward 0, the
    average player. With sd=True also returns each rating's posterior standard deviation,
    sqrt(sigma^2 * diag(A^-1)), with sigma^2 estimated from the weighted residuals.
    """
    keep = (stints["poss"] > 0).to_numpy()
    stints = stints[keep]
    players = np.unique(stints[HOME + AWAY].to_numpy())
    X = design(stints, players)
    X = sparse.hstack([sparse.csr_matrix(np.ones((len(stints), 1))), X]).tocsr()
    y = 100 * (stints["home_pts"] - stints["away_pts"]).to_numpy() / stints["poss"].to_numpy()
    w = stints["poss"].to_numpy() * (1 if weights is None else weights[keep])
    A = (X.T @ X.multiply(w[:, None])).toarray()
    A[1:, 1:] += lam * np.eye(len(players))  # home court (column 0) is not shrunk
    b = X.T @ (w * y)
    if prior is not None:
        b[1:] += lam * pd.Series(players).map(prior).fillna(0).to_numpy()
    coef = np.linalg.lstsq(A, b, rcond=None)[0]
    hca, ratings = coef[0], pd.Series(coef[1:], index=players)
    if not sd:
        return hca, ratings
    resid = y - X @ coef
    sigma2 = (w * resid**2).sum() / (len(y) - len(coef))
    return hca, ratings, pd.Series(np.sqrt(sigma2 * np.diag(np.linalg.inv(A))[1:]), index=players)


def raw_plus_minus(stints):
    """Each player's on-court net rating: his team's margin per 100 possessions while he played."""
    margin = stints["home_pts"] - stints["away_pts"]
    parts = []
    for sign, side in ((1, HOME), (-1, AWAY)):
        for c in side:
            parts.append(pd.DataFrame({"pid": stints[c], "m": sign * margin, "p": stints["poss"]}))
    t = pd.concat(parts).groupby("pid")[["m", "p"]].sum()
    return 100 * t["m"] / t["p"]


def team_net(stints):
    """Each team's net rating, points per 100 possessions."""
    margin = stints["home_pts"] - stints["away_pts"]
    t = pd.concat(
        [
            pd.DataFrame({"team": stints["home_team"], "m": margin, "p": stints["poss"]}),
            pd.DataFrame({"team": stints["away_team"], "m": -margin, "p": stints["poss"]}),
        ]
    )
    t = t.groupby("team")[["m", "p"]].sum()
    return 100 * t["m"] / t["p"]


def home_court(stints):
    """Average home margin per 100 possessions."""
    return 100 * (stints["home_pts"] - stints["away_pts"]).sum() / stints["poss"].sum()


def lineup_rating(stints, ratings, side, how):
    """Sum (adjusted ratings) or average (raw plus-minus, a team number) over a five.
    Players with no rating count as 0, an average player."""
    values = np.column_stack([stints[c].map(ratings).fillna(0).to_numpy() for c in side])
    return values.sum(axis=1) if how == "sum" else values.mean(axis=1)


def game_errors(stints, per100):
    """Predicted minus actual final margin, in points, for every game.

    `per100` is a predicted home margin per 100 possessions for every stint; each game's
    prediction adds them up over the possessions actually played by each lineup."""
    g = pd.DataFrame(
        {
            "game": stints["game_id"].to_numpy(),
            "pred": per100 * stints["poss"].to_numpy() / 100,
            "actual": (stints["home_pts"] - stints["away_pts"]).to_numpy(),
        }
    )
    g = g.groupby("game")[["pred", "actual"]].sum()
    return (g["pred"] - g["actual"]).to_numpy()


def rmse(errors):
    return float(np.sqrt(np.mean(np.square(errors))))


def z_score(errors_a, errors_b):
    """z for the difference in squared error, game by game. Below -2: a is clearly better."""
    d = np.square(errors_a) - np.square(errors_b)
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))


def predict(train, test, method, lam=None):
    """Predicted home margin per 100 possessions for every test stint, learned from train."""
    if method == "home court":
        return np.full(len(test), home_court(train))
    if method == "team net rating":
        net = team_net(train)
        return (
            home_court(train)
            + test["home_team"].map(net).fillna(0).to_numpy()
            - test["away_team"].map(net).fillna(0).to_numpy()
        )
    if method == "raw plus-minus":
        raw = raw_plus_minus(train)
        return (
            home_court(train)
            + lineup_rating(test, raw, HOME, "mean")
            - lineup_rating(test, raw, AWAY, "mean")
        )
    hca, ratings = fit(train, lam)
    return (
        hca + lineup_rating(test, ratings, HOME, "sum") - lineup_rating(test, ratings, AWAY, "sum")
    )
