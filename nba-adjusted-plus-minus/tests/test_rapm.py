import numpy as np
import pandas as pd

from nbapm.rapm import fit, game_errors, raw_plus_minus, team_net, z_score


def fake_stints(n=4000, seed=0):
    """Random lineups from 20 players, margins drawn from known true ratings plus noise."""
    rng = np.random.default_rng(seed)
    truth = pd.Series(np.linspace(-5, 5, 20), index=range(100, 120))
    rows = []
    for i in range(n):
        ten = rng.choice(truth.index, 10, replace=False)
        poss = 10.0
        margin = 3 + truth[ten[:5]].sum() - truth[ten[5:]].sum() + rng.normal(0, 10)
        rows.append([f"g{i % 200}", "AAA", "BBB", poss, margin * poss / 100, 0, *ten])
    cols = [
        "game_id",
        "home_team",
        "away_team",
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
    return pd.DataFrame(rows, columns=cols), truth


def test_least_squares_recovers_the_true_ratings_and_home_court():
    stints, truth = fake_stints()
    hca, ratings = fit(stints, 0)
    centred = ratings - ratings.mean()  # plain least squares only knows ratings up to a constant
    assert abs(hca - 3) < 0.5
    assert np.abs(centred[truth.index] - truth).max() < 0.5


def test_a_bigger_penalty_shrinks_ratings_toward_zero():
    stints, _ = fake_stints()
    _, small = fit(stints, 10)
    _, big = fit(stints, 1_000_000)
    assert big.abs().max() < small.abs().max() / 10


def test_a_big_penalty_shrinks_toward_the_prior_and_posterior_sds_are_positive():
    stints, truth = fake_stints()
    _, ratings = fit(stints, 100_000_000, prior=truth)
    assert (ratings[truth.index] - truth).abs().max() < 0.01
    _, _, sd = fit(stints, 2000, sd=True)
    assert (sd > 0).all() and sd.max() < 5


def test_raw_plus_minus_and_team_net_are_per_100_possessions():
    stints = pd.DataFrame(
        {
            "game_id": ["g"],
            "home_team": ["AAA"],
            "away_team": ["BBB"],
            "poss": [50.0],
            "home_pts": [60],
            "away_pts": [50],
            "h1": [1],
            "h2": [2],
            "h3": [3],
            "h4": [4],
            "h5": [5],
            "a1": [6],
            "a2": [7],
            "a3": [8],
            "a4": [9],
            "a5": [10],
        }
    )
    raw = raw_plus_minus(stints)
    assert raw[1] == 20 and raw[6] == -20
    assert team_net(stints)["AAA"] == 20


def test_game_errors_add_stints_up_to_a_game_margin():
    stints = pd.DataFrame(
        {
            "game_id": ["g", "g", "h"],
            "poss": [50.0, 50.0, 100.0],
            "home_pts": [55, 50, 90],
            "away_pts": [50, 50, 100],
        }
    )
    errors = game_errors(stints, np.array([10.0, 0.0, 0.0]))
    assert list(errors) == [0, 10]  # g: 5 predicted, 5 actual. h: 0 predicted, lost by 10


def test_z_score_is_negative_when_the_first_set_of_errors_is_smaller():
    rng = np.random.default_rng(1)
    big = rng.normal(0, 12, 2000)
    assert z_score(big * 0.9, big) < -2
