import numpy as np
import pandas as pd
import pytest

from personnel import defensive_backs, grouping, situation
from personnel.fit import Design


@pytest.mark.parametrize(
    ("personnel", "code"),
    [
        ("1 RB, 1 TE, 3 WR", "11"),  # 2016-2022 format
        ("2 RB, 2 TE, 1 WR", "22"),
        ("6 OL, 1 RB, 2 TE, 1 WR", "12 (6 OL)"),
        ("1 C, 2 G, 1 QB, 1 RB, 2 T, 1 TE, 3 WR", "11"),  # 2023 on
        ("1 C, 1 FB, 2 G, 1 QB, 1 RB, 2 T, 1 TE, 2 WR", "21"),  # fullback counts as a back
        ("1 C, 2 G, 1 QB, 1 RB, 3 T, 1 TE, 2 WR", "11 (6 OL)"),
        ("1 C, 1 DT, 2 G, 1 QB, 1 RB, 2 T, 2 TE, 1 WR", "12 (6 OL)"),  # DT brought in to block
        ("1 LB, 1 RB, 1 TE, 2 WR", "11"),  # an extra linebacker is not a lineman
        ("1 C, 2 G, 2 QB, 1 RB, 2 T, 1 TE, 2 WR", "11 (2 QB)"),
        ("1 RB, 1 TE, 2 WR, 1 K", "Special Teams"),
    ],
)
def test_grouping(personnel, code):
    assert grouping(personnel) == code


@pytest.mark.parametrize("personnel", [None, "", "   ", "two RB"])
def test_grouping_missing(personnel):
    assert grouping(personnel) is None


def test_defensive_backs_both_formats():
    assert defensive_backs("4 DL, 2 LB, 5 DB") == 5
    assert defensive_backs("2 CB, 1 DE, 1 DT, 1 FS, 2 ILB, 1 NT, 2 OLB, 1 SS") == 4
    assert np.isnan(defensive_backs(None))


def test_situation():
    down = pd.Series([1, 2, 2, 3, 4, 3])
    togo = pd.Series([10, 8, 4, 2, 5, 3])
    yardline = pd.Series([75, 40, 40, 30, 30, 3])
    assert list(situation(down, togo, yardline)) == [
        "1st down", "2nd & 7+", "2nd & 3-6", "3rd/4th & 1-2", "3rd/4th & 3-6", "Inside the 5"]


def plays(n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    team = rng.integers(0, 40, n)
    twelve = rng.random(n) < 0.2 + 0.01 * team  # better teams use it more
    down = rng.integers(1, 4, n)
    togo = rng.integers(1, 16, n)
    epa = 0.05 * team / 40 + 0.10 * twelve - 0.01 * togo + rng.normal(0, 1, n)
    return pd.DataFrame({"grouping": np.where(twelve, "12", "11"), "down": down, "ydstogo": togo,
                         "yardline_100": rng.integers(1, 100, n), "epa": epa, "team_season": team.astype(str),
                         "game_code": rng.integers(0, 2_000, n)})


def test_fit_recovers_effect_within_teams():
    p = plays()
    effect = Design(p, ["11", "12"]).fit()["12"]
    naive = p.epa[p.grouping == "12"].mean() - p.epa[p.grouping == "11"].mean()
    assert abs(effect - 0.10) < 0.04
    assert naive > effect  # ignoring the team inflates it


def test_game_weights_equal_repeating_the_games():
    p = plays(4_000, seed=1)
    w = np.random.default_rng(2).integers(0, 3, 2_000)
    weighted = Design(p, ["11", "12"]).fit(w)["12"]
    repeated = p.loc[p.index.repeat(w[p.game_code])].reset_index(drop=True)
    assert weighted == pytest.approx(Design(repeated, ["11", "12"]).fit()["12"])


def test_fit_matches_dummy_regression():
    p = plays(3_000, seed=3)
    effect = Design(p, ["11", "12"]).fit()["12"]
    dd = p.down.astype(str) + "_" + p.ydstogo.astype(str)
    field = np.ceil(p.yardline_100 / 5).astype(int).astype(str)
    x = pd.concat([(p.grouping == "12").astype(float).rename("12"), pd.get_dummies(dd, dtype=float),
                   pd.get_dummies(field, dtype=float, drop_first=True),
                   pd.get_dummies(p.team_season, dtype=float, drop_first=True)], axis=1)
    beta = np.linalg.lstsq(x.to_numpy(), p.epa.to_numpy(), rcond=None)[0]
    assert effect == pytest.approx(beta[0])


def test_extra_control_matches_dummy_regression():
    p = plays(3_000, seed=4).assign(box=lambda d: d.game_code % 4 + 5)
    effect = Design(p, ["11", "12"], extra=("box",)).fit()["12"]
    dd = p.down.astype(str) + "_" + p.ydstogo.astype(str)
    field = np.ceil(p.yardline_100 / 5).astype(int).astype(str)
    x = pd.concat([(p.grouping == "12").astype(float).rename("12"), pd.get_dummies(dd, dtype=float),
                   pd.get_dummies(field, dtype=float, drop_first=True),
                   pd.get_dummies(p.box.astype(str), dtype=float, drop_first=True),
                   pd.get_dummies(p.team_season, dtype=float, drop_first=True)], axis=1)
    beta = np.linalg.lstsq(x.to_numpy(), p.epa.to_numpy(), rcond=None)[0]
    assert effect == pytest.approx(beta[0])
