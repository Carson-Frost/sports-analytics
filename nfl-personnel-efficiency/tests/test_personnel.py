import numpy as np
import pandas as pd
import pytest

from personnel import defensive_backs, grouping, prepare_plays, situation
from personnel.fit import Design
from personnel.resample import GroupMean, run


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


def test_group_mean_game_weights_equal_repeating_the_games():
    p = plays(4_000, seed=5)
    w = np.random.default_rng(6).integers(0, 3, 2_000)
    weighted = GroupMean(p, "epa", "grouping").fit(w)
    repeated = p.loc[p.index.repeat(w[p.game_code])]
    assert weighted.to_dict() == pytest.approx(repeated.groupby("grouping").epa.mean().to_dict())


def toy():
    return pd.DataFrame({"grouping": ["11", "11", "12", "12", "13"], "value": [1.0, 3.0, 10.0, 20.0, 7.0],
                         "dropback": [1, 0, 1, 1, 0], "game_code": [0, 1, 1, 2, 3]})


def test_group_mean_among_a_subset():
    p = toy()
    mean = GroupMean(p, "value", "grouping", among=p.dropback == 1).fit()
    assert mean.to_dict() == {"11": 1.0, "12": 15.0}


def test_group_mean_is_nan_for_a_group_missing_from_the_draw():
    mean = GroupMean(toy(), "value", "grouping").fit(np.array([1, 1, 1, 0]))
    assert mean["12"] == 15.0
    assert np.isnan(mean["13"])


def test_non_contiguous_seasons_as_groups():
    p = pd.DataFrame({"season": [2016, 2016, 2019, 2024], "value": [1.0, 2.0, 4.0, 8.0], "game_code": [0, 1, 2, 3]})
    assert GroupMean(p, "value", "season").fit().to_dict() == {2016: 1.5, 2019: 4.0, 2024: 8.0}


def test_runner_is_reproducible_and_fits_every_estimator_on_the_same_draw():
    p = plays(2_000, seed=7)
    estimators = {"a": GroupMean(p, "epa", "grouping"), "b": GroupMean(p, "epa", "grouping"),
                  "design": Design(p, ["11", "12"])}
    first, again = run(estimators, 2_000, resamples=20, seed=1), run(estimators, 2_000, resamples=20, seed=1)
    for name in estimators:
        pd.testing.assert_frame_equal(first[name], again[name])
    pd.testing.assert_frame_equal(first["a"], first["b"])
    assert (first["a"].low < first["a"].estimate).all() and (first["a"].estimate < first["a"].high).all()
    assert list(first["design"].index) == ["12"]


class Fixed:
    def fit(self, game_weights):
        return pd.Series({"x": 2.0})


def test_an_estimator_that_ignores_weights_has_a_collapsed_interval():
    out = run({"fixed": Fixed()}, 10, resamples=5, seed=0)["fixed"]
    assert out.loc["x"].to_dict() == {"estimate": 2.0, "low": 2.0, "high": 2.0}


def test_a_group_missing_from_some_draws_is_ignored_by_the_interval():
    out = run({"mean": GroupMean(toy(), "value", "grouping")}, 4, resamples=50, seed=0)["mean"]
    assert out.loc["13"].to_dict() == {"estimate": 7.0, "low": 7.0, "high": 7.0}


def test_preparing_plays_keeps_the_win_probability_window_edges():
    p = pd.DataFrame({"grouping": "11", "wp": [0.09, 0.10, 0.50, 0.90, 0.91, 0.50], "epa": [0, 0, 0, 0, 0, np.nan]})
    kept, groupings = prepare_plays(p)
    assert list(kept.wp) == [0.10, 0.50, 0.90]
    assert groupings == ["11", "Other"]
    assert len(prepare_plays(p, window=(0.0, 1.0))[0]) == 5


def test_preparing_plays_pools_groupings_under_the_share_into_other():
    p = pd.DataFrame({"grouping": ["11"] * 197 + ["12"] + ["13"] * 2, "wp": 0.5, "epa": 0.0})  # 12 is 0.5%
    kept, groupings = prepare_plays(p, min_share=0.01)
    assert groupings == ["11", "13", "Other"]
    assert kept.grouping.value_counts().to_dict() == {"11": 197, "13": 2, "Other": 1}
    assert prepare_plays(p)[1] == ["11", "13", "12", "Other"]  # at 0.5% it keeps its own term


def test_preparing_plays_in_a_situation_pools_groupings_under_the_play_count():
    p = pd.DataFrame({"grouping": ["11"] * 60 + ["22 (6 OL)"] * 50 + ["12"] * 49, "wp": 0.5, "epa": 0.0})
    kept, groupings = prepare_plays(p, min_share=0, min_plays=50)
    assert groupings == ["11", "22 (6 OL)", "Other"]
    assert (kept.grouping == "Other").sum() == 49
