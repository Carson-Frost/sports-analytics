from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fourth_down.data import add_state, time_scaled
from fourth_down.decide import N_QUANTILES, TD_POINTS, choice, choice_wp, conversion_x, first_down_wp, grade

HERE = Path(__file__).parent
CLOCK = {"convert": 30.0, "fail": 5.0, "fg_made": 5.0, "fg_missed": 5.0, "punt": 9.0}


class Const:
    """A fitted-model stand-in that always returns the same probability."""

    def __init__(self, p):
        self.p = p

    def predict_proba(self, x):
        return np.column_stack([np.full(len(x), 1 - self.p), np.full(len(x), self.p)])


class ToyWP:
    """Win probability that rises with the lead, field position and being at home."""

    def predict_proba(self, x):
        z = x.score_differential / 7 - (x.yardline_100 - 50) / 25 + 0.2 * x.home
        p = 1 / (1 + np.exp(-z))
        return np.column_stack([1 - p, p])


def submodels(p_convert=0.5, p_fg=0.5):
    return {
        "conversion": Const(p_convert),
        "gains": np.full((15, N_QUANTILES), 5.0),  # every conversion gains 5 yards
        "fg": Const(p_fg),
        "punt": (np.array([2.5, 97.5]), np.full((2, N_QUANTILES), 80.0)),  # every punt is a touchback
        "kickoff_start": {2025: 70.0},
        "fg_longest": 70.0,
        "clock": CLOCK,
    }


def fourth_down(yardline, togo, score=0, home=1):
    return pd.DataFrame({
        "score_differential": [score], "game_seconds_remaining": [2700], "half_seconds_remaining": [900],
        "yardline_100": [yardline], "down": [4], "ydstogo": [togo], "posteam_timeouts_remaining": [3],
        "defteam_timeouts_remaining": [3], "receive_2h_ko": [0], "posteam_spread": [0.0], "season": [2025],
        "home": [home],
    })


def their_wp(score, yardline, home=1, used=CLOCK["fail"]):
    """Our win probability when they have a first down at `yardline` (their distance to goal)."""
    return 1 - first_down_wp(ToyWP(), np.array([-score]), np.array([2700 - used]), np.array([900 - used]),
                             np.array([yardline]), np.array([1]), 3, 3, np.array([0.0]), np.array([1 - home]))[0]


def test_wp_matches_nflverse():
    """The 12 inputs are built in nflfastR's order: the model reproduces nflverse's vegas_wp."""
    from fourth_down.wp import MODEL, WinProbability

    if not MODEL.exists():
        pytest.skip("run scripts/get_wp_model.py first")
    sample = pd.read_csv(HERE / "data" / "wp_sample_2025.csv")
    p = WinProbability(smooth=0).predict_proba(sample)[:, 1]
    assert np.abs(p - sample.vegas_wp).max() < 1e-6


def test_second_half_kickoff_home_and_winner():
    plays = pd.DataFrame({
        "posteam": ["H", "A", "H", "A"], "home_team": "H", "home_opening_kickoff": 1,
        "game_seconds_remaining": [3000, 3000, 1000, 2000], "result": [3, 3, 3, 0],
        "score_differential": 0, "spread_line": 2.5,
    })
    out = add_state(plays)
    assert out.receive_2h_ko.tolist() == [0, 1, 0, 1]
    assert out.home.tolist() == [1, 0, 1, 0]
    assert out.posteam_won.tolist()[:3] == [1, 0, 1]
    assert np.isnan(out.posteam_won.iloc[3])
    assert out.posteam_spread.tolist() == [2.5, -2.5, 2.5, -2.5]


def test_time_scaling():
    spread, lead = time_scaled(7, 3, 1800)
    assert spread == pytest.approx(3 * np.exp(-2))
    assert lead == pytest.approx(7 * np.exp(2))


def test_sure_conversion_from_the_one_is_a_touchdown():
    wp = choice_wp(ToyWP(), submodels(p_convert=1.0), fourth_down(yardline=1, togo=1))
    assert wp.go.iloc[0] == pytest.approx(their_wp(TD_POINTS, 70, used=CLOCK["convert"]))


def test_missed_field_goal_spot():
    close = choice_wp(ToyWP(), submodels(p_fg=0.0), fourth_down(yardline=5, togo=5))
    far = choice_wp(ToyWP(), submodels(p_fg=0.0), fourth_down(yardline=30, togo=5))
    assert close.fg.iloc[0] == pytest.approx(their_wp(0, 80))
    assert far.fg.iloc[0] == pytest.approx(their_wp(0, 100 - 38))


def test_punt_and_failed_conversion():
    wp = choice_wp(ToyWP(), submodels(p_convert=0.0), fourth_down(yardline=60, togo=4))
    assert wp.punt.iloc[0] == pytest.approx(their_wp(0, 80, used=CLOCK["punt"]))
    assert wp.go.iloc[0] == pytest.approx(their_wp(0, 40))


def test_home_flips_when_they_get_the_ball():
    """A home team that turns it over hands the ball to the away team, and the reverse."""
    at_home = choice_wp(ToyWP(), submodels(p_convert=0.0), fourth_down(60, 4, home=1)).go.iloc[0]
    away = choice_wp(ToyWP(), submodels(p_convert=0.0), fourth_down(60, 4, home=0)).go.iloc[0]
    assert at_home == pytest.approx(their_wp(0, 40, home=1))
    assert at_home > away


def test_conversion_zones():
    x = conversion_x(np.array([1, 1, 1, 1]), np.array([50, 15, 8, 3]))
    assert x[:, 9:].tolist() == [[0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1]]


def test_resampled_games_stay_separate():
    import importlib.util

    spec = importlib.util.spec_from_file_location("grade", HERE.parent / "scripts" / "grade.py")
    grade_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grade_script)
    plays = pd.DataFrame({"game_id": ["a", "a", "b"], "x": [1, 2, 3]})
    out = grade_script.resample_games(plays, np.random.default_rng(0))
    assert all(out.groupby("game_id").size().isin([1, 2]))  # a game drawn twice is two games


def test_grade():
    wp = pd.DataFrame({"go": [0.6, 0.4], "fg": [0.3, 0.3], "punt": [0.5, 0.45]})
    g = grade(wp, choice(pd.Series(["punt", "punt"])))
    assert g.best.tolist() == ["go", "punt"]
    assert g.cost.tolist() == pytest.approx([0.1, 0.0])
    assert g.go_edge.tolist() == pytest.approx([0.1, -0.05])


def test_no_field_goal_beyond_the_longest_attempt():
    far = choice_wp(ToyWP(), submodels(p_fg=1.0), fourth_down(yardline=60, togo=10))  # a 78-yard kick
    assert far.fg.iloc[0] == pytest.approx(their_wp(0, 100 - max(60 + 8, 20), used=CLOCK["fg_missed"]))


def test_smoothing_stays_inside_the_half():
    from fourth_down.wp import MODEL, WinProbability

    if not MODEL.exists():
        pytest.skip("run scripts/get_wp_model.py first")
    s = pd.read_csv(HERE / "data" / "wp_sample_2025.csv").head(50)
    late = s.assign(half_seconds_remaining=10, game_seconds_remaining=1810)
    p = WinProbability(smooth=60).predict_proba(late)[:, 1]
    assert np.all((p > 0) & (p < 1))


def test_spot_parsing():
    import importlib.util

    spec = importlib.util.spec_from_file_location("call", HERE.parent / "scripts" / "call.py")
    call = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(call)
    assert [call.yardline_100(x) for x in ("own 35", "opp 20", "50", "Own 1")] == [65, 20, 50, 99]
