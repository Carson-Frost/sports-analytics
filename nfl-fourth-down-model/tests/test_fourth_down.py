from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fourth_down.data import add_state
from fourth_down.decide import N_QUANTILES, TD_POINTS, choice, conversion_x
from fourth_down.grading import Grader

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


def their_wp(score, yardline, home=1):
    """Our win probability when they have a first down at `yardline` (their distance to goal)."""
    theirs = pd.DataFrame({"score_differential": [-score], "yardline_100": [yardline], "home": [1 - home]})
    return 1 - ToyWP().predict_proba(theirs)[0, 1]


def test_wp_matches_nflverse():
    """The 12 inputs are built in nflfastR's order: the model reproduces nflverse's vegas_wp."""
    from fourth_down.wp import WinProbability

    sample = pd.read_csv(HERE / "data" / "wp_sample_2025.csv")
    p = WinProbability(smooth=0).predict_proba(sample)[:, 1]
    assert np.abs(p - sample.vegas_wp).max() < 1e-6


def test_second_half_kickoff_and_home():
    plays = pd.DataFrame({
        "posteam": ["H", "A", "H", "A"], "home_team": "H", "home_opening_kickoff": 1,
        "game_seconds_remaining": [3000, 3000, 1000, 2000],
        "score_differential": 0, "spread_line": 2.5,
    })
    out = add_state(plays)
    assert out.receive_2h_ko.tolist() == [0, 1, 0, 1]
    assert out.home.tolist() == [1, 0, 1, 0]
    assert out.posteam_spread.tolist() == [2.5, -2.5, 2.5, -2.5]


def graded(sub, states, resamples=None):
    """Grade `states` with the toy model; with no Resamples given, every Resample is the point fit."""
    return Grader(ToyWP(), sub, resamples or [sub] * 3).grade(states)


def test_sure_conversion_from_the_one_is_a_touchdown():
    wp = graded(submodels(p_convert=1.0), fourth_down(yardline=1, togo=1))
    assert wp.go.iloc[0] == pytest.approx(their_wp(TD_POINTS, 70))


def test_missed_field_goal_spot():
    close = graded(submodels(p_fg=0.0), fourth_down(yardline=5, togo=5))
    far = graded(submodels(p_fg=0.0), fourth_down(yardline=30, togo=5))
    assert close.fg.iloc[0] == pytest.approx(their_wp(0, 80))
    assert far.fg.iloc[0] == pytest.approx(their_wp(0, 100 - 38))


def test_punt_and_failed_conversion():
    wp = graded(submodels(p_convert=0.0), fourth_down(yardline=60, togo=4))
    assert wp.punt.iloc[0] == pytest.approx(their_wp(0, 80))
    assert wp.go.iloc[0] == pytest.approx(their_wp(0, 40))


def test_home_flips_when_they_get_the_ball():
    """A home team that turns it over hands the ball to the away team, and the reverse."""
    at_home = graded(submodels(p_convert=0.0), fourth_down(60, 4, home=1)).go.iloc[0]
    away = graded(submodels(p_convert=0.0), fourth_down(60, 4, home=0)).go.iloc[0]
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


def test_cost_of_the_coachs_choice():
    states = pd.concat([fourth_down(60, 4)] * 2).assign(coach=choice(pd.Series(["punt", "pass"])).to_numpy())
    g, costs = Grader(ToyWP(), go_over_punt(0.012), [go_over_punt(0.012)] * 3).grade(states, resample_costs=True)
    assert g.cost.tolist() == pytest.approx([0.012, 0.0])
    assert costs == pytest.approx(np.array([[0.012, 0.0]] * 3))


def test_no_field_goal_beyond_the_longest_attempt():
    far = graded(submodels(p_fg=1.0), fourth_down(yardline=60, togo=10))  # a 78-yard kick
    assert far.fg.iloc[0] == pytest.approx(their_wp(0, 100 - max(60 + 8, 20)))


def test_smoothing_stays_inside_the_half():
    from fourth_down.wp import WinProbability

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


def toy_wp(score, yardline, home=1):
    """Our win probability with a first down at `yardline`."""
    return 1 - their_wp(-score, yardline, home=1 - home)


def go_over_punt(gap):
    """Submodels where going for it at 4th and 4 from our 40 beats a touchback punt by `gap`."""
    convert, fail, punt = toy_wp(0, 55), their_wp(0, 40), their_wp(0, 80)
    return submodels(p_convert=(punt + gap - fail) / (convert - fail), p_fg=0.0)


@pytest.mark.parametrize("gap, clear", [(0.012, True), (0.008, False)])
def test_clear_when_the_gap_is_at_least_a_point_in_every_resample(gap, clear):
    g = graded(go_over_punt(gap), fourth_down(60, 4)).iloc[0]
    assert g.call == "go"
    assert g.gap == pytest.approx(gap)
    assert g.gap_low == pytest.approx(gap) and g.gap_high == pytest.approx(gap)
    assert g.clear == clear


def fg_over_punt(gap):
    """Submodels where a field goal at 4th and 10 from their 30 beats a touchback punt by `gap`, and a
    failed conversion (every try fails) is clearly worse than both."""
    made, missed, punt = their_wp(3, 70), their_wp(0, 62), their_wp(0, 80)
    return submodels(p_convert=0.0, p_fg=(punt + gap - missed) / (made - missed))


def test_a_field_goal_narrowly_over_a_punt_is_a_toss_up_when_going_is_clearly_worse():
    g = graded(fg_over_punt(0.005), fourth_down(30, 10)).iloc[0]
    assert g.go < g.punt - 0.05
    assert g.call == "fg"
    assert g.gap == pytest.approx(0.005)
    assert not g.clear


def test_go_edge_clear_for_go():
    g = graded(go_over_punt(0.012), fourth_down(60, 4)).iloc[0]
    assert g.go_edge == pytest.approx(0.012)
    assert g.edge_low == pytest.approx(0.012) and g.edge_high == pytest.approx(0.012)
    assert g.clear_go and not g.clear_kick


def test_go_edge_clear_for_a_kick():
    g = graded(fg_over_punt(0.005), fourth_down(30, 10)).iloc[0]
    assert g.go_edge == pytest.approx(g.go - g.fg)
    assert g.edge_high <= -0.05
    assert g.clear_kick and not g.clear_go


def test_an_unknown_home_side_averages_win_probabilities_before_the_gap():
    """At home going wins by 0.1 points, away punting wins by 1.9: the averaged punt wins by 0.9 points,
    a Toss-up, where averaging the two gaps would give a Clear 1.0."""
    sub = go_over_punt(0.001)
    sides = pd.concat([graded(sub, fourth_down(60, 4, home=h)) for h in (1, 0)])
    unknown = graded(sub, fourth_down(60, 4, home=np.nan)).iloc[0]
    assert unknown[["go", "fg", "punt"]].tolist() == pytest.approx(sides[["go", "fg", "punt"]].mean().tolist())
    assert unknown.call == "punt"
    assert unknown.gap == pytest.approx(sides.punt.mean() - sides.go.mean())
    assert unknown.gap == pytest.approx(0.0091, abs=1e-4)
    assert sides.gap.mean() >= 0.01
    assert not unknown.clear
