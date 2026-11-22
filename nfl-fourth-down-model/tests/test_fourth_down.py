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


def test_cost_of_the_coachs_choice():
    states = pd.concat([fourth_down(60, 4)] * 2).assign(coach=choice(pd.Series(["punt", "pass"])).to_numpy())
    g = graded(go_over_punt(0.012), states)
    assert g.cost.tolist() == pytest.approx([0.012, 0.0])


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


def synthetic_plays(games=12, seed=0):
    """A small season of plays with every kind of fourth-down outcome the submodels fit on."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(games):
        clock = 3600
        for i in range(75):
            clock -= 40
            kind = ["try", "try", "field_goal", "punt", "kickoff"][i % 5]
            togo = 1 + (i // 5) % 15
            yl = 40 + (i * 7) % 50
            row = {"game_id": f"g{g}", "season": 2025, "posteam": "A", "qtr": 1 if clock > 2700 else 3,
                   "game_seconds_remaining": clock, "down": 3 + i % 2, "ydstogo": togo, "yardline_100": yl,
                   "play_type": "run", "qb_kneel": 0, "first_down_rush": 0, "first_down_pass": 0,
                   "touchdown": 0, "td_team": None, "yards_gained": 0, "kick_distance": np.nan,
                   "field_goal_result": None, "punt_blocked": 0, "return_yards": 0.0, "touchback": 0,
                   "fixed_drive": i, "drive_start_transition": None}
            if kind == "try":
                made = rng.random() < 0.5
                row.update(first_down_rush=int(made), yards_gained=togo + rng.integers(0, 10) if made else 0)
            elif kind == "field_goal":
                distance = 20 + rng.integers(0, 40)
                row.update(play_type="field_goal", down=4, kick_distance=distance,
                           field_goal_result="made" if rng.random() < 0.8 - distance / 100 else "missed")
            elif kind == "punt":
                row.update(play_type="punt", down=4, yardline_100=62, kick_distance=40 + rng.integers(0, 15))
            else:
                row.update(down=1, drive_start_transition="KICKOFF", yardline_100=65 + rng.integers(0, 15))
            rows.append(row)
    return pd.DataFrame(rows)


def test_a_fit_with_the_same_seed_gives_the_same_grades():
    plays, states = synthetic_plays(), pd.concat([fourth_down(60, 4), fourth_down(30, 10)], ignore_index=True)
    first = Grader.fit(ToyWP(), plays, resamples=3, seed=1).grade(states)
    again = Grader.fit(ToyWP(), plays, resamples=3, seed=1).grade(states)
    other = Grader.fit(ToyWP(), plays, resamples=3, seed=2).grade(states)
    pd.testing.assert_frame_equal(first, again)
    assert not first.gap_low.equals(other.gap_low)


def test_a_game_drawn_twice_counts_twice_in_the_season_summary():
    """Two clear-go spots: the team went in game a and punted in game b. Every Resample draws a twice."""
    states = pd.concat([fourth_down(60, 4)] * 2, ignore_index=True).assign(
        game_id=["a", "b"], coach=["go", "punt"])
    sub = go_over_punt(0.012)
    draws = [pd.Series({"a": 2, "b": 1})] * 3
    graded, seasons = Grader(ToyWP(), sub, [sub] * 3, draws).grade_seasons(states)
    assert graded.cost.tolist() == pytest.approx([0.0, 0.012])
    s = seasons.set_index("season").loc[2025]
    assert (s.fourth_downs, s.clear_calls, s.clear_go_spots) == (2, 2, 2)
    assert s.clear_share == pytest.approx(100)
    assert s.go_rate == pytest.approx(50)
    assert s.go_rate_low == pytest.approx(200 / 3) and s.go_rate_high == pytest.approx(200 / 3)
    assert s.wins_lost == pytest.approx(0.012)
