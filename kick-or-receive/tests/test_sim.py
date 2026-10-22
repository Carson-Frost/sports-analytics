import random
from collections import defaultdict

import numpy as np

from overtime import sim
from overtime.data import rule_for


def stub(snap=(sim.PLAY, 10, 0, 30), fourth="go", kickoff=(True, 70, 0, 0), fg=(5.0, -0.1)):
    """A model where every snap is the same and every kickoff lands at the 30."""
    return sim.Model(snaps={}, snaps_pooled=defaultdict(lambda: [snap]),
                     fourth={(d, f): [fourth] for d in range(7) for f in range(10)},
                     punts=[[(True, 80, 0, 5)]] * 20, kickoffs=[kickoff], hurry=[5],
                     fg=fg, xp=1.0, two=1.0)


def scripted(*results):
    """A drive function that plays back fixed drive results and records the modes it was asked."""
    queue, modes = list(results), []

    def drive_fn(m, yl, mode, clock, rng):
        modes.append(mode)
        result = queue.pop(0) if queue else "none"
        return result, 70, 60 if result != "clock" else 0

    return drive_fn, modes


def play(rule, *results, clock=None, go_for_two=False, model=None):
    fn, modes = scripted(*results)
    outcome, first = sim.overtime(model or stub(), rule, clock, random.Random(0), go_for_two, fn)
    return outcome, first, modes


def test_sudden_death_first_field_goal_wins():
    assert play("sudden", "fg")[:2] == (1, "fg")


def test_modified_field_goal_gives_the_other_team_a_chance():
    assert play("modified", "fg", "td")[0] == -1
    assert play("modified", "fg", "none")[0] == 1
    outcome, _, modes = play("modified", "fg", "fg", "none", "fg")
    assert outcome == -1 and modes == ["normal", "need_fg", "fg_wins", "fg_wins"]


def test_modified_first_possession_touchdown_wins():
    assert play("modified", "td")[0] == 1


def test_both_possess_touchdown_does_not_end_it():
    outcome, first, modes = play("both", "td", "td", "fg")
    assert (outcome, first) == (1, "td")
    assert modes == ["normal", "need_td", "fg_wins"]


def test_both_possess_go_for_two_after_matching_touchdown():
    assert play("both", "td", "td", go_for_two=True)[0] == -1


def test_both_possess_safety_on_first_possession_ends_it():
    assert play("both", "safety")[0] == -1


def test_clock_runs_out():
    assert play("modified", "fg", "clock", clock=600)[0] == 1
    assert play("both", "none", "none", "clock", clock=600)[0] == 0


def test_drive_scores_and_uses_the_clock():
    m = stub()
    assert sim.drive(m, 75, "normal", None, random.Random(0)) == ("td", 0, 240)
    assert sim.drive(m, 75, "normal", 300, random.Random(0))[0] == "td"  # hurries at the end
    m.hurry = [30]
    assert sim.drive(m, 75, "normal", 100, random.Random(0))[0] == "clock"


def test_drive_turnover_on_downs():
    m = stub(snap=(sim.PLAY, 0, 0, 30))
    assert sim.drive(m, 60, "need_td", None, random.Random(0)) == ("none", 40, 120)


def test_drive_punts_and_kicks():
    m = stub(snap=(sim.PLAY, 0, 0, 30), fourth="punt", fg=(50.0, 0.0))  # kicks never miss
    assert sim.drive(m, 60, "normal", None, random.Random(0))[:2] == ("none", 80)
    assert sim.drive(m, 10, "need_fg", None, random.Random(0)) == ("fg", 0, 95)


def test_penalty_short_of_the_line_repeats_the_down():
    m = stub(snap=(sim.PLAY, 0, sim.REPEAT_DOWN, 30))  # never reaches 4th down
    assert sim.drive(m, 60, "need_td", 300, random.Random(0))[0] == "clock"


def test_kick_on_any_down_in_the_last_seconds():
    m = stub(fg=(50.0, 0.0))
    assert sim.drive(m, 20, "fg_wins", 15, random.Random(0))[0] == "fg"
    assert sim.drive(m, 20, "normal", 15, random.Random(0))[0] == "td"  # keeps playing


def test_game_ends_when_a_recovered_kickoff_uses_up_the_last_chance(monkeypatch):
    kicks = iter([(0, 70, None, 0), (0, 50, None, 0)])  # then team 0 recovers its own kickoff
    monkeypatch.setattr(sim, "kickoff", lambda m, receiving, rng: next(kicks))
    outcome, _, modes = play("both", "td", "td")
    assert outcome == 1 and modes == ["normal"]


def test_game_ends_after_a_muffed_punt_on_the_last_chance():
    outcome, _, modes = play("both", "fg", "keep", "td")
    assert outcome == 1 and modes == ["normal", "need_fg"]


def test_kicking_team_recovery_counts_as_the_receivers_chance():
    muffed = stub(kickoff=(False, 50, 0, 0))  # the kicking team recovers the opening kickoff
    assert play("both", "fg", model=muffed)[0] == -1
    assert play("both", "keep", "fg")[0] == 1  # muffed punt: both teams have had the ball


def test_fit_logistic_recovers_parameters():
    rng = np.random.default_rng(0)
    x = rng.uniform(20, 60, 20_000)
    y = (rng.random(20_000) < 1 / (1 + np.exp(-(6 - 0.1 * x)))).astype(float)
    a, b = sim.fit_logistic(x, y)
    assert abs(a - 6) < 0.3 and abs(b + 0.1) < 0.01


def test_rule_history():
    assert rule_for(2005, "REG") == ("sudden", 900)
    assert rule_for(2011, "DIV") == ("modified", None)
    assert rule_for(2020, "REG") == ("modified", 600)
    assert rule_for(2022, "SB") == ("both", None)
    assert rule_for(2024, "REG") == ("modified", 600)
    assert rule_for(2025, "REG") == ("both", 600)
