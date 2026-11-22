import numpy as np
import pandas as pd

from foulup3.endgame import EndGame, down3_possessions

H = 2.0  # bandwidth for the toy end games
TIMES = np.arange(1, 25)  # a possession starting at every second


def rows(game_id, d0, delta, nxt, action="defend", timeouts=2):
    """One possession at margin d0 starting at every second, each taking a tenth of a second."""
    return pd.DataFrame(dict(game_id=game_id, d0=d0, t0=TIMES.astype(float), t1=TIMES - 0.1,
                             delta=delta, next=nxt, action=action, result="made",
                             label=f"{delta} {nxt}", timeouts=timeouts))  # fmt: skip


def toy(games=3, foul=(2, "other"), defend=(3, "none"), timeouts=(0, 2), vary=None):
    """Possessions where every margin but down 3 runs out the clock. Down 3, `foul` and `defend`
    are the trailing team's (points, who has the ball next) under each Choice, for each of its
    timeout counts. `vary(game)` can replace `defend` game by game."""
    out = []
    for g in range(games):
        out += [rows(g, d0, 0, "none") for d0 in range(-15, 16) if d0 != -3]
        for t in timeouts:
            d = vary(g) if vary else defend
            out += [rows(g, -3, *foul, "foul", t), rows(g, -3, *d, "defend", t)]
    return pd.concat(out, ignore_index=True)


def test_foul_is_a_clear_call_against_a_team_that_always_ties_when_defended():
    calls = EndGame(toy(), H, resamples=2, seed=1).calls()
    assert np.allclose(calls.foul, 1.0) and np.allclose(calls.defend, 0.0)
    assert (calls.call == "foul").all() and calls.clear.all()


def test_calls_cover_every_tenth_from_1_to_20_seconds_for_each_timeout_state():
    calls = EndGame(toy(), H, resamples=0).calls()
    for state in ["has a timeout", "no timeout"]:
        seconds = calls[calls.timeouts == state].seconds
        assert np.allclose(seconds, np.arange(10, 201) / 10)


def test_without_resamples_there_is_no_interval_and_no_clear():
    calls = EndGame(toy(), H, resamples=0).calls()
    assert {"gap_low", "gap_high", "clear"}.isdisjoint(calls.columns)
    assert np.allclose(calls.foul_ot + calls.foul + calls.foul_loss, 1.0)


def test_identical_games_collapse_the_interval_so_clear_means_a_nonzero_gap():
    same = toy(defend=(2, "other"))  # defending gives the same result as fouling
    for possessions, clear in [(toy(), True), (same, False)]:
        calls = EndGame(possessions, H, resamples=2, seed=1).calls()
        assert (calls.gap_low == calls.gap).all() and (calls.gap_high == calls.gap).all()
        assert (calls.clear == (calls.gap != 0)).all() and (calls.clear == clear).all()


def test_the_trailing_teams_timeouts_pick_which_down_3_mix_decides_the_call():
    possessions = pd.concat([
        toy(timeouts=(1,)),  # with a timeout: ties when defended, so foul
        toy(foul=(3, "none"), defend=(0, "other"), timeouts=(0,)),  # without: ties when fouled
    ], ignore_index=True)  # fmt: skip
    calls = EndGame(possessions, H, resamples=0).calls()
    assert (calls[calls.timeouts == "has a timeout"].call == "foul").all()
    assert (calls[calls.timeouts == "no timeout"].call == "defend").all()


def test_the_same_seed_gives_the_same_intervals():
    def defend(game):  # some games the trailing team ties, others it does not
        return (3, "none") if game % 3 else (0, "other")

    possessions = toy(vary=defend)
    first, again, other = (EndGame(possessions, H, resamples=3, seed=s).calls() for s in [7, 7, 8])
    pd.testing.assert_frame_equal(first, again)
    assert not np.allclose(first.gap_low, other.gap_low)


def by_timeouts():
    """Down 3, a trailing team with a timeout never wins, and one without always does (it scores
    4 and runs out the clock under either Choice)."""
    return pd.concat([
        toy(timeouts=(1,)),
        toy(foul=(4, "none"), defend=(4, "none"), timeouts=(0,)),
    ], ignore_index=True)  # fmt: skip


def test_down_3_the_trailing_teams_timeouts_pick_the_state_the_lookup_reads():
    end = EndGame(by_timeouts(), H, resamples=0)
    calls = end.calls().set_index(["timeouts", "seconds"])
    # the trailing team's chance down 3 is the team up 3's chance to lose under the Call
    has, none = (calls.loc[(s, 5.0)] for s in ["has a timeout", "no timeout"])
    has_p, none_p = (r[f"{r.call}_loss"] for r in [has, none])
    assert np.allclose([has_p, none_p], [0, 1])
    p = end.win_chance(margin=[-3, -3], seconds=[5.0, 5.0], timeouts=[2, 0], opp_timeouts=[0, 2])
    assert np.allclose(p, [has_p, none_p])


def test_at_a_tie_the_lookup_averages_both_timeout_states():
    possessions = by_timeouts()
    tied = possessions.d0 == 0
    # tied, the team with the ball always hits a three, leaving the other team down 3 with it
    possessions.loc[tied, ["delta", "next"]] = [3, "other"]
    end = EndGame(possessions, H, resamples=0)
    # the team up 3 then always wins with a timeout against it and never wins without one
    p = end.win_chance(margin=[0, 0], seconds=[5.0, 5.0], timeouts=[2, 0], opp_timeouts=[0, 2])
    assert np.allclose(p, 0.5)


def test_with_a_choice_the_lookup_gives_the_team_up_3s_chance_under_it():
    end = EndGame(toy(), H, resamples=0)  # fouling wins, defending ties
    p = end.win_chance(margin=-3, seconds=[5.0, 5.0], timeouts=1, opp_timeouts=2,
                       choice=["foul", "defend"])  # fmt: skip
    assert np.allclose(p, [1, 0])


def test_each_choices_outcome_mix_chances_sum_to_1():
    def defend(game):  # two games in three the trailing team ties, in the other it scores nothing
        return (3, "none") if game % 3 else (0, "other")

    end = EndGame(toy(vary=defend), H, resamples=0)
    for choice in ["foul", "defend"]:
        mix = end.outcome_mix(10.0, choice, "has a timeout")
        assert np.isclose(mix.chance.sum(), 1)
    mix = end.outcome_mix(10.0, "defend", "has a timeout").set_index("outcome")
    assert np.allclose(mix.chance, [1 / 3, 2 / 3])  # "0 other", then "3 none"
    assert np.allclose(mix.leader_wins_after, [1, 0]) and np.allclose(mix.seconds, 0.1)


def test_down_3_possessions_carry_the_choice_timeouts_and_buzzer_win_ending():
    columns = ["game_id", "period", "off", "d0", "delta", "how", "t0", "action", "timeouts"]
    possessions = pd.DataFrame([
        (1, 4, "A", -3, 3, "change", 8.0, "defend", 1),  # A ties: overtime
        (1, 4, "B", 0, 0, "change", 2.0, "defend", 0),
        (2, 4, "A", -3, 0, "change", 12.0, "foul to give", 2),  # not a Choice
        (2, 4, "A", -3, 0, "oreb", 9.0, "defend", 2),  # A kept the ball, not got it
        (2, 4, "A", -3, 2, "change", 6.0, "foul", 0),  # A scores 2, B runs out the clock
        (2, 4, "B", 1, 0, "change", 3.0, "defend", 2),
    ], columns=columns)  # fmt: skip
    d = down3_possessions(possessions)
    assert list(d.action) == ["defend", "foul"] and list(d.timeouts) == [1, 0]
    assert list(d.ending) == ["overtime", "win"] and list(d.leader_won) == [False, True]
