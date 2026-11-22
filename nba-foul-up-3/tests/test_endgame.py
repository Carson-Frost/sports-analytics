import numpy as np
import pandas as pd

from foulup3.endgame import EndGame, choose_bandwidth

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


def test_the_bandwidth_choice_picks_the_lowest_held_out_log_loss():
    # tied games end one way before 12 seconds and another after, so a narrow window fits
    def season(games):
        out = []
        for g in range(games):
            for t0 in TIMES:
                label = "early" if t0 <= 12 else "late"
                out.append(dict(game_id=g, d0=0, t0=float(t0), t1=t0 - 0.1, delta=0, next="none",
                                action="defend", result="made", label=label))  # fmt: skip
        return pd.DataFrame(out)

    h, scores = choose_bandwidth(season(40), season(10), [1.0, 0.05, 0.5])
    assert h == 0.05
    assert list(scores.columns) == ["bandwidth", "possessions", "log_loss"]
    assert scores.bandwidth.tolist() == [1.0, 0.05, 0.5]
    assert (scores.possessions == 10 * len(TIMES)).all()
    assert scores.log_loss[scores.bandwidth == h].item() == scores.log_loss.min()
