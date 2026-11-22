import numpy as np
import pandas as pd

from foulup3.endgame import EndGame, choose_bandwidth, down3_possessions

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


NARROW = 0.01  # with 30 games, a state draws only on possessions started within half a second


def mixes(outcome, games=30):
    """Possessions at every margin and second, where `outcome(d0, t0)` gives (points, who has the
    ball next, seconds used, result), or None to run out the clock. Down 3, both Choices and both
    timeout counts get the same mix."""
    out = []
    for g in range(games):
        for d0 in range(-15, 16):
            for t0 in TIMES:
                delta, nxt, dt, result = outcome(d0, t0) or (0, "none", 0.1, "made")
                row = dict(game_id=g, d0=d0, t0=float(t0), t1=t0 - dt, delta=delta, next=nxt,
                           result=result, label=f"{delta} {nxt}")  # fmt: skip
                choices = [("foul", 0), ("foul", 1), ("defend", 0), ("defend", 1)]
                for action, timeouts in choices if d0 == -3 else [("defend", 1)]:
                    out.append(dict(row, action=action, timeouts=timeouts))
    return pd.DataFrame(out)


def test_a_possession_seen_running_out_the_clock_with_time_left_just_uses_its_seconds():
    def outcome(d0, t0):
        if d0 == 1:  # up 1, the team holds the ball 2 s from 5 seconds up, and turns it over below
            return (0, "none", 2.0, "made") if t0 >= 5 else (0, "other", 0.1, "made")
        if d0 == -1:  # down 1, the team scores 2 at the buzzer
            return (2, "none", 24.0, "made")

    end = EndGame(mixes(outcome), NARROW, resamples=0)
    # from 6 seconds the team up 1 holds to 4, turns it over and loses at the buzzer
    assert end.win_chance(1, 6.0, 2, 2) == 0


def test_a_possession_longer_than_the_time_left_ends_at_the_buzzer():
    def outcome(d0, t0):
        if d0 == -3:  # down 3, the trailing team ties with a three that takes 5 s
            return (3, "other", 5.0, "made")
        if d0 == 0:  # tied, the team with the ball scores 2 at the buzzer
            return (2, "none", 24.0, "made")

    calls = EndGame(mixes(outcome), NARROW, resamples=0).calls()
    late = calls.seconds > 5.05  # the team up 3 gets the ball back tied with time to score
    assert np.allclose(calls.foul_ot[~late], 1) and np.allclose(calls.foul[late], 1)


def test_a_miss_at_the_buzzer_seen_with_time_left_is_rebounded_at_the_usual_share():
    def outcome(d0, t0):
        if d0 == -1:  # down 1, a miss that takes 2 s from 5 seconds up, 2 at the buzzer below
            return (0, "none", 2.0, "miss") if t0 >= 5 else (2, "none", 24.0, "made")
        if d0 == 15:  # never reached: sets the offensive rebound share to 1 in 4
            return (0, "none", 0.1, "oreb" if t0 % 4 == 0 else "dreb")

    end = EndGame(mixes(outcome), NARROW, resamples=0)
    # from 6 seconds the team down 1 misses, and wins only when it gets the rebound at 4
    assert np.isclose(end.win_chance(-1, 6.0, 2, 2), 0.25)


def test_the_end_game_matches_a_simulation_of_the_same_game():
    rng = np.random.default_rng(0)

    def mix(k=40):  # possessions with random starts, points, next team and tenths of a second
        return dict(t0=rng.uniform(0, 24, k), delta=rng.choice([0, 0, 1, 2, 3], k),
                    next=rng.choice(["other", "other", "same", "none"], k),
                    dt=rng.integers(5, 60, k))  # fmt: skip

    behind, ahead = mix(), mix()  # every margin past 9 shares one mix
    by_margin = {m: behind if m < -9 else ahead if m > 9 else mix() for m in range(-15, 16)}
    tables = []
    for m, a in by_margin.items():
        t = pd.DataFrame(dict(a, game_id=0, d0=m, t1=a["t0"] - a["dt"] / 10, result="made"))
        t = t.assign(label=t.delta.astype(str) + " " + t.next).drop(columns="dt")
        # down 3, both Choices and both timeout counts share the mix, so neither can matter
        for action, timeouts in [("foul", 0), ("foul", 1), ("defend", 0), ("defend", 1)]:
            tables.append(t.assign(action=action, timeouts=timeouts))
            if m != -3:
                break
    end = EndGame(pd.concat(tables, ignore_index=True), 1e9, resamples=0)  # all weigh the same

    def simulate(margin, seconds, n=20_000):
        """Play the game n times, drawing each possession at random: the share won at the buzzer
        by the team that starts with the ball."""
        wins = 0
        for _ in range(n):
            m, j, sign = margin, int(round(seconds * 10)), 1  # sign: +1 while it has the ball
            while j > 0:
                a = by_margin[m]
                k = rng.integers(len(a["delta"]))
                m, j = int(np.clip(m + a["delta"][k], -15, 15)), j - int(a["dt"][k])
                if j > 0 and a["next"][k] == "other":
                    m, sign = -m, -sign
            wins += sign * m > 0
        return wins / n

    for margin, seconds in [(-3, 6.0), (0, 10.0), (2, 15.0)]:
        p = end.win_chance(margin, seconds, 1, 1)
        assert abs(simulate(margin, seconds) - p) < 0.012
