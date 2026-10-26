import numpy as np

from foulup3.model import MAX_MARGIN, STEP, after, solve, state_class, terminal

CLASSES = sorted({state_class(m) for m in range(-MAX_MARGIN, MAX_MARGIN + 1)} | {"down 3, foul"})
H = 2.0  # bandwidth for the toy end games


def segment(delta, nxt, dt):
    return dict(
        t0=np.array([5.0]), delta=np.array([delta]), next=np.array([nxt]), dt=np.array([dt])
    )


def test_terminal_tie_is_neither_a_win_nor_a_loss():
    win, lose = terminal(np.array([-1, 0, 2]))
    assert list(win) == [0.0, 0.0, 1.0] and list(lose) == [1.0, 0.0, 0.0]


def test_clock_always_runs_out_so_the_buzzer_score_stands():
    result = solve({c: segment(0, "none", 1) for c in CLASSES}, H)
    margins = np.arange(-MAX_MARGIN, MAX_MARGIN + 1)
    assert np.allclose(result["value"], terminal(margins)[:, :, None])


def test_foul_beats_defending_against_a_team_that_never_misses_a_three():
    arrays = {c: segment(0, "none", 1) for c in CLASSES}
    arrays["down 3, defend"] = segment(3, "none", 1)  # ties it, then runs out the clock
    arrays["down 3, foul"] = segment(2, "other", 1)  # makes both, leader gets it up 1
    result = solve(arrays, H)
    assert np.allclose(result["defend"][1:], 0.0) and np.allclose(result["defend_loss"][1:], 0.0)
    assert np.allclose(result["foul"][1:], 1.0)


def test_borrowed_clock_ran_out_segment_passes_time_instead_of_ending_the_game():
    value = np.zeros((2, 2 * MAX_MARGIN + 1, 20))
    value[:, 1 + MAX_MARGIN, 7] = [0.42, 0.3]  # up 1 with the ball at grid point 7
    a = segment(0, "none", 3)  # ran out the clock from 0.3 s; borrowed at grid point 10
    assert list(after(value, a, 1, 10)[:, 0]) == [0.42, 0.3]


def test_borrowed_segment_longer_than_the_time_left_ends_at_the_buzzer():
    value = np.full((2, 2 * MAX_MARGIN + 1, 20), 0.45)
    a = segment(3, "other", 50)  # a tying three that took 5 s, borrowed with 1 s left
    assert list(after(value, a, -3, 10)[:, 0]) == [0.0, 0.0]  # overtime


def test_borrowed_buzzer_miss_is_rebounded_at_the_usual_share():
    value = np.zeros((2, 2 * MAX_MARGIN + 1, 20))
    value[:, -3 + MAX_MARGIN, 7] = [0.2, 0.7]  # down 3 with the ball: offense rebound
    value[:, 3 + MAX_MARGIN, 7] = [0.9, 0.05]  # up 3 with the ball: the other team rebounds
    a = dict(segment(0, "none", 3), buzzer=np.array([True]), oreb_share=0.25)
    win, lose = after(value, a, -3, 10)[:, 0]
    assert np.isclose(win, 0.25 * 0.2 + 0.75 * 0.05) and np.isclose(lose, 0.25 * 0.7 + 0.75 * 0.9)


def test_margins_are_kept_apart_until_out_of_reach():
    assert state_class(-4) == "down 4" and state_class(7) == "up 7"
    assert state_class(-12) == state_class(-10) == "down 10+"
    assert state_class(-3, "foul") == "down 3, foul"


def simulate(arrays, margin, i, rng, n):
    """Play the toy end game n times by drawing segments at random: the reference for solve().
    Returns the shares won and lost at the buzzer."""
    wins = losses = 0.0
    for _ in range(n):
        m, j, sign = margin, i, 1  # sign: +1 while the starting team has the ball
        while True:
            a = arrays[state_class(m)]
            k = rng.integers(len(a["delta"]))
            m = int(np.clip(m + a["delta"][k], -MAX_MARGIN, MAX_MARGIN))
            j = max(j - int(a["dt"][k]), 0)
            if j == 0:
                win, lose = terminal(np.array([sign * m]))[:, 0]
                wins, losses = wins + win, losses + lose
                break
            if a["next"][k] == "other":
                m, sign = -m, -sign
    return np.array([wins, losses]) / n


def test_solver_matches_a_simulation_of_the_same_game():
    rng = np.random.default_rng(0)
    arrays = {}
    for c in CLASSES:
        k = 40
        arrays[c] = dict(t0=rng.uniform(0, 24, k), delta=rng.choice([0, 0, 1, 2, 3], k),
                         next=rng.choice(["other", "other", "same", "none"], k),
                         dt=rng.integers(5, 60, k))  # fmt: skip
    arrays["down 3, foul"] = arrays["down 3, defend"]  # same mix, so the choice can't matter
    result = solve(arrays, 1e9)  # a window this wide weighs every segment equally
    for margin, seconds in [(-3, 6.0), (0, 10.0), (2, 15.0)]:
        i = int(round(seconds / STEP))
        expected = result["value"][:, margin + MAX_MARGIN, i]
        assert np.abs(simulate(arrays, margin, i, rng, 20_000) - expected).max() < 0.012


def test_timeout_split_keeps_down_3_segments_apart_and_shares_the_rest():
    import pandas as pd

    from foulup3.model import by_timeouts

    segs = pd.DataFrame(dict(d0=[-3, -3, 2, -3], timeouts=[0, 2, 0, None]))
    split = by_timeouts(segs)
    assert list(split["has a timeout"].index) == [1, 2]
    assert list(split["no timeout"].index) == [0, 2]
