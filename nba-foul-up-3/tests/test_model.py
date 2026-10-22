import numpy as np

from foulup3.model import MAX_MARGIN, after, solve, terminal

CLASSES = [
    "down 4+",
    "down 3, defend",
    "down 3, foul",
    "down 2",
    "down 1",
    "tied",
    "up 1",
    "up 2",
    "up 3",
    "up 4+",
]
H = 2.0  # bandwidth for the toy end games


def segment(delta, nxt, dt):
    return dict(
        t0=np.array([5.0]), delta=np.array([delta]), next=np.array([nxt]), dt=np.array([dt])
    )


def test_terminal():
    assert list(terminal(np.array([-1, 0, 2]))) == [0.0, 0.5, 1.0]


def test_clock_always_runs_out_so_the_buzzer_score_stands():
    result = solve({c: segment(0, "none", 1) for c in CLASSES}, H)
    margins = np.arange(-MAX_MARGIN, MAX_MARGIN + 1)
    assert np.allclose(result["value"], terminal(margins)[:, None])


def test_foul_beats_defending_against_a_team_that_never_misses_a_three():
    arrays = {c: segment(0, "none", 1) for c in CLASSES}
    arrays["down 3, defend"] = segment(3, "none", 1)  # ties it, then runs out the clock
    arrays["down 3, foul"] = segment(2, "other", 1)  # makes both, leader gets it up 1
    result = solve(arrays, H)
    assert np.allclose(result["defend"][1:], 0.5)
    assert np.allclose(result["foul"][1:], 1.0)


def test_borrowed_clock_ran_out_segment_passes_time_instead_of_ending_the_game():
    value = np.zeros((2 * MAX_MARGIN + 1, 20))
    value[1 + MAX_MARGIN, 7] = 0.42  # up 1 with the ball at grid point 7
    a = segment(0, "none", 3)  # ran out the clock from 0.3 s; borrowed at grid point 10
    assert after(value, a, 1, 10)[0] == 0.42


def test_borrowed_segment_longer_than_the_time_left_ends_at_the_buzzer():
    value = np.full((2 * MAX_MARGIN + 1, 20), 0.9)
    a = segment(3, "other", 50)  # a tying three that took 5 s, borrowed with 1 s left
    assert after(value, a, -3, 10)[0] == 0.5


def test_borrowed_buzzer_miss_is_rebounded_at_the_usual_share():
    value = np.zeros((2 * MAX_MARGIN + 1, 20))
    value[-3 + MAX_MARGIN, 7] = 0.2  # down 3 with the ball at grid point 7: offense rebound
    value[3 + MAX_MARGIN, 7] = 0.9  # up 3 with the ball: the other team rebounds
    a = dict(segment(0, "none", 3), buzzer=np.array([True]), oreb_share=0.25)
    assert np.isclose(after(value, a, -3, 10)[0], 0.25 * 0.2 + 0.75 * (1 - 0.9))
