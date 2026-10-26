"""Possession outcome probabilities from the game state, and the exact end game they imply.

The state is the margin for the team with the ball and the seconds left. For each state, the
chance of every segment outcome (points, who has the ball next, seconds used) is the observed mix
of segments from the same kind of state that started at about the same time. The end game is then
solved exactly, tenth of a second by tenth of a second, from the buzzer backwards.

Every result has three outcomes for the team with the ball: it wins at the buzzer, the game goes
to overtime, or it loses at the buzzer. A win means a win at the buzzer; overtime is not a win.
"""

import numpy as np
import pandas as pd

STEP = 0.1  # seconds per grid point
HORIZON = 24.0
MAX_MARGIN = 15  # margins beyond this are treated as this
MIN_SEGMENTS = 30  # widen the window until at least this many segments count
FLOOR = 0.5  # narrowest window, in seconds
CLOSE = 9  # every margin up to this gets its own outcome mix; wider ones share one


def state_class(margin: int, action: str = "defend") -> str:
    """Which outcome mix a state draws from. A team up 4 is fouled on most trips and a team up
    15 runs out the clock, so margins are kept apart until the game is out of reach."""
    if margin == -3:
        return "down 3, foul" if action == "foul" else "down 3, defend"
    if margin < -CLOSE:
        return f"down {CLOSE + 1}+"
    if margin > CLOSE:
        return f"up {CLOSE + 1}+"
    return "tied" if margin == 0 else f"{'up' if margin > 0 else 'down'} {abs(margin)}"


def buzzer_margin(segs: pd.DataFrame) -> pd.Series:
    """For each segment, the margin for its team with the ball when its period ended."""
    last = segs.groupby(["game_id", "period"]).tail(1)
    final = (last.d0 + last.delta).to_numpy()
    end = pd.DataFrame(dict(game_id=last.game_id, period=last.period, team=last.off, final=final))
    m = segs[["game_id", "period"]].merge(end, how="left").set_index(segs.index)
    return m.final.where(m.team == segs.off, -m.final)


TIMEOUT_STATES = ["has a timeout", "no timeout"]  # the trailing team's, at down 3


def by_timeouts(segs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The segments each timeout state is built from. Down-3 segments go to the state matching
    the trailing team's timeouts left; every other segment goes to both. Down 3, a team with a
    timeout can move the ball to half court, and one without has to go the length of the floor."""
    down3 = segs.d0 == -3
    return {"has a timeout": segs[~down3 | (segs.timeouts > 0)],
            "no timeout": segs[~down3 | (segs.timeouts == 0)]}  # fmt: skip


def decisions(segs: pd.DataFrame) -> pd.DataFrame:
    """Every time the trailing team got the ball down 3 with 24 seconds or less left, and how
    it ended for the team up 3: "win", "overtime" or "loss" at the buzzer."""
    d = segs[(segs.d0 == -3) & (segs.how == "change") & (segs.t0 > 0)]
    lead = -buzzer_margin(segs).loc[d.index]
    ending = np.where(lead > 0, "win", np.where(lead < 0, "loss", "overtime"))
    return d.assign(ending=ending, leader_won=lead > 0)


def outcome_arrays(segs: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """Per state class: each segment's start time, margin change, next possession, tenths of a
    second used and label, whether it ended in a miss at the buzzer, and the share of misses
    rebounded by the offense. Fouls with no free throws at down 3 are set aside."""
    segs = segs[(segs.t0 > 0) & (segs.result != "lost")]
    segs = segs[~((segs.d0 == -3) & (segs.action == "foul to give"))]
    # misses rebounded with time to use the ball (0.5 s or more): how often the offense gets it
    rebounded = segs.result.str.endswith(("oreb", "dreb"))
    oreb_share = segs.result[rebounded].str.endswith("oreb").mean()
    cls = [state_class(d, a) for d, a in zip(segs.d0, segs.action, strict=True)]
    out = {}
    for name, g in segs.groupby(np.array(cls)):
        out[name] = dict(
            t0=g.t0.to_numpy(float),
            delta=g.delta.to_numpy(int),
            next=g.next.to_numpy(str),
            dt=np.maximum(1, np.round((g.t0 - g.t1).to_numpy(float) / STEP)).astype(int),
            label=g.label.to_numpy(str),
            buzzer=g.result.isin(["miss", "ftmiss"]).to_numpy(),
            oreb_share=oreb_share,
        )
    return out


def weights(t0: np.ndarray, t: float, bandwidth: float) -> np.ndarray:
    """Triangular weights on segments that started near t. The window is a share of the time
    left (`bandwidth`), at least FLOOR seconds, and widened to reach MIN_SEGMENTS. It narrows near
    the buzzer, where a second changes what a possession can do. The share is chosen on held-out
    seasons in scripts/analyze.py."""
    gap = np.abs(t0 - t)
    k = min(MIN_SEGMENTS, len(gap)) - 1
    radius = max(bandwidth * t, FLOOR, np.partition(gap, k)[k] + 1e-9)
    w = np.clip(1 - gap / radius, 0, None)
    return w / w.sum()


def terminal(margin):
    """Chances to win and to lose at the buzzer for a team ahead by `margin`. A tie is neither:
    the game goes to overtime."""
    margin = np.asarray(margin)
    return np.stack([margin > 0, margin < 0]).astype(float)


def after(value: np.ndarray, a: dict, margin: int, i: int) -> np.ndarray:
    """Chances to win and to lose at the buzzer for the team that had the ball, after each
    segment in `a`, starting from `margin` at grid point i. `value` holds both, for every margin
    and grid point: value[0] to win and value[1] to lose.

    A segment borrowed from a nearby start time keeps its own seconds used. One where the clock
    ran out ("none") ended the game only because it started later than i, so here it just uses
    its seconds with the same team holding the ball. A miss at the buzzer, borrowed with time
    left, is rebounded by the offense at the usual offensive rebound share. One longer than the
    time left ends at 0."""
    m = np.clip(margin + a["delta"], -MAX_MARGIN, MAX_MARGIN)
    j = np.maximum(i - a["dt"], 0)
    keep = value[:, m + MAX_MARGIN, j]
    lose = value[::-1, -m + MAX_MARGIN, j]  # the other team's loss is this team's win
    v = np.where(a["next"] == "other", lose, keep)
    if "buzzer" in a:
        r = a["oreb_share"]
        v = np.where(a["buzzer"], r * keep + (1 - r) * lose, v)
    return np.where(j == 0, terminal(m), v)


def solve(arrays: dict, bandwidth: float) -> dict[str, np.ndarray]:
    """Chances to win and to lose at the buzzer for the team with the ball, at every margin and
    tenth of a second: value[0] and value[1].

    Also returns, for the trailing team down 3, the team up 3's chances to win and to lose at
    the buzzer when it fouls and when it defends. In every state down 3 the team up 3 takes the
    choice with the better chance to win."""
    n = int(round(HORIZON / STEP)) + 1
    margins = np.arange(-MAX_MARGIN, MAX_MARGIN + 1)
    value = np.zeros((2, len(margins), n))
    value[:, :, 0] = terminal(margins)
    out = {k: np.full(n, np.nan) for k in ["foul", "defend", "foul_loss", "defend_loss"]}

    for i in range(1, n):
        w = {c: weights(a["t0"], i * STEP, bandwidth) for c, a in arrays.items()}
        for k, margin in enumerate(margins):
            if margin == -3:
                # the trailing team's win is the team up 3's loss, and the other way round
                lose_f, win_f = after(value, arrays["down 3, foul"], -3, i) @ w["down 3, foul"]
                lose_d, win_d = after(value, arrays["down 3, defend"], -3, i) @ w["down 3, defend"]
                out["foul"][i], out["foul_loss"][i] = win_f, lose_f
                out["defend"][i], out["defend_loss"][i] = win_d, lose_d
                value[:, k, i] = [lose_f, win_f] if win_f >= win_d else [lose_d, win_d]
            else:
                c = state_class(int(margin))
                value[:, k, i] = after(value, arrays[c], margin, i) @ w[c]
    return dict(value=value, **out, seconds=np.arange(n) * STEP)
