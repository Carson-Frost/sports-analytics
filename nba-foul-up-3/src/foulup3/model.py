"""Possession outcome probabilities from the game state, and the exact end game they imply.

The state is the margin for the team with the ball and the seconds left. For each state, the
chance of every segment outcome (points, who has the ball next, seconds used) is the observed mix
of segments from the same kind of state that started at about the same time. The end game is then
solved exactly, tenth of a second by tenth of a second, from the buzzer backwards.
"""

import numpy as np
import pandas as pd

STEP = 0.1  # seconds per grid point
HORIZON = 24.0
MAX_MARGIN = 15  # margins beyond this are treated as this
MIN_SEGMENTS = 30  # widen the window until at least this many segments count


def state_class(margin: int, action: str = "defend") -> str:
    if margin == -3:
        return "down 3, foul" if action == "foul" else "down 3, defend"
    if margin <= -4:
        return "down 4+"
    if margin >= 4:
        return "up 4+"
    return {-2: "down 2", -1: "down 1", 0: "tied", 1: "up 1", 2: "up 2", 3: "up 3"}[margin]


def decisions(segs: pd.DataFrame) -> pd.DataFrame:
    """Every time the trailing team got the ball down 3 with 24 seconds or less left."""
    d = segs[(segs.d0 == -3) & (segs.how == "change") & (segs.t0 > 0)]
    return d.assign(leader_won=~d.offense_won)


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
    """Triangular weights on segments that started within `bandwidth` seconds of t, widened to
    reach MIN_SEGMENTS. The bandwidth is chosen on held-out seasons in scripts/model.py."""
    gap = np.abs(t0 - t)
    k = min(MIN_SEGMENTS, len(gap)) - 1
    radius = max(bandwidth, np.partition(gap, k)[k] + 1e-9)
    w = np.clip(1 - gap / radius, 0, None)
    return w / w.sum()


def terminal(margin):
    """Win probability for a team ahead by `margin` at the buzzer; a tie goes to overtime."""
    return np.where(margin > 0, 1.0, np.where(margin == 0, 0.5, 0.0))


def after(value: np.ndarray, a: dict, margin: int, i: int) -> np.ndarray:
    """Win probability for the team that had the ball, after each segment in `a`, starting from
    `margin` at grid point i.

    A segment borrowed from a nearby start time keeps its own seconds used. One where the clock
    ran out ("none") ended the game only because it started later than i, so here it just uses
    its seconds with the same team holding the ball. A miss at the buzzer, borrowed with time
    left, is rebounded by the offense at the usual offensive rebound share. One longer than the
    time left ends at 0."""
    m = np.clip(margin + a["delta"], -MAX_MARGIN, MAX_MARGIN)
    j = np.maximum(i - a["dt"], 0)
    keep = value[m + MAX_MARGIN, j]
    lose = 1 - value[-m + MAX_MARGIN, j]
    v = np.where(a["next"] == "other", lose, keep)
    if "buzzer" in a:
        r = a["oreb_share"]
        v = np.where(a["buzzer"], r * keep + (1 - r) * lose, v)
    return np.where(j == 0, terminal(m), v)


def solve(arrays: dict, bandwidth: float) -> dict[str, np.ndarray]:
    """Win probability for the team with the ball at every margin and tenth of a second.

    Also returns, for the trailing team down 3, the leader's win probability when the leader
    fouls and when it defends. In every later state the leader takes the better of the two."""
    n = int(round(HORIZON / STEP)) + 1
    margins = np.arange(-MAX_MARGIN, MAX_MARGIN + 1)
    value = np.zeros((len(margins), n))
    value[:, 0] = terminal(margins)
    leader_foul, leader_defend = np.full(n, np.nan), np.full(n, np.nan)

    for i in range(1, n):
        w = {c: weights(a["t0"], i * STEP, bandwidth) for c, a in arrays.items()}
        for k, margin in enumerate(margins):
            if margin == -3:
                foul = w["down 3, foul"] @ after(value, arrays["down 3, foul"], -3, i)
                defend = w["down 3, defend"] @ after(value, arrays["down 3, defend"], -3, i)
                leader_foul[i], leader_defend[i] = 1 - foul, 1 - defend
                value[k, i] = min(foul, defend)
            else:
                c = state_class(int(margin))
                value[k, i] = w[c] @ after(value, arrays[c], margin, i)
    return dict(value=value, foul=leader_foul, defend=leader_defend, seconds=np.arange(n) * STEP)
