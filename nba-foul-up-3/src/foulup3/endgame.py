"""The End game: possession outcome probabilities from the game state, and the exact end game
they imply.

The state is the margin for the team with the ball and the seconds left. For each state, the
chance of every possession outcome (points, who has the ball next, seconds used) is the observed mix
of possessions from the same kind of state that started at about the same time. The end game is then
solved exactly, tenth of a second by tenth of a second, from the buzzer backwards.

Every result has three outcomes for the team with the ball: it wins at the buzzer, the game goes
to overtime, or it loses at the buzzer. A win means a win at the buzzer; overtime is not a win.

`EndGame` is the public entry: fit it from possessions, then ask it for the Calls table.
"""

import numpy as np
import pandas as pd

_STEP = 0.1  # seconds per grid point
_HORIZON = 24.0
_MAX_MARGIN = 15  # margins beyond this are treated as this
SHOWN = (1.0, 20.0)  # seconds the Calls table covers; past 20 there are too few fouls
LOW, HIGH = 5, 95  # percentiles of a 90% interval
_MIN_POSSESSIONS = 30  # widen the window until at least this many possessions count
_FLOOR = 0.5  # narrowest window, in seconds
_CLOSE = 9  # every margin up to this gets its own outcome mix; wider ones share one


def _state_class(margin: int, action: str = "defend") -> str:
    """Which outcome mix a state draws from. A team up 4 is fouled on most trips and a team up
    15 runs out the clock, so margins are kept apart until the game is out of reach."""
    if margin == -3:
        return "down 3, foul" if action == "foul" else "down 3, defend"
    if margin < -_CLOSE:
        return f"down {_CLOSE + 1}+"
    if margin > _CLOSE:
        return f"up {_CLOSE + 1}+"
    return "tied" if margin == 0 else f"{'up' if margin > 0 else 'down'} {abs(margin)}"


def buzzer_margin(possessions: pd.DataFrame) -> pd.Series:
    """For each possession, the margin for its team with the ball when its period ended."""
    last = possessions.groupby(["game_id", "period"]).tail(1)
    final = (last.d0 + last.delta).to_numpy()
    end = pd.DataFrame(dict(game_id=last.game_id, period=last.period, team=last.off, final=final))
    m = possessions[["game_id", "period"]].merge(end, how="left").set_index(possessions.index)
    return m.final.where(m.team == possessions.off, -m.final)


TIMEOUT_STATES = ["has a timeout", "no timeout"]  # the trailing team's, at down 3


def _by_timeouts(possessions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The possessions each timeout state is built from. Down-3 possessions go to the state matching
    the trailing team's timeouts left; every other possession goes to both. Down 3, a team with a
    timeout can move the ball to half court, and one without has to go the length of the floor."""
    down3 = possessions.d0 == -3
    return {"has a timeout": possessions[~down3 | (possessions.timeouts > 0)],
            "no timeout": possessions[~down3 | (possessions.timeouts == 0)]}  # fmt: skip


def decisions(possessions: pd.DataFrame) -> pd.DataFrame:
    """Every time the trailing team got the ball down 3 with 24 seconds or less left, and how
    it ended for the team up 3: "win", "overtime" or "loss" at the buzzer."""
    d = possessions[(possessions.d0 == -3) & (possessions.how == "change") & (possessions.t0 > 0)]
    lead = -buzzer_margin(possessions).loc[d.index]
    ending = np.where(lead > 0, "win", np.where(lead < 0, "loss", "overtime"))
    return d.assign(ending=ending, leader_won=lead > 0)


def _outcome_arrays(possessions: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """Per state class: each possession's start time, margin change, next possession, tenths of a
    second used and label, whether it ended in a miss at the buzzer, and the share of misses
    rebounded by the offense. Fouls with no free throws at down 3 are set aside."""
    possessions = possessions[(possessions.t0 > 0) & (possessions.result != "lost")]
    possessions = possessions[~((possessions.d0 == -3) & (possessions.action == "foul to give"))]
    # misses rebounded with time to use the ball (0.5 s or more): how often the offense gets it
    rebounded = possessions.result.str.endswith(("oreb", "dreb"))
    oreb_share = possessions.result[rebounded].str.endswith("oreb").mean()
    cls = [_state_class(d, a) for d, a in zip(possessions.d0, possessions.action, strict=True)]
    out = {}
    for name, g in possessions.groupby(np.array(cls)):
        out[name] = dict(
            t0=g.t0.to_numpy(float),
            delta=g.delta.to_numpy(int),
            next=g.next.to_numpy(str),
            dt=np.maximum(1, np.round((g.t0 - g.t1).to_numpy(float) / _STEP)).astype(int),
            label=g.label.to_numpy(str),
            buzzer=g.result.isin(["miss", "ftmiss"]).to_numpy(),
            oreb_share=oreb_share,
        )
    return out


def _weights(t0: np.ndarray, t: float, bandwidth: float) -> np.ndarray:
    """Triangular weights on possessions that started near t. The window is a share of the time
    left (`bandwidth`), at least _FLOOR seconds, and widened to reach _MIN_POSSESSIONS. It narrows
    near the buzzer, where a second changes what a possession can do. The share is chosen on
    held-out seasons in scripts/analyze.py."""
    gap = np.abs(t0 - t)
    k = min(_MIN_POSSESSIONS, len(gap)) - 1
    radius = max(bandwidth * t, _FLOOR, np.partition(gap, k)[k] + 1e-9)
    w = np.clip(1 - gap / radius, 0, None)
    return w / w.sum()


def _terminal(margin):
    """Chances to win and to lose at the buzzer for a team ahead by `margin`. A tie is neither:
    the game goes to overtime."""
    margin = np.asarray(margin)
    return np.stack([margin > 0, margin < 0]).astype(float)


def _after(value: np.ndarray, a: dict, margin: int, i: int) -> np.ndarray:
    """Chances to win and to lose at the buzzer for the team that had the ball, after each
    possession in `a`, starting from `margin` at grid point i. `value` holds both, for every margin
    and grid point: value[0] to win and value[1] to lose.

    A possession borrowed from a nearby start time keeps its own seconds used. One where the clock
    ran out ("none") ended the game only because it started later than i, so here it just uses
    its seconds with the same team holding the ball. A miss at the buzzer, borrowed with time
    left, is rebounded by the offense at the usual offensive rebound share. One longer than the
    time left ends at 0."""
    m = np.clip(margin + a["delta"], -_MAX_MARGIN, _MAX_MARGIN)
    j = np.maximum(i - a["dt"], 0)
    keep = value[:, m + _MAX_MARGIN, j]
    lose = value[::-1, -m + _MAX_MARGIN, j]  # the other team's loss is this team's win
    v = np.where(a["next"] == "other", lose, keep)
    if "buzzer" in a:
        r = a["oreb_share"]
        v = np.where(a["buzzer"], r * keep + (1 - r) * lose, v)
    return np.where(j == 0, _terminal(m), v)


def _solve(arrays: dict, bandwidth: float) -> dict[str, np.ndarray]:
    """Chances to win and to lose at the buzzer for the team with the ball, at every margin and
    tenth of a second: value[0] and value[1].

    Also returns, for the trailing team down 3, the team up 3's chances to win and to lose at
    the buzzer when it fouls and when it defends. In every state down 3 the team up 3 takes the
    choice with the better chance to win."""
    n = int(round(_HORIZON / _STEP)) + 1
    margins = np.arange(-_MAX_MARGIN, _MAX_MARGIN + 1)
    value = np.zeros((2, len(margins), n))
    value[:, :, 0] = _terminal(margins)
    out = {k: np.full(n, np.nan) for k in ["foul", "defend", "foul_loss", "defend_loss"]}

    for i in range(1, n):
        w = {c: _weights(a["t0"], i * _STEP, bandwidth) for c, a in arrays.items()}
        for k, margin in enumerate(margins):
            if margin == -3:
                # the trailing team's win is the team up 3's loss, and the other way round
                lose_f, win_f = _after(value, arrays["down 3, foul"], -3, i) @ w["down 3, foul"]
                lose_d, win_d = _after(value, arrays["down 3, defend"], -3, i) @ w["down 3, defend"]
                out["foul"][i], out["foul_loss"][i] = win_f, lose_f
                out["defend"][i], out["defend_loss"][i] = win_d, lose_d
                value[:, k, i] = [lose_f, win_f] if win_f >= win_d else [lose_d, win_d]
            else:
                c = _state_class(int(margin))
                value[:, k, i] = _after(value, arrays[c], margin, i) @ w[c]
    return dict(value=value, **out, seconds=np.arange(n) * _STEP)


def _fit(possessions: pd.DataFrame, bandwidth: float) -> dict[str, dict]:
    """The solved end game for each timeout state of the trailing team."""
    return {s: _solve(_outcome_arrays(p), bandwidth) for s, p in _by_timeouts(possessions).items()}


class EndGame:
    """The End game fitted from possessions with a window of `bandwidth` (a share of the time
    left). With `resamples`, each Resample redraws the games with replacement (seeded by `seed`)
    and refits both timeout states on that same draw, for 90% intervals."""

    def __init__(self, possessions: pd.DataFrame, bandwidth: float, resamples: int = 200,
                 seed: int | None = None):  # fmt: skip
        self.bandwidth = bandwidth
        self._points = _fit(possessions, bandwidth)
        self._gaps = None
        if resamples:
            rows = possessions.groupby("game_id").indices
            games = np.array(list(rows))
            rng = np.random.default_rng(seed)
            gaps = {state: [] for state in TIMEOUT_STATES}
            for _ in range(resamples):
                idx = np.concatenate([rows[g] for g in rng.choice(games, len(games))])
                for state, r in _fit(possessions.iloc[idx], bandwidth).items():
                    gaps[state].append(r["foul"] - r["defend"])
            self._gaps = {state: np.array(g) for state, g in gaps.items()}

    def calls(self) -> pd.DataFrame:
        """For every tenth of a second in SHOWN and each timeout state of the trailing team: the
        team up 3's chances to win at the buzzer, go to overtime and lose under each Choice, the
        gap (foul minus defend), and the Call (the Choice with the higher chance to win). With
        Resamples, also the gap's 90% interval and Clear (the interval lies wholly on one side of
        zero)."""
        tables = []
        for state in TIMEOUT_STATES:
            p = self._points[state]
            t = pd.DataFrame(dict(timeouts=state, seconds=p["seconds"]))
            for c in ["foul", "defend"]:
                t[c], t[f"{c}_ot"] = p[c], 1 - p[c] - p[f"{c}_loss"]
                t[f"{c}_loss"] = p[f"{c}_loss"]
            t["gap"] = p["foul"] - p["defend"]
            if self._gaps is not None:
                t["gap_low"], t["gap_high"] = np.percentile(self._gaps[state], [LOW, HIGH], axis=0)
            t["call"] = np.where(t.gap >= 0, "foul", "defend")
            if self._gaps is not None:
                t["clear"] = (t.gap_low > 0) | (t.gap_high < 0)
            shown = (t.seconds >= SHOWN[0] - 1e-9) & (t.seconds <= SHOWN[1] + 1e-9)
            tables.append(t[shown])
        return pd.concat(tables, ignore_index=True)


def _held_out_losses(train: dict, held_out: dict, bandwidth: float) -> list[float]:
    """Log loss of each held-out possession's outcome under the training mix, mixed 99 to 1 with
    an even spread so no outcome scores zero."""
    losses = []
    for c, a in held_out.items():
        if c not in train:
            continue
        tr = train[c]
        spread = 1 / len(np.unique(tr["label"]))
        for t0, outcome in zip(a["t0"], a["label"], strict=True):
            p = _weights(tr["t0"], t0, bandwidth)[tr["label"] == outcome].sum()
            losses.append(-np.log(0.99 * p + 0.01 * spread))
    return losses


def choose_bandwidth(train: pd.DataFrame, held_out: pd.DataFrame,
                     bandwidths: list[float]) -> tuple[float, pd.DataFrame]:  # fmt: skip
    """Score each candidate bandwidth by the mean log loss of the held-out possessions' outcomes
    under the training possessions' mix. Returns the bandwidth with the lowest loss and the score
    table (bandwidth, possessions scored, log_loss), one row per candidate in the order given."""
    tr, ho = _outcome_arrays(train), _outcome_arrays(held_out)
    rows = []
    for h in bandwidths:
        losses = _held_out_losses(tr, ho, h)
        rows.append(dict(bandwidth=h, possessions=len(losses), log_loss=float(np.mean(losses))))
    scores = pd.DataFrame(rows)
    return float(scores.bandwidth[scores.log_loss.idxmin()]), scores
