"""Solve the end game on the model seasons (2021-22 to 2025-26), once for a trailing team with a
timeout left and once for one without, resample games for 90% intervals, and measure how often
teams up 3 fouled in every season since 1996-97.

Writes to results/: foul_rate.csv, bandwidth.csv, win_probability.csv, calibration.csv,
decision_check.csv, tree.csv, foul_share_by_second.csv and counts.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from foulup3 import MODEL_SEASONS, SEASONS
from foulup3.model import (
    STEP,
    TIMEOUT_STATES,
    after,
    buzzer_margin,
    by_timeouts,
    decisions,
    outcome_arrays,
    solve,
    weights,
)

ROOT = Path(__file__).resolve().parents[1]
SEGMENTS = ROOT / "data" / "derived" / "segments.parquet"
OUT = ROOT / "results"
RESAMPLES = 200
LOW, HIGH = 5, 95  # percentiles of a 90% interval
BANDWIDTHS = [0.15, 0.25, 0.35, 0.5, 0.75, 1.0]  # window as a share of the time left
SHOWN = (1.0, 20.0)  # past 20 seconds there are too few fouls to estimate each second
BANDS = [1, 5, 10, 20]  # clock bands for how often teams fouled
TEST_SEASONS = SEASONS[-2:]  # the calibration check is scored on these
SEED = 2026
TREE_SECONDS = 10.0  # where teams foul most


def choices(segs: pd.DataFrame, last: float = SHOWN[1]) -> pd.DataFrame:
    """Possessions starting with `last` seconds or less where the trailing team got the ball
    down 3 and the team up 3 either fouled (sending them to the line) or defended. Fouls to give
    are left out. `segs` must hold whole periods, so each game's ending is known."""
    d = decisions(segs)
    return d[d.action.isin(["foul", "defend"]) & (d.t0 >= SHOWN[0]) & (d.t0 <= last)]


def foul_rate(segs: pd.DataFrame) -> pd.DataFrame:
    d = choices(segs)
    bands = pd.cut(d.t0, BANDS, right=False)
    d = d.assign(band=bands.cat.rename_categories(lambda b: f"{b.left} to {b.right}"))
    rows = []
    for (season, band), g in d.groupby(["season", "band"], observed=True):
        rows.append(dict(season=season, band=band, possessions=len(g),
                         fouled=int((g.action == "foul").sum())))  # fmt: skip
    for season, g in d.groupby("season"):
        rows.append(dict(season=season, band="all", possessions=len(g),
                         fouled=int((g.action == "foul").sum())))  # fmt: skip
    out = pd.DataFrame(rows)
    return out.assign(share=out.fouled / out.possessions)


def outcome_loss(train: dict, valid: dict, h: float) -> list[float]:
    """Log loss of each held-out segment's outcome under the training mix, mixed 99 to 1 with
    an even spread so no outcome scores zero."""
    losses = []
    for c, a in valid.items():
        if c not in train:
            continue
        tr = train[c]
        spread = 1 / len(np.unique(tr["label"]))
        for t0, outcome in zip(a["t0"], a["label"], strict=True):
            p = weights(tr["t0"], t0, h)[tr["label"] == outcome].sum()
            losses.append(-np.log(0.99 * p + 0.01 * spread))
    return losses


def choose_bandwidth(segs: pd.DataFrame) -> float:
    """Fit on the first three model seasons and score the last two."""
    train = outcome_arrays(segs[segs.season.isin(MODEL_SEASONS[:3])])
    valid = outcome_arrays(segs[segs.season.isin(MODEL_SEASONS[3:])])
    rows = []
    for h in BANDWIDTHS:
        losses = outcome_loss(train, valid, h)
        rows.append(dict(bandwidth=h, segments=len(losses), log_loss=float(np.mean(losses))))
    scores = pd.DataFrame(rows)
    scores.to_csv(OUT / "bandwidth.csv", index=False, float_format="%.5f")
    return float(scores.bandwidth[scores.log_loss.idxmin()])


def fit(segs: pd.DataFrame, h: float) -> dict[str, dict]:
    """The solved end game for each timeout state of the trailing team."""
    return {state: solve(outcome_arrays(s), h) for state, s in by_timeouts(segs).items()}


def resampled(segs: pd.DataFrame, h: float) -> tuple[dict, dict]:
    """Point estimates, and foul minus defend (chance to win at the buzzer) on 200 redraws of
    the games, for each timeout state."""
    rows = segs.groupby("game_id").indices
    games = np.array(list(rows))
    point = fit(segs, h)
    rng = np.random.default_rng(SEED)
    gaps = {state: [] for state in TIMEOUT_STATES}
    for _ in range(RESAMPLES):
        idx = np.concatenate([rows[g] for g in rng.choice(games, len(games))])
        for state, r in fit(segs.iloc[idx], h).items():
            gaps[state].append(r["foul"] - r["defend"])
    return point, {state: np.array(g) for state, g in gaps.items()}


def calibration(segs: pd.DataFrame, h: float) -> pd.DataFrame:
    """Fit on the model seasons minus the last two; predict every possession in those two seasons
    with the score within 3 and 1 to 20 seconds left: did the team with the ball win at the
    buzzer? The comparison is the same rate in the training seasons for the same margin and
    4-second stretch of clock."""
    segs = segs.assign(won=buzzer_margin(segs) > 0)
    held = segs.season.isin(TEST_SEASONS)
    train, test = segs[~held], segs[held]
    grids = {state: r["value"][0] for state, r in fit(train, h).items()}

    def states(d):
        # down 3 is left out: there the model assumes the better choice, not the one made
        d = d[d.d0.between(-2, 3) & d.t0.between(*SHOWN) & (d.result != "lost")]
        return d.assign(bin=np.minimum(d.t0 // 4, 4).astype(int))  # 20.0 joins 16-20

    tr, te = states(train), states(test)
    at = (te.d0 + 15, np.round(te.t0 / STEP).astype(int))
    has, none = grids["has a timeout"][at], grids["no timeout"][at]
    # the timeouts that matter are the trailing team's; when tied, either team could fall behind
    trailing = np.where(te.d0 < 0, te.timeouts, te.opp_timeouts)
    p_model = np.where(te.d0 == 0, (has + none) / 2, np.where(trailing > 0, has, none))
    rate = tr.groupby(["d0", "bin"]).won.agg(lambda y: (y.sum() + 1) / (len(y) + 2))
    p_rate = rate.reindex(pd.MultiIndex.from_arrays([te.d0, te.bin])).to_numpy()
    y = te.won.to_numpy(float)

    def loss(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))

    rows = [dict(model="Possession model", possessions=len(y), log_loss=loss(p_model),
                 predicted=p_model.mean(), observed=y.mean()),
            dict(model="Win rate by margin and clock", possessions=len(y), log_loss=loss(p_rate),
                 predicted=p_rate.mean(), observed=y.mean())]  # fmt: skip
    return pd.DataFrame(rows)


def foul_share_by_second(segs: pd.DataFrame) -> pd.DataFrame:
    """How often the team up 3 fouled, for possessions starting within a second of each even
    second from 2 to 20."""
    d = choices(segs, SHOWN[1] + 1)
    rows = []
    for t in range(2, 21, 2):
        g = d[(d.t0 >= t - 1) & (d.t0 < t + 1)]
        rows.append(dict(seconds=t, possessions=len(g), fouled=int((g.action == "foul").sum())))
    out = pd.DataFrame(rows)
    return out.assign(share=out.fouled / out.possessions)


def tree(segs: pd.DataFrame, h: float) -> pd.DataFrame:
    """Every outcome of the trailing team's possession at TREE_SECONDS under each choice, when it
    has a timeout (the usual case there): its chance, the seconds it took, and the team up 3's
    chance to win at the buzzer after it."""
    arrays = outcome_arrays(by_timeouts(segs)["has a timeout"])
    value = solve(arrays, h)["value"]
    i = int(round(TREE_SECONDS / STEP))
    rows = []
    for choice in ["defend", "foul"]:
        a = arrays[f"down 3, {choice}"]
        mix = pd.DataFrame(dict(label=a["label"], w=weights(a["t0"], TREE_SECONDS, h),
                                seconds=a["dt"] * STEP,
                                leader=after(value, a, -3, i)[1]))  # fmt: skip
        for name, g in mix[mix.w > 0].groupby("label"):
            rows.append(dict(seconds_left=TREE_SECONDS, choice=choice, outcome=name,
                             chance=g.w.sum(), seconds=np.average(g.seconds, weights=g.w),
                             leader_wins_after=np.average(g.leader, weights=g.w),
                             possessions=len(g)))  # fmt: skip
    return pd.DataFrame(rows).sort_values(["choice", "chance"], ascending=[True, False])


def miss_chance(p: np.ndarray, wins: int) -> float:
    """If each game is won with its predicted chance, how often the win total lands at least
    this far from the expected total, in either direction (exact)."""
    dist = np.array([1.0])
    for x in p:
        dist = np.convolve(dist, [1 - x, x])  # dist[k]: chance of k wins
    gap = abs(wins - p.sum())
    return float(dist[np.abs(np.arange(len(dist)) - p.sum()) >= gap - 1e-9].sum())


def decision_check(segs: pd.DataFrame, h: float) -> pd.DataFrame:
    """For the choice each team actually made: wins at the buzzer against the wins the model
    expected, by choice and stretch of the clock, and how often a miss that size happens by
    chance alone."""
    split = fit(segs, h)
    d = choices(segs)
    i = np.round(d.t0.to_numpy() / STEP).astype(int)
    foul = (d.action == "foul").to_numpy()

    def win(r):
        return np.where(foul, r["foul"][i], r["defend"][i])

    has = (d.timeouts > 0).to_numpy()
    d = d.assign(p=np.where(has, win(split["has a timeout"]), win(split["no timeout"])),
                 band=pd.cut(d.t0, BANDS, right=False))  # fmt: skip
    rows = []
    for (choice, band), x in d.groupby(["action", "band"], observed=True):
        wins = int(x.leader_won.sum())
        rows.append(dict(action=choice, band=f"{band.left} to {band.right}", games=len(x),
                         won=wins, expected=x.p.sum(),
                         chance=miss_chance(x.p.to_numpy(), wins)))  # fmt: skip
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(exist_ok=True)
    every = pd.read_parquet(SEGMENTS)
    foul_rate(every).to_csv(OUT / "foul_rate.csv", index=False, float_format="%.4f")

    segs = every[every.season.isin(MODEL_SEASONS)].reset_index(drop=True)
    h = choose_bandwidth(segs)
    print(f"window {h} of the time left", flush=True)
    points, gaps = resampled(segs, h)
    tables = []
    for state in TIMEOUT_STATES:
        point = points[state]
        lo, hi = np.percentile(gaps[state], [LOW, HIGH], axis=0)
        seconds = point["seconds"]
        shown = (seconds >= SHOWN[0] - 1e-9) & (seconds <= SHOWN[1] + 1e-9)
        ot = {c: 1 - point[c] - point[f"{c}_loss"] for c in ["foul", "defend"]}
        tables.append(pd.DataFrame(dict(
            timeouts=state, seconds=seconds, foul=point["foul"], foul_ot=ot["foul"],
            foul_loss=point["foul_loss"], defend=point["defend"], defend_ot=ot["defend"],
            defend_loss=point["defend_loss"], gap=point["foul"] - point["defend"], gap_low=lo,
            gap_high=hi))[shown])  # fmt: skip
    pd.concat(tables).to_csv(OUT / "win_probability.csv", index=False, float_format="%.4f")

    calibration(segs, h).to_csv(OUT / "calibration.csv", index=False, float_format="%.4f")
    decision_check(segs, h).to_csv(OUT / "decision_check.csv", index=False, float_format="%.4f")
    tree(segs, h).to_csv(OUT / "tree.csv", index=False, float_format="%.4f")
    foul_share_by_second(segs).to_csv(OUT / "foul_share_by_second.csv", index=False,
                                      float_format="%.4f")  # fmt: skip

    d = choices(segs)
    games = pd.read_parquet(SEGMENTS.with_name("games.parquet"))
    pd.DataFrame([dict(model_first=MODEL_SEASONS[0], model_last=MODEL_SEASONS[-1],
                       model_games=int(games.season.isin(MODEL_SEASONS).sum()),
                       possessions=len(d), fouled=int((d.action == "foul").sum()),
                       no_timeout=int((d.timeouts == 0).sum()),
                       all_first=SEASONS[0], all_last=SEASONS[-1],
                       all_games=len(games))]).to_csv(OUT / "counts.csv", index=False)  # fmt: skip


if __name__ == "__main__":
    main()
