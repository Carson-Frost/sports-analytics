"""Solve the end game, test it on held-out seasons, and bootstrap the foul-minus-defend gap.

Writes to results/: bandwidth.csv, heldout.csv, model_check.csv, one_possession.csv and
win_probability.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from foulup3.model import STEP, after, decisions, outcome_arrays, solve, weights

SEGMENTS = Path("data/derived/segments.parquet")
OUT = Path("results")
VALID_SEASONS = ["2022-23", "2023-24"]  # bandwidth chosen here, trained on the seasons before
TEST_SEASONS = ["2024-25", "2025-26"]  # scored once, trained on every season before
BANDWIDTHS = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
EXAMPLE_SECONDS = 10.0
BOOTSTRAPS = 200
CHECK_BINS = [0, 5, 10, 15, 24]
BASELINE_BINS = [0, 8, 16, 24]


def choices(segs: pd.DataFrame) -> pd.DataFrame:
    """Decisions where the leader fouled (sending them to the line) or defended."""
    d = decisions(segs)
    return d[d.action.isin(["foul", "defend"])]


def predict(result: dict, d: pd.DataFrame) -> np.ndarray:
    i = np.round(d.t0.to_numpy() / STEP).astype(int)
    return np.where(d.action == "foul", result["foul"][i], result["defend"][i])


def log_loss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def baseline(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """The simpler alternative: how often leaders who made the same choice in the same stretch
    of the clock won in the training seasons (plus one win and one loss, so no bin says 100%)."""

    def binned(d):
        return d.assign(bin=pd.cut(d.t0, BASELINE_BINS).astype(str))

    rate = (
        binned(train)
        .groupby(["action", "bin"])
        .leader_won.agg(lambda y: (y.sum() + 1) / (len(y) + 2))
        .rename("p")
        .reset_index()
    )
    return binned(test).merge(rate, on=["action", "bin"], how="left").p.to_numpy()


def outcome_loss(train: dict, valid: dict, h: float) -> tuple[int, float]:
    """How well the outcome mix predicts what happened on held-out trips: mean log loss of each
    held-out segment's outcome, mixed 99 to 1 with an even spread so no outcome scores zero."""
    losses = []
    for c, a in valid.items():
        tr = train[c]
        spread = 1 / len(np.unique(tr["label"]))
        for t0, outcome in zip(a["t0"], a["label"], strict=True):
            p = weights(tr["t0"], t0, h)[tr["label"] == outcome].sum()
            losses.append(-np.log(0.99 * p + 0.01 * spread))
    return len(losses), float(np.mean(losses))


def choose_bandwidth(segs: pd.DataFrame) -> float:
    """Train on the seasons before VALID_SEASONS, keep the bandwidth whose outcome mixes predict
    the trips in VALID_SEASONS best."""
    train = outcome_arrays(segs[segs.season < VALID_SEASONS[0]])
    valid = outcome_arrays(segs[segs.season.isin(VALID_SEASONS)])
    everything = outcome_arrays(segs)
    rows = []
    for h in BANDWIDTHS:
        n, loss = outcome_loss(train, valid, h)
        r = solve(everything, h)
        gap = r["foul"] - r["defend"]
        # the crossover each window gives on all seasons, to show how much the choice matters
        crossover = max(i for i in range(1, len(gap)) if gap[i] > 0) * STEP
        rows.append(dict(bandwidth=h, segments=n, log_loss=loss, crossover=crossover))
    scores = pd.DataFrame(rows)
    scores.to_csv(OUT / "bandwidth.csv", index=False, float_format="%.5f")
    return float(scores.bandwidth[scores.log_loss.idxmin()])


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    centre = (k + z * z / 2) / (n + z * z)
    half = z * np.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
    return centre - half, centre + half


def heldout(segs: pd.DataFrame, h: float) -> None:
    train_segs = segs[~segs.season.isin(TEST_SEASONS)]
    train, test = choices(train_segs), choices(segs[segs.season.isin(TEST_SEASONS)])
    y = test.leader_won.to_numpy(int)
    p_model = predict(solve(outcome_arrays(train_segs), h), test)
    guesses = {
        "Possession Model": p_model,
        "Observed Win Rate": baseline(train, test),
        # the leaders' overall win rate in the training seasons, the same for every decision
        "Constant Guess": np.full(len(y), train.leader_won.mean()),
    }
    rows = []
    for name, p in guesses.items():
        diff = log_loss(p_model, y) - log_loss(p, y)
        z = diff.mean() / diff.std() * np.sqrt(len(y)) if diff.std() > 0 else np.nan
        rows.append(
            dict(
                model=name,
                decisions=len(y),
                log_loss=log_loss(p, y).mean(),
                z_model_minus_this=z,
                mean_predicted=p.mean(),
                observed=y.mean(),
            )
        )
    pd.DataFrame(rows).to_csv(OUT / "heldout.csv", index=False, float_format="%.5f")

    rows = []
    check = test.assign(p=p_model, bin=pd.cut(test.t0, CHECK_BINS))
    for (choice, b), g in check.groupby(["action", "bin"], observed=True):
        n, k = len(g), int(g.leader_won.sum())
        lo, hi = wilson(k, n)
        rows.append(
            dict(
                action=choice,
                low=b.left,
                high=b.right,
                decisions=n,
                predicted=g.p.mean(),
                observed=k / n,
                observed_low=lo,
                observed_high=hi,
            )
        )
    pd.DataFrame(rows).to_csv(OUT / "model_check.csv", index=False, float_format="%.4f")


def one_possession(arrays: dict, result: dict, h: float) -> pd.DataFrame:
    """What happens on the trailing team's trip at EXAMPLE_SECONDS under each choice."""
    i = int(round(EXAMPLE_SECONDS / STEP))
    rows = []
    for choice in ["defend", "foul"]:
        a = arrays[f"down 3, {choice}"]
        mix = pd.DataFrame(
            dict(
                label=a["label"],
                w=weights(a["t0"], EXAMPLE_SECONDS, h),
                seconds=a["dt"] * STEP,
                leader=1 - after(result["value"], a, -3, i),
            )
        )
        for name, g in mix[mix.w > 0].groupby("label"):
            rows.append(
                dict(
                    choice=choice,
                    outcome=name,
                    chance=g.w.sum(),
                    seconds=np.average(g.seconds, weights=g.w),
                    leader_wins_after=np.average(g.leader, weights=g.w),
                    segments=len(g),
                )
            )
    return pd.DataFrame(rows).sort_values(["choice", "chance"], ascending=[True, False])


def bootstrap(segs: pd.DataFrame, h: float, rng: np.random.Generator) -> np.ndarray:
    """Foul minus defend at every grid point, re-solved on games drawn with replacement."""
    rows = segs.groupby("game_id").indices
    games = np.array(list(rows))
    gaps = []
    for _ in range(BOOTSTRAPS):
        idx = np.concatenate([rows[g] for g in rng.choice(games, len(games))])
        r = solve(outcome_arrays(segs.iloc[idx]), h)
        gaps.append(r["foul"] - r["defend"])
    return np.array(gaps)


def main():
    OUT.mkdir(exist_ok=True)
    segs = pd.read_parquet(SEGMENTS)
    h = choose_bandwidth(segs)
    heldout(segs, h)

    arrays = outcome_arrays(segs)
    result = solve(arrays, h)
    one_possession(arrays, result, h).to_csv(
        OUT / "one_possession.csv", index=False, float_format="%.4f"
    )
    gaps = bootstrap(segs, h, np.random.default_rng(2026))[:, 1:]
    pd.DataFrame(
        dict(
            seconds=result["seconds"][1:],
            foul=result["foul"][1:],
            defend=result["defend"][1:],
            gap=(result["foul"] - result["defend"])[1:],
            gap_low=np.percentile(gaps, 2.5, axis=0),
            gap_high=np.percentile(gaps, 97.5, axis=0),
        )
    ).to_csv(OUT / "win_probability.csv", index=False, float_format="%.4f")
    print(f"bandwidth {h} s")


if __name__ == "__main__":
    main()
