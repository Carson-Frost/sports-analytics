"""Solve the end game on the model seasons (2021-22 to 2025-26), once for a trailing team with a
timeout left and once for one without, with 200 Resamples for 90% intervals, and measure how often
teams up 3 fouled in every season since 1996-97.

Writes to results/: foul_rate.csv, bandwidth.csv, win_probability.csv, calibration.csv,
decision_check.csv, tree.csv, foul_share_by_second.csv and counts.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from foulup3 import MODEL_SEASONS, SEASONS
from foulup3.endgame import (
    SHOWN,
    EndGame,
    buzzer_margin,
    choose_bandwidth,
    down3_possessions,
    in_shown,
)

ROOT = Path(__file__).resolve().parents[1]
POSSESSIONS = ROOT / "data" / "derived" / "possessions.parquet"
OUT = ROOT / "results"
BANDWIDTHS = [0.15, 0.25, 0.35, 0.5, 0.75, 1.0]  # window as a share of the time left
BANDS = [1, 5, 10, 20]  # clock bands for how often teams fouled
TEST_SEASONS = SEASONS[-2:]  # the calibration check is scored on these
SEED = 2026
TREE_SECONDS = 10.0  # where teams foul most


def shown(d: pd.DataFrame, last: float = SHOWN[1]) -> pd.DataFrame:
    """The down-3 possessions starting from SHOWN[0] to `last` seconds left."""
    return d[in_shown(d.t0, last)]


def foul_rate(possessions: pd.DataFrame) -> pd.DataFrame:
    d = shown(down3_possessions(possessions))
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


def calibration(possessions: pd.DataFrame, h: float) -> pd.DataFrame:
    """Fit on the model seasons minus the last two; predict every possession in those two seasons
    with the score within 3 and 1 to 20 seconds left: did the team with the ball win at the
    buzzer? The comparison is the same rate in the training seasons for the same margin and
    4-second stretch of clock."""
    possessions = possessions.assign(won=buzzer_margin(possessions) > 0)
    held = possessions.season.isin(TEST_SEASONS)
    train, test = possessions[~held], possessions[held]
    end = EndGame(train, h, resamples=0)

    def states(d):
        # down 3 is left out: there the model assumes the better choice, not the one made
        d = d[d.d0.between(-2, 3) & d.t0.between(*SHOWN) & (d.result != "lost")]
        return d.assign(bin=np.minimum(d.t0 // 4, 4).astype(int))  # 20.0 joins 16-20

    tr, te = states(train), states(test)
    p_model = end.win_chance(te.d0, te.t0, te.timeouts, te.opp_timeouts)
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


def foul_share_by_second(possessions: pd.DataFrame) -> pd.DataFrame:
    """How often the team up 3 fouled, for possessions starting within a second of each even
    second from 2 to 20."""
    d = shown(down3_possessions(possessions), SHOWN[1] + 1)
    rows = []
    for t in range(2, 21, 2):
        g = d[(d.t0 >= t - 1) & (d.t0 < t + 1)]
        rows.append(dict(seconds=t, possessions=len(g), fouled=int((g.action == "foul").sum())))
    out = pd.DataFrame(rows)
    return out.assign(share=out.fouled / out.possessions)


def tree(possessions: pd.DataFrame, h: float) -> pd.DataFrame:
    """Every outcome of the trailing team's possession at TREE_SECONDS under each choice, when it
    has a timeout (the usual case there): its chance, the seconds it took, and the team up 3's
    chance to win at the buzzer after it."""
    end = EndGame(possessions, h, resamples=0)
    mixes = [end.outcome_mix(TREE_SECONDS, c, "has a timeout").assign(choice=c)
             for c in ["defend", "foul"]]  # fmt: skip
    out = pd.concat(mixes, ignore_index=True).assign(seconds_left=TREE_SECONDS)
    out = out[["seconds_left", "choice", "outcome", "chance", "seconds", "leader_wins_after",
               "possessions"]]  # fmt: skip
    return out.sort_values(["choice", "chance"], ascending=[True, False])


def miss_chance(p: np.ndarray, wins: int) -> float:
    """If each game is won with its predicted chance, how often the win total lands at least
    this far from the expected total, in either direction (exact)."""
    dist = np.array([1.0])
    for x in p:
        dist = np.convolve(dist, [1 - x, x])  # dist[k]: chance of k wins
    gap = abs(wins - p.sum())
    return float(dist[np.abs(np.arange(len(dist)) - p.sum()) >= gap - 1e-9].sum())


def decision_check(possessions: pd.DataFrame, end: EndGame) -> pd.DataFrame:
    """For the choice each team actually made: wins at the buzzer against the wins the model
    expected, by choice and stretch of the clock, and how often a miss that size happens by
    chance alone."""
    d = shown(down3_possessions(possessions))
    p = end.win_chance(-3, d.t0, d.timeouts, d.opp_timeouts, choice=d.action)
    d = d.assign(p=p, band=pd.cut(d.t0, BANDS, right=False))
    rows = []
    for (choice, band), x in d.groupby(["action", "band"], observed=True):
        wins = int(x.leader_won.sum())
        rows.append(dict(action=choice, band=f"{band.left} to {band.right}", games=len(x),
                         won=wins, expected=x.p.sum(),
                         chance=miss_chance(x.p.to_numpy(), wins)))  # fmt: skip
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(exist_ok=True)
    every = pd.read_parquet(POSSESSIONS)
    foul_rate(every).to_csv(OUT / "foul_rate.csv", index=False, float_format="%.4f")

    possessions = every[every.season.isin(MODEL_SEASONS)].reset_index(drop=True)
    # fit on the first three model seasons and score the last two
    h, scores = choose_bandwidth(possessions[possessions.season.isin(MODEL_SEASONS[:3])],
                                 possessions[possessions.season.isin(MODEL_SEASONS[3:])],
                                 BANDWIDTHS)  # fmt: skip
    scores.to_csv(OUT / "bandwidth.csv", index=False, float_format="%.5f")
    print(f"window {h} of the time left", flush=True)
    end = EndGame(possessions, h, seed=SEED)
    end.calls().to_csv(OUT / "win_probability.csv", index=False, float_format="%.4f")

    calibration(possessions, h).to_csv(OUT / "calibration.csv", index=False, float_format="%.4f")
    decision_check(possessions, end).to_csv(OUT / "decision_check.csv", index=False,
                                          float_format="%.4f")  # fmt: skip
    tree(possessions, h).to_csv(OUT / "tree.csv", index=False, float_format="%.4f")
    foul_share_by_second(possessions).to_csv(OUT / "foul_share_by_second.csv", index=False,
                                      float_format="%.4f")  # fmt: skip

    d = shown(down3_possessions(possessions))
    games = pd.read_parquet(POSSESSIONS.with_name("games.parquet"))
    pd.DataFrame([dict(model_first=MODEL_SEASONS[0], model_last=MODEL_SEASONS[-1],
                       model_games=int(games.season.isin(MODEL_SEASONS).sum()),
                       possessions=len(d), fouled=int((d.action == "foul").sum()),
                       no_timeout=int((d.timeouts == 0).sum()),
                       all_first=SEASONS[0], all_last=SEASONS[-1],
                       all_games=len(games))]).to_csv(OUT / "counts.csv", index=False)  # fmt: skip


if __name__ == "__main__":
    main()
