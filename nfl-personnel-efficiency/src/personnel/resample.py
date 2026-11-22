"""Estimates with 90% intervals from game Resamples.

An estimator is anything with fit(game_weights) returning labelled values (a Series). Each Resample draws the
games with replacement once, and every estimator is fit on that same draw.
"""

import numpy as np
import pandas as pd

RESAMPLES = 200


class GroupMean:
    """The mean of one value per group, optionally among a subset of rows, under game weights.

    values is a column of plays or a Series aligned with it. by is one column or a list of columns; groups are
    their labels (sorted, or in the order of labels, which also drops rows outside them).
    """

    def __init__(self, plays: pd.DataFrame, values, by, among=None, labels=None):
        rows = plays if among is None else plays[among]
        v = rows[values] if isinstance(values, str) else values.loc[rows.index]
        keys = pd.MultiIndex.from_frame(rows[by]) if isinstance(by, list) else pd.Index(rows[by])
        self.labels = keys.unique().sort_values() if labels is None else pd.Index(labels)
        code = self.labels.get_indexer(keys)
        inside = code >= 0
        self.code = code[inside]
        self.values = v.to_numpy(dtype=float)[inside]
        self.game = rows.game_code.to_numpy()[inside]

    def fit(self, game_weights: np.ndarray | None = None) -> pd.Series:
        w = np.ones(len(self.code)) if game_weights is None else game_weights[self.game].astype(float)
        k = len(self.labels)
        with np.errstate(invalid="ignore", divide="ignore"):  # a group absent from the draw is NaN
            mean = np.bincount(self.code, w * self.values, k) / np.bincount(self.code, w, k)
        return pd.Series(mean, index=self.labels)


def run(estimators: dict, n_games: int, resamples: int = RESAMPLES, seed: int = 0) -> dict[object, pd.DataFrame]:
    """Each estimator's point estimate (all games once) and 90% interval over the Resamples.

    Returns one table per estimator name, indexed by label, with columns estimate, low and high. A label absent
    from a draw is NaN there and ignored by the interval.
    """
    point = {name: e.fit(np.ones(n_games)) for name, e in estimators.items()}
    draws = {name: [] for name in estimators}
    rng = np.random.default_rng(seed)
    for _ in range(resamples):
        w = np.bincount(rng.integers(0, n_games, n_games), minlength=n_games)
        for name, e in estimators.items():
            draws[name].append(e.fit(w).reindex(point[name].index))
    out = {}
    for name, est in point.items():
        low, high = np.nanpercentile(np.asarray(draws[name], dtype=float), [5, 95], axis=0)
        out[name] = pd.DataFrame({"estimate": est.to_numpy(), "low": low, "high": high}, index=est.index)
    return out
