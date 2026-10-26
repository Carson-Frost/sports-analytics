"""EPA per play by grouping compared with 11, within offense-season, by weighted least squares.

    epa ~ grouping + (down x yards to go) + (5-yard field bin) + [extra controls] + offense-season

Offense-season is absorbed by subtracting weighted group means. Weights are how many times each
play's game was drawn in a resample (1 for the point estimate).
"""

import numpy as np
import pandas as pd

BASELINE = "11"


class Design:
    """The regression for one set of plays, built once and refit under any game weights."""

    def __init__(self, plays: pd.DataFrame, groupings: list[str], extra: tuple[str, ...] = ()):
        self.groupings = [g for g in groupings if g != BASELINE]
        down = plays.down.clip(upper=3).astype(int).astype(str)  # 4th down shares 3rd's terms
        distance = down + "_" + plays.ydstogo.clip(upper=15).astype(int).astype(str)  # 15+ yards share one
        field = np.ceil(plays.yardline_100 / 5).astype(int).astype(str)
        g = pd.Categorical(plays.grouping, categories=[BASELINE, *self.groupings])
        parts = [
            pd.get_dummies(g, dtype=float).iloc[:, 1:].to_numpy(),
            pd.get_dummies(distance, dtype=float).iloc[:, 1:].to_numpy(),
            pd.get_dummies(field, dtype=float).iloc[:, 1:].to_numpy(),
            *(pd.get_dummies(plays[col].astype(str), dtype=float).iloc[:, 1:].to_numpy() for col in extra),
        ]
        team, _ = codes(plays.team_season)
        order = np.argsort(team, kind="stable")  # rows sorted by offense-season, so group sums are slices
        self.xy = np.column_stack([np.hstack(parts), plays.epa.to_numpy(dtype=float)])[order]
        self.x = self.xy[:, :-1]
        self.team = team[order]
        self.starts = np.flatnonzero(np.r_[True, np.diff(self.team) != 0])
        self.game = plays.game_code.to_numpy()[order]

    def fit(self, game_weights: np.ndarray | None = None) -> pd.Series:
        w = np.ones(len(self.xy)) if game_weights is None else game_weights[self.game].astype(float)
        sums = np.add.reduceat(self.xy * w[:, None], self.starts)
        total = np.add.reduceat(w, self.starts)
        means = np.divide(sums, total[:, None], out=np.zeros_like(sums), where=total[:, None] > 0)
        d = self.xy - np.repeat(means, np.diff(np.r_[self.starts, len(w)]), axis=0)
        dw = d * w[:, None]
        k = self.x.shape[1]
        beta = np.linalg.lstsq(dw[:, :k].T @ d[:, :k], dw[:, :k].T @ d[:, k], rcond=None)[0]
        present = np.array([(w * self.x[:, i]).sum() > 0 for i in range(len(self.groupings))])
        effect = np.where(present, beta[: len(self.groupings)], np.nan)
        return pd.Series(effect, index=self.groupings)


def codes(values: pd.Series) -> tuple[np.ndarray, int]:
    c = pd.Categorical(values)
    return c.codes.astype(np.int64), len(c.categories)
