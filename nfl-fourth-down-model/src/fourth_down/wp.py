"""nflfastR's win-probability model (wp_model_spread from the fastrmodels R package, MIT).

scripts/get_wp_model.py writes the model file. Its 12 inputs, in the order the model was trained on,
are built here from the same game-state columns the rest of the project uses.
"""

from pathlib import Path

import numpy as np
import xgboost as xgb

MODEL = Path(__file__).resolve().parents[2] / "models" / "wp_model_spread.ubj"


class WinProbability:
    """smooth: average over the clock up to this many seconds either side. The model is a tree
    ensemble, so its output moves in steps as the clock runs. The window shrinks to a quarter of the
    time left in the half, so late-game states aren't mixed with an expired clock. 0 reproduces
    nflverse's vegas_wp exactly."""

    def __init__(self, path=MODEL, smooth=60):
        self.booster = xgb.Booster()
        self.booster.load_model(bytearray(Path(path).read_bytes()))
        self.smooth = smooth

    def predict_proba(self, s):
        """Win probability for the team with the ball, as an sklearn-style (n, 2) array."""
        half0 = np.asarray(s.half_seconds_remaining, dtype=float)
        window = np.minimum(self.smooth, half0 / 4)
        steps = [-1, 0, 1] if self.smooth else [0]
        p = np.zeros(len(s))
        for k in steps:
            half = np.minimum(half0 + k * window, 1800)
            game = np.asarray(s.game_seconds_remaining, dtype=float) + (half - half0)
            p += self._predict(s, game, half)
        p /= len(steps)
        return np.column_stack([1 - p, p])

    def _predict(self, s, game, half):
        decay = np.exp(-4 * (3600 - game) / 3600)
        x = np.column_stack([
            s.receive_2h_ko,
            s.posteam_spread * decay,
            s.home,
            half,
            game,
            s.score_differential / decay,
            s.score_differential,
            s.down,
            s.ydstogo,
            s.yardline_100,
            s.posteam_timeouts_remaining,
            s.defteam_timeouts_remaining,
        ]).astype(float)
        return self.booster.predict(xgb.DMatrix(x))
