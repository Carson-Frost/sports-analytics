"""Fit the value curve, test it on held-out draft classes, and bootstrap its uncertainty.

Writes results/fit.csv, results/curve.csv and results/model_test.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from draft_value.curve import bootstrap_fits, fit_log, halving_pick, held_out_predictions, log_curve

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

draftees = pd.read_csv(ROOT / "data" / "draftees.csv")
pick, value = draftees["pick"].to_numpy(float), draftees["win_shares"].to_numpy()

# the curve, with a 90% band from resampling whole draft classes
a, b = fit_log(pick, value)
boot = bootstrap_fits(draftees)
picks = np.arange(1, 61)
boot_curves = log_curve(picks, boot[:, [0]], boot[:, [1]])
boot_params = np.column_stack([boot, halving_pick(boot[:, 0], boot[:, 1]), boot[:, 1] * np.log(2)])
fit = pd.DataFrame(
    {
        "parameter": ["a", "b", "halving_pick", "doubling_cost"],
        "estimate": [a, b, halving_pick(a, b), b * np.log(2)],
        "low_90": np.percentile(boot_params, 5, axis=0),
        "high_90": np.percentile(boot_params, 95, axis=0),
    }
)
fit.round(4).to_csv(RESULTS / "fit.csv", index=False)

at_pick = draftees.groupby("pick")["win_shares"].agg(["mean", "size"]).reindex(picks)
curve = pd.DataFrame(
    {
        "pick": picks,
        "drafts": at_pick["size"].to_numpy(),
        "average_win_shares": at_pick["mean"].to_numpy(),
        "curve_win_shares": log_curve(picks, a, b),
        "curve_low_90": np.percentile(boot_curves, 5, axis=0),
        "curve_high_90": np.percentile(boot_curves, 95, axis=0),
    }
)
curve["share_of_pick_1"] = curve["curve_win_shares"] / curve["curve_win_shares"].iloc[0]
curve.round(4).to_csv(RESULTS / "curve.csv", index=False)

# held-out test: squared error on each draftee's career Win Shares
sq_err = held_out_predictions(draftees).rsub(value, axis=0) ** 2
diff = sq_err.sub(sq_err["Log Curve"], axis=0)  # positive = worse than the log curve
model_test = pd.DataFrame(
    {
        "model": sq_err.columns,
        "rmse": np.sqrt(sq_err.mean()).to_numpy(),
        "z_vs_log_curve": (diff.mean() / (diff.std() / np.sqrt(len(diff)))).to_numpy(),
    }
).sort_values("rmse")
model_test.round(4).to_csv(RESULTS / "model_test.csv", index=False)

print(fit.round(3).to_string(index=False))
print(model_test.round(3).to_string(index=False))
