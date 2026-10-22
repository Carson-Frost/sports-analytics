"""The value curve: career Win Shares = a - b * ln(pick), and the models it is tested against."""

import numpy as np
import pandas as pd

EXP_RATES = np.linspace(0.001, 0.3, 3000)  # decay rates searched for the exponential curve


def fit_log(pick: np.ndarray, value: np.ndarray) -> tuple[float, float]:
    """Least squares (a, b) for value = a - b * ln(pick)."""
    slope, intercept = np.polyfit(np.log(pick), value, 1)
    return float(intercept), float(-slope)


def log_curve(pick: np.ndarray, a: float | np.ndarray, b: float | np.ndarray) -> np.ndarray:
    return a - b * np.log(pick)


def halving_pick(a: float | np.ndarray, b: float | np.ndarray) -> float | np.ndarray:
    """The pick worth half of pick 1: a - b * ln(p) = a / 2."""
    return np.exp(a / (2 * b))


def fit_exp(pick: np.ndarray, value: np.ndarray) -> tuple[float, float]:
    """Least squares (scale, rate) for value = scale * exp(-rate * (pick - 1)).

    For a fixed rate the best scale has a closed form, so search the rate on a grid.
    """
    # ponytail: grid step 0.0001 in rate, far finer than the rate's own uncertainty
    shapes = np.exp(-np.outer(EXP_RATES, pick - 1))
    scales = shapes @ value / (shapes**2).sum(axis=1)
    sse = ((value - scales[:, None] * shapes) ** 2).sum(axis=1)
    best = int(np.argmin(sse))
    return float(scales[best]), float(EXP_RATES[best])


def exp_curve(pick: np.ndarray, scale: float, rate: float) -> np.ndarray:
    return scale * np.exp(-rate * (pick - 1))


def held_out_predictions(draftees: pd.DataFrame) -> pd.DataFrame:
    """Predict each draft class from a fit on the other classes (leave one class out).

    One column per model, one row per draftee, in the order of `draftees`.
    """
    preds = pd.DataFrame(index=draftees.index, dtype=float)
    for year in draftees["year"].unique():
        train = draftees[draftees["year"] != year]
        test = draftees["year"] == year
        p_train, v_train = train["pick"].to_numpy(float), train["win_shares"].to_numpy()
        p_test = draftees.loc[test, "pick"].to_numpy(float)
        pick_means = train.groupby("pick")["win_shares"].mean()
        preds.loc[test, "Flat Average"] = v_train.mean()
        preds.loc[test, "Average at Each Pick"] = pick_means.reindex(p_test).to_numpy()
        preds.loc[test, "Exponential Curve"] = exp_curve(p_test, *fit_exp(p_train, v_train))
        preds.loc[test, "Log Curve"] = log_curve(p_test, *fit_log(p_train, v_train))
    if preds.isna().any().any():
        raise ValueError("A held-out pick never appears in the training classes")
    return preds


def bootstrap_fits(draftees: pd.DataFrame, n: int = 2000, seed: int = 2016) -> np.ndarray:
    """(a, b) refit on `n` resamples of whole draft classes, drawn with replacement."""
    rng = np.random.default_rng(seed)
    classes = [
        (g["pick"].to_numpy(float), g["win_shares"].to_numpy()) for _, g in draftees.groupby("year")
    ]
    fits = np.empty((n, 2))
    for i in range(n):
        drawn = [classes[j] for j in rng.integers(len(classes), size=len(classes))]
        picks, values = (np.concatenate(part) for part in zip(*drawn, strict=True))
        fits[i] = fit_log(picks, values)
    return fits
