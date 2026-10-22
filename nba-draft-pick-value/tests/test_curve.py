import numpy as np
import pandas as pd
import pytest

from draft_value.curve import (
    bootstrap_fits,
    exp_curve,
    fit_exp,
    fit_log,
    halving_pick,
    held_out_predictions,
    log_curve,
)

PICKS = np.arange(1, 61, dtype=float)


def exact_drafts(years: int = 3) -> pd.DataFrame:
    """Draft classes whose Win Shares sit exactly on 75 - 18 ln(pick)."""
    return pd.DataFrame(
        {
            "year": np.repeat(np.arange(years), len(PICKS)),
            "pick": np.tile(PICKS, years).astype(int),
            "win_shares": np.tile(log_curve(PICKS, 75, 18), years),
        }
    )


def test_fit_log_recovers_the_curve():
    assert fit_log(PICKS, log_curve(PICKS, 75, 18)) == pytest.approx((75, 18))


def test_halving_pick_is_worth_half_of_pick_one():
    a, b = 75, 18
    assert log_curve(halving_pick(a, b), a, b) == pytest.approx(a / 2)


def test_fit_exp_recovers_the_curve():
    scale, rate = fit_exp(PICKS, exp_curve(PICKS, 60, 0.06))
    assert scale == pytest.approx(60, rel=0.01)
    assert rate == pytest.approx(0.06, abs=1e-3)


def test_held_out_predictions_use_only_other_drafts():
    drafts = exact_drafts()
    drafts.loc[drafts["year"] == 0, "win_shares"] += 100  # only draft 0 is shifted

    preds = held_out_predictions(drafts)

    # draft 0 is predicted from drafts 1 and 2, which sit exactly on the curve
    truth = log_curve(PICKS, 75, 18)
    assert preds.loc[drafts["year"] == 0, "Log Curve"].to_numpy() == pytest.approx(truth)
    assert preds.loc[drafts["year"] == 0, "Average at Each Pick"].to_numpy() == pytest.approx(truth)
    assert list(preds.columns) == [
        "Flat Average",
        "Average at Each Pick",
        "Exponential Curve",
        "Log Curve",
    ]


def test_bootstrap_of_identical_drafts_returns_the_same_fit():
    fits = bootstrap_fits(exact_drafts(), n=20)
    assert fits == pytest.approx(np.tile([75, 18], (20, 1)))
