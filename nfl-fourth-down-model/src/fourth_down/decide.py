"""Score going for it, kicking a field goal and punting at a fourth down.

Every choice is worth the win probability of the states it leads to, weighted by how likely each
one is. The win-probability model only ever sees the team with the ball, so a state where the
other team has it is worth 1 minus their win probability.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer

FIRST_SEASON = 2015
FG_OFFSET = 18  # kick distance = yard line + 18 in nflverse (10 yards of end zone, 8 of snap)
TD_POINTS = 7
MAX_DISTANCE = 15
N_QUANTILES = 10
CHOICES = ["go", "fg", "punt"]
FIELD_ZONES = [5, 10, 20]  # conversion rates differ inside each of these
FG_KNOTS = np.array([[18.0], [30.0], [40.0], [50.0], [58.0], [66.0]])
FG_SPLINE = SplineTransformer(degree=3, knots=FG_KNOTS, extrapolation="linear").fit(FG_KNOTS)


def choice(play_type):
    """What the coach did, in the model's terms."""
    return play_type.map({"pass": "go", "run": "go", "field_goal": "fg", "punt": "punt"})


def conversion_x(ydstogo, yardline_100):
    """One rate per distance from 1 to 9 yards (10 or more share one), shifted by field zone."""
    d = np.minimum(ydstogo, 10)
    zones = [np.asarray(yardline_100) <= z for z in FIELD_ZONES]
    return np.column_stack([d == k for k in range(1, 10)] + zones)


def fg_x(distance, season):
    """A curve in distance, plus season and distance x season: kickers keep improving."""
    distance, years = np.asarray(distance, dtype=float), np.asarray(season) - 2020
    return np.column_stack([FG_SPLINE.transform(distance.reshape(-1, 1)), years, (distance - 40) * years])


def clock_used(plays):
    """Median seconds from a fourth-down snap to the next snap, by what happened.

    Measured in the 1st and 3rd quarters, where the end-of-half clock rules don't apply.
    """
    p = plays[plays.down.notna()].sort_values(["game_id", "game_seconds_remaining"], ascending=[True, False])
    nxt = p.groupby("game_id")[["game_seconds_remaining", "qtr"]].shift(-1)
    p = p.assign(used=p.game_seconds_remaining - nxt.game_seconds_remaining, same_qtr=nxt.qtr == p.qtr)
    f = p[(p.down == 4) & p.qtr.isin([1, 3]) & p.same_qtr]
    went = f.play_type.isin(["pass", "run"])
    converted = (f.first_down_rush == 1) | (f.first_down_pass == 1)
    return {
        "convert": float(f.used[went & converted].median()),
        "fail": float(f.used[went & ~converted & (f.touchdown != 1)].median()),
        "fg_made": float(f.used[(f.play_type == "field_goal") & (f.field_goal_result == "made")].median()),
        "fg_missed": float(f.used[(f.play_type == "field_goal") & (f.field_goal_result != "made")].median()),
        "punt": float(f.used[f.play_type == "punt"].median()),
    }


def fit_submodels(plays):
    """Everything besides win probability: what each choice leads to, measured from the plays."""
    tries = plays[plays.down.isin([3, 4]) & plays.play_type.isin(["pass", "run"]) & (plays.qb_kneel != 1)]
    converted = (
        (tries.first_down_rush == 1)
        | (tries.first_down_pass == 1)
        | ((tries.touchdown == 1) & (tries.td_team == tries.posteam))
    )
    conversion = LogisticRegression().fit(conversion_x(tries.ydstogo, tries.yardline_100), converted)

    q = (np.arange(N_QUANTILES) + 0.5) / N_QUANTILES
    made = tries[converted & (tries.yardline_100 >= 40)]
    distance = made.ydstogo.clip(upper=MAX_DISTANCE)
    gains = np.array([np.quantile(made.yards_gained[distance == d], q) for d in range(1, MAX_DISTANCE + 1)])

    kicks = plays[plays.play_type == "field_goal"]
    fg = LogisticRegression(C=1e4, max_iter=5000).fit(fg_x(kicks.kick_distance, kicks.season),
                                                      kicks.field_goal_result == "made")

    # punts: quantiles of where the other team takes over, per 5-yard bin of where it was kicked from
    punts = plays[(plays.play_type == "punt") & (plays.punt_blocked == 0) & plays.kick_distance.notna()]
    landing = punts.yardline_100 - punts.kick_distance + punts.return_yards.fillna(0)
    takeover = pd.Series(np.where(punts.touchback == 1, 80, 100 - landing).clip(1, 99), index=punts.index)
    bins = (punts.yardline_100 // 5) * 5 + 2.5
    punt = takeover.groupby(bins).apply(lambda t: np.quantile(t, q))
    punt = punt[takeover.groupby(bins).size() >= 50]

    first_snaps = plays[plays.down == 1].groupby(["game_id", "fixed_drive"]).head(1)
    kickoff = first_snaps[first_snaps.drive_start_transition == "KICKOFF"]

    return {
        "conversion": conversion,
        "gains": gains,
        "fg": fg,
        "fg_longest": float(kicks.kick_distance.max()),
        "punt": (punt.index.to_numpy(), np.vstack(punt.to_numpy())),
        "kickoff_start": kickoff.groupby("season").yardline_100.mean().to_dict(),
        "clock": clock_used(plays),
    }


def _logistic(params):
    m = LogisticRegression()
    m.classes_ = np.array([False, True])
    m.coef_ = np.array(params["coef"])
    m.intercept_ = np.array(params["intercept"])
    m.n_features_in_ = m.coef_.shape[1]
    return m


def submodels_to_json(sub):
    return {
        "conversion": {"coef": sub["conversion"].coef_.tolist(), "intercept": sub["conversion"].intercept_.tolist()},
        "fg": {"coef": sub["fg"].coef_.tolist(), "intercept": sub["fg"].intercept_.tolist()},
        "gains": np.round(sub["gains"], 2).tolist(),
        "fg_longest": sub["fg_longest"],
        "punt": [sub["punt"][0].tolist(), np.round(sub["punt"][1], 2).tolist()],
        "kickoff_start": {str(k): v for k, v in sub["kickoff_start"].items()},
        "clock": sub["clock"],
    }


def submodels_from_json(d):
    return {
        "conversion": _logistic(d["conversion"]),
        "fg": _logistic(d["fg"]),
        "gains": np.array(d["gains"]),
        "fg_longest": d["fg_longest"],
        "punt": (np.array(d["punt"][0]), np.array(d["punt"][1])),
        "kickoff_start": {int(k): v for k, v in d["kickoff_start"].items()},
        "clock": d["clock"],
    }
