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


def first_down_wp(model, score, game_s, half_s, yardline, receive_2h, timeouts, opp_timeouts, spread, home):
    """Win probability on first down for the team with the ball."""
    yardline = np.clip(yardline, 1, 99)
    x = pd.DataFrame({
        "score_differential": score, "game_seconds_remaining": game_s, "half_seconds_remaining": half_s,
        "yardline_100": yardline, "down": 1, "ydstogo": np.minimum(10, yardline),
        "posteam_timeouts_remaining": timeouts, "defteam_timeouts_remaining": opp_timeouts,
        "receive_2h_ko": receive_2h, "posteam_spread": spread, "home": home,
    })
    return model.predict_proba(x)[:, 1]


def choice_wp(model, sub, s):
    """Win probability of going for it, kicking and punting at each fourth down in `s`."""
    n = len(s)
    score = s.score_differential.to_numpy()
    yl = s.yardline_100.to_numpy()
    togo = s.ydstogo.to_numpy()
    to_us = s.posteam_timeouts_remaining.to_numpy()
    to_them = s.defteam_timeouts_remaining.to_numpy()
    spread = s.posteam_spread.to_numpy()
    home = s.home.to_numpy().astype(float)
    us_2h = s.receive_2h_ko.to_numpy()
    them_2h = ((s.game_seconds_remaining.to_numpy() > 1800) & (us_2h == 0)).astype(int)
    ko = s.season.map(sub["kickoff_start"]).to_numpy()
    game0 = s.game_seconds_remaining.to_numpy()
    half0 = s.half_seconds_remaining.to_numpy()

    def clock(outcome):
        used = sub["clock"][outcome]
        return np.maximum(game0 - used, 0), np.maximum(half0 - used, 0)

    def theirs(points, yardline, outcome, rows=slice(None)):
        """Our win probability when they get the ball at `yardline` (their distance to goal)."""
        g, h = clock(outcome)
        return 1 - first_down_wp(model, -score[rows] - points, g[rows], h[rows], yardline, them_2h[rows],
                                 to_them[rows], to_us[rows], -spread[rows], 1 - home[rows])

    # go for it
    p_go = sub["conversion"].predict_proba(conversion_x(togo, yl))[:, 1]
    gains = sub["gains"][np.minimum(togo, MAX_DISTANCE).astype(int) - 1]
    spot = (yl[:, None] - gains).ravel()
    rep = np.repeat(np.arange(n), N_QUANTILES)
    g, h = clock("convert")
    after_first = first_down_wp(model, score[rep], g[rep], h[rep], spot, us_2h[rep], to_us[rep],
                                to_them[rep], spread[rep], home[rep])
    td = theirs(TD_POINTS, ko, "convert")
    convert = np.where(spot <= 0, td[rep], after_first).reshape(n, N_QUANTILES).mean(axis=1)
    go = p_go * convert + (1 - p_go) * theirs(0, 100 - yl, "fail")

    # field goal
    # no make chance beyond the longest attempt in the data: the curve can't be trusted out there
    distance = yl + FG_OFFSET
    p_fg = np.where(distance > sub["fg_longest"], 0.0, sub["fg"].predict_proba(fg_x(distance, s.season))[:, 1])
    kick_spot = yl + FG_OFFSET - 10
    fg = p_fg * theirs(3, ko, "fg_made") + (1 - p_fg) * theirs(0, 100 - np.maximum(kick_spot, 20), "fg_missed")

    # punt: average over where punts from here really leave them
    centers, takeovers = sub["punt"]
    idx = np.abs(centers[None, :] - yl[:, None]).argmin(axis=1)
    spots = takeovers[idx].ravel()
    punt = theirs(0, spots, "punt", rows=rep).reshape(n, N_QUANTILES).mean(axis=1)

    return pd.DataFrame({"go": go, "fg": fg, "punt": punt}, index=s.index)


def grade(wp, coach):
    """Best choice, how much the coach's choice gave up, and how far going for it is ahead."""
    best = wp[CHOICES].idxmax(axis=1)
    chosen = wp[CHOICES].to_numpy()[np.arange(len(wp)), coach.map(CHOICES.index).to_numpy()]
    return pd.DataFrame({
        "best": best,
        "cost": wp[CHOICES].max(axis=1) - chosen,
        "go_edge": wp.go - wp[["fg", "punt"]].max(axis=1),
    }, index=wp.index)


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
