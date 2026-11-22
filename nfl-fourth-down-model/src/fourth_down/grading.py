"""Grade fourth downs: each Choice's Win probability, the Call, and whether it is Clear."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from fourth_down.decide import (
    CHOICES,
    FG_OFFSET,
    MAX_DISTANCE,
    N_QUANTILES,
    TD_POINTS,
    conversion_x,
    fg_x,
)

CLEAR = 0.01  # a Clear Call beats the next-best Choice by at least a point across the middle 90% of Resamples
LOW, HIGH = 5, 95  # percentiles of a 90% interval
RESAMPLES = 200


def situation(distance, spot, quarter, clock, score, timeouts=3, opp_timeouts=3, spread=0.0, home=None,
              gets_second_half_kickoff=False, season=2025):
    """One fourth-down state from a game situation, ready to Grade.

    spot is "own 35", "opp 20" or "50"; clock the time left in the quarter, "M:SS"; score the offense
    minus the defense; spread the points the offense was favored by; home 1 or 0, or None when unknown.
    """
    side, _, yards = spot.lower().rpartition(" ")
    minutes, seconds = clock.split(":")
    game = (4 - quarter) * 900 + int(minutes) * 60 + int(seconds)
    return pd.DataFrame({
        "score_differential": score, "game_seconds_remaining": game,
        "half_seconds_remaining": game - 1800 if quarter <= 2 else game,
        "yardline_100": 100 - int(yards) if side == "own" else int(yards), "down": 4, "ydstogo": distance,
        "posteam_timeouts_remaining": timeouts, "defteam_timeouts_remaining": opp_timeouts,
        "receive_2h_ko": int(gets_second_half_kickoff and quarter <= 2), "posteam_spread": float(spread),
        "season": season, "home": np.nan if home is None else float(home),
    }, index=[0])


class Grader:
    """Grades fourth-down states with a Win probability model and fitted submodels.

    point is the submodels fitted on every play, resamples one submodels dict per Resample. draws, when
    known, is each Resample's draw of games: how many times each game_id was drawn.
    """

    def __init__(self, model, point, resamples, draws=None):
        self.model, self.point, self.resamples, self.draws = model, point, resamples, draws

    @classmethod
    def fit(cls, model, plays, resamples=RESAMPLES, seed=2025):
        """Fit the submodels on `plays`, then refit them on each Resample's draw of games."""
        rng = np.random.default_rng(seed)
        games = plays.game_id.unique()
        draws = [pd.Series(rng.choice(games, len(games))).value_counts() for _ in range(resamples)]
        return cls(model, _fit_submodels(plays), [_fit_submodels(_redraw(plays, d)) for d in draws], draws)

    @classmethod
    def from_json(cls, model, saved):
        """A grader from `to_json`'s dict: it Grades exactly as the fitted one did, but has no draws."""
        return cls(model, _submodels_from_json(saved["point"]), [_submodels_from_json(d) for d in saved["resamples"]])

    def to_json(self):
        """The point fit and every Resample's submodels, as a JSON-ready dict."""
        return {"point": _submodels_to_json(self.point), "resamples": [_submodels_to_json(s) for s in self.resamples]}

    def grade(self, states):
        """One row per state: each Choice's Win probability; the Call and the runner-up (the next-best
        Choice), the Call's gap over it with a 90% interval, and Clear; the Go edge with its interval,
        Clear for go and Clear for a kick.

        A state with no home side (home NaN) is averaged over home and away. With a coach column (the
        coach's Choice), cost is the Win probability it gave up.
        """
        return self._grade(states)[0]

    def grade_seasons(self, states):
        """Grade the coaches' fourth downs (states with game_id, season and coach) and sum up each season.

        Returns the Grades and one row per season: fourth downs, Clear calls and their share, the go rate
        in clear-go spots with its 90% interval, how often teams went when a kick was clear, and wins
        lost on Clear calls with its interval. Rates are in percent. In each Resample the go rate weights
        every game by how often that Resample drew it; this needs the draws a fitted grader keeps.
        """
        out, costs = self._grade(states)
        went = (states.coach == "go").reset_index(drop=True)
        game_ids, seasons = states.game_id.to_numpy(), states.season.to_numpy()
        weights = np.array([pd.Series(game_ids).map(d).fillna(0).to_numpy() for d in self.draws])
        rows = []
        for season in np.unique(seasons):
            d = seasons == season
            cg = d & out.clear_go.to_numpy()
            w = weights[:, cg]
            rates = 100 * (w * went[cg].to_numpy()).sum(axis=1) / w.sum(axis=1)
            lost = costs[:, d & out.clear.to_numpy()].sum(axis=1)
            rows.append({
                "season": season,
                "fourth_downs": int(d.sum()),
                "clear_calls": int(out.clear[d].sum()),
                "clear_share": 100 * out.clear[d].mean(),
                "clear_go_spots": int(cg.sum()),
                "go_rate": 100 * went[cg].mean(),
                "go_rate_low": np.percentile(rates, LOW),
                "go_rate_high": np.percentile(rates, HIGH),
                "went_when_clear_kick": 100 * went[d & out.clear_kick.to_numpy()].mean(),
                "wins_lost": float(np.median(lost)),
                "wins_lost_low": np.percentile(lost, LOW),
                "wins_lost_high": np.percentile(lost, HIGH),
            })
        return out, pd.DataFrame(rows)

    def _grade(self, states):
        """The Grades, and with a coach column the cost of the coach's Choice in every Resample."""
        unknown = states.home.isna().to_numpy()
        both = pd.concat([states.assign(home=states.home.fillna(1)), states[unknown].assign(home=0)])
        wp, *draws = [self._choices(sub, both, unknown) for sub in [self.point, *self.resamples]]
        draws = np.array(draws)
        rows = np.arange(len(wp))
        ranked = np.argsort(-wp, axis=1, kind="stable")
        best, runner_up = ranked[:, 0], ranked[:, 1]
        gaps = draws[:, rows, best] - draws[:, rows, runner_up]
        out = pd.DataFrame(wp, columns=CHOICES, index=states.index).assign(
            call=np.array(CHOICES)[best], runner_up=np.array(CHOICES)[runner_up],
            gap=wp[rows, best] - wp[rows, runner_up])
        out["gap_low"], out["gap_high"] = np.percentile(gaps, [LOW, HIGH], axis=0)
        out["clear"] = out.gap_low >= CLEAR
        out["go_edge"] = _go_edge(wp)
        out["edge_low"], out["edge_high"] = np.percentile(_go_edge(draws), [LOW, HIGH], axis=0)
        out["clear_go"] = out.edge_low >= CLEAR
        out["clear_kick"] = out.edge_high <= -CLEAR
        if "coach" not in states:
            return out, None
        chosen = states.coach.map(CHOICES.index).to_numpy()
        out["cost"] = wp.max(axis=1) - wp[rows, chosen]
        return out, draws.max(axis=2) - draws[:, rows, chosen]

    def _choices(self, sub, both, unknown):
        """Each Choice's Win probability, averaged over home and away where the home side is unknown."""
        wp = _choice_wp(self.model, sub, both)[CHOICES].to_numpy()
        own, away = wp[:len(unknown)].copy(), wp[len(unknown):]
        own[unknown] = (own[unknown] + away) / 2
        return own


def _redraw(plays, counts):
    """The plays of each game repeated as often as it was drawn; a game drawn twice becomes two games."""
    out = plays.loc[plays.index.repeat(plays.game_id.map(counts).fillna(0).astype(int))]
    copy = out.groupby(level=0).cumcount().astype(str)
    return out.assign(game_id=out.game_id + "_" + copy).reset_index(drop=True)


def _go_edge(wp):
    """Going for it minus the better kick, on the last axis of Choices."""
    return wp[..., 0] - wp[..., 1:].max(axis=-1)


def _first_down_wp(model, score, game_s, half_s, yardline, receive_2h, timeouts, opp_timeouts, spread, home):
    """Win probability on first down for the team with the ball."""
    yardline = np.clip(yardline, 1, 99)
    x = pd.DataFrame({
        "score_differential": score, "game_seconds_remaining": game_s, "half_seconds_remaining": half_s,
        "yardline_100": yardline, "down": 1, "ydstogo": np.minimum(10, yardline),
        "posteam_timeouts_remaining": timeouts, "defteam_timeouts_remaining": opp_timeouts,
        "receive_2h_ko": receive_2h, "posteam_spread": spread, "home": home,
    })
    return model.predict_proba(x)[:, 1]


def _choice_wp(model, sub, s):
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
        return 1 - _first_down_wp(model, -score[rows] - points, g[rows], h[rows], yardline, them_2h[rows],
                                  to_them[rows], to_us[rows], -spread[rows], 1 - home[rows])

    # go for it
    p_go = sub["conversion"].predict_proba(conversion_x(togo, yl))[:, 1]
    gains = sub["gains"][np.minimum(togo, MAX_DISTANCE).astype(int) - 1]
    spot = (yl[:, None] - gains).ravel()
    rep = np.repeat(np.arange(n), N_QUANTILES)
    g, h = clock("convert")
    after_first = _first_down_wp(model, score[rep], g[rep], h[rep], spot, us_2h[rep], to_us[rep],
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


def _clock_used(plays):
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


def _fit_submodels(plays):
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
        "clock": _clock_used(plays),
    }


def _logistic(params):
    m = LogisticRegression()
    m.classes_ = np.array([False, True])
    m.coef_ = np.array(params["coef"])
    m.intercept_ = np.array(params["intercept"])
    m.n_features_in_ = m.coef_.shape[1]
    return m


def _submodels_to_json(sub):
    return {
        "conversion": {"coef": sub["conversion"].coef_.tolist(), "intercept": sub["conversion"].intercept_.tolist()},
        "fg": {"coef": sub["fg"].coef_.tolist(), "intercept": sub["fg"].intercept_.tolist()},
        "gains": sub["gains"].tolist(),
        "fg_longest": sub["fg_longest"],
        "punt": [sub["punt"][0].tolist(), sub["punt"][1].tolist()],
        "kickoff_start": {str(k): v for k, v in sub["kickoff_start"].items()},
        "clock": sub["clock"],
    }


def _submodels_from_json(d):
    return {
        "conversion": _logistic(d["conversion"]),
        "fg": _logistic(d["fg"]),
        "gains": np.array(d["gains"]),
        "fg_longest": d["fg_longest"],
        "punt": (np.array(d["punt"][0]), np.array(d["punt"][1])),
        "kickoff_start": {int(k): v for k, v in d["kickoff_start"].items()},
        "clock": d["clock"],
    }
