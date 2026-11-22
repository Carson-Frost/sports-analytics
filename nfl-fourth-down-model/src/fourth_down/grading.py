"""Grade fourth downs: each Choice's Win probability, the Call, and whether it is Clear."""

import numpy as np
import pandas as pd

from fourth_down.decide import (
    CHOICES,
    FG_OFFSET,
    MAX_DISTANCE,
    N_QUANTILES,
    TD_POINTS,
    conversion_x,
    fg_x,
    fit_submodels,
)

CLEAR = 0.01  # a Clear Call beats the next-best Choice by at least a point across the middle 90% of Resamples
LOW, HIGH = 5, 95  # percentiles of a 90% interval
RESAMPLES = 200


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
        return cls(model, fit_submodels(plays), [fit_submodels(_redraw(plays, d)) for d in draws], draws)

    def grade(self, states):
        """One row per state: each Choice's Win probability; the Call, its gap over the next-best Choice
        with a 90% interval, and Clear; the Go edge with its interval, Clear for go and Clear for a kick.

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
            call=np.array(CHOICES)[best], gap=wp[rows, best] - wp[rows, runner_up])
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
