"""Snap-by-snap overtime simulation.

A snap resamples one real regulation snap from the same down, distance and field position.
Snaps make drives, drives make an overtime played under one rule set:

- "sudden": any score wins.
- "modified": a touchdown or defensive score on the first possession wins; a first-possession
  field goal gives the other team the ball, then sudden death.
- "both": both teams get the ball (unless the kicking team scores a safety on the first
  possession), then sudden death.

A kickoff recovered by the kicking team, or a muffed punt recovered by the punting team, counts
as the other team's possession (Rule 16-1-5(c)).
"""

import math
import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

PLAY, TD, TURNOVER, SAFETY, DEF_TD = range(5)
FIRST_DOWN, REPEAT_DOWN = 1, 2  # a penalty short of the line to gain that gives these
DIST_EDGES = (1, 2, 4, 7, 10, 15)  # bands: 1, 2, 3-4, 5-7, 8-10, 11-15, 16+
MIN_BUCKET = 30  # fewer real snaps than this and the bucket ignores field position
FG_RANGE = 0.5  # a team that wants a field goal kicks when the make chance is at least this
LAST_KICK = 20  # seconds left when a team that a field goal wins or ties for kicks on any down
HURRY = 120  # seconds left in a timed overtime when the team with the ball starts hurrying


def dist_band(dist):
    return next((i for i, edge in enumerate(DIST_EDGES) if dist <= edge), len(DIST_EDGES))


def field_band(yl):
    return min(int(yl - 1) // 10, 9)


def fit_logistic(x, y, iters=25):
    """Intercept and slope of a one-variable logistic regression, by Newton's method."""
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ beta))
        beta += np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]), X.T @ (y - p))
    return beta


@dataclass
class Model:
    snaps: dict  # (down, distance band, field band) -> [(kind, yards, penalty flag, secs)]
    snaps_pooled: dict  # (down, distance band) -> same, for thin buckets
    fourth: dict  # (distance band, field band) -> ["go" | "punt" | "fg", ...]
    punts: list  # 5-yard band of the punt spot -> [(receiver has ball, spot, td, secs)]
    kickoffs: list  # [(receiver has ball, spot, td, secs)]
    hurry: list  # seconds between snaps in real two-minute drills
    fg: tuple  # logistic intercept and slope on kick distance
    xp: float
    two: float

    def fg_chance(self, yl):
        a, b = self.fg
        return 1 / (1 + math.exp(-(a + b * (yl + 17))))  # 17 = end zone plus the hold

    def snap(self, down, dist, yl, rng):
        key = (down, dist_band(dist))
        options = self.snaps.get(key + (field_band(yl),)) or self.snaps_pooled[key]
        return options[int(rng.random() * len(options))]


def regulation(plays):
    """Snaps from close regulation games, away from the end-of-half hurry-up."""
    return plays[(plays.qtr <= 4) & (plays.score_differential.abs() <= 8)
                 & (plays.half_seconds_remaining > 120)]


def kick_outcomes(kicks, receiving):
    """Kickoffs and punts as (receiving team has the ball, its spot, touchdown, seconds)."""
    out = []
    for row in kicks.itertuples():
        td = 0
        if row.touchdown == 1:
            td = 1 if row.td_team == getattr(row, receiving) else 2
        elif (not isinstance(row.next_posteam, str) or math.isnan(row.next_yardline_100)
              or row.next_play_type in ("kickoff", "punt")):
            continue  # end of a half, a missing spot, or a re-kick after a penalty
        has_ball = row.next_posteam == getattr(row, receiving)
        spot = 0 if td else int(row.next_yardline_100)
        out.append((has_ball, spot, td, min(row.secs, 30)))
    return out


def fit(plays, seasons, kick_seasons=None):
    """Fit every piece of the model on regulation plays from `seasons`.

    Kickoffs can come from other seasons, since the kickoff rule changes more often than
    anything else.
    """
    window = plays[plays.season.isin(seasons) & plays.secs.between(0, 60)]
    close = regulation(window)

    snaps, pooled = defaultdict(list), defaultdict(list)
    scrimmage = close[close.play_type.isin(["run", "pass", "no_play"]) & close.down.between(1, 3)
                      & close.next_yardline_100.notna()]
    for r in scrimmage.itertuples():
        if r.touchdown == 1:
            kind, yards = (TD, 0) if r.td_team == r.posteam else (DEF_TD, 0)
        elif r.safety == 1:
            kind, yards = SAFETY, 0
        elif (r.interception == 1 or r.fumble_lost == 1) and r.next_posteam == r.defteam:
            kind, yards = TURNOVER, int(r.next_yardline_100 - (100 - r.yardline_100))
        elif r.next_posteam == r.posteam:
            kind, yards = PLAY, int(r.yardline_100 - r.next_yardline_100)
        else:
            continue
        flag = 0
        if kind == PLAY and yards < r.ydstogo:  # a penalty: same down replayed, or a first down
            repeat = r.next_down == r.down and r.next_ydstogo == r.ydstogo - yards
            flag = REPEAT_DOWN if repeat else FIRST_DOWN if r.next_down == 1 else 0
        outcome = (kind, yards, flag, r.secs)
        down, dist = int(r.down), dist_band(r.ydstogo)
        snaps[(down, dist, field_band(r.yardline_100))].append(outcome)
        pooled[(down, dist)].append(outcome)
    snaps = {k: v for k, v in snaps.items() if len(v) >= MIN_BUCKET}

    fourth = defaultdict(list)
    choice = {"run": "go", "pass": "go", "punt": "punt", "field_goal": "fg"}
    for r in close[(close.down == 4) & close.play_type.isin(choice)].itertuples():
        fourth[(dist_band(r.ydstogo), field_band(r.yardline_100))].append(choice[r.play_type])

    drill = window[window.qtr.isin([2, 4]) & (window.half_seconds_remaining <= HURRY)
                   & window.score_differential.between(-8, 0)
                   & window.play_type.isin(["run", "pass"])]

    reg = window[window.qtr <= 4]
    punt_rows = reg[reg.play_type == "punt"]
    punts = [[] for _ in range(20)]
    for band, rows in punt_rows.groupby(((punt_rows.yardline_100 - 1) // 5).clip(0, 19)):
        punts[int(band)] = kick_outcomes(rows, "defteam")
    for i in range(18, -1, -1):  # empty bands near the goal line borrow from the next one out
        punts[i] = punts[i] or punts[i + 1]

    kick_window = plays[plays.season.isin(kick_seasons or seasons) & (plays.qtr <= 4)
                        & plays.secs.between(0, 60)]
    kicks = kick_window[(kick_window.play_type == "kickoff") & (kick_window.kick_distance >= 30)]
    kickoffs = kick_outcomes(kicks, "posteam")

    fgs = reg[(reg.play_type == "field_goal") & reg.kick_distance.notna()]
    xps = reg.extra_point_result.dropna()
    twos = reg.two_point_conv_result.dropna()
    return Model(
        snaps=snaps, snaps_pooled=dict(pooled), fourth=dict(fourth), punts=punts,
        kickoffs=kickoffs, hurry=drill.secs.tolist(),
        fg=tuple(fit_logistic(fgs.kick_distance.to_numpy(float),
                              (fgs.field_goal_result == "made").to_numpy(float))),
        xp=float((xps == "good").mean()), two=float((twos == "success").mean()),
    )


def fourth_down(m, dist, yl, mode, rng):
    """'go', 'punt' or 'fg' for one 4th down, given what the score asks of this drive."""
    in_range = m.fg_chance(yl) >= FG_RANGE
    if mode == "need_td":
        return "go"
    if mode == "need_fg":
        return "fg" if in_range else "go"
    if mode == "fg_wins" and in_range:
        return "fg"
    options = m.fourth.get((dist_band(dist), field_band(yl)), ["punt"])
    return options[int(rng.random() * len(options))]


def drive(m, yl, mode, clock, rng):
    """One possession from `yl` yards out. Returns (result, next spot, seconds used).

    result is 'td', 'fg', 'safety', 'def_td' (the other team scores a touchdown), 'keep' (the
    same team gets the ball back, a muffed punt), 'clock' (time ran out) or 'none'. The next spot
    is for whoever has the ball next, in yards from the end zone they attack.
    """
    down, dist, t = 1, min(10, yl), 0.0
    while True:
        if clock is not None and t >= clock:
            return "clock", 0, t
        call = "go"
        last_snap = clock is not None and clock - t <= LAST_KICK
        if last_snap and mode in ("fg_wins", "need_fg") and m.fg_chance(yl) >= FG_RANGE:
            call = "fg"
        elif down == 4:
            call = fourth_down(m, dist, yl, mode, rng)
        if call == "fg":
            made = rng.random() < m.fg_chance(yl)
            return ("fg", 0, t + 5) if made else ("none", min(93 - yl, 80), t + 5)
        if call == "punt":
            band = m.punts[min(int(yl - 1) // 5, 19)]
            has_ball, spot, td, secs = band[int(rng.random() * len(band))]
            if td:
                return ("def_td" if td == 1 else "td"), 0, t + secs
            return ("none" if has_ball else "keep"), spot, t + secs
        kind, yards, flag, secs = m.snap(min(down, 3), dist, yl, rng)
        if clock is not None and clock - t < HURRY:  # in overtime the offense is never ahead
            secs = m.hurry[int(rng.random() * len(m.hurry))]
        t += secs
        if kind == TD:
            return "td", 0, t
        if kind == DEF_TD:
            return "def_td", 0, t
        if kind == SAFETY:
            return "safety", 0, t
        if kind == TURNOVER:
            return "none", min(max(100 - yl + yards, 1), 99), t
        yl -= yards
        if yl <= 0:
            return "td", 0, t
        if yl >= 100:
            return "safety", 0, t
        if yards >= dist or flag == FIRST_DOWN:
            down, dist = 1, min(10, yl)
        elif flag == REPEAT_DOWN:
            dist -= yards
        elif down == 4:
            return "none", 100 - yl, t
        else:
            down, dist = down + 1, dist - yards


def game_over(rule, score, had_ball, result):
    if score[0] == score[1]:
        return False
    if rule == "sudden" or all(had_ball) or result in ("safety", "def_td"):
        return True
    return rule == "modified" and result == "td"


def drive_mode(rule, score, had_ball, team):
    """What the score asks of this drive: need_td, need_fg, fg_wins or normal."""
    lead = score[team] - score[1 - team]
    if lead < 0 and had_ball[1 - team]:  # if this drive fails, the game is over
        return "need_fg" if lead >= -3 else "need_td"
    after_fg = score.copy()
    after_fg[team] += 3
    with_ball = had_ball.copy()
    with_ball[team] = True
    if lead + 3 > 0 and game_over(rule, after_fg, with_ball, "fg"):
        return "fg_wins"
    return "normal"


def kickoff(m, receiving, rng):
    """(team with the ball, its spot, result if the kickoff itself was a touchdown, seconds)."""
    has_ball, spot, td, secs = m.kickoffs[int(rng.random() * len(m.kickoffs))]
    if td:
        return receiving, 0, ("td" if td == 1 else "def_td"), secs
    return (receiving if has_ball else 1 - receiving), spot, None, secs


def overtime(m, rule, clock, rng, go_for_two=False, drive_fn=drive, max_drives=60):
    """One overtime between equal teams. Team 0 receives the opening kickoff.

    Returns (+1 receiver wins, 0 tie, -1 kicker wins) and how the receiver's first drive ended.
    """
    score, had_ball = [0, 0], [False, False]

    def kick_to(receiving):
        team, yl, result, secs = kickoff(m, receiving, rng)
        if team != receiving:  # Rule 16-1-5(c): a kicking-team recovery uses up the chance
            had_ball[receiving] = True
        return team, yl, result, secs

    team, yl, result, t = kick_to(0)
    first = None
    for _ in range(max_drives):
        left = None if clock is None else clock - t
        if result is None:
            mode = drive_mode(rule, score, had_ball, team)
            result, next_yl, secs = drive_fn(m, yl, mode, left, rng)
            t += secs
        if result == "clock":
            break
        had_ball[team] = True
        if result == "keep":  # Rule 16-1-5(c): a muffed punt counts as the receiver's chance
            had_ball[1 - team] = True
        if team == 0 and first is None:
            first = result if result in ("td", "fg") else "none"
        if result == "td":
            score[team] += 6
            behind = score[team] - score[1 - team]
            if go_for_two and behind == -1 and had_ball[1 - team]:
                score[team] += 2 if rng.random() < m.two else 0
            else:
                score[team] += 1 if rng.random() < m.xp else 0
        elif result == "fg":
            score[team] += 3
        elif result == "safety":
            score[1 - team] += 2
        elif result == "def_td":
            score[1 - team] += 7
            had_ball[1 - team] = True
        if game_over(rule, score, had_ball, result):
            break
        if clock is not None and t >= clock:
            break
        if result in ("td", "fg"):
            team, yl, result, secs = kick_to(1 - team)
            t += secs
            if result is None and game_over(rule, score, had_ball, None):
                break  # the kicking team recovered and that was the other team's last chance
        else:
            team = team if result == "keep" else 1 - team
            yl, result = next_yl, None
    return (score[0] > score[1]) - (score[0] < score[1]), first or "none"


def simulate(m, rule, clock, n, seed=2026, go_for_two=False):
    """n overtimes. Returns shares of receiver wins, ties and kicker wins, plus the receiver's
    chance to win (ties count half) after each way its first drive can end."""
    rng = random.Random(seed)
    totals = {1: 0, 0: 0, -1: 0}
    after = defaultdict(lambda: [0, 0.0])  # first drive -> [games, receiver wins]
    for _ in range(n):
        outcome, first = overtime(m, rule, clock, rng, go_for_two)
        totals[outcome] += 1
        after[first][0] += 1
        after[first][1] += (outcome + 1) / 2
    shares = {"receiver_win": totals[1] / n, "tie": totals[0] / n, "kicker_win": totals[-1] / n}
    first = {k: (g / n, w / g) for k, (g, w) in after.items()}
    return shares, first
