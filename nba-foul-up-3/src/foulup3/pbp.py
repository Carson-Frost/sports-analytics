"""Turn NBA Stats play-by-play into segments of the 4th quarter and overtime.

A segment runs from the moment a team gets the ball (or keeps it after an offensive rebound or a
foul with no free throws) to the next of those moments. It records the score and clock at the
start, what happened, how the margin changed, who has the ball next, and the seconds it took.
"""

import re

import pandas as pd

# fouls the defense commits away from a shot: the ones a team uses to stop the clock on purpose
STOP_FOULS = {
    "Personal",
    "Loose Ball",
    "Personal Take",
    "Transition Take",
    "Clear Path",
    "Flagrant Type 1",
    "Flagrant Type 2",
}
FT_PATTERN = re.compile(r"(\d) of (\d)")
TEAM_FOULS = re.compile(r"\.(?:T(\d+)|(PN))\)")  # "(P2.T3)": 3rd team foul, "(P2.PN)": penalty
BUZZER = 0.5  # a rebound with less than this left ends the trip: there is no time to use it
LAST_TWO_MINUTES = 120


def seconds(clock: str) -> float:
    minutes, secs = re.match(r"PT(\d+)M([\d.]+)S", clock).groups()
    return int(minutes) * 60 + float(secs)


def scores(pbp: pd.DataFrame) -> pd.DataFrame:
    """Home and away points after every row. The score is read only from scoring rows (it is
    blank or "0" elsewhere) and carried forward as a running maximum, because a few games have
    scoring rows out of order or a stray "1-0" on a technical free throw."""
    made_ft = (pbp.actionType == "Free Throw") & ~pbp.description.str.startswith("MISS")
    scoring = (pbp.actionType == "Made Shot") | made_ft
    return pd.DataFrame(
        {
            side: pd.to_numeric(pbp[col].where(scoring), errors="coerce").fillna(0).cummax()
            for side, col in [("home_pts", "scoreHome"), ("away_pts", "scoreAway")]
        }
    )


def events(pbp: pd.DataFrame, names: dict[int, str]) -> pd.DataFrame:
    """4th quarter and overtime rows with the clock in seconds, a team for every row that has
    one, and the score after each row."""
    e = pbp.copy()
    e[["home_pts", "away_pts"]] = scores(pbp)
    team = e.teamId.where(e.teamId != 0)
    # team rebounds and team turnovers have no teamId, only the nickname in the description
    for team_id, name in names.items():
        named = e.description.str.lower().str.contains(name, regex=False)
        team = team.mask(team.isna() & named, team_id)
    e["team"] = team
    e["t"] = e.clock.map(seconds)
    e = e[e.period >= 4]
    return e.assign(margin_home=e.home_pts - e.away_pts)[
        ["period", "t", "team", "actionType", "subType", "description", "shotValue", "margin_home"]
    ]


def game_segments(pbp: pd.DataFrame, home: int, away: int, names: dict[int, str]) -> list[dict]:
    e = events(pbp, names)
    out = []
    for _, period in e.groupby("period", sort=True):
        out += period_segments(period.itertuples(index=False), home, away)
    return out


def period_segments(rows, home: int, away: int) -> list[dict]:
    other = {home: away, away: home}
    segs: list[dict] = []
    seg = None
    margin_home = None  # margin after the previous row
    pending = None  # "miss" or "ftmiss" while waiting for a rebound
    dead_ball = False  # a missed first free throw: the next team rebound is not real
    away_from_play = None  # the team owed a free throw that keeps the ball
    fouls = {team: dict(count=0, penalty=False, late=False) for team in (home, away)}
    limit = 5  # the 5th team foul of a quarter is a penalty foul, the 4th in overtime

    def margin(team):
        return margin_home if team == home else -margin_home

    def in_bonus(team, t):
        """Would this team's next foul send the other team to the line?"""
        f = fouls[team]
        return f["penalty"] or f["count"] >= limit - 1 or (t <= LAST_TWO_MINUTES and f["late"])

    def start(t, team, how):
        nonlocal seg
        seg = dict(
            t0=t,
            off=team,
            d0=margin(team),
            how=how,
            shot=False,
            foul=None,
            foul_t=None,
            fta=0,
            ftm=0,
            made=None,
            n=0,
            bonus=in_bonus(other[team], t),
        )

    def close(t, result, nxt):
        nonlocal seg, pending
        if seg is not None:
            segs.append(
                dict(seg, t1=t, delta=margin(seg["off"]) - seg["d0"], result=result, next=nxt)
            )
        seg, pending = None, None

    def settle_foul():
        """A stop foul with no free throws after it: the offense inbounds again."""
        if seg is not None and seg["foul_t"] is not None and seg["fta"] == 0:
            team, t = seg["off"], seg["foul_t"]
            close(t, "foul", "same")
            start(t, team, "foul")

    for r in rows:
        kind, team, t = r.actionType, r.team, r.t
        if kind == "period":
            margin_home = r.margin_home
            limit = 4 if r.period > 4 else 5
            if r.subType == "end":
                close(0.0, pending or "clock", "none")
            continue
        if margin_home is None:
            margin_home = r.margin_home
        relevant = kind in ("Made Shot", "Missed Shot", "Free Throw", "Rebound", "Turnover")
        if relevant and not (kind == "Free Throw" and seg is not None and team == seg["off"]):
            settle_foul()

        if kind in ("Made Shot", "Missed Shot") and pd.notna(team):
            if seg is None or seg["off"] != team:
                close(t, "lost", "other")
                start(t, team, "unknown")
            seg["shot"], seg["n"] = True, seg["n"] + 1
            if kind == "Made Shot":
                margin_home = r.margin_home
                seg["made"] = int(r.shotValue)
                close(t, f"made{int(r.shotValue)}", "other")
                start(t, other[team], "change")
            else:
                pending = "miss"

        elif kind == "Free Throw" and pd.notna(team):
            sub = r.subType
            if (
                "Technical" in sub
                or "Flagrant" in sub
                or "Clear Path" in sub
                or team == away_from_play
            ):
                margin_home = r.margin_home  # points count, the ball does not change
                away_from_play = None
                continue
            k, n = (int(x) for x in FT_PATTERN.search(sub).groups())
            fresh = seg is not None and seg["n"] == 0 and seg["t0"] == t
            if (
                fresh
                and seg["off"] == other[team]
                and segs
                and segs[-1]["off"] == team
                and segs[-1]["result"].startswith("made")
                and n == 1
            ):
                seg = {
                    key: v
                    for key, v in segs.pop().items()  # and-one: reopen the basket
                    if key not in ("t1", "delta", "result", "next")
                }
            if seg is None or seg["off"] != team:
                close(t, "lost", "other")
                start(t, team, "unknown")
            margin_home = r.margin_home
            made = not r.description.startswith("MISS")
            seg["fta"], seg["ftm"], seg["n"] = seg["fta"] + 1, seg["ftm"] + made, seg["n"] + 1
            dead_ball = not made and k < n
            if k == n:
                if made:
                    close(t, "ft", "other")
                    start(t, other[team], "change")
                else:
                    pending = "ftmiss"

        elif kind == "Rebound":
            if dead_ball and r.description.endswith("Rebound"):
                dead_ball = False
                continue
            dead_ball = False
            if pd.isna(team) or pending is None:
                continue  # a rebound with no live miss before it carries no information
            offense = seg["off"]
            if t < BUZZER:  # the trip ends with the miss: nobody has time to use the rebound
                close(t, pending, "none")
                start(t, team, "oreb" if team == offense else "change")
            elif team == offense:
                close(t, f"{pending} oreb", "same")
                start(t, team, "oreb")
            else:
                close(t, f"{pending} dreb", "other")
                start(t, team, "change")

        elif kind == "Turnover" and pd.notna(team):
            if seg is not None and seg["off"] == team:
                close(t, "turnover", "other")
            else:
                close(t, "lost", "other")
            start(t, other[team], "change")

        if kind == "Foul" and pd.notna(team):
            tag = TEAM_FOULS.search(r.description)
            if tag:
                f = fouls[team]
                f["count"] = int(tag.group(1)) if tag.group(1) else f["count"] + 1
                f["penalty"] = f["penalty"] or bool(tag.group(2))
                f["late"] = f["late"] or t <= LAST_TWO_MINUTES
            if r.subType == "Away From Play" and t <= LAST_TWO_MINUTES:
                away_from_play = other[team]  # one free throw, and the fouled team keeps the ball

        if kind == "Foul" and pd.notna(team) and seg is not None and team != seg["off"]:
            if (
                r.subType in STOP_FOULS | {"Shooting"}
                and seg["foul_t"] is not None
                and seg["fta"] == 0
            ):
                settle_foul()  # a foul to give, then this foul: two segments, not one
            if r.subType == "Shooting":
                seg["shot"] = True
            elif r.subType in STOP_FOULS and not seg["shot"] and seg["foul"] is None:
                seg["foul"], seg["foul_t"] = r.subType, t
            seg["n"] += 1

        if seg is None and pd.notna(team) and relevant:
            start(t, team, "unknown")
    return segs


def label(result: str, made, fta: int, ftm: int, nxt: str) -> str:
    """A plain name for what happened in a segment."""
    if made == 3 or result == "made3":
        return "Made Three"
    if made == 2 or result == "made2":
        return "Made Two"
    if result == "turnover":
        return "Turnover"
    if result == "foul":
        return "Foul, No Free Throws"
    if fta == 3:
        return "Fouled on a Three"
    if fta:
        return (
            "Missed Free Throw, Offensive Rebound"
            if nxt == "same"
            else f"Free Throws, {ftm} of {fta}"
        )
    if result == "miss oreb":
        return "Miss, Offensive Rebound"
    if result == "miss dreb":
        return "Miss, Defensive Rebound"
    return "Late Miss, Time Runs Out" if result == "miss" else "Clock Runs Out"


def action(seg: dict) -> str:
    """What the defense did: a stop foul before any shot that sent the offense to the line is
    "foul", one with no free throws is "foul to give", anything else is "defend"."""
    if seg["foul"] is None:
        return "defend"
    return "foul" if seg["fta"] > 0 else "foul to give"
