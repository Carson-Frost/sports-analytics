"""Turn NBA Stats play-by-play into possessions of the 4th quarter and overtime.

A possession runs from the moment a team gets the ball (or keeps it after an offensive rebound or a
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
TIMEOUT_RULES_FROM = "2017-18"  # 7 a game, at most 4 in the 4th, at most 2 in its last 3 minutes
MAX_SECONDS = 24  # only possessions that start with 24 seconds or less left in the period


def parse_game(
    pbp: pd.DataFrame, log: pd.DataFrame, game_id: str, season: str
) -> tuple[dict, list[dict]]:
    """One game's record and its finished possessions, from its raw play-by-play rows and its two
    game-log rows. Only possessions starting with MAX_SECONDS or less left are kept. From
    TIMEOUT_RULES_FROM each carries the timeouts both teams have left, before that None."""
    pbp = _clean(pbp)
    home_id = _home_team(pbp)
    h, a = log[log.TEAM_ID == home_id].iloc[0], log[log.TEAM_ID != home_id].iloc[0]
    final = tuple(_scores(pbp).iloc[-1])
    game = dict(game_id=game_id, season=season, date=h.GAME_DATE, home=h.TEAM_ID,
                away=a.TEAM_ID, home_team=h.TEAM_NAME, away_team=a.TEAM_NAME,
                home_pts=h.PTS, away_pts=a.PTS,
                score_matches=final == (h.PTS, a.PTS))  # fmt: skip
    # team rebounds and turnovers are written with the nickname: "CELTICS Rebound"
    names = {t.TEAM_ID: t.TEAM_NAME.split()[-1].lower() for t in (h, a)}
    other = {h.TEAM_ID: a.TEAM_ID, a.TEAM_ID: h.TEAM_ID}
    left = {team: _timeouts_left(pbp, nick) for team, nick in names.items()}
    known = season >= TIMEOUT_RULES_FROM
    rows = []
    for p in _game_possessions(pbp, h.TEAM_ID, a.TEAM_ID, names):
        if p["t0"] > MAX_SECONDS:
            continue
        timeouts = left[p["off"]](p["period"], p["t0"]) if known else None
        opp = left[other[p["off"]]](p["period"], p["t0"]) if known else None
        row = dict(p, game_id=game_id, season=season, action=_action(p),
                   label=_label(p["result"], p["made"], p["fta"], p["ftm"], p["next"]),
                   timeouts=timeouts, opp_timeouts=opp)  # fmt: skip
        del row["n"], row["foul_t"]
        rows.append(row | {k: int(row[k]) for k in ("d0", "delta", "off")})
    return game, rows


def _seconds(clock: str) -> float:
    minutes, secs = re.match(r"PT(\d+)M([\d.]+)S", clock).groups()
    return int(minutes) * 60 + float(secs)


def _timeouts_left(pbp: pd.DataFrame, nickname: str):
    """A function giving how many timeouts a team has left at a period and clock time, under the
    rules since 2017-18: 7 a game, at most 4 in the 4th quarter and at most 2 in its last 3
    minutes, 2 in each overtime. Timeout rows name the team: "CELTICS Timeout: Regular"."""
    to = pbp[(pbp.actionType == "Timeout") & ~pbp.description.str.contains("Official")]
    mine = to[to.description.str.lower().str.split("timeout").str[0].str.contains(nickname)]
    period, t = mine.period.to_numpy(), mine.clock.map(_seconds).to_numpy()

    def left(p: int, now: float) -> int:
        before = (period < p) | ((period == p) & (t > now))
        this = before & (period == p)
        if p > 4:
            return max(2 - this.sum(), 0)
        late = this & (t <= 180)
        return max(min(7 - (before & (period <= 4)).sum(), 4 - this.sum(), 2 - late.sum()), 0)

    return left


def _scores(pbp: pd.DataFrame) -> pd.DataFrame:
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


def _clean(pbp: pd.DataFrame) -> pd.DataFrame:
    """Some 2000s games pad the text columns with spaces ("Made Shot      ")."""
    pbp = pbp.copy()
    for col in ["actionType", "subType", "description"]:
        pbp[col] = pbp[col].fillna("").astype(str).str.strip()
    return pbp


def _home_team(pbp: pd.DataFrame) -> int:
    """The team whose baskets raise scoreHome. The game log can't say for neutral-site games
    (Paris, Mexico City, the NBA Cup in Las Vegas), where both rows read "away @ home"."""
    made = pbp[pbp.actionType == "Made Shot"]
    raised = pd.to_numeric(made.scoreHome, errors="coerce").diff().fillna(1) > 0
    return int(made.teamId[raised].mode().iloc[0])


def _events(pbp: pd.DataFrame, names: dict[int, str]) -> pd.DataFrame:
    """4th quarter and overtime rows with the clock in seconds, a team for every row that has
    one, and the score after each row."""
    e = pbp.copy()
    e[["home_pts", "away_pts"]] = _scores(pbp)
    team = e.teamId.where(e.teamId != 0)
    # team rebounds and team turnovers have no teamId, only the nickname in the description
    for team_id, name in names.items():
        named = e.description.str.lower().str.contains(name, regex=False)
        team = team.mask(team.isna() & named, team_id)
    e["team"] = team
    e["t"] = e.clock.map(_seconds)
    e = e[e.period >= 4]
    return e.assign(margin_home=e.home_pts - e.away_pts)[
        ["period", "t", "team", "actionType", "subType", "description", "shotValue", "margin_home"]
    ]


def _game_possessions(pbp: pd.DataFrame, home: int, away: int, names: dict[int, str]) -> list[dict]:
    e = _events(pbp, names)
    out = []
    for number, period in e.groupby("period", sort=True):
        possessions = _period_possessions(period.itertuples(index=False), home, away)
        out += [dict(s, period=int(number)) for s in possessions]
    return out


def _period_possessions(rows, home: int, away: int) -> list[dict]:
    other = {home: away, away: home}
    possessions: list[dict] = []
    poss = None
    margin_home = None  # margin after the previous row
    pending = None  # "miss" or "ftmiss" while waiting for a rebound
    dead_ball = False  # a missed first free throw: the next team rebound is not real
    away_from_play = None  # the team owed a free throw that keeps the ball
    fouls = {team: dict(count=0, penalty=False, late=False) for team in (home, away)}
    limit = 5  # the 5th team foul of a quarter is a penalty foul, the 4th in overtime

    def margin(team):
        return margin_home if team == home else -margin_home

    def in_penalty(team, t):
        """Would this team's next foul send the other team to the line?"""
        f = fouls[team]
        return f["penalty"] or f["count"] >= limit - 1 or (t <= LAST_TWO_MINUTES and f["late"])

    def start(t, team, how):
        nonlocal poss
        poss = dict(
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
            penalty=in_penalty(other[team], t),
        )

    def close(t, result, nxt):
        nonlocal poss, pending
        if poss is not None:
            possessions.append(
                dict(poss, t1=t, delta=margin(poss["off"]) - poss["d0"], result=result, next=nxt)
            )
        poss, pending = None, None

    def settle_foul():
        """A stop foul with no free throws after it: the offense inbounds again."""
        if poss is not None and poss["foul_t"] is not None and poss["fta"] == 0:
            team, t = poss["off"], poss["foul_t"]
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
        if relevant and not (kind == "Free Throw" and poss is not None and team == poss["off"]):
            settle_foul()

        if kind in ("Made Shot", "Missed Shot") and pd.notna(team):
            if poss is None or poss["off"] != team:
                close(t, "lost", "other")
                start(t, team, "unknown")
            poss["shot"], poss["n"] = True, poss["n"] + 1
            if kind == "Made Shot":
                margin_home = r.margin_home
                poss["made"] = int(r.shotValue)
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
            fresh = poss is not None and poss["n"] == 0 and poss["t0"] == t
            if (
                fresh
                and poss["off"] == other[team]
                and possessions
                and possessions[-1]["off"] == team
                and possessions[-1]["result"].startswith("made")
                and n == 1
            ):
                poss = {
                    key: v
                    for key, v in possessions.pop().items()  # and-one: reopen the basket
                    if key not in ("t1", "delta", "result", "next")
                }
            if poss is None or poss["off"] != team:
                close(t, "lost", "other")
                start(t, team, "unknown")
            margin_home = r.margin_home
            made = not r.description.startswith("MISS")
            poss["fta"], poss["ftm"], poss["n"] = poss["fta"] + 1, poss["ftm"] + made, poss["n"] + 1
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
            offense = poss["off"]
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
            if poss is not None and poss["off"] == team:
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

        if kind == "Foul" and pd.notna(team) and poss is not None and team != poss["off"]:
            if (
                r.subType in STOP_FOULS | {"Shooting"}
                and poss["foul_t"] is not None
                and poss["fta"] == 0
            ):
                settle_foul()  # a foul to give, then this foul: two possessions, not one
            if r.subType == "Shooting":
                poss["shot"] = True
            elif r.subType in STOP_FOULS and not poss["shot"] and poss["foul"] is None:
                poss["foul"], poss["foul_t"] = r.subType, t
            poss["n"] += 1

        if poss is None and pd.notna(team) and relevant:
            start(t, team, "unknown")
    return possessions


def _label(result: str, made, fta: int, ftm: int, nxt: str) -> str:
    """A plain name for what happened in a possession."""
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


def _action(poss: dict) -> str:
    """What the defense did: a stop foul before any shot that sent the offense to the line is
    "foul", one with no free throws is "foul to give", anything else is "defend"."""
    if poss["foul"] is None:
        return "defend"
    return "foul" if poss["fta"] > 0 else "foul to give"
