"""Turn one game's play-by-play into stints: stretches of play with the same ten players."""

import re
import unicodedata

SUB = re.compile(r"SUB: (.+?) FOR (.+)")
CLOCK = re.compile(r"PT(\d+)M([\d.]+)S")
# events that put a player on the floor when they happen (steals and blocks have an empty type)
ON_FLOOR = {
    "Made Shot",
    "Missed Shot",
    "Free Throw",
    "Rebound",
    "Turnover",
    "Foul",
    "Jump Ball",
    "Violation",
    "Heave",
    "",
}


def elapsed(period, clock):
    """Seconds of game time elapsed at this clock reading."""
    m = CLOCK.match(clock)
    left = int(m[1]) * 60 + float(m[2])
    if period <= 4:
        return 720 * period - left
    return 2880 + 300 * (period - 4) - left


def minutes_to_seconds(text):
    """Box score minutes, '34:21', as seconds. Empty means did not play."""
    if not text:
        return 0.0
    mins, secs = text.split(":")
    return int(mins) * 60 + float(secs)


def plain(text):
    """Strip accents, since substitution text writes 'Doncic' where the box score has 'Dončić'."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().strip()


def roster(box):
    """Players who got on the floor, from boxscoretraditionalv3: {person_id: info}."""
    b = box["boxScoreTraditional"]
    players = {}
    for side in ("homeTeam", "awayTeam"):
        team = b[side]
        for p in team["players"]:
            seconds = minutes_to_seconds(p["statistics"]["minutes"])
            if seconds > 0:
                players[p["personId"]] = {
                    "team": team["teamId"],
                    "name": f"{p['firstName']} {p['familyName']}".strip(),
                    "first": plain(p["firstName"]),
                    "family": plain(p["familyName"]),
                    "keys": {
                        plain(k)
                        for k in (
                            p["familyName"],
                            p["nameI"],
                            f"{p['firstName']} {p['familyName']}",
                        )
                    },
                    "starter": p["position"] != "",
                    "seconds": seconds,
                }
    return b["homeTeamId"], b["awayTeamId"], players


def names_match(name, p):
    """'Black', 'A. Black' or 'Anthony Black' all mean him, and so does 'St. Curry' for Stephen
    when two Currys share 'S. Curry'."""
    if name in p["keys"]:
        return True
    start, _, family = name.partition(". ")
    return family == p["family"] and p["first"].startswith(start)


def resolve(name, team, players, exclude=()):
    """The one player on `team` the substitution text `name` can mean, or None."""
    name = plain(name)
    team_players = {
        pid: p for pid, p in players.items() if p["team"] == team and pid not in exclude
    }
    hits = [pid for pid, p in team_players.items() if names_match(name, p)]
    if not hits:  # given name first in the box score, family name in the text: 'Hansen' for Yang
        hits = [pid for pid, p in team_players.items() if p["first"] == name]
    return hits[0] if len(hits) == 1 else None


def is_on_floor_event(a, players):
    return (
        a["actionType"] in ON_FLOOR and a["personId"] in players and "Technical" not in a["subType"]
    )


def period_starters(actions, team, players, previous):
    """The five players on `team` at the start of a period.

    A player who is subbed out, or does something, before being subbed in started the period.
    Players who played the whole period without a recorded event are filled from `previous`
    (the five at the end of the last period, or the box score starters), then by minutes.
    """
    first = {}
    for a in actions:
        if a["actionType"] == "Substitution" and a["teamId"] == team:
            first.setdefault(a["personId"], "on")
            m = SUB.match(a["description"])
            incoming = m and resolve(m[1], team, players)
            if incoming:
                first.setdefault(incoming, "in")
        elif is_on_floor_event(a, players) and players[a["personId"]]["team"] == team:
            first.setdefault(a["personId"], "on")
    starters = [pid for pid, how in first.items() if how == "on" and pid in players][:5]
    if len(starters) < 5:
        silent = [pid for pid, p in players.items() if p["team"] == team and pid not in first]
        silent.sort(key=lambda pid: (pid not in previous, -players[pid]["seconds"]))
        starters += silent[: 5 - len(starters)]
    return set(starters)


def game_stints(pbp, box):
    """One row per stint: who was on the floor, how long, possessions and points each side."""
    home, away, players = roster(box)
    actions = pbp["game"]["actions"]
    for a in actions:  # also know each player by the play-by-play's own names for him
        if a["personId"] in players:
            keys = players[a["personId"]]["keys"]
            keys |= {plain(a["playerName"]), plain(a["playerNameI"])}
            m = SUB.match(a["description"]) if a["actionType"] == "Substitution" else None
            if m:
                keys.add(plain(m[2]))
    periods = sorted({a["period"] for a in actions})
    # box score starters only break ties: before 2015-16 every player in it has a position
    on = {
        t: {pid for pid, p in players.items() if p["team"] == t and p["starter"]}
        for t in (home, away)
    }
    stints = []
    score = {home: 0, away: 0}

    for period in periods:
        acts = [a for a in actions if a["period"] == period]
        on = {t: period_starters(acts, t, players, on[t]) for t in (home, away)}
        start = elapsed(period, "PT12M00.00S" if period <= 4 else "PT05M00.00S")
        stint = new_stint(on, home, away, start)
        last_miss = None

        for a in acts:
            t = elapsed(period, a["clock"])
            if a["actionType"] == "Substitution":
                m = SUB.match(a["description"])
                team = a["teamId"]
                if team not in on or a["personId"] not in on[team] or not m:
                    continue  # a substitution we cannot place is skipped, not guessed
                incoming = resolve(m[1], team, players, exclude=on[team])
                if not incoming:
                    continue
                stints.append(close(stint, t))
                on[team] = (on[team] - {a["personId"]}) | {incoming}
                stint = new_stint(on, home, away, t)
                continue

            team = a["teamId"]
            if team not in (home, away):
                continue
            side = "h" if team == home else "a"
            if a["isFieldGoal"] == 1:
                stint[side + "fga"] += 1
                last_miss = team if a["shotResult"] == "Missed" else None
            elif a["actionType"] == "Free Throw":
                stint[side + "fta"] += 1
            elif a["actionType"] == "Turnover":
                stint[side + "tov"] += 1
            elif a["actionType"] == "Rebound":
                if last_miss == team:
                    stint[side + "oreb"] += 1
                last_miss = None
            # older seasons write 0-0 on every non-scoring row instead of leaving it blank
            if a["scoreHome"] != "" and (a["scoreHome"], a["scoreAway"]) != ("0", "0"):
                new = {home: int(a["scoreHome"]), away: int(a["scoreAway"])}
                stint["home_pts"] += new[home] - score[home]
                stint["away_pts"] += new[away] - score[away]
                score = new
        end = elapsed(period, "PT00M00.00S")
        stints.append(close(stint, end))

    return [s for s in stints if s["seconds"] > 0 or s["home_pts"] or s["away_pts"]]


def new_stint(on, home, away, start):
    s = {
        "start": start,
        "home": sorted(on[home]),
        "away": sorted(on[away]),
        "home_pts": 0,
        "away_pts": 0,
    }
    for side in ("h", "a"):
        for stat in ("fga", "fta", "tov", "oreb"):
            s[side + stat] = 0
    return s


def close(s, end):
    """Finish a stint: possessions are FGA - OREB + TOV + 0.44 FTA, averaged over both teams."""
    poss = [s[x + "fga"] - s[x + "oreb"] + s[x + "tov"] + 0.44 * s[x + "fta"] for x in ("h", "a")]
    return {
        "start": s["start"],
        "seconds": end - s["start"],
        "poss": sum(poss) / 2,
        "home_pts": s["home_pts"],
        "away_pts": s["away_pts"],
        "home": s["home"],
        "away": s["away"],
    }
