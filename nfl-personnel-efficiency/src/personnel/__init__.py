"""Personnel groupings from nflverse participation strings."""

import re

import numpy as np
import pandas as pd

OFFENSIVE_LINE = {"OL", "C", "G", "T"}
DEFENSIVE_LINE = {"DL", "DT", "DE", "NT"}  # brought in to block, so they count as linemen
BACKS = {"RB", "FB", "HB"}
KICKING = {"K", "P", "LS"}
DEFENSIVE_BACKS = {"DB", "CB", "S", "SS", "FS"}
SITUATIONS = [
    "1st down",
    "2nd & 7+",
    "2nd & 3-6",
    "2nd & 1-2",
    "3rd/4th & 7+",
    "3rd/4th & 3-6",
    "3rd/4th & 1-2",
    "Inside the 5",
]
MAIN = ["11", "12", "13", "21", "22"]
HEAVIER = MAIN[1:]
WINDOW = (0.10, 0.90)  # Win probability where both teams still play normally; outside it is garbage time
MIN_SHARE = 0.005  # groupings under 0.5% of plays are pooled as Other
MIN_PLAYS = 50  # in one situation, groupings with fewer plays there are pooled as Other


def counts(personnel: str | None) -> dict[str, int] | None:
    """'1 RB, 2 TE, 2 WR' -> {'RB': 1, 'TE': 2, 'WR': 2}. None if empty or unreadable."""
    if not isinstance(personnel, str) or not personnel.strip():
        return None
    out: dict[str, int] = {}
    for token in personnel.split(","):
        match = re.fullmatch(r"\s*(\d+)\s+(\w+)\s*", token)
        if match is None:
            return None
        out[match[2]] = out.get(match[2], 0) + int(match[1])
    return out


def grouping(personnel: str | None) -> str | None:
    """'1 C, 2 G, 1 QB, 1 RB, 2 T, 2 TE, 2 WR' or '1 RB, 2 TE, 2 WR' -> '12'.

    Backs (RB + FB) then tight ends. Six or more linemen adds '(6 OL)', a second quarterback
    '(2 QB)'. Through 2022 the five linemen are implied, so a string with none means five.
    A kicker, punter or long snapper makes it 'Special Teams' (a fake).
    """
    c = counts(personnel)
    if c is None:
        return None
    if KICKING & c.keys():
        return "Special Teams"
    backs = sum(c.get(p, 0) for p in BACKS)
    line = (sum(c.get(p, 0) for p in OFFENSIVE_LINE) or 5) + sum(c.get(p, 0) for p in DEFENSIVE_LINE)
    extras = ["6 OL"] * (line >= 6) + ["2 QB"] * (c.get("QB", 0) >= 2)
    code = f"{backs}{c.get('TE', 0)}"
    return f"{code} ({', '.join(extras)})" if extras else code


def defensive_backs(personnel: str | None) -> float:
    """'4 DL, 2 LB, 5 DB' -> 5. NaN if unreadable."""
    c = counts(personnel)
    return np.nan if c is None else float(sum(c.get(p, 0) for p in DEFENSIVE_BACKS))


def situation(down: pd.Series, togo: pd.Series, yardline_100: pd.Series) -> pd.Series:
    """Down and distance bucket, or 'Inside the 5' for any down there."""
    length = np.select([togo <= 2, togo <= 6], ["1-2", "3-6"], "7+")
    prefix = np.where(down >= 3, "3rd/4th & ", "2nd & ")
    bucket = np.where(down == 1, "1st down", np.char.add(prefix.astype(str), length))
    return pd.Series(np.where(yardline_100 <= 5, "Inside the 5", bucket), index=down.index)


def prepare_plays(plays: pd.DataFrame, window=WINDOW, min_share=MIN_SHARE, min_plays=0):
    """Drop garbage time and plays without EPA, then pool groupings under min_share of the plays left, or under
    min_plays of them, into Other.

    Returns the plays and the grouping list: the kept groupings, most used first, then Other.
    """
    kept = plays[plays.wp.between(*window) & plays.epa.notna()]
    share, count = kept.grouping.value_counts(normalize=True), kept.grouping.value_counts()
    groupings = [g for g in share.index if share[g] >= min_share and count[g] >= min_plays]
    return kept.assign(grouping=kept.grouping.where(kept.grouping.isin(groupings), "Other")), [*groupings, "Other"]
