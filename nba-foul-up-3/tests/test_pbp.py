import pandas as pd

from foulup3.pbp import parse_game

HOME, AWAY = 1, 2
GAME = "0022300001"


def pbp(rows):
    cols = [
        "clock",
        "period",
        "teamId",
        "scoreHome",
        "scoreAway",
        "description",
        "actionType",
        "subType",
        "shotValue",
    ]
    return pd.DataFrame([dict(zip(cols, r, strict=True)) for r in rows])


def log(home_pts=0, away_pts=0, matchup=("BOS vs. ATL", "ATL @ BOS")):
    """The two game-log rows of the game, home first."""
    return pd.DataFrame(
        {
            "GAME_ID": GAME,
            "TEAM_ID": [HOME, AWAY],
            "TEAM_NAME": ["Boston Celtics", "Atlanta Hawks"],
            "GAME_DATE": "2024-01-01",
            "MATCHUP": list(matchup),
            "PTS": [home_pts, away_pts],
        }
    )


def run(rows, season="2023-24"):
    return parse_game(pbp(rows), log(), GAME, season)[1]


def test_foul_up_three_then_free_throws_and_a_team_rebound():
    possessions = run(
        [
            ("PT12M00.00S", 4, 0, "", "", "Start of 4th Period", "period", "start", 0),
            (
                "PT00M20.00S",
                4,
                HOME,
                "100",
                "97",
                "Brown 3PT Jump Shot",
                "Made Shot",
                "Jump Shot",
                3,
            ),
            ("PT00M18.50S", 4, HOME, "", "", "Holiday P.FOUL", "Foul", "Personal", 0),
            (
                "PT00M18.50S",
                4,
                AWAY,
                "100",
                "98",
                "Young Free Throw 1 of 2",
                "Free Throw",
                "Free Throw 1 of 2",
                0,
            ),
            (
                "PT00M18.50S",
                4,
                AWAY,
                "",
                "",
                "MISS Young Free Throw 2 of 2",
                "Free Throw",
                "Free Throw 2 of 2",
                0,
            ),
            ("PT00M17.00S", 4, 0, "", "", "HAWKS Rebound", "Rebound", "Unknown", 0),
            ("PT00M00.00S", 4, 0, "", "", "End of 4th Period", "period", "end", 0),
        ]
    )
    foul = next(s for s in possessions if s["t0"] == 20.0 and s["off"] == AWAY)
    assert (foul["off"], foul["d0"], foul["delta"]) == (AWAY, -3, 1)
    assert (foul["fta"], foul["ftm"], foul["result"], foul["next"]) == (2, 1, "ftmiss oreb", "same")
    assert foul["action"] == "foul"
    assert possessions[-1]["how"] == "oreb" and possessions[-1]["result"] == "clock"


def test_and_one_stays_with_the_basket_and_dead_ball_rebound_is_skipped():
    possessions = run(
        [  # a 3rd-quarter basket tells the parser which team is home
            ("PT00M30.00S", 3, HOME, "90", "90", "Brown Dunk", "Made Shot", "Dunk", 2),
            ("PT12M00.00S", 4, 0, "", "", "Start of 4th Period", "period", "start", 0),
            ("PT00M15.00S", 4, AWAY, "90", "92", "Young Layup", "Made Shot", "Layup", 2),
            ("PT00M15.00S", 4, HOME, "", "", "Brown S.FOUL", "Foul", "Shooting", 0),
            (
                "PT00M15.00S",
                4,
                AWAY,
                "90",
                "93",
                "Young Free Throw 1 of 1",
                "Free Throw",
                "Free Throw 1 of 1",
                0,
            ),
            ("PT00M10.00S", 4, AWAY, "", "", "Okongwu S.FOUL", "Foul", "Shooting", 0),
            (
                "PT00M10.00S",
                4,
                HOME,
                "",
                "",
                "MISS Tatum Free Throw 1 of 2",
                "Free Throw",
                "Free Throw 1 of 2",
                0,
            ),
            ("PT00M10.00S", 4, 0, "", "", "CELTICS Rebound", "Rebound", "Unknown", 0),
            (
                "PT00M10.00S",
                4,
                HOME,
                "91",
                "93",
                "Tatum Free Throw 2 of 2",
                "Free Throw",
                "Free Throw 2 of 2",
                0,
            ),
            ("PT00M00.00S", 4, 0, "", "", "End of 4th Period", "period", "end", 0),
        ]
    )
    basket = next(s for s in possessions if s["t0"] == 15.0 and s["off"] == AWAY)
    assert (basket["delta"], basket["made"], basket["result"]) == (3, 2, "ft")
    home = next(s for s in possessions if s["off"] == HOME and s["fta"] == 2)
    assert (home["d0"], home["delta"], home["next"], home["action"]) == (-3, 1, "other", "defend")
    assert possessions[-1]["off"] == AWAY and possessions[-1]["d0"] == 2


def test_foul_with_no_free_throws_keeps_the_ball():
    possessions = run(
        [
            ("PT12M00.00S", 4, 0, "", "", "Start of 4th Period", "period", "start", 0),
            ("PT00M09.00S", 4, HOME, "100", "97", "Brown Dunk", "Made Shot", "Dunk", 2),
            ("PT00M07.00S", 4, HOME, "", "", "Holiday P.FOUL", "Foul", "Personal", 0),
            (
                "PT00M04.00S",
                4,
                AWAY,
                "",
                "",
                "MISS Young 3PT Jump Shot",
                "Missed Shot",
                "Jump Shot",
                3,
            ),
            ("PT00M03.00S", 4, HOME, "", "", "Brown REBOUND", "Rebound", "Unknown", 0),
            ("PT00M00.00S", 4, 0, "", "", "End of 4th Period", "period", "end", 0),
        ]
    )
    give, shot = (s for s in possessions if s["off"] == AWAY)
    assert (give["result"], give["next"], give["action"]) == ("foul", "same", "foul to give")
    assert (shot["t0"], shot["how"], shot["result"]) == (7.0, "foul", "miss dreb")


def test_miss_rebounded_at_the_buzzer_ends_the_trip():
    possessions = run(
        [
            ("PT12M00.00S", 4, 0, "", "", "Start of 4th Period", "period", "start", 0),
            ("PT00M05.00S", 4, HOME, "100", "97", "Brown Dunk", "Made Shot", "Dunk", 2),
            (
                "PT00M00.60S",
                4,
                AWAY,
                "",
                "",
                "MISS Young 3PT Jump Shot",
                "Missed Shot",
                "Jump Shot",
                3,
            ),
            ("PT00M00.30S", 4, 0, "", "", "HAWKS Rebound", "Rebound", "Unknown", 0),
            ("PT00M00.00S", 4, 0, "", "", "End of 4th Period", "period", "end", 0),
        ]
    )
    shot = next(s for s in possessions if s["off"] == AWAY and s["t0"] == 5.0)
    assert (shot["result"], shot["next"], shot["t1"]) == ("miss", "none", 0.3)


def test_away_from_play_free_throw_keeps_the_ball():
    possessions = run(
        [
            ("PT12M00.00S", 4, 0, "", "", "Start of 4th Period", "period", "start", 0),
            ("PT00M10.00S", 4, HOME, "100", "97", "Brown Dunk", "Made Shot", "Dunk", 2),
            (
                "PT00M09.00S",
                4,
                HOME,
                "",
                "",
                "Holiday AWAY.FROM.PLAY.FOUL (P1.PN)",
                "Foul",
                "Away From Play",
                0,
            ),
            (
                "PT00M09.00S",
                4,
                AWAY,
                "100",
                "98",
                "Young Free Throw 1 of 1",
                "Free Throw",
                "Free Throw 1 of 1",
                0,
            ),
            (
                "PT00M04.00S",
                4,
                AWAY,
                "",
                "",
                "MISS Young 3PT Jump Shot",
                "Missed Shot",
                "Jump Shot",
                3,
            ),
            ("PT00M03.00S", 4, HOME, "", "", "Brown REBOUND", "Rebound", "Unknown", 0),
            ("PT00M00.00S", 4, 0, "", "", "End of 4th Period", "period", "end", 0),
        ]
    )
    possession = next(s for s in possessions if s["off"] == AWAY and s["t0"] == 10.0)
    assert (possession["delta"], possession["fta"], possession["result"], possession["action"]) == (
        1,
        0,
        "miss dreb",
        "defend",
    )


def test_penalty_follows_the_team_foul_count_and_the_last_two_minutes():
    possessions = run(
        [
            ("PT12M00.00S", 4, 0, "", "", "Start of 4th Period", "period", "start", 0),
            ("PT03M00.00S", 4, HOME, "", "", "Holiday P.FOUL (P1.T2)", "Foul", "Personal", 0),
            ("PT00M20.00S", 4, HOME, "100", "97", "Brown Dunk", "Made Shot", "Dunk", 2),
            ("PT00M18.00S", 4, HOME, "", "", "Holiday P.FOUL (P2.T3)", "Foul", "Personal", 0),
            (
                "PT00M15.00S",
                4,
                AWAY,
                "",
                "",
                "MISS Young 3PT Jump Shot",
                "Missed Shot",
                "Jump Shot",
                3,
            ),
            ("PT00M14.00S", 4, HOME, "", "", "Brown REBOUND", "Rebound", "Unknown", 0),
            ("PT00M00.00S", 4, 0, "", "", "End of 4th Period", "period", "end", 0),
        ]
    )
    first, second = (s for s in possessions if s["off"] == AWAY)
    assert (first["penalty"], first["result"]) == (False, "foul")  # 3rd team foul: a foul to give
    assert second["penalty"] is True  # a team foul in the last two minutes: the next one shoots


def test_foul_to_give_then_a_penalty_foul_makes_two_possessions():
    # the end of Bucks at Rockets, 0022400831: a foul to give at 6.0, then a penalty foul at 3.7
    possessions = run(
        [  # a 3rd-quarter basket tells the parser which team is home
            ("PT00M30.00S", 3, HOME, "98", "97", "Green Dunk", "Made Shot", "Dunk", 2),
            ("PT12M00.00S", 4, 0, "", "", "Start of 4th Period", "period", "start", 0),
            (
                "PT00M08.10S",
                4,
                HOME,
                "100",
                "97",
                "Green Free Throw 2 of 2",
                "Free Throw",
                "Free Throw 2 of 2",
                0,
            ),
            (
                "PT00M06.00S",
                4,
                HOME,
                "",
                "",
                "Eason Personal Take Foul (P2.T4)",
                "Foul",
                "Personal Take",
                0,
            ),
            ("PT00M03.70S", 4, HOME, "", "", "Eason P.FOUL (P3.PN)", "Foul", "Personal", 0),
            (
                "PT00M03.70S",
                4,
                AWAY,
                "100",
                "98",
                "Lillard Free Throw 1 of 2",
                "Free Throw",
                "Free Throw 1 of 2",
                0,
            ),
            (
                "PT00M03.70S",
                4,
                AWAY,
                "100",
                "99",
                "Lillard Free Throw 2 of 2",
                "Free Throw",
                "Free Throw 2 of 2",
                0,
            ),
            ("PT00M00.00S", 4, 0, "", "", "End of 4th Period", "period", "end", 0),
        ]
    )
    give, line = (s for s in possessions if s["off"] == AWAY)
    assert (give["t0"], give["t1"], give["action"]) == (8.1, 6.0, "foul to give")
    assert (line["t0"], line["fta"], line["penalty"], line["action"]) == (6.0, 2, True, "foul")


def timeout(clock, period, kind="Regular"):
    return (clock, period, 0, "", "", f"CELTICS Timeout: {kind}", "Timeout", kind, 0)


def celtics_timeouts(*timeouts, period=4, season="2023-24"):
    """Timeouts the Celtics have left when they get the ball after a Hawks basket with 20 seconds
    left in the period, given the Celtics timeouts called earlier in the game."""
    rows = [("PT11M40.00S", 1, HOME, "2", "0", "Brown Dunk", "Made Shot", "Dunk", 2)]
    for p in range(1, period + 1):
        called = [t for t in timeouts if t[1] == p]
        if p < 4:
            rows += called
            continue
        length = "PT12M00.00S" if p == 4 else "PT05M00.00S"
        rows += [
            (length, p, 0, "", "", "Start", "period", "start", 0),
            *called,
            ("PT00M20.00S", p, AWAY, "2", "2", "Young Layup", "Made Shot", "Layup", 2),
            ("PT00M00.00S", p, 0, "", "", "End", "period", "end", 0),
        ]
    poss = next(
        x for x in run(rows, season) if x["off"] == HOME and x["period"] == period and x["t0"] == 20
    )
    return poss["timeouts"]


def test_two_left_late_in_the_4th_with_none_used():
    assert celtics_timeouts() == 2


def test_4th_quarter_cap_of_4():
    used = [timeout(c, 4) for c in ("PT08M00.00S", "PT07M00.00S", "PT06M00.00S")]
    assert celtics_timeouts(*used) == 1  # 4 in the 4th less 3 used, though 4 are left in the game


def test_last_3_minutes_cap_of_2():
    assert celtics_timeouts(timeout("PT02M00.00S", 4)) == 1  # 2 in the last 3 minutes less 1


def test_game_cap_of_7_binds_before_the_4th_quarter_cap():
    used = [timeout(c, p) for p in (1, 2, 3) for c in ("PT06M00.00S", "PT05M00.00S")]
    assert celtics_timeouts(*used) == 1  # 7 less 6, though the 4th allows 2 this late


def test_2_per_overtime_not_carried_over():
    assert celtics_timeouts(timeout("PT03M00.00S", 5), period=5) == 1
    assert celtics_timeouts(timeout("PT03M00.00S", 5), period=6) == 2  # OT2 starts fresh
    assert celtics_timeouts(period=6) == 2  # an unused OT1 timeout is gone


def test_official_timeouts_ignored():
    assert celtics_timeouts(timeout("PT02M00.00S", 4, kind="Official")) == 2


def test_a_timeout_at_the_possessions_start_time_not_yet_counted():
    assert celtics_timeouts(timeout("PT00M20.00S", 4)) == 2


def test_no_timeouts_before_2017_18():
    assert celtics_timeouts(timeout("PT02M00.00S", 4), season="2016-17") is None


def test_possessions_starting_before_the_24_second_mark_are_left_out():
    possessions = run(
        [
            ("PT11M40.00S", 1, HOME, "2", "0", "Brown Dunk", "Made Shot", "Dunk", 2),
            ("PT12M00.00S", 4, 0, "", "", "Start", "period", "start", 0),
            ("PT00M40.00S", 4, AWAY, "2", "2", "Young Layup", "Made Shot", "Layup", 2),
            ("PT00M24.00S", 4, HOME, "4", "2", "Brown Dunk", "Made Shot", "Dunk", 2),
            ("PT00M00.00S", 4, 0, "", "", "End", "period", "end", 0),
        ]
    )
    assert [(p["off"], p["t0"]) for p in possessions] == [(AWAY, 24.0)]


def test_home_team_comes_from_scoring_rows_on_a_neutral_site():
    rows = [
        ("PT11M40.00S", 1, AWAY, "0", "2", "Young Layup", "Made Shot", "Layup", 2),
        ("PT11M20.00S", 1, HOME, "2", "2", "Brown Dunk", "Made Shot", "Dunk", 2),
        ("PT11M00.00S", 1, HOME, "4", "2", "Brown Dunk", "Made Shot", "Dunk", 2),
    ]
    neutral = log(4, 2, matchup=("BOS @ ATL", "ATL @ BOS")).iloc[::-1]  # away row first
    game, _ = parse_game(pbp(rows), neutral, GAME, "2023-24")
    assert game == dict(
        game_id=GAME,
        season="2023-24",
        date="2024-01-01",
        home=HOME,
        away=AWAY,
        home_team="Boston Celtics",
        away_team="Atlanta Hawks",
        home_pts=4,
        away_pts=2,
        score_matches=True,
    )


def test_fouled_on_a_three():
    possessions = run(
        [
            ("PT11M40.00S", 1, HOME, "2", "0", "Brown Dunk", "Made Shot", "Dunk", 2),
            ("PT12M00.00S", 4, 0, "", "", "Start", "period", "start", 0),
            ("PT00M20.00S", 4, HOME, "5", "2", "Brown 3PT Jump Shot", "Made Shot", "Jump Shot", 3),
            ("PT00M12.00S", 4, HOME, "", "", "Holiday S.FOUL", "Foul", "Shooting", 0),
        ]
        + [
            (
                "PT00M12.00S",
                4,
                AWAY,
                "5",
                str(2 + k),
                f"Young Free Throw {k} of 3",
                "Free Throw",
                f"Free Throw {k} of 3",
                0,
            )  # fmt: skip
            for k in (1, 2, 3)
        ]
        + [("PT00M00.00S", 4, 0, "", "", "End", "period", "end", 0)]
    )
    three = next(p for p in possessions if p["off"] == AWAY)
    assert (three["label"], three["action"], three["delta"]) == ("Fouled on a Three", "defend", 3)
