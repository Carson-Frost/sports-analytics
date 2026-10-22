import pandas as pd

from foulup3.pbp import action, game_segments, seconds

HOME, AWAY = 1, 2
NAMES = {HOME: "celtics", AWAY: "hawks"}


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


def run(rows):
    return game_segments(pbp(rows), HOME, AWAY, NAMES)


def test_seconds():
    assert seconds("PT01M05.50S") == 65.5


def test_foul_up_three_then_free_throws_and_a_team_rebound():
    segs = run(
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
    foul = next(s for s in segs if s["t0"] == 20.0 and s["off"] == AWAY)
    assert (foul["off"], foul["d0"], foul["delta"]) == (AWAY, -3, 1)
    assert (foul["fta"], foul["ftm"], foul["result"], foul["next"]) == (2, 1, "ftmiss oreb", "same")
    assert action(foul) == "foul"
    assert segs[-1]["how"] == "oreb" and segs[-1]["result"] == "clock"


def test_and_one_stays_with_the_basket_and_dead_ball_rebound_is_skipped():
    segs = run(
        [
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
    basket = next(s for s in segs if s["t0"] == 15.0 and s["off"] == AWAY)
    assert (basket["delta"], basket["made"], basket["result"]) == (3, 2, "ft")
    home = next(s for s in segs if s["off"] == HOME and s["fta"] == 2)
    assert (home["d0"], home["delta"], home["next"], action(home)) == (-3, 1, "other", "defend")
    assert segs[-1]["off"] == AWAY and segs[-1]["d0"] == 2


def test_foul_with_no_free_throws_keeps_the_ball():
    segs = run(
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
    give, shot = (s for s in segs if s["off"] == AWAY)
    assert (give["result"], give["next"], action(give)) == ("foul", "same", "foul to give")
    assert (shot["t0"], shot["how"], shot["result"]) == (7.0, "foul", "miss dreb")


def test_miss_rebounded_at_the_buzzer_ends_the_trip():
    segs = run(
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
    shot = next(s for s in segs if s["off"] == AWAY and s["t0"] == 5.0)
    assert (shot["result"], shot["next"], shot["t1"]) == ("miss", "none", 0.3)


def test_away_from_play_free_throw_keeps_the_ball():
    segs = run(
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
    trip = next(s for s in segs if s["off"] == AWAY and s["t0"] == 10.0)
    assert (trip["delta"], trip["fta"], trip["result"], action(trip)) == (
        1,
        0,
        "miss dreb",
        "defend",
    )


def test_bonus_follows_the_team_foul_count_and_the_last_two_minutes():
    segs = run(
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
    first, second = (s for s in segs if s["off"] == AWAY)
    assert (first["bonus"], first["result"]) == (False, "foul")  # 3rd team foul: a foul to give
    assert second["bonus"] is True  # a team foul in the last two minutes: the next one is a penalty


def test_foul_to_give_then_a_bonus_foul_makes_two_segments():
    # the end of Bucks at Rockets, 0022400831: a foul to give at 6.0, then a bonus foul at 3.7
    segs = run(
        [
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
    give, line = (s for s in segs if s["off"] == AWAY)
    assert (give["t0"], give["t1"], action(give)) == (8.1, 6.0, "foul to give")
    assert (line["t0"], line["fta"], line["bonus"], action(line)) == (6.0, 2, True, "foul")
