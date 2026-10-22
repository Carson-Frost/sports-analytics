from nbapm.stints import elapsed, game_stints, names_match, plain, resolve

HOME, AWAY = 1, 2


def player(pid, first, family, starter=True, minutes="24:00"):
    return {
        "personId": pid,
        "firstName": first,
        "familyName": family,
        "nameI": f"{first[0]}. {family}",
        "position": "G" if starter else "",
        "statistics": {"minutes": minutes, "points": 0},
    }


def box(home_players, away_players):
    return {
        "boxScoreTraditional": {
            "homeTeamId": HOME,
            "awayTeamId": AWAY,
            "homeTeam": {"teamId": HOME, "players": home_players},
            "awayTeam": {"teamId": AWAY, "players": away_players},
        }
    }


def act(period, clock, kind, team=0, pid=0, desc="", score=("", ""), fg=0, result="", name=""):
    return {
        "period": period,
        "clock": clock,
        "actionType": kind,
        "subType": "",
        "teamId": team,
        "personId": pid,
        "playerName": name,
        "playerNameI": name,
        "description": desc,
        "scoreHome": score[0],
        "scoreAway": score[1],
        "isFieldGoal": fg,
        "shotResult": result,
    }


HOME_FIVE = [player(10 + i, f"Home{i}", f"Homer{i}") for i in range(5)]
AWAY_FIVE = [player(20 + i, f"Away{i}", f"Awayer{i}") for i in range(5)]
BENCH = player(15, "Seth", "Curry", starter=False)


def test_elapsed_counts_up_through_overtime():
    assert elapsed(1, "PT12M00.00S") == 0
    assert elapsed(2, "PT06M00.00S") == 1080
    assert elapsed(4, "PT00M00.00S") == 2880
    assert elapsed(5, "PT00M00.00S") == 3180


def test_names_match_accents_and_two_letter_initials():
    seth = {"keys": {"Curry", "S. Curry", "Seth Curry"}, "first": "Seth", "family": "Curry"}
    stephen = {
        "keys": {"Curry", "S. Curry", "Stephen Curry"},
        "first": "Stephen",
        "family": "Curry",
    }
    assert names_match("Se. Curry", seth) and not names_match("Se. Curry", stephen)
    assert plain("Dončić") == "Doncic"


def test_resolve_falls_back_to_the_first_name_for_given_name_first_order():
    # the box score has firstName 'Hansen', familyName 'Yang'; the substitution says 'Hansen'
    players = {
        1: {
            "team": HOME,
            "keys": {"Yang", "H. Yang", "Hansen Yang"},
            "first": "Hansen",
            "family": "Yang",
        },
        2: {"team": HOME, "keys": {"Clingan"}, "first": "Donovan", "family": "Clingan"},
    }
    assert resolve("Hansen", HOME, players) == 1


def test_resolve_refuses_an_ambiguous_name():
    players = {
        1: {"team": HOME, "keys": {"Curry"}, "first": "Seth", "family": "Curry"},
        2: {"team": HOME, "keys": {"Curry"}, "first": "Stephen", "family": "Curry"},
    }
    assert resolve("Curry", HOME, players) is None
    assert resolve("Curry", HOME, players, exclude={2}) == 1


def test_a_substitution_splits_the_game_into_two_stints_with_the_right_points():
    pbp = {
        "game": {
            "actions": [
                act(1, "PT11M00.00S", "Made Shot", HOME, 10, score=("2", "0"), fg=1, result="Made"),
                act(
                    1,
                    "PT10M00.00S",
                    "Missed Shot",
                    AWAY,
                    20,
                    score=("0", "0"),
                    fg=1,
                    result="Missed",
                ),
                act(1, "PT09M59.00S", "Rebound", AWAY, 21, score=("0", "0")),
                act(
                    1,
                    "PT06M00.00S",
                    "Substitution",
                    HOME,
                    10,
                    desc="SUB: Se. Curry FOR Homer0",
                    name="Homer0",
                ),
                act(1, "PT05M00.00S", "Made Shot", AWAY, 21, score=("2", "3"), fg=1, result="Made"),
            ]
        }
    }
    stints = game_stints(pbp, box(HOME_FIVE + [BENCH], AWAY_FIVE))
    assert [s["seconds"] for s in stints] == [360, 360]
    assert (stints[0]["home_pts"], stints[0]["away_pts"]) == (2, 0)
    assert (stints[1]["home_pts"], stints[1]["away_pts"]) == (0, 3)  # the old 0-0 rows are ignored
    assert 10 in stints[0]["home"] and 15 in stints[1]["home"] and 10 not in stints[1]["home"]
    assert stints[0]["poss"] == (1 + 0) / 2  # away offensive rebound: 1 FGA - 1 OREB


def test_period_starters_come_from_first_appearances_and_fill_silent_players():
    # in the 2nd period the bench player is on the floor from the start (he does something before
    # any substitution), Homer0 is subbed in later, and Homer1 to Homer3 never appear at all
    pbp = {
        "game": {
            "actions": [
                act(1, "PT00M00.00S", "period"),
                act(2, "PT11M00.00S", "Missed Shot", HOME, 15, fg=1, result="Missed"),
                act(2, "PT10M00.00S", "Turnover", HOME, 14),
                act(2, "PT06M00.00S", "Substitution", HOME, 15, desc="SUB: Homer0 FOR Curry"),
            ]
        }
    }
    stints = game_stints(pbp, box(HOME_FIVE + [BENCH], AWAY_FIVE))
    second = [s for s in stints if s["start"] >= 720]
    assert set(second[0]["home"]) == {15, 14, 11, 12, 13}
    assert set(second[1]["home"]) == {10, 14, 11, 12, 13}
