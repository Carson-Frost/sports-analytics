import pandas as pd

from overtime import data, sim

ROWS = [  # play_id, play_type, posteam, down, ydstogo, yardline_100, seconds left, kick distance
    (1, "kickoff", "B", None, 0, 35, 3600, 65),
    (2, "no_play", None, None, 0, None, 3600, None),  # a timeout row, no team
    (3, "no_play", "B", 1, 10, 75, 3598, None),  # false start: 1st and 15
    (4, "run", "B", 1, 15, 80, 3595, None),
    (5, "no_play", "B", 2, 6, 71, 3560, None),  # offside: 5 yards, still 2nd down
    (6, "pass", "B", 2, 1, 66, 3555, None),
    (7, "run", "B", 1, 10, 60, 3520, None),
    (8, "field_goal", "B", 4, 3, 30, 3480, 48),
    (9, "kickoff", "A", None, 0, 35, 3475, 65),
    (10, "run", "A", 1, 10, 75, 3470, None),
    (11, "field_goal", "A", 4, 2, 25, 3430, 43),
    (12, "field_goal", "A", 4, 2, 35, 3400, 53),
]
MADE = {8: "made", 11: "missed", 12: "made"}


def plays():
    df = pd.DataFrame(ROWS, columns=["play_id", "play_type", "posteam", "down", "ydstogo",
                                     "yardline_100", "game_seconds_remaining", "kick_distance"])
    teams = {"A": "B", "B": "A"}
    return df.assign(
        game_id="2025_01_A_B", season=2025, season_type="REG", week=1, qtr=1, game_half="Half1",
        half_seconds_remaining=df.game_seconds_remaining - 1800, defteam=df.posteam.map(teams),
        score_differential=0, touchdown=0, td_team=None, interception=0, fumble_lost=0, safety=0,
        field_goal_result=df.play_id.map(MADE), extra_point_result=None,
        two_point_conv_result=None, fixed_drive=1, fixed_drive_result=None)


def test_next_play_skips_team_less_rows():
    p = data.add_next_play(plays())
    assert 2 not in p.play_id.tolist()
    kick = p[p.play_id == 1].iloc[0]
    assert (kick.next_posteam, kick.next_yardline_100, kick.secs) == ("B", 75, 2)


def test_fit_reads_snaps_kicks_and_repeat_downs():
    m = sim.fit(data.add_next_play(plays()), [2025])
    assert (sim.PLAY, 5, sim.REPEAT_DOWN, 5) in m.snaps_pooled[(2, sim.dist_band(6))]
    assert (sim.PLAY, -5, sim.REPEAT_DOWN, 3) in m.snaps_pooled[(1, sim.dist_band(10))]
    assert (sim.PLAY, 9, 0, 35) in m.snaps_pooled[(1, sim.dist_band(15))]
    assert m.kickoffs == [(True, 75, 0, 2), (True, 75, 0, 5)]
    assert m.fourth[(sim.dist_band(3), sim.field_band(30))] == ["fg"]


def test_overtimes_come_from_the_play_by_play(tmp_path):
    (tmp_path / "games.csv").write_text(
        "game_id,season,game_type,week,home_team,away_team,result,overtime\n"
        "2021_12_LV_DAL,2021,REG,12,DAL,LV,-3,0\n"  # the schedule's flag misses this one
        "2021_12_X_Y,2021,REG,12,Y,X,7,0\n")
    ot = pd.DataFrame({"game_id": ["2021_12_LV_DAL"], "qtr": [5], "play_type": ["kickoff"],
                       "posteam": ["LV"], "play_id": [4000]})
    games = data.build_ot_games(ot, raw=tmp_path)
    assert games.game_id.tolist() == ["2021_12_LV_DAL"]
    assert games.iloc[0].receiver == "LV" and games.iloc[0].outcome == 1
