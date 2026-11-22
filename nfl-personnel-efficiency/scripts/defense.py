"""How defenses line up against heavier personnel, and what that means for runs and play-action.

Writes results/defense.csv: for each main grouping, the run gap to 11 personnel before and after holding the number
of defenders in the box fixed, and the play-action share of dropbacks (2022-2025, FTN charting).
Writes results/play_types.csv: EPA per play of play-action dropbacks, other dropbacks and runs, by season (2022-2025).
Writes results/base_by_season.csv: how often defenses were in base (4 or fewer defensive backs) against 11 and 12
personnel, by season.
Writes results/dropbacks_11.csv: EPA per dropback out of 11 personnel, by season.
Writes results/team_weeks_2025.csv: each 2025 offense's share of plays in each main grouping, game by game.
90% intervals from 200 game resamples, as in analyze.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from personnel import MAIN, prepare_plays
from personnel.fit import Design, codes
from personnel.resample import GroupMean, run

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
FTN_FIRST = 2022  # FTN's play-action charting starts in 2022


def main():
    plays = pd.read_parquet(ROOT / "data" / "plays.parquet")
    plays["game_code"], n_games = codes(plays.game_id)
    n, groupings = prepare_plays(plays)

    runs = n[(n["pass"] == 0) & n.defenders_in_box.notna()]  # both fits on the same runs
    runs = runs.assign(box=runs.defenders_in_box.clip(4, 9).astype(int))  # 4 or fewer, 9 or more share a term
    recent = n[n.season >= FTN_FIRST]
    recent = recent.assign(play_type=np.select([recent["pass"] == 0, recent.play_action == 1],
                                               ["Runs", "Play-action dropbacks"], "Other dropbacks"))
    out = run({
        "run_gap": Design(runs, groupings),
        "run_gap_same_box": Design(runs, groupings, extra=("box",)),
        "play_action_share": GroupMean(recent, "play_action", "grouping", among=recent["pass"] == 1, labels=MAIN),
        "play_types": GroupMean(recent, "epa", ["season", "play_type"]),
        "base_defense": GroupMean(n, n.defensive_backs <= 4, ["season", "grouping"],
                                  among=n.grouping.isin(["11", "12"])),
        "dropbacks_11": GroupMean(n, "epa", "season", among=(n["pass"] == 1) & (n.grouping == "11")),
    }, n_games, seed=2027)

    rows = []
    for name in ["run_gap", "run_gap_same_box", "play_action_share"]:
        for g, r in out[name].iterrows():
            if g in MAIN:
                rows.append({"grouping": g, "measure": name, "value": r.estimate, "low": r.low, "high": r.high})
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(rows).round(4).to_csv(OUT / "defense.csv", index=False)

    types = out["play_types"].rename(columns={"estimate": "epa"}).rename_axis(["season", "play_type"]).reset_index()
    types.round(4).to_csv(OUT / "play_types.csv", index=False)
    base = out["base_defense"].rename(columns={"estimate": "base_defense"}).reset_index()
    base.round(4).to_csv(OUT / "base_by_season.csv", index=False)
    out["dropbacks_11"].rename(columns={"estimate": "epa"}).rename_axis("season").reset_index().round(4).to_csv(
        OUT / "dropbacks_11.csv", index=False)
    games = plays[plays.season == 2025].assign(grouping=lambda d: d.grouping.where(d.grouping.isin(MAIN), "Other"))
    games = games.groupby(["posteam", "week"]).grouping.value_counts(normalize=True).unstack(fill_value=0)
    games["plays"] = plays[plays.season == 2025].groupby(["posteam", "week"]).size()
    games[[*MAIN, "Other", "plays"]].round(4).rename_axis(["team", "week"]).to_csv(OUT / "team_weeks_2025.csv")
    print(pd.read_csv(OUT / "defense.csv").to_string(index=False))
    print(base.round(3).to_string(index=False))
    print(pd.read_csv(OUT / "play_types.csv").to_string(index=False))


if __name__ == "__main__":
    main()
