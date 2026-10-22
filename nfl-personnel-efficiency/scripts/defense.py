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
from analyze import HIGH, LOW, MAIN, MIN_SHARE, RESAMPLES, interval, neutral, pooled

from personnel.fit import Design, codes

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
FTN_FIRST = 2022  # FTN's play-action charting starts in 2022


def weighted_mean(values, groups, weights, n_groups):
    return np.bincount(groups, weights=weights * values, minlength=n_groups) / np.bincount(
        groups, weights=weights, minlength=n_groups)


def main():
    plays = pd.read_parquet(ROOT / "data" / "plays.parquet")
    plays["game_code"], n_games = codes(plays.game_id)
    raw = neutral(plays)
    common = raw.grouping.value_counts(normalize=True)
    keep = [g for g in common.index if common[g] >= MIN_SHARE]
    n = pooled(raw, keep)
    groupings = [*keep, "Other"]

    runs = n[(n["pass"] == 0) & n.defenders_in_box.notna()]  # both fits on the same runs
    runs = runs.assign(box=runs.defenders_in_box.clip(4, 9).astype(int))  # 4 or fewer, 9 or more share a term
    plain, boxed = Design(runs, groupings), Design(runs, groupings, extra=("box",))

    drops = n[(n["pass"] == 1) & (n.season >= FTN_FIRST) & n.grouping.isin(MAIN)]
    d_group = pd.Categorical(drops.grouping, categories=MAIN).codes
    d_pa = drops.play_action.to_numpy(dtype=float)

    recent = n[n.season >= FTN_FIRST]
    kind = np.select([recent["pass"] == 0, recent.play_action == 1], ["Runs", "Play-action dropbacks"],
                     "Other dropbacks")
    labels = [f"{s} {k}" for s, k in zip(recent.season, kind, strict=True)]
    cells = pd.Categorical(labels)
    cell_code, n_cells = cells.codes, len(cells.categories)
    r_epa = recent.epa.to_numpy(dtype=float)

    pair = n[n.grouping.isin(["11", "12"])]
    seasons = sorted(pair.season.unique())
    pair_code = (pair.season - seasons[0]).to_numpy() * 2 + (pair.grouping == "12").to_numpy()
    pair_base = (pair.defensive_backs <= 4).to_numpy(dtype=float)

    drop11 = n[(n["pass"] == 1) & (n.grouping == "11")]
    drop11_code = (drop11.season - seasons[0]).to_numpy()
    drop11_epa = drop11.epa.to_numpy(dtype=float)

    def one(w):
        return (plain.fit(w), boxed.fit(w),
                weighted_mean(d_pa, d_group, w[drops.game_code.to_numpy()], len(MAIN)),
                weighted_mean(r_epa, cell_code, w[recent.game_code.to_numpy()], n_cells),
                weighted_mean(pair_base, pair_code, w[pair.game_code.to_numpy()], 2 * len(seasons)),
                weighted_mean(drop11_epa, drop11_code, w[drop11.game_code.to_numpy()], len(seasons)))

    point = one(np.ones(n_games))
    rng = np.random.default_rng(2027)
    draws = [one(np.bincount(rng.integers(0, n_games, n_games), minlength=n_games)) for _ in range(RESAMPLES)]

    rows = []
    for i, name in enumerate(["run_gap", "run_gap_same_box"]):
        lo, hi = interval([d[i].reindex(point[i].index) for d in draws])
        for j, g in enumerate(point[i].index):
            if g in MAIN:
                rows.append({"grouping": g, "measure": name, "value": point[i][g], "low": lo[j], "high": hi[j]})
    lo, hi = interval([d[2] for d in draws])
    for j, g in enumerate(MAIN):
        rows.append({"grouping": g, "measure": "play_action_share", "value": point[2][j], "low": lo[j], "high": hi[j]})
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(rows).round(4).to_csv(OUT / "defense.csv", index=False)

    lo, hi = interval([d[3] for d in draws])
    types = pd.DataFrame({"cell": cells.categories, "epa": point[3], "low": lo, "high": hi})
    types[["season", "play_type"]] = types.cell.str.split(" ", n=1, expand=True)
    types[["season", "play_type", "epa", "low", "high"]].round(4).to_csv(OUT / "play_types.csv", index=False)
    lo, hi = interval([d[4] for d in draws])
    base = pd.DataFrame({"season": np.repeat(seasons, 2), "grouping": ["11", "12"] * len(seasons),
                         "base_defense": point[4], "low": lo, "high": hi})
    base.round(4).to_csv(OUT / "base_by_season.csv", index=False)
    lo, hi = interval([d[5] for d in draws])
    pd.DataFrame({"season": seasons, "epa": point[5], "low": lo, "high": hi}).round(4).to_csv(
        OUT / "dropbacks_11.csv", index=False)
    games = plays[plays.season == 2025].assign(grouping=lambda d: d.grouping.where(d.grouping.isin(MAIN), "Other"))
    games = games.groupby(["posteam", "week"]).grouping.value_counts(normalize=True).unstack(fill_value=0)
    games["plays"] = plays[plays.season == 2025].groupby(["posteam", "week"]).size()
    games[[*MAIN, "Other", "plays"]].round(4).rename_axis(["team", "week"]).to_csv(OUT / "team_weeks_2025.csv")
    print(pd.read_csv(OUT / "defense.csv").to_string(index=False))
    print(base.round(3).to_string(index=False))
    print(pd.read_csv(OUT / "play_types.csv").to_string(index=False))
    assert (LOW, HIGH) == (5, 95)


if __name__ == "__main__":
    main()
