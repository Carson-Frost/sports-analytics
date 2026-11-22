"""Usage and EPA per play by personnel grouping, with 90% intervals from 200 game resamples.

Usage counts every run and dropback. EPA compares each grouping with 11 on plays with win
probability 10-90%, within the same offense-season, down and yards to go, and 5-yard field bin.

Writes results/effects.csv, situations.csv, groupings.csv, teams_2025.csv, usage_by_season.csv,
usage_by_team_season.csv and wp_window.csv.
"""

from pathlib import Path

import pandas as pd

from personnel import HEAVIER, MAIN, MIN_PLAYS, SITUATIONS, WINDOW, prepare_plays
from personnel.fit import Design, codes
from personnel.resample import GroupMean, run

ROOT = Path(__file__).resolve().parents[1]
PLAYS = ROOT / "data" / "plays.parquet"
OUT = ROOT / "results"
SEASON = 2025
SEASONS = range(2016, 2026)


def pooled(plays, keep):
    return plays.assign(grouping=plays.grouping.where(plays.grouping.isin(keep), "Other"))


def shares(plays, by):
    n = plays.groupby([*by, "grouping"]).size().rename("plays").reset_index()
    n["share"] = n.plays / n.groupby(by).plays.transform("sum")
    return n


def main():
    plays = pd.read_parquet(PLAYS)
    plays["game_code"], n_games = codes(plays.game_id)
    n, groupings = prepare_plays(plays)

    effects = {
        ("all", "2016-2025"): Design(n, groupings),
        ("run", "2016-2025"): Design(n[n["pass"] == 0], groupings),
        ("dropback", "2016-2025"): Design(n[n["pass"] == 1], groupings),
        ("all", "2016-2020"): Design(n[n.season <= 2020], groupings),
        ("all", "2021-2025"): Design(n[n.season >= 2021], groupings),
    }
    for season in SEASONS:  # each season on its own, to see how each gap changed
        part = n[n.season == season]
        effects["all", str(season)] = Design(part, groupings)
        effects["run", str(season)] = Design(part[part["pass"] == 0], groupings)
        effects["dropback", str(season)] = Design(part[part["pass"] == 1], groupings)
    estimators = dict(effects)
    sit_plays = {}
    for sit in SITUATIONS:  # goal-line groupings get their own term
        sit_plays[sit] = prepare_plays(plays[plays.situation == sit], min_share=0, min_plays=MIN_PLAYS)
        estimators["situation", sit] = Design(*sit_plays[sit])

    # rates per grouping, and each 2025 offense's EPA per play
    dropback = n["pass"] == 1
    rates = {"dropback": ("pass", None), "base": (n.defensive_backs <= 4, None), "epa_run": ("epa", ~dropback),
             "epa_dropback": ("epa", dropback), "epa_all": ("epa", None),
             "box": ("defenders_in_box", n.defenders_in_box.notna())}
    for name, (values, among) in rates.items():
        estimators["rate", name] = GroupMean(n, values, "grouping", among=among, labels=groupings)
    estimators["team"] = GroupMean(n, "epa", "posteam", among=n.season == SEASON)

    out = run(estimators, n_games, seed=2026)

    OUT.mkdir(exist_ok=True)
    rows = []
    for (kind, period) in effects:
        est = out[kind, period]
        part = n if kind == "all" else n[n["pass"] == (kind == "dropback")]
        bounds = [int(year) for year in period.split("-")]  # "2016-2025" or one season, "2016"
        part = part[part.season.between(bounds[0], bounds[-1])]
        count = part.grouping.value_counts()
        for g, r in est.iterrows():
            rows.append({"grouping": g, "play_type": kind, "seasons": period, "plays": int(count.get(g, 0)),
                         "effect": r.estimate, "low": r.low, "high": r.high})
    pd.DataFrame(rows).round(4).to_csv(OUT / "effects.csv", index=False)

    rows = []
    for sit in SITUATIONS:
        part, sit_groupings = sit_plays[sit]
        count = part.grouping.value_counts()
        all_sit = plays[plays.situation == sit]
        used = pooled(all_sit, sit_groupings).grouping.value_counts() / len(all_sit)
        rows.append({"situation": sit, "grouping": "11", "neutral_plays": int(count.get("11", 0)),
                     "share": used.get("11", 0.0), "effect": 0.0, "low": 0.0, "high": 0.0})
        for g, r in out["situation", sit].iterrows():
            rows.append({"situation": sit, "grouping": g, "neutral_plays": int(count.get(g, 0)),
                         "share": used.get(g, 0.0), "effect": r.estimate, "low": r.low, "high": r.high})
    pd.DataFrame(rows).round(4).to_csv(OUT / "situations.csv", index=False)

    all_counts = pooled(plays, groupings).grouping.value_counts()
    g = pd.DataFrame({
        "grouping": groupings,
        "plays": [int(all_counts.get(x, 0)) for x in groupings],
        "share": [all_counts.get(x, 0) / len(plays) for x in groupings],
        "neutral_plays": [int((n.grouping == x).sum()) for x in groupings],
    })
    point_names = {"dropback": "dropback_rate", "base": "base_defense"}
    for name in rates:
        r = out["rate", name].to_numpy().T
        g[point_names.get(name, name)], g[f"{name}_low"], g[f"{name}_high"] = r
    g.round(4).to_csv(OUT / "groupings.csv", index=False)

    mix = shares(plays[plays.season == SEASON].assign(
        grouping=lambda d: d.grouping.where(d.grouping.isin(MAIN), "Other")), ["posteam"])
    mix = mix.pivot(index="posteam", columns="grouping", values="share")[[*MAIN, "Other"]].fillna(0)
    team = out["team"]
    teams_out = mix.loc[team.index].assign(epa=team.estimate, low=team.low, high=team.high)
    teams_out.sort_values("epa", ascending=False).round(4).rename_axis("team").to_csv(OUT / "teams_2025.csv")

    family = plays.grouping.where(plays.grouping.isin(MAIN), "Other")
    six = plays.grouping.str.contains("6 OL") & (family == "Other")
    by_season = shares(plays.assign(grouping=family), ["season"]).pivot(index="season", columns="grouping",
                                                                         values="share")[[*MAIN, "Other"]]
    by_season["other_6ol"] = six.groupby(plays.season).mean()  # share of all plays: Other with a sixth lineman
    by_season.round(4).to_csv(OUT / "usage_by_season.csv")
    by_team = shares(plays.assign(grouping=family), ["posteam", "season"])
    by_team = by_team.pivot(index=["posteam", "season"], columns="grouping", values="share")[[*MAIN, "Other"]]
    by_team.fillna(0).round(4).rename_axis(["team", "season"]).to_csv(OUT / "usage_by_team_season.csv")

    window = []
    for lo_wp, hi_wp in [(0.0, 1.0), (0.05, 0.95), WINDOW, (0.2, 0.8)]:
        part, part_groupings = prepare_plays(plays, window=(lo_wp, hi_wp))
        est = Design(part, part_groupings).fit()
        window.append({"wp_low": lo_wp, "wp_high": hi_wp, "plays": len(part), **{f"effect_{x}": est[x] for x in
                                                                                    HEAVIER}})
    pd.DataFrame(window).round(4).to_csv(OUT / "wp_window.csv", index=False)
    print(pd.read_csv(OUT / "effects.csv").query("seasons == '2016-2025'").to_string(index=False))


if __name__ == "__main__":
    main()
