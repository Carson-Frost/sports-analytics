"""Usage and EPA per play by personnel grouping, with 90% intervals from 200 game resamples.

Usage counts every run and dropback. EPA compares each grouping with 11 on plays with win
probability 10-90%, within the same offense-season, down and yards to go, and 5-yard field bin.

Writes results/effects.csv, situations.csv, groupings.csv, teams_2025.csv, usage_by_season.csv,
usage_by_team_season.csv and wp_window.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from personnel import SITUATIONS
from personnel.fit import Design, codes

ROOT = Path(__file__).resolve().parents[1]
PLAYS = ROOT / "data" / "plays.parquet"
OUT = ROOT / "results"
RESAMPLES = 200
LOW, HIGH = 5, 95  # percentiles of a 90% interval
WP = (0.10, 0.90)  # both teams still playing normally
MIN_SHARE = 0.005  # groupings under 0.5% of plays are pooled as Other
MIN_PLAYS = 50  # in one situation, groupings with fewer plays there are pooled as Other
SEASON = 2025
MAIN = ["11", "12", "13", "21", "22"]
SEASONS = range(2016, 2026)


def neutral(plays, lo=WP[0], hi=WP[1]):
    return plays[plays.wp.between(lo, hi) & plays.epa.notna()].copy()


def pooled(plays, keep):
    return plays.assign(grouping=plays.grouping.where(plays.grouping.isin(keep), "Other"))


def interval(draws):
    return np.nanpercentile(np.asarray(draws, dtype=float), [LOW, HIGH], axis=0)


def shares(plays, by):
    n = plays.groupby([*by, "grouping"]).size().rename("plays").reset_index()
    n["share"] = n.plays / n.groupby(by).plays.transform("sum")
    return n


def main():
    plays = pd.read_parquet(PLAYS)
    plays["game_code"], n_games = codes(plays.game_id)
    raw = neutral(plays)
    common = raw.grouping.value_counts(normalize=True)
    keep = [g for g in common.index if common[g] >= MIN_SHARE]
    n = pooled(raw, keep)
    groupings = [*keep, "Other"]

    designs = {
        ("all", "2016-2025"): Design(n, groupings),
        ("run", "2016-2025"): Design(n[n["pass"] == 0], groupings),
        ("dropback", "2016-2025"): Design(n[n["pass"] == 1], groupings),
        ("all", "2016-2020"): Design(n[n.season <= 2020], groupings),
        ("all", "2021-2025"): Design(n[n.season >= 2021], groupings),
    }
    for season in SEASONS:  # each season on its own, to see how each gap changed
        part = n[n.season == season]
        designs["all", str(season)] = Design(part, groupings)
        designs["run", str(season)] = Design(part[part["pass"] == 0], groupings)
        designs["dropback", str(season)] = Design(part[part["pass"] == 1], groupings)
    sit_designs = {}
    for sit in SITUATIONS:
        part = raw[raw.situation == sit]
        big = part.grouping.value_counts()
        sit_keep = [g for g in big.index if big[g] >= MIN_PLAYS]  # goal-line packages get their own term
        sit_designs[sit] = Design(pooled(part, sit_keep), [*sit_keep, "Other"])

    # rates per grouping, and each 2025 offense's EPA per play
    g_code = pd.Categorical(n.grouping, categories=groupings).codes
    dropback = n["pass"].to_numpy(dtype=float)
    base = (n.defensive_backs <= 4).to_numpy(dtype=float)
    box = n.defenders_in_box.to_numpy(dtype=float)
    epa = n.epa.to_numpy(dtype=float)
    t25 = n[n.season == SEASON]
    team_code, _ = codes(t25.posteam)
    teams = sorted(t25.posteam.unique())

    def rates(w):
        w = w[n.game_code.to_numpy()]
        def mean(v, among):
            return np.bincount(g_code, weights=w * among * v, minlength=len(groupings)) / np.bincount(
                g_code, weights=w * among, minlength=len(groupings))
        everything = np.ones(len(w))
        return [mean(dropback, everything), mean(base, everything), mean(epa, 1 - dropback), mean(epa, dropback),
                mean(epa, everything)]

    def box_mean(w):
        w = w[n.game_code.to_numpy()] * ~np.isnan(box)
        return np.bincount(g_code, weights=w * np.nan_to_num(box), minlength=len(groupings)) / np.bincount(
            g_code, weights=w, minlength=len(groupings))

    def team_epa(w):
        w = w[t25.game_code.to_numpy()]
        return np.bincount(team_code, weights=w * t25.epa.to_numpy()) / np.bincount(team_code, weights=w)

    ones = np.ones(n_games)
    point = {k: d.fit() for k, d in designs.items()}
    sit_point = {k: d.fit() for k, d in sit_designs.items()}
    point_rates, point_box, point_team = rates(ones), box_mean(ones), team_epa(ones)

    rng = np.random.default_rng(2026)
    draws = {k: [] for k in designs}
    sit_draws = {k: [] for k in sit_designs}
    rate_draws, box_draws, team_draws = [], [], []
    for b in range(RESAMPLES):
        w = np.bincount(rng.integers(0, n_games, n_games), minlength=n_games)
        for k, d in designs.items():
            draws[k].append(d.fit(w))
        for k, d in sit_designs.items():
            sit_draws[k].append(d.fit(w))
        rate_draws.append(rates(w))
        box_draws.append(box_mean(w))
        team_draws.append(team_epa(w))
        if b % 25 == 0:
            print("resample", b, flush=True)

    OUT.mkdir(exist_ok=True)
    rows = []
    for (kind, period), est in point.items():
        lo, hi = interval([d.reindex(est.index) for d in draws[kind, period]])
        part = n if kind == "all" else n[n["pass"] == (kind == "dropback")]
        bounds = [int(year) for year in period.split("-")]  # "2016-2025" or one season, "2016"
        part = part[part.season.between(bounds[0], bounds[-1])]
        count = part.grouping.value_counts()
        for i, g in enumerate(est.index):
            rows.append({"grouping": g, "play_type": kind, "seasons": period, "plays": int(count.get(g, 0)),
                         "effect": est[g], "low": lo[i], "high": hi[i]})
    pd.DataFrame(rows).round(4).to_csv(OUT / "effects.csv", index=False)

    usage = shares(plays, ["situation"])
    rows = []
    for sit, est in sit_point.items():
        lo, hi = interval([d.reindex(est.index) for d in sit_draws[sit]])
        part = raw[raw.situation == sit]
        count = pooled(part, [*est.index, "11"]).grouping.value_counts()
        used = usage[usage.situation == sit].set_index("grouping").share
        rows.append({"situation": sit, "grouping": "11", "neutral_plays": int(count.get("11", 0)),
                     "share": used.get("11", 0.0), "effect": 0.0, "low": 0.0, "high": 0.0})
        for i, g in enumerate(est.index):
            share = used.drop(index=list(est.index.drop("Other")) + ["11"], errors="ignore").sum() if g == "Other" \
                else used.get(g, 0.0)
            rows.append({"situation": sit, "grouping": g, "neutral_plays": int(count.get(g, 0)), "share": share,
                         "effect": est[g], "low": lo[i], "high": hi[i]})
    pd.DataFrame(rows).round(4).to_csv(OUT / "situations.csv", index=False)

    all_counts = pooled(plays, keep).grouping.value_counts()
    rate_lo, rate_hi = interval([np.array(r) for r in rate_draws])
    box_lo, box_hi = interval(box_draws)
    g = pd.DataFrame({
        "grouping": groupings,
        "plays": [int(all_counts.get(x, 0)) for x in groupings],
        "share": [all_counts.get(x, 0) / len(plays) for x in groupings],
        "neutral_plays": [int((n.grouping == x).sum()) for x in groupings],
        "dropback_rate": point_rates[0], "dropback_low": rate_lo[0], "dropback_high": rate_hi[0],
        "base_defense": point_rates[1], "base_low": rate_lo[1], "base_high": rate_hi[1],
        "epa_run": point_rates[2], "epa_run_low": rate_lo[2], "epa_run_high": rate_hi[2],
        "epa_dropback": point_rates[3], "epa_dropback_low": rate_lo[3], "epa_dropback_high": rate_hi[3],
        "epa_all": point_rates[4], "epa_all_low": rate_lo[4], "epa_all_high": rate_hi[4],
        "box": point_box, "box_low": box_lo, "box_high": box_hi,
    })
    g.round(4).to_csv(OUT / "groupings.csv", index=False)

    mix = shares(plays[plays.season == SEASON].assign(
        grouping=lambda d: d.grouping.where(d.grouping.isin(MAIN), "Other")), ["posteam"])
    mix = mix.pivot(index="posteam", columns="grouping", values="share")[[*MAIN, "Other"]].fillna(0)
    lo, hi = interval(team_draws)
    teams_out = mix.loc[teams].assign(epa=point_team, low=lo, high=hi).sort_values("epa", ascending=False)
    teams_out.round(4).rename_axis("team").to_csv(OUT / "teams_2025.csv")

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
    for lo_wp, hi_wp in [(0.0, 1.0), (0.05, 0.95), WP, (0.2, 0.8)]:
        part = pooled(neutral(plays, lo_wp, hi_wp), keep)
        est = Design(part, groupings).fit()
        window.append({"wp_low": lo_wp, "wp_high": hi_wp, "plays": len(part), **{f"effect_{x}": est[x] for x in
                                                                                    ["12", "13", "21", "22"]}})
    pd.DataFrame(window).round(4).to_csv(OUT / "wp_window.csv", index=False)
    print(pd.read_csv(OUT / "effects.csv").query("seasons == '2016-2025'").to_string(index=False))


if __name__ == "__main__":
    main()
