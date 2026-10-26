"""Write README.md from the results."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
NAMES = {"ARI": "Cardinals", "ATL": "Falcons", "BAL": "Ravens", "BUF": "Bills", "CAR": "Panthers", "CHI": "Bears",
         "CIN": "Bengals", "CLE": "Browns", "DAL": "Cowboys", "DEN": "Broncos", "DET": "Lions", "GB": "Packers",
         "HOU": "Texans", "IND": "Colts", "JAX": "Jaguars", "KC": "Chiefs", "LA": "Rams", "LAC": "Chargers",
         "LV": "Raiders", "MIA": "Dolphins", "MIN": "Vikings", "NE": "Patriots", "NO": "Saints", "NYG": "Giants",
         "NYJ": "Jets", "PHI": "Eagles", "PIT": "Steelers", "SEA": "Seahawks", "SF": "49ers", "TB": "Buccaneers",
         "TEN": "Titans", "WAS": "Commanders"}


def ordinal(n):
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def main():
    teams = pd.read_csv(R / "teams_2025.csv", index_col="team")
    rank = pd.Series(range(1, len(teams) + 1), index=teams.index)  # sorted by EPA per play

    e = pd.read_csv(R / "effects.csv", dtype={"grouping": str})
    e = e.set_index(["play_type", "seasons", "grouping"]).sort_index()
    run12, pass12 = e.loc[("run", "2016-2025", "12"), "effect"], e.loc[("dropback", "2016-2025", "12"), "effect"]
    usage = pd.read_csv(R / "usage_by_season.csv", index_col="season")
    last = usage.index[-1]
    yearly = e[[len(x[1]) == 4 for x in e.index]]  # one-season rows
    yearly = yearly[yearly.index.get_level_values("grouping").isin(["12", "13", "21", "22"])]
    clear = (yearly.low > 0) | (yearly.high < 0)
    n_run12 = int(clear.loc["run"].xs("12", level="grouping").sum())
    n_seasons = len(usage)
    in_last = [x[1] == str(last) and x[0] in ("run", "dropback") for x in clear.index]
    assert not clear[in_last].any(), "the 2025 sentence no longer holds"
    assert not clear.loc[("run", str(last), "12")], "the 12 sentence no longer holds"
    g = pd.read_csv(R / "groupings.csv", dtype={"grouping": str}).set_index("grouping")
    base12, base11 = 100 * g.loc["12", "base_defense"], 100 * g.loc["11", "base_defense"]
    box12, box11 = g.loc["12", "box"], g.loc["11", "box"]
    dfn = pd.read_csv(R / "defense.csv", dtype={"grouping": str}).set_index(["measure", "grouping"])
    same = dfn.loc["run_gap_same_box"]
    same12, max_same = same.loc["12", "value"], same.drop(index="12").value.abs().max()
    assert same.loc["12", "low"] < 0 < same.loc["12", "high"], "the same-box sentence no longer holds"
    pa12, pa11 = (100 * dfn.loc[("play_action_share", x), "value"] for x in ("12", "11"))
    types = pd.read_csv(R / "play_types.csv").pivot(index="season", columns="play_type", values="epa")
    assert (types["Play-action dropbacks"] > types[["Other dropbacks", "Runs"]].max(axis=1)).all()
    pa_last, other_last, run_last = types.loc[last, ["Play-action dropbacks", "Other dropbacks", "Runs"]]
    bs = pd.read_csv(R / "base_by_season.csv", dtype={"grouping": str})
    b12 = bs[bs.grouping == "12"].set_index("season").base_defense * 100
    first_season, low_season = b12.index[0], b12.idxmin()
    b_first, b_low, b_last = b12.iloc[0], b12.min(), b12.loc[last]
    assert b_first > b_last > b_low, "the base-defense sentence no longer holds"
    e11 = pd.read_csv(R / "dropbacks_11.csv", index_col="season").epa
    peak_season = e11.idxmax()
    low_season11 = e11.loc[peak_season:].idxmin()
    e_peak, e_low = e11.loc[peak_season], e11.loc[low_season11]
    heavy = ["12", "13", "21", "22"]
    runs, drops = e.loc[("run", "2016-2025")].loc[heavy], e.loc[("dropback", "2016-2025")].loc[heavy]
    assert (runs.high < 0).all() and (drops.effect > 0).all(), "the results sentence no longer holds"
    assert [x for x in heavy if drops.loc[x, "low"] > 0] == ["12"], "the results sentence no longer holds"
    halves = e.loc["all"].effect.unstack("seasons")
    lo, hi = e.loc["all"].low.unstack("seasons"), e.loc["all"].high.unstack("seasons")
    clear = (lo > 0) | (hi < 0)
    faded = [x for x in ["12", "21", "13", "22"] if clear.loc[x, "2016-2020"] and not clear.loc[x, "2021-2025"]]
    s = pd.read_csv(R / "situations.csv", dtype={"grouping": str}).set_index(["situation", "grouping"])
    goal = s.loc[("Inside the 5", "22 (6 OL)")]
    assert goal.low > 0, "the inside-the-5 limit no longer holds"
    assert halves.notna().all().all()
    ts = pd.read_csv(R / "usage_by_team_season.csv").sort_values("13", ascending=False)
    top13 = ts.iloc[0]
    rams_rank = ordinal(rank["LA"])
    rams = pd.read_csv(R / "team_weeks_2025.csv", dtype={"team": str}).query("team == 'LA'").reset_index(drop=True)
    start = int((rams["13"] > 0).idxmax())  # first game with any 13 personnel
    late = rams.iloc[start:]
    rams_late = (f"{100 * (late['13'] * late.plays).sum() / late.plays.sum():.0f}% over their last {len(late)} games, "
                 f"after none in their first {start}")
    assert (top13.team, top13.season) == ("LA", 2025), "the Rams sentence no longer holds"
    most11 = teams["11"].nlargest(10).index
    bottom = teams.epa.nsmallest(10).index
    n_bad = len(set(most11) & set(bottom))
    season = pd.read_csv(R / "usage_by_season.csv", index_col="season")
    assert teams[["11", "12", "13", "21", "22"]].sum().idxmax() == "11"
    six_of_other = season.loc[2025, "other_6ol"] / season.loc[2025, "Other"]
    assert 0.7 <= six_of_other < 0.75, "the Other sentence no longer holds"

    text = f"""# NFL Personnel Groupings

The NFL seems to be transitioning towards a new era of offense, one that uses heavier personnel packages more
frequently. Where does this shift come from and is it justified?

![How often each 2025 offense used each personnel grouping](figures/teams_2025.png)

Notably, the Rams used 13 personnel on {100 * top13["13"]:.0f}% of their plays ({rams_late}), the most by any team since
2016, and a significant part of their dominant 2025 offense ({rams_rank} in EPA per play).

While 11 personnel was still clearly the most used personnel last season, {n_bad} of the top 10 teams in 11 personnel
usage were in the bottom 10 in efficiency.

Note, nearly three quarters of the Other category is groupings with a sixth offensive lineman.

The impacts of all this were felt in the offseason, where 9 TEs were drafted in the first 95 picks
([2026 draft](https://en.wikipedia.org/wiki/2026_NFL_draft)), showing teams' intentions to gain personnel flexibility.

So, is this trend towards heavier personnel packages justified, or are teams overreacting?

## Method

- We use every regular-season run and dropback from 2016 to 2025, with personnel data from nflverse. Play-action
  comes from FTN's charting, which starts in 2022.
- We drop garbage time: plays where either team had over a 90% chance to win.
- We measure efficiency with expected points added (EPA) per play: how much a play raised or lowered the offense's
  expected points.
- We compare each grouping to the same team's 11 personnel plays from that season, on similar downs, distances and
  field position. For example, we compare the Rams' 13 personnel plays to the Rams' own 11 personnel plays.
- To see how much of each difference could be luck, we rerun the comparison on 200 random redraws of the games.
  The middle 90% of those reruns is the 90% interval. We call a difference clear when that whole interval is above
  or below zero.

## Results

![EPA per run and per dropback by personnel grouping](figures/run_pass.png)

In every heavier grouping, offenses were less efficient running and more efficient dropping back than in 11
personnel, though only the dropback gap in 12 personnel is clear. Dropbacks out of 12 personnel gained {pass12:.3f}
more EPA than dropbacks out of 11 personnel, and runs out of 12 personnel gained {abs(run12):.3f} less.

Most of that running gap goes away once you account for how defenses line up. Against 12 personnel, defenses were
in base on {base12:.0f}% of snaps, compared with {base11:.0f}% against 11 personnel. They also put {box12:.1f} defenders
in the box on average, instead of {box11:.1f}. Against the same number of defenders in the box, runs out of 12
personnel trailed runs out of 11 personnel by {abs(same12):.3f} EPA per play, and the gap is no longer clear. For 13,
21 and 22 personnel, it is within {max_same:.3f} of zero.

That loaded box is what play-action takes advantage of. From 2022 to 2025, offenses used play-action on {pa12:.0f}% of
dropbacks out of 12 personnel and {pa11:.0f}% out of 11 personnel. Play-action dropbacks gained more EPA per play than
other dropbacks or runs in every one of those seasons. In {last}, they gained {pa_last:+.3f} EPA per play, compared with
{other_last:+.3f} for other dropbacks and {run_last:+.3f} for runs.

So the value of heavier personnel is less about putting a player labeled as a 'TE' on the field instead of one
labeled as a 'WR', and more about what it forces the defense to do. Making a defense respect the run, then throwing
against it, is how offenses are attacking defenses best. That makes personnel flexibility, and marrying the run and
the pass, perhaps more important than living in any one grouping. But as more offenses have leaned on heavier
personnel, have those gaps held up?

![Each heavier grouping's gap to 11 personnel on runs and dropbacks, by season](figures/over_time.png)

For 12 personnel, they have narrowed. Runs out of 12 personnel were clearly less efficient than runs out
of 11 personnel in {n_run12} of {n_seasons} seasons, but not in {last}. In {last}, no heavier grouping clearly
differed from 11 personnel, on runs or on dropbacks.

Defenses have also changed how they line up against 12 personnel. They were in base on {b_first:.0f}% of its
snaps in {first_season}, and {b_low:.0f}% in {low_season}, before rising to {b_last:.0f}% in {last}. A lighter
box leaves runs less to overcome, and play-action less to exploit.

Defenses have adjusted to 11 personnel too, mostly by finding ways to take away its passing game. EPA per dropback
out of 11 personnel fell from {e_peak:+.3f} in {peak_season} to {e_low:+.3f} in {low_season11}, before partly
recovering. This may help explain why other personnel packages, which force the defense to stay sound in the run
game, have found more efficiency in the pass game, even while giving up some in the run game.

So there is no clean answer to whether the shift toward heavier personnel is justified. More than anything, this
shows how NFL schemes on both sides of the ball are constantly evolving, reacting to each other and to the rest of
the league.

## Limits

- Teams choose their personnel, so these gaps describe what happened, not what a grouping causes. The comparison
  accounts for the team and the situation, but not the play call or how the defense lines up.
- Offenses were clearly less efficient in {" and ".join(faded)} personnel than in 11 personnel from 2016 to 2020, but
  not from 2021 to 2025.
- Inside the 5-yard line, offenses gained {goal.effect:.2f} more EPA per play in 22 personnel with a sixth lineman
  than in 11 personnel (90% interval: {goal.low:.2f} to {goal.high:.2f}).

## Run it

```bash
pip install -e ".[dev]"
python scripts/pull.py
python scripts/build.py
python scripts/analyze.py    # will take a long time
python scripts/defense.py
python scripts/figures.py
python scripts/readme.py
pytest && ruff check .
```

## Credits

[nflverse](https://github.com/nflverse/nflverse-data) play-by-play (CC BY 4.0) and participation, from NFL Next
Gen Stats for 2016-2022 and FTN Data for 2023-2025 (CC BY-SA 4.0). The results and charts are CC BY-SA 4.0.
"""
    (ROOT / "README.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
