"""Run every simulation and write the results CSVs.

results/drive_check.csv   model fit on 2021-2024 vs every close 2025 regulation drive
results/model_check.csv   each rule set on its own era vs the real overtime games
results/scenarios.csv     today's game under each rule set, playoff and regular season clocks
results/first_drive.csv   how the receiver's first drive ends, and what happens after
"""

import random
from pathlib import Path

import pandas as pd

from overtime import sim
from overtime.data import DERIVED

N = 100_000
TODAY_SNAPS, TODAY_KICKS = range(2023, 2027), range(2025, 2027)
RULES = ["sudden", "modified", "both"]
CLOCKS = {0: "Untimed", 600: "10 Min", 900: "15 Min"}
OUT = Path("results")


def drive_check(plays):
    m = sim.fit(plays, range(2021, 2025))
    p = plays[(plays.season == 2025) & (plays.qtr <= 4) & plays.play_type.isin(["run", "pass"])]
    starts = p.sort_values("play_id").groupby(["game_id", "fixed_drive"]).first()
    starts = starts[(starts.score_differential.abs() <= 8) & (starts.half_seconds_remaining > 300)
                    & (starts.fixed_drive_result != "End of half")]
    rng = random.Random(2025)
    rows = []
    for yl, result in zip(starts.yardline_100, starts.fixed_drive_result, strict=True):
        sims = [sim.drive(m, int(yl), "normal", None, rng)[0] for _ in range(20)]
        rows.append({"yl": yl, "actual_td": result == "Touchdown",
                     "actual_fg": result == "Field goal",
                     "model_td": sims.count("td") / 20, "model_fg": sims.count("fg") / 20})
    d = pd.DataFrame(rows)
    d["band"] = pd.cut(d.yl, [0, 39, 59, 79, 99],
                       labels=["Opponent 39 In", "Own 41 to Opp 40", "Own 21 to 40", "Own 1 to 20"])
    cols = ["actual_td", "model_td", "actual_fg", "model_fg"]
    out = d.groupby("band", observed=True)[cols].mean()
    out.insert(0, "drives", d.groupby("band", observed=True).size())
    total = d[cols].mean().to_frame("All Drives").T
    total.insert(0, "drives", len(d))
    return pd.concat([out, total]).round(4).rename_axis("start").reset_index()


def model_check(plays, games):
    rows = []
    for (rule, clock), g in games.groupby(["rule", "clock"]):
        seasons = sorted(g.season.unique())
        shares, _ = sim.simulate(sim.fit(plays, seasons), rule, clock or None, N)
        rows.append({"rule": rule, "clock": CLOCKS[clock],
                     "seasons": f"{seasons[0]}-{seasons[-1]}", "games": len(g),
                     "rec_wins": (g.outcome == 1).sum(), "ties": (g.outcome == 0).sum(),
                     "kick_wins": (g.outcome == -1).sum(),
                     "model_share": shares["receiver_win"] + shares["tie"] / 2,
                     "model_tie": shares["tie"]})
    d = pd.DataFrame(rows)
    eras = []
    for rule, g in d.groupby("rule"):
        w = g.games / g.games.sum()
        eras.append({"rule": rule, "clock": "All",
                     "seasons": f"{min(s[:4] for s in g.seasons)}-{max(s[5:] for s in g.seasons)}",
                     "games": g.games.sum(), "rec_wins": g.rec_wins.sum(), "ties": g.ties.sum(),
                     "kick_wins": g.kick_wins.sum(), "model_share": (w * g.model_share).sum(),
                     "model_tie": (w * g.model_tie).sum()})
    d = pd.concat([d, pd.DataFrame(eras)], ignore_index=True)
    d["observed_share"] = (d.rec_wins + d.ties / 2) / d.games
    p, n, z = d.observed_share, d.games, 1.96  # Wilson 95% interval, a tie as half a win
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / (1 + z * z / n)
    d["observed_low"], d["observed_high"] = center - half, center + half
    return d.round(4)


def today(plays):
    m = sim.fit(plays, TODAY_SNAPS, TODAY_KICKS)
    scenarios, firsts = [], []
    for rule in RULES:
        for clock in (0, 600):
            for two in ([False, True] if rule == "both" else [False]):
                shares, first = sim.simulate(m, rule, clock or None, N, go_for_two=two)
                scenarios.append({"rule": rule, "clock": CLOCKS[clock], "go_for_two": two,
                                  **shares})
                if clock == 0 and not two:
                    firsts += [{"rule": rule, "first_drive": k, "share": s, "receiver_win": w}
                               for k, (s, w) in first.items()]
    starts = [100 - spot for has, spot, td, _ in m.kickoffs if has and not td]
    fit_facts = {"snaps": sum(len(v) for v in m.snaps_pooled.values()),
                 "kickoffs": len(m.kickoffs), "kickoff_start": sum(starts) / len(starts),
                 "fg_50": m.fg_chance(33), "xp": m.xp, "two": m.two}
    return (pd.DataFrame(scenarios).round(4), pd.DataFrame(firsts).round(4),
            pd.DataFrame([fit_facts]).round(4))


def main():
    plays = pd.read_parquet(DERIVED / "plays.parquet")
    games = pd.read_csv(DERIVED / "ot_games.csv")
    OUT.mkdir(exist_ok=True)
    drive_check(plays).to_csv(OUT / "drive_check.csv", index=False)
    model_check(plays, games).to_csv(OUT / "model_check.csv", index=False)
    scenarios, firsts, facts = today(plays)
    scenarios.to_csv(OUT / "scenarios.csv", index=False)
    firsts.to_csv(OUT / "first_drive.csv", index=False)
    facts.to_csv(OUT / "today_model.csv", index=False)
    print(scenarios.to_string(index=False))


if __name__ == "__main__":
    main()
