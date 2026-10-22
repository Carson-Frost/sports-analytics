"""Every number the README quotes, from the results CSVs -> results/readme_numbers.csv."""

from pathlib import Path

import pandas as pd

from overtime.data import DERIVED

RESULTS = Path("results")


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    s = pd.read_csv(RESULTS / "scenarios.csv")
    f = pd.read_csv(RESULTS / "first_drive.csv").set_index(["rule", "first_drive"])
    c = pd.read_csv(RESULTS / "model_check.csv").set_index(["rule", "clock"])
    d = pd.read_csv(RESULTS / "drive_check.csv").set_index("start")
    t = pd.read_csv(RESULTS / "today_model.csv").iloc[0]
    plays = pd.read_parquet(DERIVED / "plays.parquet", columns=["season"])
    games = pd.read_csv(DERIVED / "ot_games.csv")

    rows = {
        "plays": f"{len(plays):,}",
        "ot_games": f"{len(games)}",
        "today_snaps": f"{int(t.snaps):,}",
        "today_kickoffs": f"{int(t.kickoffs):,}",
        "kickoff_start": f"{t.kickoff_start:.1f}",
        "fg_50_yards": pct(t.fg_50),
        "xp": pct(t.xp),
        "two_point": pct(t.two),
    }
    for row in s.itertuples():
        key = f"{row.rule}_{row.clock.replace(' ', '').lower()}"
        key += "_go2" if row.go_for_two else ""
        rows[f"{key}_receive"] = pct(row.receiver_win)
        rows[f"{key}_tie"] = pct(row.tie)
        rows[f"{key}_kick"] = pct(row.kicker_win)
    for (rule, first), row in f.iterrows():
        rows[f"first_{rule}_{first}_share"] = pct(row.share)
        rows[f"first_{rule}_{first}_receiver_win"] = pct(row.receiver_win)
    for rule in ["sudden", "modified", "both"]:
        era = c.loc[(rule, "All")]
        rows[f"check_{rule}_games"] = f"{int(era.games)}"
        rows[f"check_{rule}_seasons"] = era.seasons
        rows[f"check_{rule}_real"] = pct(era.observed_share)
        rows[f"check_{rule}_real_low"] = pct(max(era.observed_low, 0))
        rows[f"check_{rule}_real_high"] = pct(min(era.observed_high, 1))
        rows[f"check_{rule}_model"] = pct(era.model_share)
    timed = c[c.index.get_level_values("clock").isin(["10 Min", "15 Min"])]
    rows["ties_timed_games"] = f"{int(timed.games.sum())}"
    rows["ties_timed_model"] = f"{(timed.games * timed.model_tie).sum():.0f}"
    rows["ties_timed_real"] = f"{int(timed.ties.sum())}"
    for col in ["actual_td", "model_td", "actual_fg", "model_fg"]:
        rows[f"drive_{col}"] = pct(d.loc["All Drives", col])
    rows["drive_count"] = f"{int(d.loc['All Drives', 'drives']):,}"

    out = pd.DataFrame(rows.items(), columns=["name", "value"])
    out.to_csv(RESULTS / "readme_numbers.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
