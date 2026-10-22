"""Write every number the README quotes to results/numbers.csv, as the README prints it."""

from pathlib import Path

import numpy as np
import pandas as pd
from model import BOOTSTRAPS, MIN_MINUTES
from model import TEST_SEASON as SEASON

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def signed(x):
    return f"{x:+.1f}"


def main():
    checks = pd.read_csv(RESULTS / "data_checks.csv")
    tuning = pd.read_csv(RESULTS / "lambda_tuning.csv")
    test = pd.read_csv(RESULTS / "model_test.csv")
    ratings = pd.read_csv(RESULTS / "ratings_2025_26.csv")
    stints = pd.read_csv(ROOT / "artifacts" / "stints.csv.gz")
    season = stints[stints["season"] == SEASON]
    n = {}

    # the data
    bundles = [
        ROOT / "data" / "raw" / "bundles" / f"nba_stats_json_{y}.tar.gz" for y in range(2015, 2027)
    ]
    n["bundles_gb"] = f"{sum(b.stat().st_size for b in bundles) / 1e9:.1f} GB"
    n["seasons"] = len(checks)
    n["tuning_seasons"] = len(checks) - 1
    n["min_minutes"] = f"{MIN_MINUTES:,}"
    n["bootstraps"] = BOOTSTRAPS
    n["games_all"] = f"{checks['games'].sum():,}"
    n["games_tuning"] = f"{checks.loc[checks['season'] != SEASON, 'games'].sum():,}"
    n["games_dropped"] = checks["games_dropped"].sum()
    n["rotation_hours"] = f"{checks['rotation_hours'].sum():,.0f}"
    hours = checks["rotation_hours"]
    match = (hours * checks["rotation_match"]).sum() / hours.sum()
    n["rotation_match"] = f"{100 * match:.1f}%"
    n["rotation_match_worst"] = f"{100 * checks['rotation_match'].min():.1f}%"
    n["games_season"] = f"{season['game_id'].nunique():,}"
    n["stints_season"] = f"{len(season):,}"
    n["players_season"] = len(ratings)
    q = ratings[ratings["minutes"] >= MIN_MINUTES].copy()
    n["qualified"] = len(q)

    # choosing lambda
    best = tuning.loc[tuning["rmse"].idxmin()]
    n["lambda"] = f"{best['lambda']:,.0f}"
    n["tuning_games"] = f"{int(best['games']):,}"
    n["tuning_rmse_best"] = f"{best['rmse']:.2f}"
    n["tuning_rmse_ols"] = f"{tuning.loc[tuning['lambda'] == 0, 'rmse'].iloc[0]:.2f}"
    near = tuning[tuning["rmse"] <= best["rmse"] + 0.05]["lambda"]
    n["lambda_flat_low"], n["lambda_flat_high"] = f"{near.min():,.0f}", f"{near.max():,.0f}"

    # the test
    short = {
        "home court": "home",
        "team net rating": "team",
        "raw plus-minus": "raw",
        "adjusted plus-minus": "apm",
        "regularized adjusted plus-minus": "rapm",
    }
    for _, row in test.iterrows():
        tag = "test" if row["sample"] == SEASON else "past"
        n[f"{tag}_rmse_{short[row['method']]}"] = f"{row['rmse']:.2f}"
        if not np.isnan(row["z_rapm_vs_method"]):
            n[f"{tag}_z_{short[row['method']]}"] = f"{row['z_rapm_vs_method']:.1f}"
        n[f"{tag}_games"] = f"{row['games']:,}"

    # raw plus-minus is mostly the team
    n["r2_raw_team"] = f"{100 * np.corrcoef(q['raw_plus_minus'], q['team_net'])[0, 1] ** 2:.0f}%"
    n["r2_rapm_team"] = f"{100 * np.corrcoef(q['rapm'], q['team_net'])[0, 1] ** 2:.0f}%"
    q["raw_rank"] = q["raw_plus_minus"].rank(ascending=False, method="min").astype(int)
    q["rapm_rank"] = q["rapm"].rank(ascending=False, method="min").astype(int)
    for tag, name in (("dort", "Luguentz Dort"), ("markkanen", "Lauri Markkanen")):
        p = q[q["name"] == name].iloc[0]
        n[f"{tag}_raw"] = signed(p["raw_plus_minus"])
        n[f"{tag}_raw_rank"] = p["raw_rank"]
        n[f"{tag}_rapm"] = signed(p["rapm"])
        n[f"{tag}_rapm_rank"] = p["rapm_rank"]
        n[f"{tag}_team_net"] = signed(p["team_net"])
    top_raw = q.sort_values("raw_plus_minus", ascending=False).iloc[0]
    n["top_raw_name"], n["top_raw"] = top_raw["name"], signed(top_raw["raw_plus_minus"])
    n["top_raw_rapm_rank"] = top_raw["rapm_rank"]

    # the leaders
    top = q.sort_values("rapm", ascending=False)
    first = top.iloc[0]
    n["first_name"], n["first_rapm"] = first["name"], signed(first["rapm"])
    n["first_low"], n["first_high"] = signed(first["low"]), signed(first["high"])
    for i, tag in ((1, "second"), (2, "third")):
        n[f"{tag}_name"], n[f"{tag}_rapm"] = top.iloc[i]["name"], signed(top.iloc[i]["rapm"])
    by_firsts = q.sort_values("first_in_resamples", ascending=False)
    for i in range(3):
        n[f"firsts_{i + 1}_name"] = by_firsts.iloc[i]["name"]
        n[f"firsts_{i + 1}"] = int(by_firsts.iloc[i]["first_in_resamples"])
    n["median_interval_width"] = f"{(q['high'] - q['low']).median():.1f}"

    # does last season help?
    last = pd.read_csv(RESULTS / "last_season_test.csv").set_index("variant")
    for tag, variant in (
        ("last_games", "with 2024-25 games"),
        ("last_prior", "shrunk toward 2024-25 ratings"),
    ):
        n[f"{tag}_rmse"] = f"{last.loc[variant, 'rmse']:.2f}"
        n[f"{tag}_z"] = f"{last.loc[variant, 'z_vs_team_net']:.1f}"

    out = pd.DataFrame(list(n.items()), columns=["name", "value"])
    out.to_csv(RESULTS / "numbers.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
