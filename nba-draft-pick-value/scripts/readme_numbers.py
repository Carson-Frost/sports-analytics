"""Every number the README quotes, rounded as quoted, into results/numbers.csv."""

from pathlib import Path

import pandas as pd

from draft_value.curve import fit_log, halving_pick

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

draftees = pd.read_csv(ROOT / "data" / "draftees.csv")
fit = pd.read_csv(RESULTS / "fit.csv").set_index("parameter")
curve = pd.read_csv(RESULTS / "curve.csv").set_index("pick")
model_test = pd.read_csv(RESULTS / "model_test.csv").set_index("model")
first, second = draftees[draftees["round"] == 1], draftees[draftees["round"] == 2]

rows = []


def add(name: str, value: float, decimals: int = 1) -> None:
    rows.append({"name": name, "value": f"{float(value):.{decimals}f}"})


def fit_for(subset: pd.DataFrame, column: str = "win_shares") -> tuple[float, float]:
    return fit_log(subset["pick"].to_numpy(float), subset[column].to_numpy())


# the data
add("draftees", len(draftees), 0)
add("drafts", draftees["year"].nunique(), 0)
add("never_played", (draftees["games"] == 0).sum(), 0)
sizes = draftees.groupby("year")["pick"].max()
add("picks_per_draft_1990", sizes[1990], 0)
add("last_draft_with_54_picks", sizes[sizes == 54].index.max(), 0)
add("first_draft_with_60_picks", sizes[sizes == 60].index.min(), 0)
add("first_round_picks_1990", (first["year"] == 1990).sum(), 0)
no_29 = sorted(set(sizes.index) - set(draftees.loc[draftees["pick"] == 29, "year"]))
rows.append({"name": "drafts_missing_pick_29", "value": " ".join(map(str, no_29))})

# the curve
for p in ["a", "halving_pick", "doubling_cost"]:
    for col in ["estimate", "low_90", "high_90"]:
        add(f"{p}_{col}", fit.loc[p, col])
add("b", fit.loc["b", "estimate"], 2)
for p in [1, 2, 8, 10, 30, 60]:
    add(f"curve_pick_{p}", curve.loc[p, "curve_win_shares"])
for p in [6, 9]:
    add(f"average_pick_{p}", curve.loc[p, "average_win_shares"])
add("pick_3_plus_pick_14", curve.loc[3, "curve_win_shares"] + curve.loc[14, "curve_win_shares"])

# the rounds, as drafted
add("first_round_average", first["win_shares"].mean())
add("second_round_average", second["win_shares"].mean())
add("second_round_median", second["win_shares"].median())
add("first_round_never_played_pct", 100 * (first["games"] == 0).mean())
add("second_round_never_played_pct", 100 * (second["games"] == 0).mean(), 0)

# held-out test
for model in model_test.index:
    key = model.lower().replace(" ", "_")
    add(f"rmse_{key}", model_test.loc[model, "rmse"], 2)
    if model != "Log Curve":
        add(f"z_{key}", model_test.loc[model, "z_vs_log_curve"], 1)

# caveats
eras = {"1990_2005": draftees["year"] <= 2005, "2006_2016": draftees["year"] >= 2006}
for label, years in eras.items():
    a_part, b_part = fit_for(draftees[years])
    add(f"pick_1_value_{label}", a_part)
    add(f"halving_pick_{label}", halving_pick(a_part, b_part))
add("halving_pick_vorp", halving_pick(*fit_for(draftees, "vorp")))
pick_15 = draftees[draftees["pick"] == 15]
add("pick_15_average", pick_15["win_shares"].mean())
add("pick_15_median", pick_15["win_shares"].median())
brown = draftees[draftees["player"] == "Jaylen Brown"]
add("win_shares_brown_snapshot", brown["win_shares"].item())
for player in ["Steve Nash", "Giannis Antetokounmpo", "Kawhi Leonard"]:
    key = player.split()[-1].lower()
    add(f"win_shares_{key}", pick_15.loc[pick_15["player"] == player, "win_shares"].item())

numbers = pd.DataFrame(rows)
numbers.to_csv(RESULTS / "numbers.csv", index=False)
print(numbers.to_string(index=False))
