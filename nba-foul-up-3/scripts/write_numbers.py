"""Every number the README quotes, written to results/numbers.csv (plus results/foul_rate.csv)."""

from pathlib import Path

import numpy as np
import pandas as pd

from foulup3 import SEASONS
from foulup3.model import decisions, outcome_arrays

SEGMENTS = Path("data/derived/segments.parquet")
RAW = Path("data/raw")
EXAMPLE_GAME = "0022400942"  # Bucks at Pacers, March 11, 2025
RESULTS = Path("results")
RATE_BINS = [0, 3, 6, 9, 12, 16, 20, 24]
HEADLINE_SECONDS = [2, 4, 6, 8, 10, 12, 15, 20]


def main():
    segs = pd.read_parquet(SEGMENTS)
    wp = pd.read_csv(RESULTS / "win_probability.csv")
    held = pd.read_csv(RESULTS / "heldout.csv").set_index("model")
    band = pd.read_csv(RESULTS / "bandwidth.csv")
    one = pd.read_csv(RESULTS / "one_possession.csv")
    rows = []

    def add(name, value, note):
        rows.append(dict(name=name, value=value, note=note))

    add("seasons", segs.season.nunique(), "regular seasons of play-by-play")
    add("first_season", segs.season.min(), "")
    add("last_season", segs.season.max(), "")
    pulled = sum(
        pd.read_parquet(RAW / f"games_{x[:4]}{x[-2:]}_RegularSeason.parquet").GAME_ID.nunique()
        for x in SEASONS
    )
    add("games_pulled", pulled, "regular season games pulled")
    add("games", segs.game_id.nunique(), "games with a trip starting at 24 seconds or less")
    add("segments", len(segs), "segments starting with 24 seconds or less")
    arrays = outcome_arrays(segs)
    for c in ["down 3, defend", "down 3, foul"]:
        add(f"segments_{c.split(', ')[1]}", len(arrays[c]["t0"]), f"segments in class {c}")

    d = decisions(segs)
    add("decisions", len(d[d.action != "foul to give"]), "trailing team got the ball down 3")
    for a in ["defend", "foul", "foul to give"]:
        g = d[d.action == a]
        key = a.replace(" ", "_")
        add(f"decisions_{key}", len(g), f"decisions where the leader chose {a}")
        add(f"observed_win_{key}", round(100 * g.leader_won.mean(), 1), "leader win %, raw")
    choice = d[d.action != "foul to give"]
    add(
        "bonus_share",
        round(100 * choice.bonus.mean(), 1),
        "% of decisions where the leader was already in the bonus",
    )
    fouled = d[d.action != "defend"]
    add(
        "bonus_share_fouled",
        round(100 * fouled.bonus.mean(), 1),
        "% of leader fouls (with or without free throws) made in the bonus",
    )
    add("foul_share", round(100 * (choice.action == "foul").mean(), 1), "% of decisions fouled")

    rate = choice.assign(bin=pd.cut(choice.t0, RATE_BINS)).groupby("bin", observed=True)
    rate = pd.DataFrame(
        dict(
            low=[b.left for b in rate.groups],
            high=[b.right for b in rate.groups],
            decisions=rate.size().values,
            foul_share=rate.action.agg(lambda a: (a == "foul").mean()).values,
        )
    )
    rate.to_csv(RESULTS / "foul_rate.csv", index=False, float_format="%.4f")
    for r in rate.itertuples():
        add(
            f"foul_share_{r.low}_{r.high}s",
            round(100 * r.foul_share, 1),
            f"% fouled with {r.low} to {r.high} seconds left",
        )

    for s in HEADLINE_SECONDS:
        r = wp.iloc[(wp.seconds - s).abs().idxmin()]
        foul, defend = round(100 * r.foul, 1), round(100 * r.defend, 1)
        add(f"foul_{s}s", foul, f"leader win % fouling, {s} seconds left")
        add(f"defend_{s}s", defend, f"leader win % defending, {s} s left")
        add(f"gap_{s}s", round(foul - defend, 1), f"foul minus defend, points, {s} s")
        add(f"gap_low_{s}s", round(100 * r.gap_low, 1), "bootstrap 2.5th percentile")
        add(f"gap_high_{s}s", round(100 * r.gap_high, 1), "bootstrap 97.5th percentile")
    whole = wp[np.isclose(wp.seconds, wp.seconds.round())]
    add(
        "crossover",
        round(wp[wp.gap > 0].seconds.max(), 1),
        "last tenth of a second where fouling is ahead",
    )
    better = whole[whole.gap > 0].seconds
    add(
        "foul_better_up_to",
        int(better.max()) if len(better) else 0,
        "largest whole second where fouling is ahead",
    )
    sure = whole[whole.gap_low > 0].seconds
    add(
        "foul_clearly_better_up_to",
        int(sure.max()) if len(sure) else 0,
        "largest whole second where the 95% interval is above zero",
    )
    worse = whole[whole.gap_high < 0].seconds
    add(
        "foul_clearly_worse_from",
        int(worse.min()) if len(worse) else 0,
        "smallest whole second where the 95% interval is below zero",
    )

    for _, r in one.iterrows():
        key = f"{r.choice}_{r.outcome}".lower().replace(", ", "_").replace(" ", "_")
        add(f"chance_{key}", round(100 * r.chance, 1), "% of trips at 10 seconds")
        add(f"after_{key}", round(100 * r.leader_wins_after, 1), "leader win % after it")
        add(f"seconds_{key}", round(r.seconds, 1), "seconds the trip took")

    ex = d[d.game_id == EXAMPLE_GAME].iloc[0]  # the game in The Question
    games = pd.read_parquet(RAW / "games_202425_RegularSeason.parquet")
    pts = games[games.GAME_ID == EXAMPLE_GAME].set_index("TEAM_ID").PTS
    day = pd.Timestamp(games[games.GAME_ID == EXAMPLE_GAME].GAME_DATE.iloc[0])
    add("example_date", f"{day:%B} {day.day}, {day.year}", "")
    add("example_seconds", ex.t0, "seconds left when the trailing team got the ball")
    add("example_shot_seconds", ex.t1, "seconds left at the shot")
    add("example_points", ex.delta, "points on the trip")
    add("example_trailer_before", pts[ex.off] - ex.delta, "")
    add("example_trailer_final", pts[ex.off], "")
    add("example_leader_final", pts.drop(ex.off).iloc[0], "")

    best = band.loc[band.log_loss.idxmin()]
    add("bandwidth", best.bandwidth, "seconds, chosen on the validation seasons")
    add("validation_segments", int(best.segments), "trips in the validation seasons")
    add("validation_loss_best", round(best.log_loss, 3), "outcome log loss at that bandwidth")
    near = band[band.log_loss <= best.log_loss + 0.001]  # windows that do as well as the best
    add("bandwidth_tied_low", near.bandwidth.min(), "smallest window within 0.001 of the best")
    add("bandwidth_tied_high", near.bandwidth.max(), "largest window within 0.001 of the best")
    add(
        "crossover_low",
        round(band.crossover[band.bandwidth.between(3, 6)].min(), 1),
        "smallest crossover over windows of 3 to 6 seconds",
    )
    add(
        "crossover_high",
        round(band.crossover[band.bandwidth.between(3, 6)].max(), 1),
        "largest crossover over windows of 3 to 6 seconds",
    )
    narrow = band.loc[band.bandwidth.idxmin()]
    add("validation_loss_narrowest", round(narrow.log_loss, 3), f"at {narrow.bandwidth:g} s")
    model = held.loc["Possession Model"]
    add("test_decisions", int(model.decisions), "held-out decisions")
    for name, key in [
        ("Possession Model", "model"),
        ("Observed Win Rate", "baseline"),
        ("Constant Guess", "constant"),
    ]:
        add(f"test_loss_{key}", round(held.loc[name, "log_loss"], 4), f"log loss, {name}")
    for name, key in [("Observed Win Rate", "baseline"), ("Constant Guess", "constant")]:
        add(
            f"test_z_{key}",
            round(held.loc[name, "z_model_minus_this"], 2),
            f"z of model minus {name}, paired by decision",
        )
    add("test_predicted", round(100 * model.mean_predicted, 1), "mean predicted leader win %")
    add("test_observed", round(100 * model.observed, 1), "observed leader win %")

    pd.DataFrame(rows).to_csv(RESULTS / "numbers.csv", index=False)


if __name__ == "__main__":
    main()
