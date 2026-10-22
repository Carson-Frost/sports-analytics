"""The call for one fourth down.

    python scripts/call.py --distance 2 --spot "opp 45" --quarter 4 --clock 2:00 --score -3
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fourth_down.decide import FG_OFFSET, choice_wp, submodels_from_json
from fourth_down.wp import WinProbability

ROOT = Path(__file__).resolve().parents[1]
SEASON = 2025
CLEAR = 0.01


def yardline_100(spot):
    """'own 35' -> 65, 'opp 20' -> 20, '50' -> 50: yards to the opponent's goal line."""
    side, _, yards = spot.lower().rpartition(" ")
    yards = int(yards)
    return 100 - yards if side == "own" else yards


def state(a):
    m, s = a.clock.split(":")
    left = int(m) * 60 + int(s)
    game = (4 - a.quarter) * 900 + left
    half = game - 1800 if a.quarter <= 2 else game
    home = [1, 0] if a.home is None else [int(a.home)]  # unknown: average home and away
    return pd.DataFrame({
        "score_differential": a.score, "game_seconds_remaining": game, "half_seconds_remaining": half,
        "yardline_100": yardline_100(a.spot), "down": 4, "ydstogo": a.distance,
        "posteam_timeouts_remaining": a.timeouts, "defteam_timeouts_remaining": a.opp_timeouts,
        "receive_2h_ko": int(a.gets_second_half_kickoff and a.quarter <= 2), "posteam_spread": a.spread,
        "season": SEASON, "home": home,
    })


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--distance", type=int, required=True, help="yards to go")
    p.add_argument("--spot", required=True, help='"own 35", "opp 20" or "50"')
    p.add_argument("--quarter", type=int, required=True, choices=[1, 2, 3, 4])
    p.add_argument("--clock", required=True, help="time left in the quarter, M:SS")
    p.add_argument("--score", type=int, required=True, help="offense minus defense")
    p.add_argument("--timeouts", type=int, default=3)
    p.add_argument("--opp-timeouts", type=int, default=3)
    p.add_argument("--spread", type=float, default=0.0, help="points the offense was favored by")
    p.add_argument("--home", type=int, choices=[0, 1], help="1 if the offense is at home")
    p.add_argument("--gets-second-half-kickoff", action="store_true")
    a = p.parse_args()

    saved = json.loads((ROOT / "models" / "submodels.json").read_text())
    wp, s = WinProbability(), state(a)
    point = choice_wp(wp, submodels_from_json(saved["point"]), s).mean()
    edges = []
    for d in saved["resamples"]:
        w = choice_wp(wp, submodels_from_json(d), s).mean()
        edges.append(w.go - max(w.fg, w.punt))
    low, high = np.percentile(edges, [5, 95]) * 100

    kick = "fg" if point.fg > point.punt else "punt"
    edge = (point.go - point[kick]) * 100
    kick_yards = yardline_100(a.spot) + FG_OFFSET
    in_range = kick_yards <= saved["point"]["fg_longest"]

    print(f"go    {100 * point.go:.1f}%")
    fg = f"{100 * point.fg:.1f}%" if in_range else "-"
    print(f"fg    {fg}  ({kick_yards:.0f} yards)")
    print(f"punt  {100 * point.punt:.1f}%")
    name = {"go": "Go for it", "fg": "Field goal", "punt": "Punt"}
    other = {"go": "going for it", "fg": "a field goal", "punt": "a punt"}
    if edge > 0:
        best, vs, by, lo, hi = "go", kick, edge, low, high
    else:
        best, vs, by, lo, hi = kick, "go", -edge, -high, -low
    clear = "Clear call." if lo >= CLEAR * 100 else "Too close to call."
    print(f"\n{name[best]}: +{by:.1f} over {other[vs]} (90%: {lo:+.1f} to {hi:+.1f}). {clear}")


if __name__ == "__main__":
    main()
