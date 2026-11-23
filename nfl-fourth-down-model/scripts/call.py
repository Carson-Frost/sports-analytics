"""The call for one fourth down.

    python scripts/call.py --distance 2 --spot "opp 45" --quarter 4 --clock 2:00 --score -3
"""

import argparse
import json
from pathlib import Path

from fourth_down.grading import FG_OFFSET, NAME, Grader, situation
from fourth_down.wp import WinProbability

ROOT = Path(__file__).resolve().parents[1]
OTHER = {"go": "going for it", "fg": "a field goal", "punt": "a punt"}


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
    args = p.parse_args()

    submodels = json.loads((ROOT / "models" / "submodels.json").read_text(encoding="utf-8"))
    grader = Grader.from_json(WinProbability(), submodels)
    state = situation(args.distance, args.spot, args.quarter, args.clock, args.score, args.timeouts,
                      args.opp_timeouts, args.spread, args.home, args.gets_second_half_kickoff)
    graded = grader.grade(state).iloc[0]
    kick_yards = state.yardline_100.iloc[0] + FG_OFFSET

    print(f"go    {100 * graded.go:.1f}%")
    fg = f"{100 * graded.fg:.1f}%" if kick_yards <= grader.point["fg_longest"] else "-"
    print(f"fg    {fg}  ({kick_yards:.0f} yards)")
    print(f"punt  {100 * graded.punt:.1f}%")
    verdict = "Clear call." if graded.clear else "Toss-up."
    print(f"\n{NAME[graded.call]}: +{100 * graded.gap:.1f} over {OTHER[graded.runner_up]} "
          f"(90%: {100 * graded.gap_low:+.1f} to {100 * graded.gap_high:+.1f}). {verdict}")


if __name__ == "__main__":
    main()
