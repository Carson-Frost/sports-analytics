"""The two README charts, from the CSVs in results/."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb

from fourth_down.charts import ACCENT, GREY, INK, LIGHT, MUTED, WIDTH, margins, save, titled

ROOT = Path(__file__).resolve().parents[1]
R, F = ROOT / "results", ROOT / "figures"
SOURCE = "Source: nflverse play-by-play; win probability from nflfastR"
CLEAR = 0.01  # same rule as scripts/grade.py
FG_COLOUR = "#C79A3B"
FIELD = {80: "Own 20", 70: "Own 30", 60: "Own 40", 50: "Midfield", 40: "Opp 40", 30: "Opp 30", 20: "Opp 20",
         10: "Opp 10"}


def blend(colour, share):
    return tuple(1 - share * (1 - c) for c in to_rgb(colour))


def go_rate(seasons):
    fig, ax = plt.subplots(figsize=(WIDTH, 4.2))
    x = seasons.season
    ax.vlines(x, seasons.go_rate_low, seasons.go_rate_high, color=ACCENT, lw=2, alpha=0.45)
    ax.plot(x, seasons.go_rate, color=ACCENT, lw=2.2, marker="o", ms=5)
    for season in (x.iloc[0], x.iloc[-1]):
        v = seasons.set_index("season").loc[season, "go_rate"]
        ax.text(season, v + 6, f"{v:.0f}%", ha="center", color=INK, fontweight="bold")
    ax.set_xticks(range(x.iloc[0], x.iloc[-1] + 1, 2))
    ax.set_xlim(x.iloc[0] - 0.6, x.iloc[-1] + 0.6)
    ax.set_ylim(0, 70)
    ax.set_yticks(range(0, 61, 20))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.grid(axis="y", color=LIGHT, lw=0.8)
    ax.set_axisbelow(True)
    margins(fig, top=0.95, left=0.6)
    titled(fig, "Since 2015, Teams Go for It Twice as Often When the Model Says Go",
           SOURCE + ". Bars are 90% intervals.",
           units="Share of fourth downs where the model clearly favored going for it and the team went")
    save(fig, F / "go_rate.png")


def recommendation_map(grid):
    """The best call in each square: full colour when it's a clear call, pale when it's too close."""
    from matplotlib.patches import Patch

    calls = ["go", "fg", "punt"]
    colours = {"go": ACCENT, "punt": GREY, "fg": FG_COLOUR}
    names = {"go": "Go for it", "punt": "Punt", "fg": "Field goal"}
    grid = grid.set_index(["ydstogo", "yardline_100"])
    rows = sorted(grid.index.get_level_values(0).unique())
    cols = sorted(grid.index.get_level_values(1).unique(), reverse=True)
    fig, ax = plt.subplots(figsize=(WIDTH, 6.1))
    for i, togo in enumerate(rows):
        for j, yl in enumerate(cols):
            r = grid.loc[(togo, yl)]
            ranked = r[calls].sort_values(ascending=False)
            best, margin = ranked.index[0], 100 * (ranked.iloc[0] - ranked.iloc[1])
            clear = r.low >= CLEAR or r.high <= -CLEAR
            if clear:
                fill = blend(colours[best], min(margin / 5, 1) * 0.6 + 0.4)
                ink = "white" if best == "go" and margin > 2.5 else INK
            else:
                fill, ink = blend(colours[best], 0.25), MUTED
            ax.add_patch(plt.Rectangle((j + 0.04, i + 0.06), 0.92, 0.88, facecolor=fill, lw=0))
            ax.text(j + 0.5, i + 0.5, f"+{margin:.1f}", ha="center", va="center", color=ink,
                    fontweight="bold" if clear else "normal")
    ax.set_xlim(0, len(cols))
    ax.set_ylim(len(rows), 0)
    ax.set_xticks(np.arange(len(cols)) + 0.5, [FIELD[yl] for yl in cols])
    ax.set_yticks(np.arange(len(rows)) + 0.5, [f"4th & {t}" for t in rows])
    ax.xaxis.tick_top()
    margins(fig, top=1.35, bottom=0.85, left=0.85)
    fig.legend(handles=[Patch(facecolor=colours[c], label=names[c]) for c in ("go", "punt", "fg")],
               loc="lower left", bbox_to_anchor=(0.11, 0.045), ncol=3, frameon=False, handlelength=1.2,
               labelcolor=INK)
    titled(fig, "Go for It, Punt, or Kick a Field Goal?",
           SOURCE + ". Pale squares: too close to call.",
           units="Model's best call by yards to go and field position, in a tie game at the start of the 2nd quarter.\n"
                 "Number: win probability gained over the next-best call, in percentage points.")
    save(fig, F / "recommendation_map.png")


def main():
    seasons = pd.read_csv(R / "by_season.csv")
    go_rate(seasons)
    recommendation_map(pd.read_csv(R / "recommendation_map.csv"))
    print("2 charts ->", F)


if __name__ == "__main__":
    main()
