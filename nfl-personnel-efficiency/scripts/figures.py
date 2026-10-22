"""The two README charts, from the CSVs in results/."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from personnel.charts import INK, MUTED, SERIES, WIDTH, margins, save, titled

LIGHT_GREY = "#b4b2a9"
TITLE_OVER_TIME = "In 2025, No Heavier Grouping Clearly Differed From 11 Personnel"

ROOT = Path(__file__).resolve().parents[1]
R, F = ROOT / "results", ROOT / "figures"
SOURCE = "Source: nflverse play-by-play and participation (NFL Next Gen Stats 2016-2022, FTN 2023-2025)"
MAIN = ["11", "12", "13", "21", "22"]


RAMP = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]  # light to dark blue


def shade(value, top):
    """A step of the blue ramp, darker as value nears the column's top value."""
    return RAMP[min(int(round(value / top * (len(RAMP) - 1))), len(RAMP) - 1)]


def teams(t):
    cols = [*MAIN, "Other"]
    fig, ax = plt.subplots(figsize=(WIDTH, 10.2))
    start = 1.15  # the EPA column comes first, since teams are sorted by it
    for i, team in enumerate(t.index):
        ax.text(0.8, i + 0.5, f"{round(t.loc[team, 'epa'], 2) + 0.0:+.2f}", ha="right", va="center", fontsize=9.5)
        for j, code in enumerate(cols):
            v = t.loc[team, code]
            fill = shade(v, t[code].max())
            ax.add_patch(plt.Rectangle((start + j + 0.03, i + 0.06), 0.94, 0.88, facecolor=fill, edgecolor="none"))
            dark = RAMP.index(fill) >= 4
            ax.text(start + j + 0.5, i + 0.5, f"{100 * v:.0f}%", ha="center", va="center", fontsize=9,
                    color="white" if dark else INK)
    ax.set_xlim(0, start + len(cols))
    ax.set_ylim(len(t), 0)
    names = [c if c == "Other" else f"{c}\npersonnel" for c in cols]
    ax.set_xticks([0.5, *(start + np.arange(len(cols)) + 0.5)], ["EPA/play", *names], fontweight="bold")
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(t)) + 0.5, t.index)
    margins(fig, top=1.25, left=0.6, right=0.15, bottom=0.4)
    titled(fig, "How Often Each Offense Used Each Personnel Grouping, 2025",
           "Source: nflverse play-by-play and participation (FTN Data)",
           units="Share of each team's plays. Teams sorted by EPA per play.")
    save(fig, F / "teams_2025.png")


def run_pass(e, g):
    """EPA per run and per dropback for each main grouping: 11's EPA plus each grouping's adjusted gap to 11."""
    e = e[e.seasons == "2016-2025"].set_index(["play_type", "grouping"])
    g = g.set_index("grouping")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 4.2), sharey=True)
    y = np.arange(len(MAIN))
    for ax, kind, name, colour in [(axes[0], "run", "Runs", SERIES[1]), (axes[1], "dropback", "Dropbacks", SERIES[0])]:
        base = g.loc["11", f"epa_{kind}"]
        d = e.loc[kind].reindex(MAIN)[["effect", "low", "high"]].fillna(0.0)  # 11 is the baseline
        level = base + d.effect
        colours = [LIGHT_GREY] + [colour] * (len(MAIN) - 1)
        ax.barh(y, level, height=0.62, color=colours)
        ax.hlines(y[1:], base + d.low.iloc[1:], base + d.high.iloc[1:], color=INK, lw=1.1)
        ax.axvline(base, color=MUTED, lw=0.9, ls=(0, (3, 3)))
        outer = np.where(level >= 0, np.maximum(level, base + d.high), np.minimum(level, base + d.low))
        for yi, v, edge in zip(y, level, outer, strict=True):
            ax.text(edge + (0.006 if v >= 0 else -0.006), yi, f"{v:+.3f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=9.5)
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_title(name, fontsize=12, fontweight="bold", color=colour, loc="center", pad=8)
        ticks = [-0.15, -0.1, -0.05, 0] if kind == "run" else [0, 0.05, 0.1, 0.15]
        ax.set_xticks(ticks, [f"{t:+.2f}" if t else "0" for t in ticks], fontsize=8.5, color=MUTED)
        ax.set_xlabel("EPA per play", fontsize=9.5)
        ax.set_xlim(-0.17, 0.005) if kind == "run" else ax.set_xlim(-0.005, 0.19)
    axes[0].set_yticks(y, [f"{code} personnel" for code in MAIN], fontweight="bold")
    axes[0].set_ylim(len(MAIN) - 0.5, -0.6)
    margins(fig, top=1.15, left=1.25, right=0.2, bottom=0.85, wspace=0.03)
    titled(fig, "Run vs. Dropback Efficiency by Personnel Package", SOURCE,
           units="EPA per play, 2016-2025, for the same offense, down, distance and field position as 11 personnel.\n"
                 "Grey bars and dashed lines: 11 personnel. Lines on bars: 90% intervals.")
    save(fig, F / "run_pass.png")


def over_time(e, usage):
    """Each heavier grouping's gap to 11 on runs and on dropbacks, season by season."""
    seasons = list(usage.index)
    e = e.set_index(["play_type", "seasons", "grouping"]).sort_index()
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 6.6), sharex=True, sharey=True)
    x = np.arange(len(seasons))
    for ax, code in zip(axes.flat, MAIN[1:], strict=True):
        for kind, colour in [("dropback", SERIES[0]), ("run", SERIES[1])]:
            d = e.loc[kind].xs(code, level="grouping").loc[[str(y) for y in seasons]]
            ax.fill_between(x, d.low, d.high, color=colour, alpha=0.15, lw=0)
            ax.plot(x, d.effect, color=colour, lw=2, marker="o", ms=3.5, clip_on=True)
        ax.axhline(0, color=MUTED, lw=0.9, ls=(0, (3, 3)))
        start, end = 100 * usage[code].iloc[0], 100 * usage[code].iloc[-1]
        share = f"{start:.0f}% of plays in {seasons[0]}, {end:.0f}% in {seasons[-1]}"
        ax.set_title(f"{code} personnel", loc="left", fontsize=11, fontweight="bold")
        ax.text(1, 1.02, share, transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=MUTED)
        ax.yaxis.grid(color="#E3E3E0", lw=0.6)
        ax.set_axisbelow(True)
    ax0 = axes[0, 0]
    ax0.set_ylim(-0.25, 0.25)
    ax0.set_yticks([-0.2, -0.1, 0, 0.1, 0.2], ["-0.2", "-0.1", "0", "+0.1", "+0.2"])
    for ax in axes[:, 0]:
        ax.set_ylabel("EPA per play vs. 11 personnel", fontsize=9.5)
    key_y = 0.985 - 0.5 / fig.get_figheight()  # beside the shorter second subtitle line
    fig.text(0.98, key_y, "Runs", color=SERIES[1], fontweight="bold", ha="right", va="top")
    fig.text(0.91, key_y, "Dropbacks", color=SERIES[0], fontweight="bold", ha="right", va="top")
    for ax in axes[1]:
        ax.set_xticks(x[::3], [str(y) for y in seasons[::3]])
    margins(fig, top=1.15, left=0.8, right=0.15, bottom=0.6)
    fig.subplots_adjust(hspace=0.3, wspace=0.08)
    titled(fig, TITLE_OVER_TIME, SOURCE,
           units="Difference in EPA per play from 11 personnel, for the same offense, down, distance and field "
                 "position.\nAbove the dashed line: more EPA per play than 11 personnel. Bands: 90% intervals.")
    save(fig, F / "over_time.png")


def main():
    teams(pd.read_csv(R / "teams_2025.csv", index_col="team"))
    run_pass(pd.read_csv(R / "effects.csv", dtype={"grouping": str}),
             pd.read_csv(R / "groupings.csv", dtype={"grouping": str}))
    over_time(pd.read_csv(R / "effects.csv", dtype={"grouping": str}),
              pd.read_csv(R / "usage_by_season.csv", index_col="season"))
    print("3 charts ->", F.relative_to(ROOT))


if __name__ == "__main__":
    main()
