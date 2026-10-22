"""Draw the four charts in figures/ from the CSVs in results/.

Refuses to save a chart with any text that starts with a lowercase letter.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

from foulup3.charts import check_capitals  # noqa: E402

RESULTS = Path("results")
OUT = Path("figures")
ACCENT, GREY, LIGHT, INK, GRID = "#C8102E", "#8A8A86", "#C9C7C0", "#15171D", "#E3E3E0"
TITLE, LABEL, SMALL = 15, 11, 9  # the only three font sizes
WIDTH, DPI = 7.5, 200

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": LABEL,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.edgecolor": GRID,
        "figure.facecolor": "white",
    }
)


def source() -> str:
    n = pd.read_csv(RESULTS / "numbers.csv").set_index("name").value
    return (
        f"Source: NBA.com Stats Play-by-Play, Regular Seasons {n['first_season']} to "
        f"{n['last_season']}"
    )


def percent(x) -> str:
    return f"{100 * x:.0f}%"


def frame(height: float, title: str):
    fig, ax = plt.subplots(figsize=(WIDTH, height))
    fig.text(0.02, 1 - 0.32 / height, title, size=TITLE, weight="bold", ha="left", va="top")
    fig.text(0.02, 0.12 / height, source(), size=SMALL, color=GREY, ha="left", va="bottom")
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    return fig, ax


def save(fig, name: str) -> None:
    check_capitals(fig, name)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, dpi=DPI, facecolor="white")
    plt.close(fig)


def foul_edge() -> None:
    wp = pd.read_csv(RESULTS / "win_probability.csv")
    wp = wp[wp.seconds >= 1]
    fig, ax = frame(4.6, "Fouling Up 3, Win Probability Gained Over Defending")
    fig.subplots_adjust(left=0.1, right=0.97, top=0.84, bottom=0.2)
    ax.fill_between(wp.seconds, 100 * wp.gap_low, 100 * wp.gap_high, color=ACCENT, alpha=0.15, lw=0)
    ax.axhline(0, color=INK, lw=0.8)
    ax.plot(wp.seconds, 100 * wp.gap, color=ACCENT, lw=2.6)
    ax.text(23.6, 0.6, "Foul Better", color=GREY, weight="bold", va="bottom")
    ax.text(23.6, -0.6, "Defend Better", color=GREY, weight="bold", va="top")
    ax.set_xlim(24, 1)
    ax.set_xticks([24, 20, 16, 12, 8, 4, 1])
    ax.set_xlabel("Seconds Left When the Trailing Team Gets the Ball", color=GREY)
    ax.set_ylabel("Percentage Points, With 95% Interval", color=GREY)
    low, high = 100 * wp.gap_low.min(), 100 * wp.gap_high.max()
    ax.set_ylim(low - 1.5, high + 1.5)  # keeps the lowest tick off the corner
    ax.set_yticks(range(int(low) // 3 * 3 + 3, int(high) + 1, 3))
    ax.tick_params(axis="x", pad=6)
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:+.0f}" if y else "0")
    ax.grid(axis="y", color=GRID, lw=0.6)
    save(fig, "foul_edge.png")


def one_possession() -> None:
    one = pd.read_csv(RESULTS / "one_possession.csv")
    one = one[one.chance >= 0.01]
    blocks = [("Defend", one[one.choice == "defend"]), ("Foul", one[one.choice == "foul"])]
    n_rows = sum(len(b) + 1 for _, b in blocks)
    height = 1.3 + 0.32 * n_rows
    fig, ax = frame(height, "One Trip Down 3 With 10 Seconds Left")
    fig.subplots_adjust(left=0.02, right=0.98, top=1 - 0.75 / height, bottom=0.42 / height)
    ax.set_xlim(0, 1)
    ax.set_ylim(n_rows + 0.6, -0.4)
    ax.set_xticks([])
    ax.set_yticks([])
    cols = {"Chance": 0.53, "Seconds Used": 0.7, "Leader Wins After": 0.89}
    for name, x in cols.items():
        ax.text(x, -0.15, name, ha="center", va="center", weight="bold")
    y = 0.6
    for head, block in blocks:
        ax.text(0.0, y + 0.1, head, weight="bold", va="center")
        ax.plot([0, 1], [y + 0.45, y + 0.45], color=GRID, lw=0.8)
        y += 1
        for r in block.itertuples():
            shade = [1 - (1 - c) * min(1, r.chance / 0.6) for c in to_rgb(ACCENT)]
            ax.add_patch(
                plt.Rectangle((cols["Chance"] - 0.07, y - 0.42), 0.14, 0.84, color=shade, lw=0)
            )
            dark = r.chance > 0.33
            ax.text(0.02, y, r.outcome, va="center")
            ax.text(
                cols["Chance"],
                y,
                percent(r.chance),
                ha="center",
                va="center",
                color="white" if dark else INK,
                weight="bold",
            )
            ax.text(cols["Seconds Used"], y, f"{r.seconds:.1f}", ha="center", va="center")
            ax.text(
                cols["Leader Wins After"], y, percent(r.leader_wins_after), ha="center", va="center"
            )
            y += 1
    save(fig, "one_possession.png")


def foul_rate() -> None:
    rate = pd.read_csv(RESULTS / "foul_rate.csv")
    labels = [f"{lo:g} to {hi:g} Seconds" for lo, hi in zip(rate.low, rate.high, strict=True)]
    fig, ax = frame(4.0, "How Often a Leader Up 3 Fouls")
    fig.subplots_adjust(left=0.24, right=0.94, top=0.83, bottom=0.14)
    ax.barh(labels, rate.foul_share, color=ACCENT, height=0.6)
    for i, r in enumerate(rate.itertuples()):
        ax.text(r.foul_share + 0.006, i, percent(r.foul_share), va="center", weight="bold")
    ax.invert_yaxis()
    ax.set_xticks([])
    save(fig, "foul_rate.png")


def model_check() -> None:
    check = pd.read_csv(RESULTS / "model_check.csv")
    check = check.sort_values(["action", "low"], ascending=[True, False]).reset_index(drop=True)
    labels = [
        f"{a.title()}, {lo:g} to {hi:g} Seconds ({n})"
        for a, lo, hi, n in zip(check.action, check.low, check.high, check.decisions, strict=True)
    ]
    fig, ax = frame(
        0.9 + 0.42 * len(check) + 0.5, "Predicted vs Observed, 2024-25 and 2025-26 Seasons"
    )
    fig.subplots_adjust(
        left=0.36, right=0.95, top=1 - 0.95 / fig.get_figheight(), bottom=0.75 / fig.get_figheight()
    )
    for i, r in check.iterrows():
        ax.plot(
            [r.observed_low, r.observed_high], [i, i], color=LIGHT, lw=3, solid_capstyle="round"
        )
        ax.plot(r.observed, i, "o", color=GREY, ms=8)
        ax.plot(r.predicted, i, "o", color=ACCENT, ms=8)
    top = check.iloc[0]
    left, right = sorted([(top.predicted, "Predicted", ACCENT), (top.observed, "Observed", GREY)])
    ax.text(left[0], -0.5, left[1], color=left[2], ha="right", weight="bold")
    ax.text(right[0], -0.5, right[1], color=right[2], ha="left", weight="bold")
    ax.set_yticks(range(len(check)), labels)
    ax.set_ylim(len(check) - 0.5, -1.1)
    lowest = min(check.observed_low.min(), check.predicted.min())
    ax.set_xticks([x / 20 for x in range(int(lowest * 20), 21)])
    ax.set_xlim(lowest - 0.02, 1.02)
    ax.xaxis.set_major_formatter(lambda x, _: percent(x))
    ax.set_xlabel("Leader's Win Probability", color=GREY)
    ax.grid(axis="x", color=GRID, lw=0.6)
    save(fig, "model_check.png")


if __name__ == "__main__":
    foul_edge()
    one_possession()
    foul_rate()
    model_check()
