"""The three README charts, from results/. Refuses to save a chart with lowercase text."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from draft_value.charts import check_capitals

plt.switch_backend("Agg")  # files only: no GUI window per figure

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

ACCENT, GRAY, LIGHT, PALE, INK = "#D4561C", "#8A8A86", "#C4C2BA", "#F3F2EF", "#15171D"
TITLE, LABEL, SMALL = 15, 11, 9  # the only three font sizes
SOURCE = "Source: Basketball Reference, 1990 to 2016 NBA Drafts"

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": LABEL,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def finish(fig: plt.Figure, title: str, name: str) -> None:
    fig.text(0.012, 0.965, title, fontsize=TITLE, fontweight="bold", va="top")
    fig.text(0.012, 0.02, SOURCE, fontsize=SMALL, color=GRAY)
    check_capitals(fig)
    fig.savefig(FIGURES / name, dpi=200)
    plt.close(fig)


def value_curve(curve: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.17)
    ax.grid(axis="y", color="#E6E5E1", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.fill_between(
        curve["pick"],
        curve["curve_low_90"],
        curve["curve_high_90"],
        color=ACCENT,
        alpha=0.16,
        linewidth=0,
    )
    ax.scatter(curve["pick"], curve["average_win_shares"], s=26, color=LIGHT, zorder=2)
    ax.plot(curve["pick"], curve["curve_win_shares"], color=ACCENT, linewidth=2.6, zorder=3)
    for p, dx, dy in [(1, 2.0, 3), (8, 2.0, 7.5), (30, 1.5, 7), (60, -1.0, 11)]:
        y = curve.loc[curve["pick"] == p, "curve_win_shares"].item()
        ax.scatter([p], [y], s=40, color=ACCENT, zorder=4)
        ax.text(
            p + dx,
            y + dy,
            f"Pick {p}: {y:.1f}",
            fontsize=LABEL,
            fontweight="bold",
            ha="right" if dx < 0 else "left",
            color=INK,
        )
    ax.text(10.6, 52, "Average at Each Pick", fontsize=LABEL, color=GRAY, va="center")
    ax.text(3.0, 66, "Fitted Curve", fontsize=LABEL, color=ACCENT, fontweight="bold")
    ax.set_xlim(-1, 62)
    ax.set_ylim(-3, 85)
    ax.set_xticks([1, 10, 20, 30, 40, 50, 60])
    ax.tick_params(length=0)
    ax.set_xlabel("Draft Pick", fontsize=LABEL, color=GRAY)
    ax.set_ylabel("Career Win Shares", fontsize=LABEL, color=GRAY)
    finish(fig, "Career Win Shares by Draft Pick", "value_curve.png")


def pick_value_table(curve: pd.DataFrame) -> None:
    picks = [1, 2, 3, 5, 8, 10, 14, 20, 30, 31, 45, 60]
    rows = curve.set_index("pick").loc[picks]
    fig, ax = plt.subplots(figsize=(7.5, 6.4))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.9, bottom=0.07)
    for x, header in [(0.42, "Expected Career Win Shares"), (0.76, "Share of the No. 1 Pick")]:
        ax.text(x, len(picks) - 0.1, header, ha="center", fontsize=LABEL, fontweight="bold")
    for i, (pick, row) in enumerate(rows.iterrows()):
        y = len(picks) - 1 - i
        share = row["share_of_pick_1"]
        ax.text(0.15, y, f"Pick {pick}", ha="right", va="center", fontsize=LABEL)
        ax.add_patch(plt.Rectangle((0.255, y - 0.42), 0.33, 0.84, color=PALE, linewidth=0))
        ax.add_patch(
            plt.Rectangle(
                (0.595, y - 0.42), 0.33, 0.84, color=ACCENT, alpha=0.08 + 0.92 * share, linewidth=0
            )
        )
        ax.text(
            0.42,
            y,
            f"{row['curve_win_shares']:.1f}",
            ha="center",
            va="center",
            fontsize=LABEL,
            fontweight="bold",
        )
        ax.text(
            0.76,
            y,
            f"{100 * share:.0f}%",
            ha="center",
            va="center",
            fontsize=LABEL,
            fontweight="bold",
            color="white" if share > 0.55 else INK,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(picks) + 0.2)
    ax.axis("off")
    finish(fig, "What Each Pick Is Worth", "pick_value_table.png")


def model_test(results: pd.DataFrame) -> None:
    results = results.sort_values("rmse", ascending=False)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    fig.subplots_adjust(left=0.25, right=0.95, top=0.84, bottom=0.24)
    colors = [ACCENT if m == "Log Curve" else LIGHT for m in results["model"]]
    ax.barh(results["model"], results["rmse"], color=colors, height=0.6)
    for y, value in enumerate(results["rmse"]):
        ax.text(value + 0.4, y, f"{value:.2f}", va="center", fontsize=LABEL, fontweight="bold")
    ax.set_xlim(0, 34)
    ax.grid(axis="x", color="#E6E5E1", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_xlabel("Held-Out RMSE in Career Win Shares", fontsize=LABEL, color=GRAY)
    finish(fig, "Error Predicting Unseen Drafts", "model_test.png")


if __name__ == "__main__":
    curve = pd.read_csv(RESULTS / "curve.csv")
    value_curve(curve)
    pick_value_table(curve)
    model_test(pd.read_csv(RESULTS / "model_test.csv"))
    print("Wrote figures/value_curve.png, figures/pick_value_table.png, figures/model_test.png")
