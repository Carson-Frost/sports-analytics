"""Draw the README charts from results/*.csv into figures/.

Refuses to save a chart if any word in its text starts lowercase, other than small connecting
words after the first word and the source's own lowercase names.
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from model import MIN_MINUTES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

ACCENT = "#C8102E"
GREY = "#B4B2A9"
INK = "#15171D"
MUTED = "#8A8A86"
GRID = "#E3E3E0"
TITLE, LABEL, SMALL = 15, 11, 9  # the only three font sizes
SOURCE = "Data: stats.nba.com via sportsdataverse hoopR-nba-stats-raw"
CONNECTING = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "per",
    "the",
    "to",
    "via",
    "vs",
    "with",
}
DASHES = ("\u2014", "\u2013")  # em and en dash, refused anywhere on a chart
LOWERCASE_NAMES = {"stats.nba.com", "sportsdataverse", "hoopR-nba-stats-raw"}  # as they spell it

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": LABEL,
        "text.color": INK,
        "axes.edgecolor": INK,
        "axes.labelcolor": MUTED,
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


def start(height, title, note=None, ncols=1):
    fig, axes = plt.subplots(1, ncols, figsize=(7.5, height))
    fig.text(0.02, 0.98, title, fontsize=TITLE, fontweight="bold", va="top")
    if note:
        fig.text(0.02, 0.98 - 0.33 / height, note, fontsize=LABEL, color=MUTED, va="top")
    fig.text(0.02, 0.02, SOURCE, fontsize=SMALL, color=MUTED, va="bottom")
    for ax in fig.axes:
        ax.tick_params(length=0)
    return fig, axes


def lowercase_word(text):
    """The first word that breaks title case, or None.

    Every word must start with a capital except the small connecting words, which must be
    lowercase unless they come first. Words starting with a digit ('3rd', '90%') are skipped,
    but the first word that starts with a letter counts as first: '12 players' fails. Words are
    split on spaces, hyphens, slashes and opening parentheses, so 'Held-out' and '(tie' fail."""
    first = True
    for token in text.split():
        if token.strip("(\"'[,.:;)") in LOWERCASE_NAMES:
            continue
        for word in re.split(r"[-/(]", token):
            word = word.lstrip("\"'[")
            if not word or not word[0].isalpha():
                continue
            small = word.rstrip(",.:;)").lower() in CONNECTING
            if (word[0].islower() and (first or not small)) or (
                not first and small and word[0].isupper()
            ):
                return word
            first = False
    return None


def save(fig, name, top):
    """Lay out, draw (tick labels only exist once drawn), check every text for title case, save."""
    fig.tight_layout(rect=(0, 0.05, 1, top))
    fig.canvas.draw()
    for t in fig.findobj(Text):
        if any(dash in t.get_text() for dash in DASHES):
            raise ValueError(f"{name}: em or en dash in {t.get_text()!r}")
        word = lowercase_word(t.get_text())
        if word:
            raise ValueError(f"{name}: lowercase word {word!r} in {t.get_text()!r}")
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name, dpi=200)
    plt.close(fig)
    print("wrote", name)


def top_players():
    """Dot plot: the 15 best 2025-26 players by RAPM, with 90% posterior intervals."""
    r = pd.read_csv(RESULTS / "ratings_2025_26.csv")
    top = r[r["minutes"] >= MIN_MINUTES].head(15).iloc[::-1]
    fig, ax = start(
        6.2,
        "Top 15 Players by Adjusted Plus Minus, 2025-26",
        "Dots Are Ratings, Bars Are 90% Intervals",
    )
    y = range(len(top))
    colors = [GREY] * (len(top) - 1) + [ACCENT]  # the leader, at the top, in the accent
    ax.hlines(y, top["low"], top["high"], color=colors, linewidth=2.5, alpha=0.6)
    ax.scatter(top["rapm"], y, color=colors, s=45, zorder=3)
    ax.set_yticks(list(y), [f"{n}, {t}" for n, t in zip(top["name"], top["team"], strict=True)])
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_xlim(left=min(0, top["low"].min()) - 0.5)
    ax.xaxis.set_major_formatter(lambda v, _: f"+{v:g}" if v > 0 else f"{v:g}")
    ax.set_xlabel("Points per 100 Possessions Above an Average Player")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    save(fig, "top_rapm.png", 0.88)


def team_scatter():
    """Two panels: raw and adjusted plus-minus against team net rating, with a fitted line."""
    r = pd.read_csv(RESULTS / "ratings_2025_26.csv")
    q = r[r["minutes"] >= MIN_MINUTES]
    fig, axes = start(4.4, "Plus Minus Against Team Net Rating, 2025-26", ncols=2)
    for ax, col, label in (
        (axes[0], "raw_plus_minus", "Raw Plus Minus"),
        (axes[1], "rapm", "Adjusted Plus Minus"),
    ):
        slope, intercept = np.polyfit(q["team_net"], q[col], 1)
        ax.scatter(q["team_net"], q[col], color=GREY, s=14, alpha=0.8)
        xs = np.array([q["team_net"].min(), q["team_net"].max()])
        ax.plot(xs, intercept + slope * xs, color=ACCENT, linewidth=2)
        ax.set_title(label, fontsize=LABEL, fontweight="bold", loc="left")
        ax.set_xlabel("Team Net Rating (Points per 100)")
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Points per 100 Possessions")
    save(fig, "team_scatter.png", 0.9)


def choosing_lambda():
    """Line: held-out error against the ridge penalty, the chosen value in the accent colour."""
    t = pd.read_csv(RESULTS / "lambda_tuning.csv")
    t = t[t["lambda"] > 0]  # plain least squares is far off this scale; the test chart shows it
    best = t.loc[t["rmse"].idxmin()]
    fig, ax = start(4.4, "Choosing Lambda")
    ax.plot(t["lambda"], t["rmse"], color=GREY, linewidth=2, marker="o", markersize=5)
    ax.scatter([best["lambda"]], [best["rmse"]], color=ACCENT, s=110, zorder=3)
    ax.annotate(
        f"Lambda = {best['lambda']:,.0f}",
        (best["lambda"], best["rmse"]),
        xytext=(0, 12),
        textcoords="offset points",
        ha="center",
        fontweight="bold",
    )
    ax.set_xscale("log")
    ax.set_xticks(
        [250, 500, 1000, 2000, 4000, 8000, 16000],
        ["250", "500", "1,000", "2,000", "4,000", "8,000", "16,000"],
    )
    ax.minorticks_off()
    ax.set_xlabel("Lambda (Log Scale)")
    ax.set_ylabel("Held-Out Error (Points per Game)")
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    save(fig, "choosing_lambda.png", 0.9)


def model_test():
    """Bars: 2025-26 held-out error for each method, RAPM in the accent colour."""
    t = pd.read_csv(RESULTS / "model_test.csv")
    t = t[t["sample"] == "2025-26"].iloc[::-1]
    names = {
        "home court": "Home Court Only",
        "team net rating": "Team Net Rating",
        "raw plus-minus": "Raw Plus Minus",
        "adjusted plus-minus": "Adjusted Plus Minus",
        "regularized adjusted plus-minus": "Regularized Adjusted Plus Minus",
    }
    fig, ax = start(
        4.2,
        "Error Predicting the Last Third of 2025-26",
        "Root Mean Squared Error of Final Margins (Points)",
    )
    colors = [ACCENT if m.startswith("regularized") else GREY for m in t["method"]]
    ax.barh([names[m] for m in t["method"]], t["rmse"], color=colors, height=0.6)
    for i, v in enumerate(t["rmse"]):
        ax.text(v + 0.3, i, f"{v:.2f}", va="center", fontweight="bold")
    ax.set_xlim(0, t["rmse"].max() * 1.12)
    ax.set_xticks([])
    save(fig, "model_test.png", 0.86)


if __name__ == "__main__":
    top_players()
    team_scatter()
    choosing_lambda()
    model_test()
