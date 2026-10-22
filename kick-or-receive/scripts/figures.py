"""Results CSVs -> figures/*.png. Refuses to save a chart with any lowercase word in it."""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.text import Text  # noqa: E402

ACCENT, GREY, INK, MUTED, GRID = "#1F5FA6", "#B4B2A9", "#15171D", "#8A8A86", "#E3E3E0"
TIE = "#E2E0DA"
TITLE, LABEL, SMALL = 15, 11, 9  # the only three font sizes
RULES = {"sudden": "Sudden Death", "modified": "Modified Sudden Death",
         "both": "Both Teams Possess"}
TODAY = "Data: nflverse Play-by-Play 2023-2026  |  100,000 Simulated Overtimes per Rule"
RESULTS, FIGURES = Path("results"), Path("figures")

plt.rcParams.update({"font.family": "Arial", "font.size": LABEL, "axes.edgecolor": GRID,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})


def start(height, title, caption, left, top_space=0.75):
    """A figure with the title and caption on its left edge, like hers. Margins in inches."""
    fig, ax = plt.subplots(figsize=(7.5, height))
    fig.subplots_adjust(left=left, right=0.96, top=1 - top_space / height,
                        bottom=0.65 / height)
    fig.text(0.02, 1 - 0.15 / height, title, va="top", fontsize=TITLE, fontweight="bold",
             color=INK)
    fig.text(0.02, 0.12 / height, caption, fontsize=SMALL, color=MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, colors=INK, labelsize=LABEL)
    return fig, ax


SMALL_WORDS = {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "per", "the",
               "to", "vs", "with"}
DASHES = ("\u2014", "\u2013")  # em and en dash, written as codes so none appears here
LOWERCASE_NAMES = {"nflverse"}  # the project spells its own name in lowercase


def lowercase_word(line):
    """The first word in a line of chart text that should be capitalised and isn't, or None.

    Every word that starts with a letter counts, including one after a leading number
    ("12 personnel"); small connecting words are allowed lowercase except as the first word.
    """
    words = [w for w in re.split(r"[\s\-/(]+", line) if w]  # "Coin-toss" and "(tie" count
    for i, word in enumerate(words):
        if not word[0].isalpha() or word[0].isupper() or word in LOWERCASE_NAMES:
            continue
        if i == 0 or word not in SMALL_WORDS:
            return word
    return None


def save(fig, name):
    fig.canvas.draw()  # tick labels only exist once the figure is drawn
    for text in fig.findobj(Text):
        if any(dash in text.get_text() for dash in DASHES):
            raise ValueError(f"{name}: em or en dash in {text.get_text()!r}")
        for line in text.get_text().splitlines():
            word = lowercase_word(line)
            if word:
                raise ValueError(f"{name}: lowercase word {word!r} in {line!r}")
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name, dpi=200)
    plt.close(fig)
    print(f"wrote figures/{name}")


def receive_or_kick():
    s = pd.read_csv(RESULTS / "scenarios.csv")
    s = s[~s.go_for_two].set_index(["rule", "clock"])
    rows = [(r, "Untimed", f"{RULES[r]}\nPlayoffs") for r in RULES]
    rows.append(("both", "10 Min", "Both Teams Possess\nRegular Season"))
    fig, ax = start(3.9, "Coin-Toss Winner's Chance to Win in Overtime", TODAY, 0.27)
    for yi, (rule, clock, _) in zip(range(len(rows))[::-1], rows, strict=True):
        rec, tie, kick = s.loc[(rule, clock), ["receiver_win", "tie", "kicker_win"]]
        ax.barh(yi, rec, color=ACCENT, height=0.6)
        ax.barh(yi, tie, left=rec, color=TIE, height=0.6)
        ax.barh(yi, kick, left=rec + tie, color=GREY, height=0.6)
        ax.text(0.02, yi, f"Receive {rec:.0%}", va="center", color="white", fontweight="bold")
        if tie > 0:
            ax.text(rec + tie / 2, yi, f"Tie {tie:.0%}", va="center", ha="center", color=INK,
                    fontweight="bold", zorder=3,
                    bbox={"facecolor": TIE, "edgecolor": "none", "pad": 1})
        ax.text(0.98, yi, f"{kick:.0%} Kick", va="center", ha="right", color=INK,
                fontweight="bold")
    ax.axvline(0.5, color=INK, linestyle="--", linewidth=0.8)
    ax.set_yticks(range(len(rows))[::-1], [label for _, _, label in rows])
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    save(fig, "receive_or_kick.png")


def first_drive():
    f = pd.read_csv(RESULTS / "first_drive.csv")
    rows = {"td": "Touchdown", "fg": "Field Goal", "none": "No Score"}
    table = f.pivot(index="first_drive", columns="rule", values="receiver_win")
    table = table.loc[list(rows), list(RULES)]
    share = f.pivot(index="first_drive", columns="rule", values="share").loc[list(rows),
                                                                            list(RULES)]
    title = "Receiver's Chance to Win After Its First Drive, Playoff Overtime"
    fig, ax = start(3.6, title, TODAY, 0.24, 1.05)
    shade = LinearSegmentedColormap.from_list("shade", ["#EEF3FA", ACCENT])
    ax.imshow(table.to_numpy(), cmap=shade, vmin=0, vmax=1, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(RULES)):
            v, ink = table.iat[i, j], "white" if table.iat[i, j] > 0.55 else INK
            ax.text(j, i - 0.1, f"{v:.0%}", ha="center", va="center", fontweight="bold",
                    color=ink)
            ax.text(j, i + 0.25, f"{share.iat[i, j]:.0%} of First Drives", ha="center",
                    va="center", fontsize=SMALL, color=ink)
    headers = ["Sudden\nDeath", "Modified\nSudden Death", "Both Teams\nPossess"]
    ax.set_xticks(range(len(RULES)), headers, fontweight="bold")
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)), [f"First Drive {r}" for r in rows.values()])
    ax.set_xticks([x + 0.5 for x in range(len(RULES) - 1)], minor=True)
    ax.set_yticks([y + 0.5 for y in range(len(rows) - 1)], minor=True)
    ax.grid(which="minor", color="white", linewidth=4)
    ax.tick_params(which="minor", length=0)
    save(fig, "first_drive.png")


def model_check():
    c = pd.read_csv(RESULTS / "model_check.csv")
    c = c[c.clock == "All"].set_index("rule").loc[list(RULES)]
    caption = "Data: nflverse Play-by-Play 1999-2026  |  100,000 Simulated Overtimes per Clock"
    fig, ax = start(3.6, "Receiver's Win Share, Model vs Real Overtimes", caption, 0.3)
    y = list(range(len(c)))[::-1]
    ax.hlines(y, c.observed_low.clip(lower=0.3), c.observed_high.clip(upper=0.9),
              color=GREY, linewidth=3)
    ax.plot(c.observed_share, y, "o", color=GREY, markersize=9)
    ax.plot(c.model_share, y, "o", color=ACCENT, markersize=9)
    ax.axvline(0.5, color=MUTED, linewidth=0.8, zorder=0)
    top = y[0]  # labels on the top row only, Model above its dot and Real below, so never touch
    ax.annotate("Model", (c.model_share.iloc[0], top), xytext=(0, 9), textcoords="offset points",
                ha="center", va="bottom", color=ACCENT, fontweight="bold")
    ax.annotate("Real", (c.observed_share.iloc[0], top), xytext=(0, -9),
                textcoords="offset points", ha="center", va="top", color=MUTED,
                fontweight="bold")
    labels = [f"{RULES[r]}\n{row.seasons}, {row.games} Games" for r, row in c.iterrows()]
    ax.set_yticks(y, labels)
    ax.set_ylim(-0.6, len(c) - 0.3)
    ax.set_xlim(0.3, 0.9)
    ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                  ["30%", "40%", "50%", "60%", "70%", "80%", "90%"])
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    save(fig, "model_check.png")


if __name__ == "__main__":
    receive_or_kick()
    first_drive()
    model_check()
