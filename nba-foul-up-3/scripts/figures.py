"""The README charts, from the CSVs in results/."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from foulup3.charts import BANDS, DECADES, GRID, INK, MUTED, WIDTH, margins, save, titled
from foulup3.outcomes import KINDS, TREE, flatten

ROOT = Path(__file__).resolve().parents[1]
R, F = ROOT / "results", ROOT / "figures"
SOURCE = "Source: NBA.com Stats play-by-play via sportsdataverse, regular seasons {}"
BAND_ORDER = ["1 to 5", "5 to 10", "10 to 20"]


def season_ticks(seasons, every=5):
    """Every fifth season, counted back from the latest so it always gets a label."""
    idx = list(range(len(seasons) - 1, -1, -every))[::-1]
    return idx, [seasons[i] for i in idx]


def foul_rate(rate):
    seasons = sorted(rate.season.unique())
    x = np.arange(len(seasons))
    fig, ax = plt.subplots(figsize=(WIDTH, 4.4))
    for band, colour in zip(BAND_ORDER, BANDS, strict=True):
        r = rate[rate.band == band].set_index("season").reindex(seasons)
        ax.plot(x, 100 * r.share, color=colour, lw=2, marker="o", ms=3)
        ax.text(x[-1] + 0.6, 100 * r.share.iloc[-1], f"{band} seconds", color=colour,
                fontweight="bold", va="center", fontsize=9.5)  # fmt: skip
    idx, labels = season_ticks(seasons)
    ax.set_xticks(idx, labels)
    ax.set_xlim(-0.5, len(seasons) + 4.6)
    ax.set_ylim(-1.5, 43)
    ax.tick_params(axis="x", pad=6)
    ax.set_yticks(range(0, 46, 10), [f"{v}%" for v in range(0, 46, 10)])
    ax.set_ylabel("Share of possessions the team up 3 fouled")
    ax.yaxis.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    margins(fig, top=0.8, left=0.75, right=0.15, bottom=0.6)
    titled(fig, "Teams Up 3 Foul Far More Often Than They Used To",
           SOURCE.format(f"{seasons[0]} to {seasons[-1]}"),
           units="When the trailing team gets the ball down 3, by seconds left. "
                 "Fouls to give are left out.")  # fmt: skip
    save(fig, F / "foul_rate.png")


def tree(leaves, wp, era):
    """The trailing team's possession at one moment, as a tree, with totals by kind of outcome."""
    seconds = leaves.seconds_left.iloc[0]
    at = wp[wp.timeouts == "has a timeout"].set_index("seconds").loc[seconds]
    leaves = leaves.set_index(["choice", "outcome"])
    blocks = {c: flatten(TREE[c], leaves.loc[c], 1) for c in TREE}
    wins = {"defend": at.defend, "foul": at.foul}
    endings = [("Team up 3 wins", wins), ("Goes to overtime", {c: at[f"{c}_ot"] for c in wins}),
               ("Team up 3 loses", {c: at[f"{c}_loss"] for c in wins})]
    rows = []
    for choice in ["defend", "foul"]:
        rows.append([0, choice.title(), None, wins[choice], "head"])
        rows += blocks[choice]
    n_tree, n_total = len(rows), len(KINDS) + 4
    fig, ax = plt.subplots(figsize=(WIDTH, 1.5 + 0.3 * (n_tree + n_total + 1.5)))
    margins(fig, top=1.05, left=0.1, right=0.1, bottom=0.4)
    ax.set_xlim(0, 1)
    ax.set_ylim(n_tree + n_total + 1.2, -1.3)
    ax.axis("off")
    indent, x_chance, x_after = 0.045, 0.74, 0.97
    ax.text(x_chance, -1.0, "Chance", fontweight="bold", ha="right", va="center", fontsize=9.5)
    ax.text(x_after, -1.0, "Team up 3\nwins after", fontweight="bold", ha="right", va="center",
            fontsize=9.5, linespacing=1.1)  # fmt: skip
    colour = {k: c for k, _, c in KINDS}
    for i, (depth, label, chance, after, kind) in enumerate(rows):
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((0, i - 0.5), 1, 1, color="#F4F4F1", lw=0, zorder=0))
        x = depth * indent + 0.035
        if kind == "head":
            ax.text(0.0, i, label, va="center", fontsize=11, fontweight="bold")
            ax.text(x_after, i, f"{100 * after:.1f}%", va="center", ha="right", fontsize=11,
                    fontweight="bold")  # fmt: skip
            continue
        parent = max(j for j in range(i) if rows[j][0] == depth - 1)
        px = (depth - 1) * indent + 0.035 + 0.01 if depth > 1 else 0.012
        ax.plot([px, px, x - 0.008], [parent + 0.35, i, i], color=MUTED, lw=0.8, zorder=1)
        leaf = kind is not None
        if leaf:
            ax.scatter([x + 0.008], [i], s=42, marker="s", color=colour[kind], zorder=3)
        tx = x + (0.025 if leaf else 0)
        ax.text(tx, i, label, va="center", fontsize=9.5, color=INK if leaf else MUTED,
                fontweight="normal" if leaf else "bold")  # fmt: skip
        ax.plot([0.5, x_chance - 0.055], [i, i], color=GRID, lw=0.8, ls=(0, (1, 2)), zorder=1)
        style = dict(va="center", ha="right", fontsize=9.5, color=INK if leaf else MUTED)
        ax.text(x_chance, i, f"{100 * chance:.0f}%", **style)
        ax.text(x_after, i, f"{100 * after:.0f}%", **style)
    # totals by kind of outcome, one column per choice
    y0 = n_tree + 1.0
    ax.plot([0, 1], [y0 - 0.6, y0 - 0.6], color=INK, lw=0.8)
    ax.text(0.0, y0, "Totals", va="center", fontsize=11, fontweight="bold")
    ax.text(x_chance, y0, "Defend", fontweight="bold", ha="right", va="center", fontsize=9.5)
    ax.text(x_after, y0, "Foul", fontweight="bold", ha="right", va="center", fontsize=9.5)
    for k, (key, name, c) in enumerate(KINDS, start=1):
        y = y0 + k
        ax.scatter([0.043], [y], s=42, marker="s", color=c)
        ax.text(0.07, y, name, va="center", fontsize=9.5)
        for x_col, choice in [(x_chance, "defend"), (x_after, "foul")]:
            share = sum(r[2] for r in blocks[choice] if r[4] == key)
            ax.text(x_col, y, f"{100 * share:.0f}%", va="center", ha="right", fontsize=9.5)
    for k, (name, share) in enumerate(endings):
        y = y0 + len(KINDS) + 1 + k
        weight = "bold" if k == 0 else "normal"
        ax.text(0.07, y, name, va="center", fontsize=9.5, fontweight=weight)
        for x_col, choice in [(x_chance, "defend"), (x_after, "foul")]:
            ax.text(x_col, y, f"{100 * share[choice]:.1f}%", va="center", ha="right", fontsize=9.5,
                    fontweight=weight)  # fmt: skip
    titled(fig, f"Every Way a Possession Down 3 Can Go With {seconds:.0f} Seconds Left",
           SOURCE.format(era),
           units=f"{era}, trailing team with a timeout left. Chances are within each choice;\n"
                 "grey labels sum the rows under them. Wins are wins at the buzzer. Outcomes under "
                 "1% are left out.")  # fmt: skip
    save(fig, F / "tree.png")


FOUL, DEFEND = "#d9541e", "#2a78d6"  # the two choices, checked for color-blind separation
CALL_SECONDS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
RISKY = 4  # asterisk on a foul call at this many seconds or fewer


def tint(colour, share):
    """`colour` mixed with white: share 0 is white, 1 is the full colour."""
    rgb = np.array(plt.matplotlib.colors.to_rgb(colour))
    return tuple(1 - share * (1 - rgb))


def calls(wp, era):
    """The foul-or-defend table as a chart, one panel per timeout state of the trailing team:
    each choice's chances, the difference shaded toward the better choice, and the call."""
    cols = [("Seconds\nleft", None), ("Win", "foul"), ("Overtime", "foul_ot"),
            ("Loss", "foul_loss"), ("Win", "defend"), ("Overtime", "defend_ot"),
            ("Loss", "defend_loss"), ("Difference\nin Win%", "gap"), ("Call", None)]  # fmt: skip
    widths = np.array([0.9, 0.8, 0.95, 0.8, 0.8, 0.95, 0.8, 1.1, 1.0])
    x = np.concatenate([[0], np.cumsum(widths)])
    wp = wp[np.isclose(wp.seconds, wp.seconds.round())]
    panels = [("Trailing team has a timeout", "has a timeout"),
              ("Trailing team has no timeout", "no timeout")]  # fmt: skip
    n = len(CALL_SECONDS)
    head = 3.2  # rows of space above each panel's cells
    fig, ax = plt.subplots(figsize=(WIDTH, 1.5 + 0.36 * 2 * (n + head)))
    margins(fig, top=0.95, left=0.15, right=0.15, bottom=0.4)
    ax.set_xlim(0, x[-1])
    ax.set_ylim(2 * (n + head) - head - 0.45, -head)
    ax.axis("off")
    big = wp.gap.abs().max()
    gap = 0.03  # white space between cells
    for p, (title, state) in enumerate(panels):
        top = p * (n + head)  # y of the panel's first row
        rows = wp[wp.timeouts == state].set_index("seconds").loc[CALL_SECONDS]
        ax.text(0, top - 2.75, title, ha="left", va="center", fontsize=11.5, fontweight="bold",
                color=INK)  # fmt: skip
        for label, c0, c1, colour in [("Foul", 1, 4, FOUL), ("Defend", 4, 7, DEFEND)]:
            ax.text((x[c0] + x[c1]) / 2, top - 1.85, label, ha="center", va="center",
                    fontsize=10.5, fontweight="bold", color=INK)  # fmt: skip
            ax.plot([x[c0] + 0.08, x[c1] - 0.08], [top - 1.5, top - 1.5], color=colour, lw=2.5,
                    solid_capstyle="round")  # fmt: skip
        for k, (label, _) in enumerate(cols):
            ax.text((x[k] + x[k + 1]) / 2, top - 0.62, label, ha="center", va="bottom",
                    fontsize=9, color=MUTED, linespacing=1.1)  # fmt: skip
        for row, (t, r) in enumerate(rows.iterrows()):
            y = top + row
            call = "Foul" if r.gap_low > 0 else "Defend" if r.gap_high < 0 else "Toss-up"
            better = "foul" if r.foul > r.defend else "defend"
            for k, (_, key) in enumerate(cols):
                face, text, weight, ink = "#F4F4F1", "", "normal", INK
                if k == 0:
                    face, text = "white", f"{t:.0f}"
                elif key == "gap":
                    face = tint(FOUL if r.gap > 0 else DEFEND, 0.08 + 0.6 * abs(r.gap) / big)
                    text = f"{100 * r.gap:+.1f}"
                elif key is None:
                    face = {"Foul": FOUL, "Defend": DEFEND, "Toss-up": "#E3E3E0"}[call]
                    ink = "white" if call != "Toss-up" else INK
                    text, weight = call + ("*" if call == "Foul" and t <= RISKY else ""), "bold"
                else:
                    text = f"{100 * r[key]:.1f}%"
                    if key == better:
                        face = tint(FOUL if key == "foul" else DEFEND, 0.2)
                        weight = "bold"
                ax.add_patch(plt.Rectangle((x[k] + gap, y - 0.5 + gap), widths[k] - 2 * gap,
                                           1 - 2 * gap, color=face, lw=0))  # fmt: skip
                ax.text((x[k] + x[k + 1]) / 2, y, text, ha="center", va="center", fontsize=9.5,
                        color=ink, fontweight=weight)  # fmt: skip
    titled(fig, "Foul or Defend, by Seconds Left", SOURCE.format(era),
           units=f"Chances for the team up 3 at the buzzer, {era}. Overtime is not a win. "
                 "Shaded Win% is the higher one.\n"
                 "*Very risky this late. Toss-up: the 90% interval of the difference crosses "
                 "zero.")  # fmt: skip
    save(fig, F / "calls.png")


def foul_by_second(share, era):
    fig, ax = plt.subplots(figsize=(WIDTH, 3.6))
    ax.bar(share.seconds, 100 * share.share, width=1.5, color=DECADES[-1])
    for r in share.itertuples():
        ax.text(r.seconds, 100 * r.share + 0.8, f"{100 * r.share:.0f}%", ha="center", fontsize=9)
    ax.set_xlim(21.2, 0.8)
    ax.set_xticks(share.seconds)
    ax.set_xlabel("Seconds left when the trailing team gets the ball (within 1 second)")
    ax.set_ylabel("Share of possessions\nthe team up 3 fouled")
    ax.set_ylim(0, 42)
    ax.set_yticks(range(0, 41, 10), [f"{v}%" for v in range(0, 41, 10)])
    ax.yaxis.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    margins(fig, top=0.8, left=0.95, right=0.15, bottom=0.75)
    titled(fig, "How Often Teams Up 3 Foul, by Seconds Left", SOURCE.format(era),
           units=f"When the trailing team gets the ball down 3, {era}. Fouls to give are left out.")
    save(fig, F / "foul_by_second.png")


def main():
    foul_rate(pd.read_csv(R / "foul_rate.csv"))
    c = pd.read_csv(R / "counts.csv").iloc[0]
    seasons = f"{c.model_first} to {c.model_last}"
    tree(pd.read_csv(R / "tree.csv"), pd.read_csv(R / "win_probability.csv"), seasons)
    foul_by_second(pd.read_csv(R / "foul_share_by_second.csv"), seasons)
    calls(pd.read_csv(R / "win_probability.csv"), seasons)


if __name__ == "__main__":
    main()
