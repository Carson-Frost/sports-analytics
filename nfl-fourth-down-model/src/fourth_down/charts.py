"""Chart style shared by the figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ACCENT = "#2F6DB5"
GREY = "#B4B2A9"
LIGHT = "#EDECE8"
INK = "#15171D"
MUTED = "#8A8A86"
TITLE, LABEL, CAPTION = 14, 10.5, 8.5
WIDTH, DPI = 7.5, 200

plt.rcParams.update({
    "font.family": ["Liberation Sans", "Arial", "DejaVu Sans"],
    "font.size": LABEL,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "axes.labelcolor": MUTED,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def titled(fig, title, source, units=None):
    """Left-aligned title, an optional units line, and one source caption."""
    fig.text(0.02, 0.98, title, fontsize=TITLE, fontweight="bold", color=INK, va="top")
    if units:
        fig.text(0.02, 0.98 - 0.32 / fig.get_figheight(), units, fontsize=LABEL, color=MUTED, va="top")
    fig.text(0.02, 0.012, source, fontsize=CAPTION, color=MUTED, va="bottom")


def margins(fig, top, bottom=0.55, left=0.9, right=0.25):
    """Margins in inches."""
    w, h = fig.get_size_inches()
    fig.subplots_adjust(left=left / w, right=1 - right / w, top=1 - top / h, bottom=bottom / h)


def save(fig, path):
    Path(path).parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
