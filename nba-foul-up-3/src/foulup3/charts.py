"""Chart style shared by the figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

DECADES = ["#b4b2a9", "#7fa6d9", "#2a78d6"]  # oldest to newest
BANDS = ["#eb6834", "#1baf7a", "#2a78d6"]  # 1-5, 5-10 and 10-20 seconds left
ACCENT = "#2a78d6"
INK = "#15171D"
MUTED = "#6f6e69"
GRID = "#E3E3E0"
TITLE, LABEL, CAPTION = 14, 10, 8.5
WIDTH, DPI = 7.5, 200
INSTALLED = {f.name for f in font_manager.fontManager.ttflist}
# Liberation Sans has the same metrics as Arial
FONT = next((f for f in ["Arial", "Liberation Sans"] if f in INSTALLED), "DejaVu Sans")

plt.rcParams.update({
    "font.family": FONT,
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
})  # fmt: skip


def titled(fig, title, source, units=None):
    """Left-aligned title, an optional units line, and one source caption."""
    fig.text(0.02, 0.985, title, fontsize=TITLE, fontweight="bold", color=INK, va="top")
    if units:
        fig.text(0.02, 0.985 - 0.32 / fig.get_figheight(), units, fontsize=LABEL, color=MUTED,
                 va="top", linespacing=1.4)  # fmt: skip
    fig.text(0.02, 0.012, source, fontsize=CAPTION, color=MUTED, va="bottom")


def margins(fig, top, bottom=0.45, left=0.9, right=0.25, wspace=None):
    """Margins in inches."""
    w, h = fig.get_size_inches()
    fig.subplots_adjust(left=left / w, right=1 - right / w, top=1 - top / h, bottom=bottom / h,
                        wspace=wspace)  # fmt: skip


def save(fig, path):
    Path(path).parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
