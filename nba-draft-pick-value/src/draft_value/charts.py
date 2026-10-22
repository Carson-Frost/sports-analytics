"""The text rules for every chart: no word starts lowercase, and no em or en dashes."""

import re

import matplotlib.pyplot as plt
from matplotlib.text import Text

DASHES = ("\u2014", "\u2013")  # em and en dash, banned on every chart
LOWER_OK = {"a", "an", "and", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "vs"}


def check_capitals(fig: plt.Figure) -> None:
    """Raise if any word of any text starts lowercase, except small connecting words."""
    fig.canvas.draw()  # category tick labels only exist after a draw
    for text in fig.findobj(Text):
        if any(dash in text.get_text() for dash in DASHES):
            raise ValueError(f"Em or en dash on chart: {text.get_text()!r}")
        # a word starts at a letter and ends at anything but a letter, digit or apostrophe,
        # so hyphens, slashes and parentheses split words ("Coin-toss" is two words)
        words = re.findall(r"[A-Za-z][\w']*", text.get_text())
        for i, word in enumerate(words):
            if word[0].islower() and (i == 0 or word not in LOWER_OK):
                raise ValueError(f"Lowercase text on chart: {text.get_text()!r}")
