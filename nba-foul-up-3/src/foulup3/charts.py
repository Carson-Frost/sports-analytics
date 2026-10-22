"""The rules every chart is held to before it is saved."""

from matplotlib.text import Text

# words that stay lowercase inside a title-case label
SMALL_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "but",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "or",
    "per",
    "the",
    "to",
    "vs",
    "via",
    "with",
}
DASHES = {chr(0x2014), chr(0x2013)}  # em and en dash


def title_case_problem(text: str) -> str | None:
    """What breaks the rules in one piece of text, or None. Refused: an em or en dash anywhere,
    and any lowercase word or hyphenated part that is not a small connecting word (the very
    first word must be capitalised even if it is one). Numbers are skipped, so the word after
    a leading number ("3 seconds") is still checked, and so is each part of "Coin-toss"."""
    for dash in DASHES:
        if dash in text:
            return dash
    for i, token in enumerate(text.split()):
        for k, part in enumerate(token.strip("()[],:;.'\"").split("-")):
            first = i == 0 and k == 0
            if (
                part
                and part[0].isalpha()
                and part[0].islower()
                and (first or part not in SMALL_WORDS)
            ):
                return part
    return None


def check_capitals(fig, name: str = "chart") -> None:
    """Raise if any text on the figure breaks the rules. Draws first, because category tick
    labels only exist once matplotlib has drawn the figure."""
    fig.canvas.draw()
    for t in fig.findobj(Text):
        problem = title_case_problem(t.get_text())
        if problem:
            raise ValueError(f"{name}: {problem!r} in {t.get_text()!r}")
