"""Write README.md from the results."""

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
ESPN = ("https://www.espn.com/nba/story/_/id/48582233/"
        "nba-playoffs-2026-foul-3-san-antonio-spurs-portland-trail-blazers-impact")  # fmt: skip
HOOPR = "https://github.com/sportsdataverse/hoopR-nba-stats-raw"


def wrap(text: str, width: int = 100) -> str:
    """Wrap prose paragraphs and bullets; leave headings, images and code blocks alone."""
    out, code = [], False
    for block in text.split("\n\n"):
        if block.startswith("```") or code:
            code = block.count("```") % 2 == 1 if not code else block.count("```") % 2 == 0
            out.append(block)
        elif block.startswith(("#", "![", "|")):
            out.append(block)
        elif block.startswith("- "):
            items = ["- " + i for i in block[2:].split("\n- ")]
            out.append("\n".join(textwrap.fill(" ".join(i.split()), width, subsequent_indent="  ",
                                                break_long_words=False, break_on_hyphens=False)
                                  for i in items))  # fmt: skip
        else:
            out.append(textwrap.fill(" ".join(block.split()), width, break_long_words=False,
                                     break_on_hyphens=False))  # fmt: skip
    return "\n\n".join(out)


def markdown_table(head, rows, right) -> str:
    """A pipe table padded so its columns line up in the source too."""
    widths = [max(len(c) for c in col) for col in zip(head, *rows, strict=True)]

    def line(cells):
        return "| " + " | ".join(c.rjust(w) if r else c.ljust(w)
                                 for c, w, r in zip(cells, widths, right, strict=True)) + " |"  # fmt: skip

    rule = "| " + " | ".join(("-" * (w - 1) + ":") if r else "-" * w
                             for w, r in zip(widths, right, strict=True)) + " |"  # fmt: skip
    return "\n".join([line(head), rule, *(line(r) for r in rows)])


def signed(x) -> str:
    """0.0149 -> '+1.5'; never '-0.0'."""
    v = round(100 * x, 1) + 0.0
    return f"{v:+.1f}" if v else "0.0"


def pct(x, digits=0):
    return f"{100 * x:.{digits}f}"


def main():
    c = pd.read_csv(R / "counts.csv").iloc[0]
    model_seasons = f"{c.model_first} to {c.model_last}"
    rate = pd.read_csv(R / "foul_rate.csv")
    seasons = sorted(rate.season.unique())
    overall = rate[rate.band == "all"].set_index("season").share
    early_years, late_years = seasons[:10], seasons[-10:]
    mid = rate[rate.band == "5 to 10"]

    def share(years):
        x = mid[mid.season.isin(years)]
        return x.fouled.sum() / x.possessions.sum()

    mid_then, mid_now = share(early_years), share(late_years)
    assert mid_now > 2 * mid_then, "the 5-to-10 sentence no longer holds"

    wp = pd.read_csv(R / "win_probability.csv")
    wp = wp[np.isclose(wp.seconds, wp.seconds.round())]
    has, none = (wp[wp.timeouts == s].set_index("seconds") for s in ["has a timeout", "no timeout"])
    assert set(has.index[has.gap_low > 0]) == set(range(1, 7)), "the with-timeout foul range changed"
    assert set(has.index[has.gap_high < 0]) == set(range(15, 21)), "the with-timeout defend range changed"
    assert set(none.index[none.gap_low > 0]) == set(range(2, 6)), "the no-timeout foul range changed"
    assert set(none.index[none.gap_high < 0]) == set(range(14, 21)), "the no-timeout defend range changed"
    assert has.gap.loc[2] > none.gap.loc[2] + 0.04, "the 'fouling gains less' sentence no longer holds"

    fouled = pd.read_csv(R / "foul_share_by_second.csv").set_index("seconds")
    leaves = pd.read_csv(R / "tree.csv").set_index(["choice", "outcome"])
    tree_t = int(leaves.seconds_left.iloc[0])
    assert fouled.share.idxmax() == tree_t, "the tree is no longer where teams foul most"
    cal = pd.read_csv(R / "calibration.csv").set_index("model")
    model, baseline = cal.loc["Possession model"], cal.loc["Win rate by margin and clock"]
    assert abs(model.log_loss - baseline.log_loss) < 0.01, "the calibration sentence no longer holds"
    band = pd.read_csv(R / "bandwidth.csv")
    window = band.bandwidth[band.log_loss.idxmin()]

    # the check against real games
    check = pd.read_csv(R / "decision_check.csv").set_index(["action", "band"])
    check = check.reindex([(c, b) for c in ["foul", "defend"]
                           for b in ["1 to 5", "5 to 10", "10 to 20"]])  # fmt: skip
    check_rows = [[c.title(), b, f"{int(r.games)}", f"{int(r.won)}", f"{r.expected:.1f}",
                   f"{r.won - r.expected:+.1f}", f"{r.chance:.2f}"]
                  for (c, b), r in check.iterrows()]  # fmt: skip
    check_table = markdown_table(["Choice", "Seconds left", "Games", "Won", "Model expected",
                                  "Difference", "p-value"], check_rows,
                                 right=[False, False, True, True, True, True, True])  # fmt: skip
    assert (check.chance > 0.3).all(), "the 'about one time in three or more' sentence no longer holds"
    assert check.chance.idxmin() in [("foul", "10 to 20"), ("defend", "5 to 10")]

    text = f"""# NBA Fouling Up 3

Teams up 3 in the final seconds foul far more often than they used to. Should they, and when?

![How often teams up 3 fouled, by season and seconds left](figures/foul_rate.png)

Teams up 3 fouled on {pct(overall.iloc[0])}% of these possessions in {seasons[0]} and
{pct(overall.iloc[-1])}% in {seasons[-1]}. Most of the rise came with 5 to 10 seconds left: from
{pct(mid_then)}% of possessions in {early_years[0]} to {early_years[-1]} to {pct(mid_now)}% in
{late_years[0]} to {late_years[-1]}.

Raw win rates can't settle whether it works, because teams choose when to foul. In an
[ESPN study]({ESPN}) of the last five seasons, teams up 3 with 3 to 17 seconds left won 92.0% of
the time when they fouled and 91.7% when they defended. So this project builds the end of the
game one possession at a time.

## Method

- Data: the {int(c.model_games):,} regular-season games from {model_seasons}, the five seasons
  since fouling jumped on the chart above, from NBA.com play-by-play.
- Every possession that starts with 24 seconds or less left in the 4th quarter or overtime counts.
  An offensive rebound starts a new one.
- Fouling means fouling before a shot and sending the trailing team to the line. Fouls to give
  don't count.
- For each score and second, the model uses what happened on possessions with the same score that
  started near that second. The window is {pct(window)}% of the time left, the width that best
  predicted held-out seasons.
- It then works back from the buzzer, a tenth of a second at a time, to get each choice's chances
  to win, go to overtime and lose. Overtime is not a win.
- Down 3, the trailing team's outcomes come only from possessions where it had a timeout left, or
  only from ones where it had none. Timeouts are counted under the rules since 2017-18.
- Whenever the team up 3 is up 3 again later in the game, the model has it take the choice with
  the better Win%, and the trailing team's timeout state stays the same.
- The 90% intervals come from a bootstrap: 200 rebuilds of the model, each on the games resampled
  with replacement. A choice is clearly better when the whole interval is on one side of zero.
- Results stop at 20 seconds, past which there are too few fouls to estimate each second.
- Fit without {seasons[-2]} and {seasons[-1]}, the model predicted who won at the buzzer on the
  {int(model.possessions):,} close-game possessions in those seasons about as well as the win rate
  for the same score and clock.

## Results

This is every way the trailing team's possession can go at {tree_t} seconds, where teams foul most,
when it has a timeout left:

![Every way a possession down 3 can go with {tree_t} seconds left]\
(figures/tree.png)

![How often teams up 3 fouled, by seconds left, {model_seasons}](figures/foul_by_second.png)

Fouling gains more when the trailing team has a timeout left: at 2 seconds,
{signed(has.gap.loc[2])} points, against {signed(none.gap.loc[2])} when it has none.

- Trailing team has a timeout: foul with 2 to 6 seconds left, defend from 15 seconds on.
- Trailing team has no timeout: foul with 2 to 5 seconds left, defend from 14 seconds on.
- In between, neither is clearly better.

![Foul or defend, by seconds left, with each choice's chances to win, go to overtime and lose]\
(figures/calls.png)

## Checking the model

Wins at the buzzer against the model's expected wins, for every possession in {model_seasons}
where the team up 3 fouled or defended:

{check_table}

## Limits

- The foul side is built only from fouls teams chose to make. If they foul in easier spots, an
  average foul would do worse.
- Every team is treated as average. The model doesn't know who is on the floor, or the team up
  3's timeouts.

## Run it

```bash
pip install -e ".[dev]"
python scripts/pull.py       # will take a long time
python scripts/build.py      # will take a long time
python scripts/analyze.py    # will take a long time
python scripts/figures.py
python scripts/readme.py
pytest && ruff check .
```

## Credits

NBA.com Stats play-by-play and game logs, used under NBA.com's terms of use and read from the
sportsdataverse [hoopR-nba-stats-raw]({HOOPR}) archive (CC BY 4.0). No raw data is committed;
`scripts/pull.py` downloads it.
"""
    text = wrap(text) + "\n"
    (ROOT / "README.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
