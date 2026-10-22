import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from foulup3.charts import check_capitals, title_case_problem  # noqa: E402


@pytest.mark.parametrize(
    "text, problem",
    [
        ("defend", "defend"),
        ("3 seconds", "seconds"),
        ("Leader Wins after", "after"),
        ("the End", "the"),
        ("Coin-toss Winner", "toss"),
        ("Source: NBA.com Stats play-by-play", "play"),
        (f"Foul {chr(0x2014)} Defend", chr(0x2014)),
        (f"Picks 31{chr(0x2013)}60", chr(0x2013)),
        ("Free Throws, 2 of 2", None),
        ("Miss at the Buzzer", None),
        ("Source: NBA.com Stats Play-by-Play, Regular Seasons 2020-21 to 2025-26", None),
        ("Leader's Win Probability Up 3, Foul vs Defend", None),
        ("92%", None),
        ("Defend, 0 to 5 Seconds (80)", None),
    ],
)
def test_title_case_problem(text, problem):
    assert title_case_problem(text) == problem


def test_lowercase_category_label_is_refused():
    fig, ax = plt.subplots()
    ax.barh(["Foul", "3 seconds"], [1, 2])  # tick labels only exist after a draw
    with pytest.raises(ValueError, match="seconds"):
        check_capitals(fig)
    plt.close(fig)


def test_title_case_chart_passes():
    fig, ax = plt.subplots()
    ax.barh(["Foul", "Defend"], [1, 2])
    ax.set_title("Leader's Win Probability")
    ax.text(0, 0, "92%")
    check_capitals(fig)
    plt.close(fig)
