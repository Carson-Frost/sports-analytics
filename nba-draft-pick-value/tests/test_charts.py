import matplotlib.pyplot as plt
import pytest

from draft_value.charts import check_capitals

plt.switch_backend("Agg")  # no GUI window per figure


def test_lowercase_category_tick_label_is_refused():
    fig, ax = plt.subplots()
    ax.barh(["Log Curve", "flat average"], [25.7, 30.3])
    with pytest.raises(ValueError, match="flat average"):
        check_capitals(fig)
    plt.close(fig)


def test_lowercase_title_is_refused():
    fig, ax = plt.subplots()
    ax.set_title("Career win shares by draft pick")
    with pytest.raises(ValueError, match="Career win shares"):
        check_capitals(fig)
    plt.close(fig)


def test_title_case_with_small_words_passes():
    fig, ax = plt.subplots()
    ax.set_title("Share of the No. 1 Pick")
    ax.barh(["Average at Each Pick"], [26.0])
    check_capitals(fig)
    plt.close(fig)


@pytest.mark.parametrize("label", ["12 personnel", "1990 drafts"])
def test_lowercase_word_after_a_leading_number_is_refused(label):
    fig, ax = plt.subplots()
    ax.set_xlabel(label)
    with pytest.raises(ValueError, match=label):
        check_capitals(fig)
    plt.close(fig)


@pytest.mark.parametrize("label", ["Coin-toss Winner", "Chance (tie Counted Half)", "Wins/game"])
def test_lowercase_word_after_a_hyphen_parenthesis_or_slash_is_refused(label):
    fig, ax = plt.subplots()
    ax.set_title(label)
    with pytest.raises(ValueError):
        check_capitals(fig)
    plt.close(fig)


def test_small_word_inside_a_hyphenated_label_passes():
    fig, ax = plt.subplots()
    ax.set_title("Play-by-Play Held-Out Error")
    check_capitals(fig)
    plt.close(fig)


@pytest.mark.parametrize("label", ["Picks 1 \u2014 30", "Picks 1 \u2013 30"])
def test_em_or_en_dash_is_refused(label):
    fig, ax = plt.subplots()
    ax.set_title(label)
    with pytest.raises(ValueError, match="dash"):
        check_capitals(fig)
    plt.close(fig)
