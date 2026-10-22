import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import figures  # noqa: E402


def test_title_case_check_catches_a_lowercase_word_anywhere():
    assert figures.lowercase_word("12 players") == "players"
    assert figures.lowercase_word("Held-out Error") == "out"
    assert figures.lowercase_word("Games Decided (tie Excluded)") == "tie"
    assert figures.lowercase_word("Error Predicting the last Third") == "last"
    assert figures.lowercase_word("of Players") == "of"  # a connecting word can't lead
    assert figures.lowercase_word("Players With Minutes") == "With"  # small words stay small
    assert figures.lowercase_word("Held-Out Error (Points per Game)") is None
    assert figures.lowercase_word("Shai Gilgeous-Alexander, OKC") is None
    assert (
        figures.lowercase_word("Data: stats.nba.com via sportsdataverse hoopR-nba-stats-raw")
        is None
    )
    assert figures.lowercase_word("+8.2") is None


def test_a_lowercase_category_label_stops_the_chart(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "FIGURES", tmp_path)
    fig, ax = figures.start(3, "Fine Title")
    ax.barh(["Home Court Only", "12 players"], [1, 2])  # tick labels exist only once drawn
    with pytest.raises(ValueError, match="players"):
        figures.save(fig, "bad.png", 0.9)
    assert not (tmp_path / "bad.png").exists()


def test_an_em_or_en_dash_stops_the_chart(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "FIGURES", tmp_path)
    for dash in ("\u2014", "\u2013"):
        fig, _ = figures.start(3, f"Fine Title {dash} Second Part")
        with pytest.raises(ValueError, match="dash"):
            figures.save(fig, "bad.png", 0.9)
    assert not (tmp_path / "bad.png").exists()
