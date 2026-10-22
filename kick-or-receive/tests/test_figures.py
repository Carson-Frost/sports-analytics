import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

spec = importlib.util.spec_from_file_location(
    "figures", Path(__file__).parents[1] / "scripts" / "figures.py")
figures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figures)


def test_lowercase_category_label_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "FIGURES", tmp_path)
    fig, ax = plt.subplots()
    ax.barh(["Sudden Death", "both teams possess"], [0.6, 0.5])  # filled in only at draw time
    with pytest.raises(ValueError, match="both teams possess"):
        figures.save(fig, "bad.png")
    assert not (tmp_path / "bad.png").exists()


def test_lowercase_word_after_a_number_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "FIGURES", tmp_path)
    fig, ax = plt.subplots()
    ax.barh(["11 Personnel", "12 personnel"], [0.6, 0.5])
    with pytest.raises(ValueError, match="personnel"):
        figures.save(fig, "bad.png")


def test_em_and_en_dashes_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "FIGURES", tmp_path)
    for dash in ("\u2014", "\u2013"):
        fig, ax = plt.subplots()
        ax.set_title(f"Receive {dash} Kick")
        with pytest.raises(ValueError, match="dash"):
            figures.save(fig, "bad.png")


def test_lowercase_word_rules():
    assert figures.lowercase_word("3rd down") == "down"
    assert figures.lowercase_word("Sudden Death, 1999-2011") is None
    assert figures.lowercase_word("Receiver's Chance to Win vs the Kicker") is None
    assert figures.lowercase_word("to Win") == "to"
    assert figures.lowercase_word("Data: nflverse Play-by-Play  |  100,000 per Rule") is None
    assert figures.lowercase_word("Coin-toss Winner") == "toss"
    assert figures.lowercase_word("Receive 47% (tie 14%)") == "tie"
    assert figures.lowercase_word("1999-2011, 214 Games") is None


def test_capitalised_chart_is_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "FIGURES", tmp_path)
    fig, ax = plt.subplots()
    ax.barh(["Sudden Death", "Both Teams Possess"], [0.6, 0.5])
    figures.save(fig, "good.png")
    assert (tmp_path / "good.png").exists()
