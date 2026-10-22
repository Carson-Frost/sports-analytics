import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from model import split  # noqa: E402


def test_split_never_lets_a_test_game_happen_before_a_training_game():
    # 30 games over 10 days, three a day, two stints each, listed out of order
    games = [(f"g{i:02d}", f"2026-01-{1 + i // 3:02d}") for i in range(30)]
    rows = [{"game_id": g, "date": d} for g, d in reversed(games) for _ in range(2)]
    train, test = split(pd.DataFrame(rows))
    assert train["date"].max() < test["date"].min()
    assert set(train["game_id"]).isdisjoint(test["game_id"])
    assert train["game_id"].nunique() == 18 and test["game_id"].nunique() == 12  # cut on a day
