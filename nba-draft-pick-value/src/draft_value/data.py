"""Read Basketball Reference draft pages into one row per draftee."""

import io
import re
from pathlib import Path

import pandas as pd

URL = "https://www.basketball-reference.com/draft/NBA_{year}.html"
YEARS = range(1990, 2017)  # every class has at least 10 seasons of career by 2025-26


def parse_page(html: str, year: int) -> pd.DataFrame:
    """One row per draftee from a draft page's `stats` table."""
    table = pd.read_html(io.StringIO(html), attrs={"id": "stats"})[0]
    table.columns = [bottom for _, bottom in table.columns]
    table = table.loc[:, ~table.columns.duplicated()]  # Per Game repeats MP, PTS, TRB, AST
    # the round comes from the page's "Round 2" header row, not from the pick number
    table["round"] = 1 + table.astype(str).eq("Round 2").any(axis=1).cumsum()
    picks = pd.to_numeric(table["Pk"], errors="coerce")
    table = table[picks.notna()]  # drops the header rows repeated inside the table
    out = pd.DataFrame(
        {
            "year": year,
            "pick": pd.to_numeric(table["Pk"]).astype(int),
            "round": table["round"],
            "player": table["Player"],
            "games": pd.to_numeric(table["G"], errors="coerce").fillna(0).astype(int),
            # a draftee who never played has blank stats: he added nothing, so 0, not missing
            "win_shares": pd.to_numeric(table["WS"], errors="coerce").fillna(0.0),
            "vorp": pd.to_numeric(table["VORP"], errors="coerce").fillna(0.0),
        }
    )
    return out.reset_index(drop=True)


def read_raw(raw_dir: Path) -> pd.DataFrame:
    """Every `NBA_<year>.html` in `raw_dir`, stacked."""
    pages = sorted(raw_dir.glob("NBA_*.html"))
    if not pages:
        raise FileNotFoundError(f"No NBA_<year>.html pages in {raw_dir}; run scripts/pull.py")
    frames = []
    for page in pages:
        year = int(re.search(r"NBA_(\d{4})", page.name).group(1))
        frames.append(parse_page(page.read_text(encoding="utf-8"), year))
    return pd.concat(frames, ignore_index=True)
