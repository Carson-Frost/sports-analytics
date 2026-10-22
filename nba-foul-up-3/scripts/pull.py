"""Pull NBA Stats game logs and play-by-play into data/raw, skipping files already present.

One request at a time with a pause between them, so it is slow on purpose. The API stops
answering after about 600 requests in a row; when it does, this rests for five minutes, and after
three rests in a row it stops. Rerun it and it picks up where it stopped.
"""

import argparse
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder, playbyplayv3

from foulup3 import SEASONS

COLUMNS = [
    "gameId",
    "actionNumber",
    "clock",
    "period",
    "teamId",
    "teamTricode",
    "scoreHome",
    "scoreAway",
    "description",
    "actionType",
    "subType",
    "shotValue",
    "shotResult",
]
PAUSE = 1.5  # seconds between requests
TRIES = 3
REST = 5 * 60  # seconds to wait when a game fails every try
MAX_RESTS = 3  # rests in a row before giving up for now


def fetch(call):
    for attempt in range(TRIES):
        try:
            return call()
        except Exception as error:  # the API times out or drops connections now and then
            if attempt == TRIES - 1:
                raise
            wait = 30 * 2**attempt
            print(f"  retrying in {wait}s after {type(error).__name__}", flush=True)
            time.sleep(wait)


def save(df: pd.DataFrame, path: Path) -> None:
    """Write to a temporary name first, so an interrupted run never leaves half a file."""
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def pull_season(season: str, out: Path, limit: int | None = None) -> bool:
    """Pull one season; False if the API stopped answering."""
    games_path = out / f"games_{season[:4]}{season[-2:]}_RegularSeason.parquet"
    if not games_path.exists():
        save(
            fetch(
                lambda: leaguegamefinder.LeagueGameFinder(
                    season_nullable=season,
                    league_id_nullable="00",
                    season_type_nullable="Regular Season",
                    timeout=60,
                ).get_data_frames()[0]
            ),
            games_path,
        )
        time.sleep(PAUSE)
    game_ids = sorted(pd.read_parquet(games_path, columns=["GAME_ID"]).GAME_ID.unique())[:limit]
    (out / "pbp").mkdir(parents=True, exist_ok=True)
    missing = [g for g in game_ids if not (out / "pbp" / f"{g}.parquet").exists()]
    print(f"{season}: {len(game_ids)} games, {len(missing)} to pull", flush=True)
    rests = 0
    for i, game_id in enumerate(missing, 1):
        try:
            pbp = fetch(
                lambda g=game_id: playbyplayv3.PlayByPlayV3(
                    game_id=g, timeout=60
                ).get_data_frames()[0]
            )
            save(pbp[COLUMNS], out / "pbp" / f"{game_id}.parquet")
            rests = 0
        except Exception as error:
            rests += 1
            print(f"  {game_id} failed ({type(error).__name__}), rest {rests}", flush=True)
            if rests == MAX_RESTS:
                return False
            time.sleep(REST)
        if i % 100 == 0:
            print(f"  {i}/{len(missing)} {time.strftime('%H:%M')}", flush=True)
        time.sleep(PAUSE)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="+", default=SEASONS)
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument("--limit", type=int, help="pull only the first N games of each season")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for season in args.seasons:
        if not pull_season(season, args.out, args.limit):
            print("stopped: the API is not answering, rerun later to resume")
            return
    print("done")


if __name__ == "__main__":
    main()
