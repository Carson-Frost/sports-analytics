"""Raw nflverse files -> data/derived/plays.parquet and data/derived/ot_games.csv."""

from overtime.data import DERIVED, build_ot_games, build_plays


def main():
    DERIVED.mkdir(parents=True, exist_ok=True)
    plays = build_plays()
    plays.to_parquet(DERIVED / "plays.parquet", index=False)
    games = build_ot_games(plays)
    games.to_csv(DERIVED / "ot_games.csv", index=False)
    print(f"{len(plays):,} plays, {len(games)} overtime games")
    print(games.groupby(["rule", "clock"]).size().to_string())


if __name__ == "__main__":
    main()
