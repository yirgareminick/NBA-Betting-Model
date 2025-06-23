# build_features.py

import polars as pl
import sqlite3
from pathlib import Path

def load_table(db_path: Path, table: str) -> pl.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pl.read_database(f"SELECT * FROM {table}", conn)
    conn.close()
    return df

def build_features(db_path: Path, lookback: int = 5) -> pl.DataFrame:
    # Load raw tables
    games = load_table(db_path, "games")
    teams = load_table(db_path, "team_stats")
    odds = load_table(db_path, "odds")

    # Parse dates
    games = games.with_columns([
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d")
    ])

    # Sort for rolling calculations
    games = games.sort("date")

    # Explode into per-team rows
    home_games = games.select([
        pl.col("game_id"),
        pl.col("date"),
        pl.col("home_team").alias("team"),
        pl.col("home_pts").alias("pts"),
        pl.col("away_pts").alias("opp_pts"),
        (pl.col("home_pts") > pl.col("away_pts")).cast(pl.Int8).alias("win"),
    ])
    away_games = games.select([
        pl.col("game_id"),
        pl.col("date"),
        pl.col("away_team").alias("team"),
        pl.col("away_pts").alias("pts"),
        pl.col("home_pts").alias("opp_pts"),
        (pl.col("away_pts") > pl.col("home_pts")).cast(pl.Int8).alias("win"),
    ])
    long_games = pl.concat([home_games, away_games])

    # Rolling stats
    rolling = (
        long_games.sort(["team", "date"])
        .groupby_dynamic(index_column="date", every="1d", by="team", period=f"{lookback}d")
        .agg([
            pl.col("win").mean().alias("win_pct_last_n"),
            pl.col("pts").mean().alias("avg_pts_last_n"),
            pl.col("opp_pts").mean().alias("avg_opp_pts_last_n"),
        ])
    )

    # Merge into original games
    # Placeholder: will need a smarter join per team-game-date
    features = games.join(rolling, left_on=["home_team", "date"], right_on=["team", "date"], how="left")

    return features

if __name__ == "__main__":
    DB_PATH = Path("data/interim/nba_games.sqlite")
    features = build_features(DB_PATH)
    features.write_parquet("data/processed/features.parquet")