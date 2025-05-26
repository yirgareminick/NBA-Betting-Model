"""
Download season-level team metrics (ORtg, DRtg, Pace, SOS) from Basketball Reference.
Each season is saved into data/raw/team_stats_{season}.csv.
"""

from __future__ import annotations
from pathlib import Path
import polars as pl
import requests
from bs4 import BeautifulSoup

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def fetch_bbref_table(season: int) -> pl.DataFrame:
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_ratings.html"
    print(f"[fetch] Requesting: {url}")
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")
    table = soup.select_one("table#ratings")
    if table is None:
        raise ValueError(f"Could not find 'ratings' table on page for season {season}")
    print("[fetch] Ratings table found.")

    # Use pandas as bridge since polars cannot read HTML directly
    import pandas as pd
    from io import StringIO
    pd_df = pd.read_html(StringIO(str(table)), header=[0, 1])[0]
    pd_df.columns = [
        f"{col[0]}_{col[1]}".strip().replace(" ", "_").lower()
        if isinstance(col, tuple) else str(col).strip().replace(" ", "_").lower()
        for col in pd_df.columns
    ]
    print("[debug] Flattened columns:", pd_df.columns.tolist())
    df = pl.from_pandas(pd_df)
    print(f"[debug] Parsed dataframe shape: {df.shape}")
    return df

def fetch_season(season: int) -> pl.DataFrame:
    csv_path = RAW_DIR / f"team_stats_{season}.csv"
    if csv_path.exists():
        print(f"[cache] {csv_path} exists — deleting for fresh scrape")
        csv_path.unlink()

    df = fetch_bbref_table(season)
    print(f"[save] Writing data to: {csv_path}")
    df.write_csv(csv_path)
    return df

def pull_team_stats(seasons: list[int]) -> pl.DataFrame:
    print(f"[start] Pulling team stats for seasons: {seasons}")
    return pl.concat([fetch_season(s) for s in seasons])

if __name__ == "__main__":
    import argparse
    a = argparse.ArgumentParser()
    a.add_argument("--seasons", nargs="+", required=True, type=int)
    args = a.parse_args()
    print(pull_team_stats(args.seasons).head())