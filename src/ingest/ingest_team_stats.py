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
    
    # Add headers to avoid 403 errors
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        resp = requests.get(url, timeout=15, headers=headers)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            print(f"[error] Access forbidden (403) - possibly rate limited or off-season data not available")
            raise ValueError(f"Access forbidden for season {season} - possibly off-season") from e
        raise
    except requests.exceptions.Timeout:
        print(f"[error] Request timed out for season {season}")
        raise ValueError(f"Request timed out fetching data for season {season}")
    except requests.exceptions.RequestException as e:
        print(f"[error] Failed to fetch data: {e}")
        raise ValueError(f"Failed to fetch data for season {season}") from e

    soup = BeautifulSoup(resp.text, "lxml")
    try:
        table = soup.select_one("table#ratings")
        if table is None:
            raise ValueError(f"Could not find 'ratings' table on page for season {season}")
        print("[fetch] Ratings table found.")

        # Use pandas as bridge since polars cannot read HTML directly
        import pandas as pd
        from io import StringIO
        try:
            pd_df = pd.read_html(StringIO(str(table)), header=[0, 1])[0]
        except Exception as e:
            print(f"[error] Failed to parse HTML table: {e}")
            raise ValueError(f"Failed to parse data table for season {season}") from e
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
    
    # Check if file already exists and is recent (within 7 days)
    if csv_path.exists():
        from datetime import datetime, timedelta
        file_age = datetime.now() - datetime.fromtimestamp(csv_path.stat().st_mtime)
        
        if file_age < timedelta(days=7):
            print(f"[cache] Using recent {csv_path} (age: {file_age.days} days)")
            return pl.read_csv(csv_path)
        else:
            print(f"[cache] {csv_path} exists but is old ({file_age.days} days) — attempting fresh scrape")

    try:
        df = fetch_bbref_table(season)
        print(f"[save] Writing data to: {csv_path}")
        df.write_csv(csv_path)
        return df
    except Exception as e:
        print(f"[error] Failed to fetch season {season}: {e}")
        
        # Fallback to existing file if scraping fails
        if csv_path.exists():
            print(f"[fallback] Using existing {csv_path} as fallback")
            return pl.read_csv(csv_path)
        else:
            print(f"[error] No fallback data available for season {season}")
            raise e

def pull_team_stats(seasons: list[int]) -> pl.DataFrame:
    print(f"[start] Pulling team stats for seasons: {seasons}")
    
    all_data = []
    failed_seasons = []
    
    for season in seasons:
        try:
            df = fetch_season(season)
            all_data.append(df)
        except Exception as e:
            print(f"[warning] Failed to fetch season {season}: {e}")
            failed_seasons.append(season)
            continue
    
    if not all_data:
        raise ValueError(f"Failed to fetch data for all seasons: {failed_seasons}")
    
    if failed_seasons:
        print(f"[warning] Some seasons failed: {failed_seasons}")
        
    return pl.concat(all_data)

if __name__ == "__main__":
    import argparse
    a = argparse.ArgumentParser()
    a.add_argument("--seasons", nargs="+", required=True, type=int)
    args = a.parse_args()
    print(pull_team_stats(args.seasons).head())