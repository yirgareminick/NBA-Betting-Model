from pathlib import Path
from datetime import datetime
import subprocess
import polars as pl

RAW_DIR = Path("data/raw")

def _kaggle_download():
    """Grab the CSV file from Kaggle if it isn’t cached locally."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / "kaggle_games.csv"
    if target.exists():
        return target
    print("Downloading Kaggle historical NBA data …")
    subprocess.run(
        ["kaggle", "datasets", "download",
         "-d", "bkeps/nba-dataset"],
        check=True,
        cwd=RAW_DIR,
    )

    #kaggle zip extracts to cwd; rename the big CSV to a stable name
    subprocess.run(["unzip", "*.zip"], shell=True, check=True, cwd=RAW_DIR)
    (RAW_DIR / "games.csv").rename(target)
    return target

def pull_games(start: str, end: str) -> pl.DataFrame:
    csv_path = _kaggle_download()
    df = pl.read_csv(csv_path)
    df = (
        df.filter(pl.col("GAME_DATE") >= start)
          .filter(pl.col("GAME_DATE") <= end)
          .with_columns(pl.col("GAME_DATE").str.strptime(pl.Date, "%Y-%m-%d"))
    )
    return df

if __name__ == "__main__":
    import argparse
    a = argparse.ArgumentParser()
    a.add_argument("--start", required=True)
    a.add_argument("--end", required=True)
    args = a.parse_args()
    print(pull_games(args.start, args.end).head())