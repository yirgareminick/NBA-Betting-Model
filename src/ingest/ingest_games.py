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
"""
ingest_games.py

Utility for downloading historical NBA game results from the Kaggle dataset
`wyattowalsh/basketball` using KaggleHub, then returning a filtered Polars
DataFrame.  The CSV is cached to data/raw/ so subsequent calls reuse the file.

Usage (CLI):
    poetry run python -m src.ingest.ingest_games --start 2023-10-01 --end 2024-06-30
"""

from __future__ import annotations

from pathlib import Path
from datetime import date
import polars as pl

RAW_DIR = Path("data/raw")
DATASET = "wyattowalsh/basketball"
CSV_PATH_IN_DATASET = "games/games.csv"


def _kaggle_download(file_path: str = CSV_PATH_IN_DATASET) -> Path:
    """
    Download *file_path* from the Kaggle dataset via KaggleHub and cache locally.
    Returns the local Path of the CSV.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / Path(file_path).name

    # reuse cached copy if it exists
    if target.exists():
        return target

    try:
        from kagglehub import KaggleDatasetAdapter, load_dataset
    except ImportError as exc:
        raise ImportError(
            "kagglehub is not installed. Install with "
            "`poetry add kagglehub[pandas-datasets]`."
        ) from exc

    print(f"⬇️  Downloading {file_path} from {DATASET} via KaggleHub …")
    df = load_dataset(
        KaggleDatasetAdapter.PANDAS,
        DATASET,
        file_path,
    )
    df.to_csv(target, index=False)
    return target


def pull_games(
    start: str,
    end: str,
    *,
    file_path: str = CSV_PATH_IN_DATASET,
) -> pl.DataFrame:
    """
    Return a Polars DataFrame with games between *start* and *end* (inclusive).

    Parameters
    ----------
    start, end : str
        ISO date strings ``YYYY-MM-DD`` bounding the desired window.
    file_path : str, optional
        Alternative relative path inside the Kaggle dataset.
    """
    csv_path = _kaggle_download(file_path)
    df = pl.read_csv(csv_path, try_parse_dates=True)

    return (
        df.filter(pl.col("GAME_DATE") >= start)
          .filter(pl.col("GAME_DATE") <= end)
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--path",
        default=CSV_PATH_IN_DATASET,
        help="File path inside Kaggle dataset (default: games/games.csv)",
    )
    args = parser.parse_args()

    print(pull_games(args.start, args.end, file_path=args.path).head())