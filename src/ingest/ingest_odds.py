"""
ingest_odds.py

Fetch NBA moneyline odds from The Odds API and return a cleaned Polars DataFrame.

Usage:
    poetry run python src/ingest/ingest_odds.py --regions us --bookmakers draftkings,fan
"""

import os
import requests
from pathlib import Path
import polars as pl
import argparse
from dotenv import load_dotenv


load_dotenv()

API_KEY = os.getenv("ODDS_API_KEY")
if not API_KEY:
    raise ValueError("Set ODDS_API_KEY in .env file. Get your free key at: https://the-odds-api.com/")

SPORT = "basketball_nba"
ENDPOINT = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

def fetch_odds(regions="us", markets="h2h", odds_format="american", bookmakers=None):
    params = {
        "apiKey": API_KEY,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": "iso"
    }
    if bookmakers:
        filtered = [b for b in bookmakers if b]
        if filtered:
            params["bookmakers"] = ",".join(filtered)
    resp = requests.get(ENDPOINT, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()

def parse_odds(json_data):
    records = []
    for game in json_data:
        game_id = game.get("id")
        ct = game.get("commence_time")
        home = game.get("home_team")
        away = game.get("away_team")
        for book in game.get("bookmakers", []):
            book_name = book.get("key")
            h2h = next((m for m in book.get("markets", []) if m["key"] == "h2h"), None)
            if not h2h:
                continue
            odds = {o["name"]: o["price"] for o in h2h["outcomes"]}
            home_odds = odds.get(home)
            away_odds = odds.get(away)
            # Calculate implied probabilities for American odds
            def implied_prob(odds_val):
                if odds_val is None:
                    return None
                try:
                    odds_val = float(odds_val)
                except (ValueError, TypeError):
                    # Invalid odds format, return None
                    return None
                if odds_val > 0:
                    return 100 / (odds_val + 100)
                else:
                    return abs(odds_val) / (abs(odds_val) + 100)
            home_prob = implied_prob(home_odds)
            away_prob = implied_prob(away_odds)
            records.append({
                "game_id": game_id,
                "commence_time": ct,
                "home_team": home,
                "away_team": away,
                "bookmaker": book_name,
                "home_odds": home_odds,
                "away_odds": away_odds,
                "home_prob": home_prob,
                "away_prob": away_prob,
            })
    df = pl.DataFrame(records)
    # Parse commence_time as datetime
    if "commence_time" in df.columns:
        df = df.with_columns([
            pl.col("commence_time").str.strptime(pl.Datetime, strict=False).alias("commence_time")
        ])
    return df

def pull_odds_to_csv(regions="us", bookmakers=None, out_csv=None):
    data = fetch_odds(regions=regions, bookmakers=bookmakers)
    df = parse_odds(data)
    path = out_csv or RAW_DIR / f"odds_{SPORT}_{regions}.csv"
    df.write_csv(path)
    return df

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--regions", default="us", help="Bookmaker regions (csv)")
    p.add_argument("--bookmakers", default=None, help="Specific bookmakers (csv keys)")
    p.add_argument("--out_csv", default=None, help="Optional output file path")
    args = p.parse_args()

    df = pull_odds_to_csv(regions=args.regions, bookmakers=(args.bookmakers or "").split(","), out_csv=args.out_csv)
    print(df.head())