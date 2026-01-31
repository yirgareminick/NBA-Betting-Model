"""
Constants for NBA Betting Model

Centralized configuration constants used across the project.
"""

# Betting configuration
DEFAULT_BANKROLL = 10000

# Model configuration
MODEL_RANDOM_STATE = 42
MODEL_TEST_SIZE = 0.2

# Feature engineering
DEFAULT_LOOKBACK_GAMES = 10
MIN_GAMES_FOR_ROLLING = 3

# Odds API configuration
ODDS_API_SPORT = "basketball_nba"
ODDS_API_ENDPOINT = "https://api.the-odds-api.com/v4"

# Date formats
DATE_FORMAT = "%Y-%m-%d"
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
