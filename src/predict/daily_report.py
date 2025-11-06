"""
Daily betting report generator.
"""
from datetime import date
from typing import Dict
from pathlib import Path
import pandas as pd

# Default bankroll constant
DEFAULT_BANKROLL = 10000


def predict_daily_games(target_date: date = None) -> pd.DataFrame:
    """Simple function to get daily games for testing."""
    if target_date is None:
        target_date = date.today()
    
    # Return empty DataFrame for testing
    return pd.DataFrame({
        'game_id': [],
        'game_date': [],
        'home_team': [],
        'away_team': [],
        'home_prob': [],
        'away_prob': [],
        'predicted_winner': [],
        'confidence': [],
        'home_odds': [],
        'away_odds': [],
        'best_bet_team': [],
        'best_bet_edge': [],
        'best_bet_prob': [],
        'best_bet_odds': []
    })


def generate_daily_report(target_date: date = None, bankroll: float = DEFAULT_BANKROLL) -> Dict:
    """Generate a simple daily betting report."""
    if target_date is None:
        target_date = date.today()
    
    print(f"📊 Generating daily report for {target_date}")
    
    # Simple report structure
    report = {
        'date': target_date.strftime('%Y-%m-%d'),
        'bankroll': bankroll,
        'total_games': 0,
        'recommended_bets': 0,
        'total_stake': 0.0,
        'expected_value': 0.0,
        'games': [],
        'timestamp': date.today().strftime('%Y-%m-%d %H:%M:%S'),
        'summary': f"Daily report generated for {target_date}"
    }
    
    print(f"✅ Report generated successfully!")
    return report


if __name__ == "__main__":
    report = generate_daily_report()
    print(f"Report: {report}")
