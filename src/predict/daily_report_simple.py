"""
Simple daily betting report generator for testing purposes.
"""
from datetime import date
from typing import Dict
from pathlib import Path
import pandas as pd

# Default bankroll constant
DEFAULT_BANKROLL = 10000


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
