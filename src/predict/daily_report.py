"""
Daily betting report generator with multiple output formats.
"""
from datetime import date, datetime
from typing import Dict, Optional
from pathlib import Path
import pandas as pd
import json
from .predict_games import predict_daily_games

# Default bankroll constant
DEFAULT_BANKROLL = 10000


class ReportFormatter:
    """Formats betting reports in various output formats."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.reports_dir = self.project_root / "reports"
        self.reports_dir.mkdir(exist_ok=True)
    
    def format_currency(self, amount: float) -> str:
        """Format currency values."""
        return f"${amount:,.2f}"
    
    def format_percentage(self, value: float) -> str:
        """Format percentage values."""
        return f"{value * 100:.2f}%"


def generate_daily_report(target_date: date = None, bankroll: float = DEFAULT_BANKROLL) -> Dict:
    """Generate a daily betting report."""
    from stake.kelly_criterion import calculate_daily_bets
    
    if target_date is None:
        target_date = date.today()
    
    print(f"📊 Generating daily report for {target_date}")
    
    # Get daily games predictions
    predictions = predict_daily_games(target_date)
    total_games = len(predictions)
    
    # Initialize default values
    recommended_bets = 0
    total_stake = 0.0
    expected_value = 0.0
    
    # Calculate betting recommendations if we have predictions
    if not predictions.empty:
        try:
            betting_recommendations, simulation_results = calculate_daily_bets(predictions, bankroll)
            recommended_bets = betting_recommendations['recommended_bet'].sum()
            total_stake = betting_recommendations['stake_amount'].sum()
            expected_value = betting_recommendations['expected_value'].sum()
        except Exception:
            # If calculation fails, keep defaults
            pass
    
    # Simple report structure
    report = {
        'date': target_date.strftime('%Y-%m-%d'),
        'bankroll': bankroll,
        'total_games': total_games,
        'recommended_bets': recommended_bets,
        'total_stake': total_stake,
        'expected_value': expected_value,
        'games': predictions.to_dict('records') if not predictions.empty else [],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'summary': f"Daily report generated for {target_date}"
    }
    
    print(f"✅ Report generated successfully!")
    return report


if __name__ == "__main__":
    report = generate_daily_report()
    print(f"Report: {report}")
