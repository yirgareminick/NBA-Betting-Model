"""
Daily Betting Report Generator

This module creates comprehensive daily reports with predictions, betting recommendations,
and performance analysis for NBA games.
"""

import pandas as pd
import numpy as np
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Optional
import yaml
import json
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src"))

try:
    from predict.predict_games import predict_daily_games
    from stake.kelly_criterion import calculate_daily_bets
except ImportError as e:
    print(f"Warning: Could not import dependencies: {e}")
    predict_daily_games = None
    calculate_daily_bets = None


class DailyReportGenerator:
    """Generates daily betting reports with predictions and recommendations."""

    def __init__(self, bankroll: float = 10000):
        self.bankroll = bankroll
        self.project_root = Path(__file__).parent.parent.parent
        self.reports_dir = self.project_root / "reports"
        self.reports_dir.mkdir(exist_ok=True)

    def generate_daily_report(self, target_date: date = None) -> Dict:
        """Generate complete daily betting report."""
        if target_date is None:
            target_date = date.today()

        print("=" * 80)
        print(f"📊 GENERATING DAILY REPORT - {target_date}")
        print("=" * 80)

        # Get predictions
        predictions = predict_daily_games(target_date)

        if predictions.empty:
            return self._generate_empty_report(target_date)

        # Calculate betting recommendations
        betting_recommendations, simulation_results = calculate_daily_bets(
            predictions, self.bankroll
        )

        # Generate report data
        report_data = {
            'date': target_date.isoformat(),
            'bankroll': self.bankroll,
            'total_games': len(predictions),
            'recommended_bets': int(betting_recommendations['recommended_bet'].sum()),
            'total_stake': float(betting_recommendations['stake_amount'].sum()),
            'expected_value': float(betting_recommendations['expected_value'].sum()),
            'simulation_results': simulation_results,
            'games': self._format_games_data(betting_recommendations),
            'summary': self._generate_summary(betting_recommendations, simulation_results)
        }

        # Save reports
        self._save_json_report(report_data, target_date)
        self._save_html_report(report_data, target_date)
        self._save_csv_data(betting_recommendations, target_date)

        print("=" * 80)
        print("✅ DAILY REPORT COMPLETED")
        print(f"📁 Reports saved to: {self.reports_dir}")
        print("=" * 80)

        return report_data

    def _generate_empty_report(self, target_date: date) -> Dict:
        """Generate report when no games are available."""
        return {
            'date': target_date.isoformat(),
            'bankroll': self.bankroll,
            'total_games': 0,
            'recommended_bets': 0,
            'total_stake': 0.0,
            'expected_value': 0.0,
            'games': [],
            'summary': 'No games available for betting today.'
        }

    def _format_games_data(self, betting_df: pd.DataFrame) -> list:
        """Format games data for report."""
        games_data = []

        for _, row in betting_df.iterrows():
            game_data = {
                'matchup': f"{row['away_team']} @ {row['home_team']}",
                'predicted_winner': row['predicted_winner'],
                'confidence': f"{row['confidence']:.1%}",
                'home_prob': f"{row['home_prob']:.1%}",
                'away_prob': f"{row['away_prob']:.1%}",
                'home_odds': f"{row['home_odds']:.2f}",
                'away_odds': f"{row['away_odds']:.2f}",
                'best_bet_team': row['best_bet_team'],
                'best_bet_edge': f"{row['best_bet_edge']:.1%}",
                'recommended': row['recommended_bet'],
                'stake_amount': f"${row['stake_amount']:.2f}",
                'expected_value': f"${row['expected_value']:.2f}"
            }
            games_data.append(game_data)

        return games_data

    def _generate_summary(self, betting_df: pd.DataFrame, simulation_results: Dict) -> str:
        """Generate text summary of daily recommendations."""
        total_games = len(betting_df)
        recommended_bets = betting_df['recommended_bet'].sum()
        total_stake = betting_df['stake_amount'].sum()
        expected_value = betting_df['expected_value'].sum()

        if recommended_bets == 0:
            return "No betting opportunities identified today. All games lack sufficient edge."

        summary_lines = [
            f"Daily Analysis Summary:",
            f"• {recommended_bets} of {total_games} games meet betting criteria",
            f"• Total recommended stake: ${total_stake:,.2f} ({total_stake/self.bankroll:.1%} of bankroll)",
            f"• Expected value: ${expected_value:,.2f}",
            f"• Probability of profit: {simulation_results['prob_profit']:.1%}",
            f"• Expected return range: ${simulation_results['percentile_5']:,.2f} to ${simulation_results['percentile_95']:,.2f}"
        ]

        return "\n".join(summary_lines)

    def _save_json_report(self, report_data: Dict, target_date: date):
        """Save report as JSON file."""
        filename = f"daily_report_{target_date.strftime('%Y%m%d')}.json"
        filepath = self.reports_dir / filename

        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

        print(f"💾 JSON report saved: {filepath}")

    def _save_html_report(self, report_data: Dict, target_date: date):
        """Save report as HTML file."""
        html_content = self._generate_html_report(report_data)

        filename = f"daily_report_{target_date.strftime('%Y%m%d')}.html"
        filepath = self.reports_dir / filename

        with open(filepath, 'w') as f:
            f.write(html_content)

        print(f"🌐 HTML report saved: {filepath}")

    def _save_csv_data(self, betting_df: pd.DataFrame, target_date: date):
        """Save detailed betting data as CSV."""
        filename = f"betting_data_{target_date.strftime('%Y%m%d')}.csv"
        filepath = self.reports_dir / filename

        betting_df.to_csv(filepath, index=False)
        print(f"📊 CSV data saved: {filepath}")

    def _generate_html_report(self, report_data: Dict) -> str:
        """Generate HTML report content."""
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>NBA Betting Report - {date}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ margin: 20px 0; padding: 15px; background-color: #e8f4fd; border-radius: 5px; }}
        .games-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .games-table th, .games-table td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        .games-table th {{ background-color: #f2f2f2; }}
        .recommended {{ background-color: #d4edda; }}
        .not-recommended {{ background-color: #f8d7da; }}
        .simulation {{ margin: 20px 0; padding: 15px; background-color: #fff3cd; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>NBA Betting Report</h1>
        <h2>Date: {date}</h2>
        <p><strong>Bankroll:</strong> ${bankroll:,.2f}</p>
    </div>

    <div class="summary">
        <h3>Daily Summary</h3>
        <p><strong>Total Games:</strong> {total_games}</p>
        <p><strong>Recommended Bets:</strong> {recommended_bets}</p>
        <p><strong>Total Stake:</strong> ${total_stake:,.2f}</p>
        <p><strong>Expected Value:</strong> ${expected_value:,.2f}</p>
    </div>

    {simulation_section}

    <h3>Game Analysis</h3>
    {games_table}

    <div style="margin-top: 30px; font-size: 0.9em; color: #666;">
        <p>Generated on {timestamp}</p>
        <p>This report is for informational purposes only. Please gamble responsibly.</p>
    </div>
</body>
</html>
        """

        # Generate simulation section
        if 'simulation_results' in report_data:
            sim = report_data['simulation_results']
            simulation_section = f"""
            <div class="simulation">
                <h3>Risk Analysis (1000 simulations)</h3>
                <p><strong>Expected Return:</strong> ${sim['mean_return']:,.2f}</p>
                <p><strong>Probability of Profit:</strong> {sim['prob_profit']:.1%}</p>
                <p><strong>5th Percentile:</strong> ${sim['percentile_5']:,.2f}</p>
                <p><strong>95th Percentile:</strong> ${sim['percentile_95']:,.2f}</p>
            </div>
            """
        else:
            simulation_section = ""

        # Generate games table
        if report_data['games']:
            table_rows = []
            for game in report_data['games']:
                row_class = "recommended" if game['recommended'] else "not-recommended"
                row = f"""
                <tr class="{row_class}">
                    <td>{game['matchup']}</td>
                    <td>{game['predicted_winner']}</td>
                    <td>{game['confidence']}</td>
                    <td>{game['best_bet_team']}</td>
                    <td>{game['best_bet_edge']}</td>
                    <td>{game['stake_amount']}</td>
                    <td>{game['expected_value']}</td>
                </tr>
                """
                table_rows.append(row)

            games_table = f"""
            <table class="games-table">
                <thead>
                    <tr>
                        <th>Matchup</th>
                        <th>Predicted Winner</th>
                        <th>Confidence</th>
                        <th>Best Bet</th>
                        <th>Edge</th>
                        <th>Stake</th>
                        <th>Expected Value</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(table_rows)}
                </tbody>
            </table>
            """
        else:
            games_table = "<p>No games available today.</p>"

        return html_template.format(
            date=report_data['date'],
            bankroll=report_data['bankroll'],
            total_games=report_data['total_games'],
            recommended_bets=report_data['recommended_bets'],
            total_stake=report_data['total_stake'],
            expected_value=report_data['expected_value'],
            simulation_section=simulation_section,
            games_table=games_table,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )


def generate_daily_report(target_date: date = None, bankroll: float = 10000) -> Dict:
    """Main function to generate daily betting report."""
    generator = DailyReportGenerator(bankroll)
    return generator.generate_daily_report(target_date)


if __name__ == "__main__":
    report = generate_daily_report()
    print(f"Report generated with {report['recommended_bets']} recommended bets")
