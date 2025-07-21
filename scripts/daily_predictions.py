#!/usr/bin/env python3
"""
Daily NBA Predictions Script

This script generates daily NBA game predictions, calculates betting recommendations,
and creates comprehensive reports. It's designed to be run daily during NBA season.

Usage:
    python scripts/daily_predictions.py
    python scripts/daily_predictions.py --date 2024-01-15
    python scripts/daily_predictions.py --bankroll 5000
    python scripts/daily_predictions.py --config custom_config.yml
"""

import argparse
import sys
from datetime import date, datetime
from pathlib import Path
import yaml

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from predict.daily_report import generate_daily_report
from predict.predict_games import predict_daily_games
from stake.kelly_criterion import calculate_daily_bets


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate daily NBA betting predictions")
    
    parser.add_argument(
        "--date", 
        type=str, 
        help="Target date for predictions (YYYY-MM-DD). Defaults to today."
    )
    
    parser.add_argument(
        "--bankroll", 
        type=float, 
        default=10000,
        help="Betting bankroll amount. Default: $10,000"
    )
    
    parser.add_argument(
        "--config", 
        type=str,
        help="Path to custom configuration file"
    )
    
    parser.add_argument(
        "--predictions-only", 
        action="store_true",
        help="Generate predictions only, skip betting calculations"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str,
        help="Custom output directory for reports"
    )
    
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Suppress detailed output"
    )
    
    return parser.parse_args()


def load_config(config_path: str = None) -> dict:
    """Load configuration from file."""
    if config_path:
        config_file = Path(config_path)
    else:
        config_file = Path(__file__).parent.parent / "configs" / "model.yml"
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    else:
        print(f"⚠️  Config file not found: {config_file}")
        return {}


def print_predictions_summary(predictions_df):
    """Print a summary of predictions."""
    if len(predictions_df) == 0:
        print("📭 No games found for prediction")
        return
    
    print(f"\n🔮 PREDICTIONS SUMMARY")
    print("=" * 50)
    
    for _, game in predictions_df.iterrows():
        print(f"{game['away_team']} @ {game['home_team']}")
        print(f"  Predicted winner: {game['predicted_winner']} ({game['confidence']:.1%} confidence)")
        print(f"  Model probabilities: {game['home_team']} {game['home_prob']:.1%}, {game['away_team']} {game['away_prob']:.1%}")
        if 'best_bet_edge' in game:
            print(f"  Best bet: {game['best_bet_team']} (Edge: {game['best_bet_edge']:.1%})")
        print()


def print_betting_summary(betting_df, simulation_results):
    """Print a summary of betting recommendations."""
    recommended_bets = betting_df[betting_df['recommended_bet']]
    
    if len(recommended_bets) == 0:
        print("💰 No betting opportunities identified")
        return
    
    print(f"\n💰 BETTING RECOMMENDATIONS")
    print("=" * 50)
    
    total_stake = recommended_bets['stake_amount'].sum()
    total_ev = recommended_bets['expected_value'].sum()
    
    print(f"Recommended bets: {len(recommended_bets)}")
    print(f"Total stake: ${total_stake:,.2f}")
    print(f"Expected value: ${total_ev:,.2f}")
    print(f"Probability of profit: {simulation_results['prob_profit']:.1%}")
    print()
    
    for _, bet in recommended_bets.iterrows():
        print(f"{bet['away_team']} @ {bet['home_team']}")
        print(f"  Bet: {bet['best_bet_team']} (Edge: {bet['best_bet_edge']:.1%})")
        print(f"  Stake: ${bet['stake_amount']:.2f}")
        print(f"  Expected value: ${bet['expected_value']:.2f}")
        print()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Parse target date
    if args.date:
        try:
            target_date = datetime.strptime(args.date, '%Y-%m-%d').date()
        except ValueError:
            print("❌ Invalid date format. Use YYYY-MM-DD")
            sys.exit(1)
    else:
        target_date = date.today()
    
    # Load configuration
    config = load_config(args.config)
    
    if not args.quiet:
        print("🏀 NBA DAILY PREDICTIONS")
        print("=" * 80)
        print(f"Date: {target_date}")
        print(f"Bankroll: ${args.bankroll:,.2f}")
        print("=" * 80)
    
    try:
        if args.predictions_only:
            # Generate predictions only
            predictions = predict_daily_games(target_date)
            
            if not args.quiet:
                print_predictions_summary(predictions)
            
            print(f"✅ Predictions generated for {len(predictions)} games")
            
        else:
            # Generate full report with betting recommendations
            if args.output_dir:
                # TODO: Implement custom output directory
                print("⚠️  Custom output directory not yet implemented")
            
            report = generate_daily_report(target_date, args.bankroll)
            
            if not args.quiet and report['total_games'] > 0:
                # Load the detailed data for summary display
                predictions = predict_daily_games(target_date)
                betting_recommendations, simulation_results = calculate_daily_bets(
                    predictions, args.bankroll
                )
                
                print_predictions_summary(predictions)
                print_betting_summary(betting_recommendations, simulation_results)
            
            print(f"✅ Complete report generated:")
            print(f"   • {report['total_games']} games analyzed")
            print(f"   • {report['recommended_bets']} betting recommendations")
            print(f"   • Expected value: ${report['expected_value']:,.2f}")
    
    except Exception as e:
        print(f"❌ Error generating predictions: {str(e)}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
