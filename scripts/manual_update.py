#!/usr/bin/env python3
"""
Manual Update Script for NBA Betting Model

This script allows for manual, on-demand updates when needed.
Useful for testing, emergency updates, or custom date ranges.

Usage: 
    python scripts/manual_update.py                           # Update current season
    python scripts/manual_update.py --years 2023 2024        # Update specific years
    python scripts/manual_update.py --odds-only              # Update only odds
    python scripts/manual_update.py --games-only --years 2024 # Update only 2024 games
    python scripts/manual_update.py --features-only          # Rebuild features only
"""

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import List, Optional

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.automation_base import AutomationBase


class ManualUpdater(AutomationBase):
    """Manual update automation for NBA betting model with flexible options."""
    
    def __init__(self, 
                 start_year: Optional[int] = None,
                 end_year: Optional[int] = None,
                 odds_only: bool = False,
                 games_only: bool = False,
                 teams_only: bool = False,
                 features_only: bool = False,
                 bookmakers: List[str] = None,
                 lookback_days: int = 10):
        super().__init__("manual_update")
        
        # Set default years if not provided
        current_year = date.today().year
        self.start_year = start_year or current_year
        self.end_year = end_year or current_year
        
        # Update flags
        self.odds_only = odds_only
        self.games_only = games_only
        self.teams_only = teams_only
        self.features_only = features_only
        
        # Configuration
        self.bookmakers = bookmakers or ["draftkings", "fanduel"]
        self.lookback_days = lookback_days
        
        # Validate configuration
        self._validate_configuration()
        
    def _validate_configuration(self):
        """Validate the configuration parameters."""
        # Check year range
        if self.start_year > self.end_year:
            raise ValueError(f"Start year ({self.start_year}) cannot be greater than end year ({self.end_year})")
            
        current_year = date.today().year
        if self.end_year > current_year + 1:
            self.logger.warning(f"End year ({self.end_year}) is in the future")
            
        # Check mutually exclusive flags
        exclusive_flags = [self.odds_only, self.games_only, self.teams_only, self.features_only]
        if sum(exclusive_flags) > 1:
            raise ValueError("Only one of --odds-only, --games-only, --teams-only, --features-only can be specified")
            
    def update_odds(self):
        """Update betting odds data."""
        bookmakers_str = ",".join(self.bookmakers)
        self.run_python_script(
            "src/ingest/ingest_odds.py",
            ["--regions", "us", "--bookmakers", bookmakers_str],
            f"Odds: {bookmakers_str}"
        )
        
    def update_games(self):
        """Update games data for specified year range."""
        self.run_python_script(
            "src/ingest/ingest_games_new.py",
            [str(self.start_year), str(self.end_year)],
            f"Games: {self.start_year}-{self.end_year}"
        )
        
    def update_team_stats(self):
        """Update team statistics for specified years."""
        seasons = [str(year) for year in range(self.start_year, self.end_year + 1)]
        self.run_python_script(
            "src/ingest/ingest_team_stats.py",
            ["--seasons"] + seasons,
            f"Team stats: {'-'.join(seasons)}"
        )
        
    def rebuild_features(self):
        """Rebuild feature engineering."""
        self.run_python_script(
            "src/features/build_features.py",
            ["--lookback", str(self.lookback_days)],
            f"Features: {self.lookback_days}d lookback"
        )
        
    def run_full_update(self):
        """Run a complete update of all data sources."""
        try:
            self.update_games()
            self.update_team_stats()
            self.update_odds()
            self.rebuild_features()
        except Exception as e:
            self.logger.error(f"Update failed: {str(e)}")
            raise
            
    def run(self, send_notification: bool = False, dry_run: bool = False):
        """Execute the manual update process."""
        try:
            self.logger.info(f"Manual update: {self.start_year}-{self.end_year}{' (DRY RUN)' if dry_run else ''}")
            
            if dry_run:
                return True
            
            # Execute based on flags
            if self.odds_only:
                self.update_odds()
            elif self.games_only:
                self.update_games()
            elif self.teams_only:
                self.update_team_stats()
            elif self.features_only:
                self.rebuild_features()
            else:
                # Full update
                self.run_full_update()
                
            # Generate report
            quality_report = self.check_data_quality()
            
            # Check for issues and log summary
            issues = []
            if quality_report.get('feature_records') == "missing":
                issues.append("No features")
            if quality_report.get('games_records') == "missing":
                issues.append("No games")
                
            if issues:
                self.logger.warning(f"Issues: {', '.join(issues)}")
            else:
                records = quality_report.get('feature_records', 'N/A')
                self.logger.info(f"Update complete: {records} records")
                
            # Finalize
            self.finalize(success=True, send_notification=send_notification)
            return True
            
        except Exception as e:
            self.logger.error(f"Manual update failed: {str(e)}")
            self.logger.exception("Full error details:")
            self.finalize(success=False, send_notification=send_notification)
            return False


def main():
    """Main entry point for the manual update script."""
    parser = argparse.ArgumentParser(
        description="Manual update script for NBA betting model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/manual_update.py                           # Update current season
    python scripts/manual_update.py --years 2023 2024        # Update specific years
    python scripts/manual_update.py --odds-only              # Update only odds
    python scripts/manual_update.py --games-only --years 2024 # Update only 2024 games
    python scripts/manual_update.py --features-only          # Rebuild features only
    python scripts/manual_update.py --bookmakers draftkings fanduel bovada
    python scripts/manual_update.py --dry-run                # Test without changes

Environment Variables:
    KAGGLE_USERNAME   Your Kaggle username
    KAGGLE_KEY        Your Kaggle API key
    ODDS_API_KEY      Your Odds API key
    SLACK_WEBHOOK_URL Slack webhook for notifications
        """
    )
    
    # Year specification
    parser.add_argument(
        "--years",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Specify start and end years (e.g., --years 2023 2024)"
    )
    
    # Update type flags (mutually exclusive)
    update_group = parser.add_mutually_exclusive_group()
    update_group.add_argument(
        "--odds-only",
        action="store_true",
        help="Update only betting odds data"
    )
    update_group.add_argument(
        "--games-only",
        action="store_true",
        help="Update only games data"
    )
    update_group.add_argument(
        "--teams-only",
        action="store_true",
        help="Update only team statistics"
    )
    update_group.add_argument(
        "--features-only",
        action="store_true",
        help="Rebuild features only (no data ingestion)"
    )
    
    # Configuration options
    parser.add_argument(
        "--bookmakers",
        nargs="+",
        default=["draftkings", "fanduel"],
        help="Specify bookmakers for odds (default: draftkings fanduel)"
    )
    
    parser.add_argument(
        "--lookback",
        type=int,
        default=10,
        help="Lookback days for feature engineering (default: 10)"
    )
    
    # Execution options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing"
    )
    
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Send notification on completion (requires webhook URL)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Parse years
    start_year = None
    end_year = None
    if args.years:
        start_year, end_year = args.years
        
    # Validate bookmakers
    valid_bookmakers = ["draftkings", "fanduel", "bovada", "betmgm", "caesars", "pointsbet"]
    for bookmaker in args.bookmakers:
        if bookmaker not in valid_bookmakers:
            parser.error(f"Unknown bookmaker: {bookmaker}. Valid options: {', '.join(valid_bookmakers)}")
            
    # Run the manual update
    try:
        updater = ManualUpdater(
            start_year=start_year,
            end_year=end_year,
            odds_only=args.odds_only,
            games_only=args.games_only,
            teams_only=args.teams_only,
            features_only=args.features_only,
            bookmakers=args.bookmakers,
            lookback_days=args.lookback
        )
        
        # Set log level
        updater.logger.setLevel(args.log_level)
        for handler in updater.logger.handlers:
            handler.setLevel(args.log_level)
        
        success = updater.run(send_notification=args.notify, dry_run=args.dry_run)
        sys.exit(0 if success else 1)
        
    except ValueError as e:
        parser.error(str(e))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
