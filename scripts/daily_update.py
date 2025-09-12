#!/usr/bin/env python3
"""
Daily Update Script for NBA Betting Model

This script performs daily data updates during the NBA season.
Run this script once per day, preferably in the evening after games complete.

Usage: 
    python scripts/daily_update.py
    python scripts/daily_update.py --force-season     # Force season mode
    python scripts/daily_update.py --force-offseason  # Force off-season mode
    python scripts/daily_update.py --notify           # Send notification
"""

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.automation_base import AutomationBase


class DailyUpdater(AutomationBase):
    """Daily update automation for NBA betting model."""
    
    def __init__(self, force_season: bool = False, force_offseason: bool = False):
        super().__init__("daily_update")
        self.force_season = force_season
        self.force_offseason = force_offseason
        
    def is_update_day_for_team_stats(self) -> bool:
        """Check if today is a team stats update day (Tuesday or Wednesday)."""
        return date.today().weekday() in [1, 2]  # 0=Monday, 1=Tuesday, 2=Wednesday
        
    def run_season_updates(self):
        """Run updates during NBA season."""
        self.logger.info("NBA season detected - running full updates")
        current_year = date.today().year
        
        # 1. Update betting odds (most important for daily betting)
        self.run_python_script(
            "src/ingest/ingest_odds.py",
            ["--regions", "us", "--bookmakers", "draftkings,fanduel"],
            "Updating betting odds"
        )
        
        # 2. Update recent games (completed games from yesterday)
        self.run_python_script(
            "src/ingest/ingest_games_new.py",
            [str(current_year), str(current_year)],
            "Updating recent games"
        )
        
        # 3. Update team stats (only on Tuesdays and Wednesdays)
        if self.is_update_day_for_team_stats():
            self.run_python_script(
                "src/ingest/ingest_team_stats.py",
                ["--seasons", str(current_year)],
                "Updating team statistics",
                allow_failure=True  # Allow this to fail without stopping pipeline
            )
        else:
            self.logger.info("Skipping team stats update (only runs Tuesday/Wednesday)")
            
        # 4. Rebuild features with updated data
        self.run_python_script(
            "src/features/build_features.py",
            ["--lookback", "10"],
            "Rebuilding features"
        )
        
        # 5. Retrain model if enough new data (placeholder - implement when model exists)
        # self.run_python_script(
        #     "src/models/train_model.py",
        #     [],
        #     "Retraining model"
        # )
        
    def run_offseason_updates(self):
        """Run minimal updates during off-season."""
        self.logger.info("Off-season detected - running maintenance updates only")
        current_year = date.today().year
        
        self.logger.info("Running off-season maintenance")
        
        # Update team stats for roster changes/trades
        self.run_python_script(
            "src/ingest/ingest_team_stats.py",
            ["--seasons", str(current_year)],
            "Updating team statistics (off-season)",
            allow_failure=True  # Allow this to fail without stopping pipeline
        )
        
        # Rebuild features
        self.run_python_script(
            "src/features/build_features.py",
            ["--lookback", "10"],
            "Rebuilding features (off-season)"
        )
        
    def run(self, send_notification: bool = False):
        """Execute the daily update process."""
        try:
            self.logger.info("Starting NBA Betting Model Daily Update")
            self.logger.info(f"Project root: {self.project_root}")
            self.logger.info(f"Log file: {self.log_file}")
            
            # Determine if we're in season
            if self.force_season:
                in_season = True
                self.logger.info("Forced season mode")
            elif self.force_offseason:
                in_season = False
                self.logger.info("Forced off-season mode")
            else:
                in_season = self.is_nba_season()
                
            # Run appropriate updates
            if in_season:
                self.run_season_updates()
            else:
                self.run_offseason_updates()
                
            # Generate daily report
            self.logger.info("Generating daily report...")
            quality_report = self.check_data_quality()
            
            # Log quality metrics
            if quality_report.get('feature_records') not in ["missing", "error"]:
                if isinstance(quality_report['feature_records'], int):
                    self.logger.info(f"Feature records: {quality_report['feature_records']:,}")
                else:
                    self.logger.info(f"Feature records: {quality_report['feature_records']}")
            if quality_report.get('latest_date') not in ["missing", "error"]:
                self.logger.info(f"Latest game date: {quality_report['latest_date']}")
                
            # Finalize with success
            self.finalize(success=True, send_notification=send_notification)
            return True
            
        except Exception as e:
            self.logger.error(f"Daily update failed: {str(e)}")
            self.logger.exception("Full error details:")
            self.finalize(success=False, send_notification=send_notification)
            return False


def main():
    """Main entry point for the daily update script."""
    parser = argparse.ArgumentParser(
        description="Daily update script for NBA betting model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/daily_update.py
    python scripts/daily_update.py --force-season
    python scripts/daily_update.py --notify
        """
    )
    
    parser.add_argument(
        "--force-season",
        action="store_true",
        help="Force season mode regardless of current date"
    )
    
    parser.add_argument(
        "--force-offseason", 
        action="store_true",
        help="Force off-season mode regardless of current date"
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
    
    # Validate arguments
    if args.force_season and args.force_offseason:
        parser.error("Cannot specify both --force-season and --force-offseason")
        
    # Run the daily update
    updater = DailyUpdater(
        force_season=args.force_season,
        force_offseason=args.force_offseason
    )
    
    # Set log level
    updater.logger.setLevel(args.log_level)
    for handler in updater.logger.handlers:
        handler.setLevel(args.log_level)
    
    success = updater.run(send_notification=args.notify)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
