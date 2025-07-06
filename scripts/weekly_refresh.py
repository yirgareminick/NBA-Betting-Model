#!/usr/bin/env python3
"""
Weekly Refresh Script for NBA Betting Model

This script performs a comprehensive weekly refresh of all data sources.
Run this script once per week, preferably on Monday morning.

Usage: 
    python scripts/weekly_refresh.py
    python scripts/weekly_refresh.py --backup        # Create backup before refresh
    python scripts/weekly_refresh.py --notify        # Send notification
    python scripts/weekly_refresh.py --quick         # Skip quality checks
"""

import argparse
import sys
import shutil
from datetime import date, datetime
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.automation_base import AutomationBase


class WeeklyRefresher(AutomationBase):
    """Weekly refresh automation for NBA betting model."""
    
    def __init__(self, create_backup: bool = False, quick_mode: bool = False):
        super().__init__("weekly_refresh")
        self.create_backup = create_backup
        self.quick_mode = quick_mode
        
    def create_data_backup(self):
        """Create a backup of current data before refresh."""
        if not self.create_backup:
            return
            
        self.logger.info("Creating data backup...")
        
        backup_dir = self.project_root / "backups" / f"weekly_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup processed data
        processed_dir = self.project_root / "data" / "processed"
        if processed_dir.exists():
            shutil.copytree(processed_dir, backup_dir / "processed", dirs_exist_ok=True)
            
        # Backup raw data
        raw_dir = self.project_root / "data" / "raw"
        if raw_dir.exists():
            shutil.copytree(raw_dir, backup_dir / "raw", dirs_exist_ok=True)
            
        self.logger.info(f"Backup created at: {backup_dir}")
        
    def run_comprehensive_refresh(self):
        """Run a comprehensive refresh of all data sources."""
        current_year = date.today().year
        previous_year = current_year - 1
        
        # 1. Full games data refresh (current and previous season)
        self.run_python_script(
            "src/ingest/ingest_games_new.py",
            [str(previous_year), str(current_year)],
            "Refreshing games data (full)"
        )
        
        # 2. Update team statistics for current seasons
        self.run_python_script(
            "src/ingest/ingest_team_stats.py",
            ["--seasons", str(previous_year), str(current_year)],
            "Refreshing team statistics"
        )
        
        # 3. Update betting odds with more bookmakers
        self.run_python_script(
            "src/ingest/ingest_odds.py",
            ["--regions", "us", "--bookmakers", "draftkings,fanduel,bovada"],
            "Refreshing betting odds"
        )
        
        # 4. Rebuild features from scratch
        self.run_python_script(
            "src/features/build_features.py",
            ["--lookback", "10"],
            "Rebuilding all features"
        )
        
    def run_quality_checks(self) -> dict:
        """Run comprehensive data quality checks."""
        if self.quick_mode:
            self.logger.info("Quick mode: Skipping detailed quality checks")
            return self.check_data_quality()
            
        self.logger.info("Running comprehensive data quality checks...")
        
        quality_report = self.check_data_quality()
        current_year = date.today().year
        previous_year = current_year - 1
        
        # Additional quality checks for weekly refresh
        
        # Check games file with expected naming
        games_file = self.project_root / "data" / "processed" / f"games_{previous_year}_{current_year}.csv"
        if games_file.exists():
            try:
                with open(games_file, 'r') as f:
                    line_count = sum(1 for _ in f) - 1
                quality_report['games_full_records'] = line_count
                self.logger.info(f"Games file (full): {line_count:,} records")
            except Exception as e:
                self.logger.warning(f"Could not read full games file: {e}")
                quality_report['games_full_records'] = "error"
        else:
            quality_report['games_full_records'] = "missing"
            
        # Check team stats for both years
        for year in [previous_year, current_year]:
            team_stats_file = self.project_root / "data" / "raw" / f"team_stats_{year}.csv"
            if team_stats_file.exists():
                try:
                    with open(team_stats_file, 'r') as f:
                        line_count = sum(1 for _ in f) - 1
                    quality_report[f'team_stats_{year}'] = line_count
                    self.logger.info(f"Team stats ({year}): {line_count} teams")
                except Exception as e:
                    self.logger.warning(f"Could not read team stats {year}: {e}")
                    quality_report[f'team_stats_{year}'] = "error"
            else:
                quality_report[f'team_stats_{year}'] = "missing"
                
        # Check odds data freshness
        odds_file = self.project_root / "data" / "raw" / "odds_basketball_nba_us.csv"
        if odds_file.exists():
            try:
                # Get file modification time
                mod_time = datetime.fromtimestamp(odds_file.stat().st_mtime)
                hours_old = (datetime.now() - mod_time).total_seconds() / 3600
                quality_report['odds_age_hours'] = round(hours_old, 1)
                self.logger.info(f"Odds data is {hours_old:.1f} hours old")
                
                if hours_old > 24:
                    self.logger.warning("WARNING: Odds data is more than 24 hours old")
                    
            except Exception as e:
                self.logger.warning(f"Could not check odds file age: {e}")
                quality_report['odds_age_hours'] = "error"
                
        return quality_report
        
    def cleanup_old_backups(self, days: int = 14):
        """Clean up old backup directories."""
        backup_root = self.project_root / "backups"
        if not backup_root.exists():
            return
            
        cutoff_time = datetime.now().timestamp() - (days * 24 * 60 * 60)
        cleaned_count = 0
        
        for backup_dir in backup_root.iterdir():
            if backup_dir.is_dir() and backup_dir.stat().st_mtime < cutoff_time:
                try:
                    shutil.rmtree(backup_dir)
                    cleaned_count += 1
                except Exception as e:
                    self.logger.warning(f"Could not delete old backup {backup_dir}: {e}")
                    
        if cleaned_count > 0:
            self.logger.info(f"🧹 Cleaned up {cleaned_count} old backups")
            
    def run(self, send_notification: bool = False):
        """Execute the weekly refresh process."""
        try:
            self.logger.info("Starting NBA Betting Model Weekly Refresh")
            self.logger.info(f"Project root: {self.project_root}")
            self.logger.info(f"Log file: {self.log_file}")
            
            # Create backup if requested
            self.create_data_backup()
            
            # Run comprehensive refresh
            self.run_comprehensive_refresh()
            
            # Run quality checks
            quality_report = self.run_quality_checks()
            
            # Log summary statistics
            self.logger.info("Weekly Refresh Summary:")
            for key, value in quality_report.items():
                if isinstance(value, (int, float)):
                    if 'records' in key:
                        self.logger.info(f"  {key}: {value:,}")
                    else:
                        self.logger.info(f"  {key}: {value}")
                else:
                    self.logger.info(f"  {key}: {value}")
                    
            # Check for data quality issues
            issues = []
            if quality_report.get('feature_records') == "missing":
                issues.append("Missing feature data")
            if quality_report.get('games_full_records') == "missing":
                issues.append("Missing games data")
            if quality_report.get('odds_age_hours', 0) > 48:
                issues.append("Stale odds data")
                
            if issues:
                self.logger.warning(f"Quality issues detected: {', '.join(issues)}")
            else:
                self.logger.info("All quality checks passed")
                
            # Cleanup old backups
            self.cleanup_old_backups()
            
            # Finalize with success
            self.finalize(success=True, send_notification=send_notification)
            return True
            
        except Exception as e:
            self.logger.error(f"Weekly refresh failed: {str(e)}")
            self.logger.exception("Full error details:")
            self.finalize(success=False, send_notification=send_notification)
            return False


def main():
    """Main entry point for the weekly refresh script."""
    parser = argparse.ArgumentParser(
        description="Weekly refresh script for NBA betting model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/weekly_refresh.py
    python scripts/weekly_refresh.py --backup --notify
    python scripts/weekly_refresh.py --quick
        """
    )
    
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup of data before refresh"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: skip detailed quality checks"
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
    
    # Run the weekly refresh
    refresher = WeeklyRefresher(
        create_backup=args.backup,
        quick_mode=args.quick
    )
    
    # Set log level
    refresher.logger.setLevel(args.log_level)
    for handler in refresher.logger.handlers:
        handler.setLevel(args.log_level)
    
    success = refresher.run(send_notification=args.notify)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
