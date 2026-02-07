"""
Base module for NBA Betting Model automation scripts.

This module provides common functionality for logging, error handling,
configuration, and utility functions used across all automation scripts.
"""

import logging
import subprocess
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Optional, List, Dict, Any
import yaml
import json


class AutomationBase:
    """Base class for automation scripts with common functionality."""
    
    def __init__(self, script_name: str, log_level: str = "INFO"):
        self.script_name = script_name
        self.project_root = Path(__file__).parent.parent.absolute()
        self.log_dir = self.project_root / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.log_file = self.log_dir / f"{script_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.setup_logging(log_level)
        
        # Load configuration
        self.config = self.load_config()
        
        # Determine Python command (poetry vs system)
        self.python_cmd = self.get_python_command()
        
    def setup_logging(self, level: str):
        """Setup logging to both file and console."""
        log_level = getattr(logging, level.upper())
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Setup file handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        
        # Setup console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        
        # Setup logger
        self.logger = logging.getLogger(self.script_name)
        self.logger.setLevel(log_level)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML files."""
        config = {}
        
        # Load main config files
        config_files = [
            self.project_root / "configs" / "paths.yml",
            self.project_root / "configs" / "model.yml"
        ]
        
        for config_file in config_files:
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    file_config = yaml.safe_load(f)
                    if file_config:
                        config.update(file_config)
        
        return config
        
    def get_python_command(self) -> List[str]:
        """Determine whether to use poetry or system Python."""
        try:
            result = subprocess.run(['poetry', '--version'], 
                                  capture_output=True, text=True, cwd=self.project_root)
            if result.returncode == 0:
                return ['poetry', 'run', 'python']
        except FileNotFoundError:
            pass
            
        return ['python']
        
    def run_command(self, description: str, command: List[str], 
                   check: bool = True, cwd: Optional[Path] = None, 
                   allow_failure: bool = False) -> subprocess.CompletedProcess:
        """Run a command with proper logging and error handling."""
        if cwd is None:
            cwd = self.project_root
            
        self.logger.debug(f"Command: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=check
            )
            
            self.logger.info(f"✓ {description}")
            if result.stdout.strip():
                self.logger.debug(f"Output: {result.stdout.strip()}")
            return result
            
        except subprocess.CalledProcessError as e:
            if allow_failure:
                self.logger.warning(f"Failed (continuing): {description}")
                self.logger.warning(f"Error code: {e.returncode}")
                if e.stdout:
                    self.logger.warning(f"Stdout: {e.stdout.strip()}")
                if e.stderr:
                    self.logger.warning(f"Stderr: {e.stderr.strip()}")
                return e  # Return the error as a result
            else:
                self.logger.error(f"Failed: {description}")
                self.logger.error(f"Error code: {e.returncode}")
                if e.stdout:
                    self.logger.error(f"Stdout: {e.stdout.strip()}")
                if e.stderr:
                    self.logger.error(f"Stderr: {e.stderr.strip()}")
                raise
            
    def run_python_script(self, script_path: str, args: List[str] = None, 
                         description: str = None, allow_failure: bool = False) -> subprocess.CompletedProcess:
        """Run a Python script with the configured Python command."""
        if args is None:
            args = []
            
        if description is None:
            description = f"Running {script_path}"
            
        command = self.python_cmd + [script_path] + args
        return self.run_command(description, command, allow_failure=allow_failure)
        
    def is_nba_season(self) -> bool:
        """Check if we're currently in NBA season (October through June)."""
        current_month = date.today().month
        return current_month >= 10 or current_month <= 6
        
    def get_current_season_years(self) -> tuple[int, int]:
        """Get the current and previous season years."""
        current_year = date.today().year
        
        # If it's January-June, we're in the same season that started last year
        if date.today().month <= 6:
            return current_year - 1, current_year
        else:
            return current_year, current_year + 1
            
    def check_data_quality(self) -> Dict[str, Any]:
        """Run basic data quality checks and return results."""
        quality_report = {}
        
        # Check games data
        games_file = self.project_root / "data" / "processed" / "games_2020_2023.csv"
        if games_file.exists():
            try:
                with open(games_file, 'r') as f:
                    line_count = sum(1 for _ in f) - 1  # Subtract header
                quality_report['games_records'] = line_count
                self.logger.info(f"Games file: {line_count:,} records")
            except Exception as e:
                self.logger.warning(f"Could not read games file: {e}")
                quality_report['games_records'] = "error"
        else:
            quality_report['games_records'] = "missing"
            
        # Check features data
        features_file = self.project_root / "data" / "processed" / "nba_features.parquet"
        if features_file.exists():
            try:
                # Try to get record count using polars
                cmd = self.python_cmd + ["-c", 
                    "import polars as pl; "
                    f"df = pl.read_parquet('{features_file}'); "
                    f"print(len(df), df['game_date'].max(), sep=',')"]
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
                
                if result.returncode == 0:
                    parts = result.stdout.strip().split(',')
                    if len(parts) >= 2:
                        count, latest_date = parts[0], parts[1]
                        quality_report['feature_records'] = int(count)
                        quality_report['latest_date'] = latest_date
                        self.logger.info(f"Feature records: {count}")
                        self.logger.info(f"Latest game date: {latest_date}")
                    else:
                        quality_report['feature_records'] = "parse_error"
                        quality_report['latest_date'] = "parse_error"
                else:
                    quality_report['feature_records'] = "error"
                    quality_report['latest_date'] = "error"
            except Exception as e:
                self.logger.warning(f"Could not read features file: {e}")
                quality_report['feature_records'] = "error"
                quality_report['latest_date'] = "error"
        else:
            quality_report['feature_records'] = "missing"
            quality_report['latest_date'] = "missing"
            
        # Check team stats
        current_year = date.today().year
        team_stats_file = self.project_root / "data" / "raw" / f"team_stats_{current_year}.csv"
        if team_stats_file.exists():
            try:
                with open(team_stats_file, 'r') as f:
                    line_count = sum(1 for _ in f) - 1  # Subtract header
                quality_report['team_stats_records'] = line_count
                self.logger.info(f"Team stats: {line_count} teams")
            except Exception as e:
                self.logger.warning(f"Could not read team stats file: {e}")
                quality_report['team_stats_records'] = "error"
        else:
            quality_report['team_stats_records'] = "missing"
            
        return quality_report
        
    def cleanup_old_logs(self, days: int = 30):
        """Clean up log files older than specified days."""
        cutoff_time = datetime.now().timestamp() - (days * 24 * 60 * 60)
        
        cleaned_count = 0
        for log_file in self.log_dir.glob("*.log"):
            if log_file.stat().st_mtime < cutoff_time:
                try:
                    log_file.unlink()
                    cleaned_count += 1
                except Exception as e:
                    self.logger.warning(f"Could not delete old log file {log_file}: {e}")
                    
        if cleaned_count > 0:
            self.logger.info(f"Cleaned {cleaned_count} old logs")
            
    def send_notification(self, message: str, webhook_url: Optional[str] = None):
        """Send notification via webhook (Slack, Discord, etc.)."""
        if not webhook_url:
            return
            
        try:
            import requests
            payload = {"text": f"NBA Model: {message}"}
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            self.logger.info("Notification sent")
        except Exception as e:
            self.logger.warning(f"Failed to send notification: {e}")
            
    def finalize(self, success: bool = True, send_notification: bool = False):
        """Finalize the script execution with cleanup and reporting."""
        status = "completed" if success else "failed"
        self.logger.info(f"{self.script_name}: {status}")
        
        # Cleanup old logs
        self.cleanup_old_logs()
        
        # Send notification if requested
        if send_notification:
            quality_report = self.check_data_quality()
            message = f"NBA Model {self.script_name} {status}. "
            if 'feature_records' in quality_report:
                message += f"Records: {quality_report['feature_records']:,}, "
                message += f"Latest: {quality_report['latest_date']}"
            
            # You can set this via environment variable
            import os
            webhook_url = os.getenv('SLACK_WEBHOOK_URL') or os.getenv('DISCORD_WEBHOOK_URL')
            self.send_notification(message, webhook_url)
