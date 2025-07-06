#!/usr/bin/env python3
"""
Scheduler Script for NBA Betting Model

This script provides Python-based scheduling functionality as an alternative to cron.
It can run continuously and handle multiple scheduled tasks with built-in retry logic.

Usage: 
    python scripts/scheduler.py --config configs/schedule.yml
    python scripts/scheduler.py --daemon                     # Run as background daemon
    python scripts/scheduler.py --run-once daily            # Run single task and exit
"""

import argparse
import sys
import time
import signal
import threading
from datetime import datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml
import subprocess

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.automation_base import AutomationBase


class ScheduledTask:
    """Represents a scheduled task with timing and retry logic."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.script = config['script']
        self.schedule = config.get('schedule', {})
        self.enabled = config.get('enabled', True)
        self.max_retries = config.get('max_retries', 2)
        self.retry_delay = config.get('retry_delay', 300)  # 5 minutes
        self.timeout = config.get('timeout', 3600)  # 1 hour
        self.args = config.get('args', [])
        
        # Parse schedule
        self.days = self.schedule.get('days', [])  # 0=Monday, 6=Sunday
        self.time = self.schedule.get('time', '00:00')  # HH:MM format
        self.interval_hours = self.schedule.get('interval_hours')  # For interval-based
        
        # State tracking
        self.last_run = None
        self.next_run = None
        self.is_running = False
        self.consecutive_failures = 0
        
        self._calculate_next_run()
        
    def _calculate_next_run(self):
        """Calculate the next run time based on schedule."""
        now = datetime.now()
        
        if self.interval_hours:
            # Interval-based scheduling
            if self.last_run:
                self.next_run = self.last_run + timedelta(hours=self.interval_hours)
            else:
                self.next_run = now
        else:
            # Day/time based scheduling
            hour, minute = map(int, self.time.split(':'))
            target_time = dt_time(hour, minute)
            
            # Find next occurrence
            for days_ahead in range(8):  # Check next 7 days
                candidate_date = now.date() + timedelta(days=days_ahead)
                candidate_datetime = datetime.combine(candidate_date, target_time)
                
                # Skip if in the past today
                if candidate_datetime <= now:
                    continue
                    
                # Check if this day is in our schedule
                if not self.days or candidate_date.weekday() in self.days:
                    self.next_run = candidate_datetime
                    break
                    
    def should_run(self) -> bool:
        """Check if this task should run now."""
        if not self.enabled or self.is_running:
            return False
            
        now = datetime.now()
        return self.next_run and now >= self.next_run
        
    def run(self, logger) -> bool:
        """Execute the task and return success status."""
        if self.is_running:
            logger.warning(f"Task {self.name} is already running")
            return False
            
        self.is_running = True
        success = False
        
        try:
            logger.info(f"Starting scheduled task: {self.name}")
            
            # Build command
            command = ['python', self.script] + self.args
            
            # Execute with timeout
            result = subprocess.run(
                command,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode == 0:
                logger.info(f"Task {self.name} completed successfully")
                self.consecutive_failures = 0
                success = True
            else:
                logger.error(f"Task {self.name} failed with code {result.returncode}")
                if result.stderr:
                    logger.error(f"Error output: {result.stderr.strip()}")
                self.consecutive_failures += 1
                
        except subprocess.TimeoutExpired:
            logger.error(f"Task {self.name} timed out after {self.timeout} seconds")
            self.consecutive_failures += 1
        except Exception as e:
            logger.error(f"Task {self.name} failed with exception: {str(e)}")
            self.consecutive_failures += 1
        finally:
            self.is_running = False
            self.last_run = datetime.now()
            self._calculate_next_run()
            
        return success


class TaskScheduler(AutomationBase):
    """Main scheduler class for managing and executing scheduled tasks."""
    
    def __init__(self, config_file: Optional[Path] = None):
        super().__init__("scheduler")
        self.tasks: Dict[str, ScheduledTask] = {}
        self.running = False
        self.check_interval = 60  # Check every minute
        
        # Load configuration
        self.config_file = config_file or self.project_root / "configs" / "schedule.yml"
        self.load_schedule_config()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def load_schedule_config(self):
        """Load scheduled tasks from configuration file."""
        if not self.config_file.exists():
            self.logger.warning(f"Schedule config not found: {self.config_file}")
            self.create_default_config()
            
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
                
            tasks_config = config.get('tasks', {})
            for task_name, task_config in tasks_config.items():
                self.tasks[task_name] = ScheduledTask(task_name, task_config)
                self.logger.info(f"Loaded task: {task_name}")
                
        except Exception as e:
            self.logger.error(f"Failed to load schedule config: {e}")
            raise
            
    def create_default_config(self):
        """Create a default schedule configuration file."""
        default_config = {
            'tasks': {
                'daily_update': {
                    'script': 'scripts/daily_update.py',
                    'schedule': {
                        'time': '02:00',  # 2 AM
                        'days': [0, 1, 2, 3, 4, 5, 6]  # Every day
                    },
                    'enabled': True,
                    'max_retries': 2,
                    'retry_delay': 300,
                    'timeout': 3600,
                    'args': ['--notify']
                },
                'weekly_refresh': {
                    'script': 'scripts/weekly_refresh.py',
                    'schedule': {
                        'time': '01:00',  # 1 AM
                        'days': [0]  # Monday
                    },
                    'enabled': True,
                    'max_retries': 1,
                    'retry_delay': 600,
                    'timeout': 7200,
                    'args': ['--backup', '--notify']
                }
            },
            'global': {
                'log_retention_days': 30,
                'max_concurrent_tasks': 2,
                'notification_webhook': '${SLACK_WEBHOOK_URL}'
            }
        }
        
        self.config_file.parent.mkdir(exist_ok=True)
        with open(self.config_file, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
            
        self.logger.info(f"Created default schedule config: {self.config_file}")
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        
    def run_single_task(self, task_name: str) -> bool:
        """Run a single task by name and exit."""
        if task_name not in self.tasks:
            self.logger.error(f"Task not found: {task_name}")
            return False
            
        task = self.tasks[task_name]
        return task.run(self.logger)
        
    def get_status(self) -> Dict[str, Any]:
        """Get current status of all tasks."""
        status = {
            'scheduler_running': self.running,
            'total_tasks': len(self.tasks),
            'enabled_tasks': sum(1 for t in self.tasks.values() if t.enabled),
            'tasks': {}
        }
        
        for name, task in self.tasks.items():
            status['tasks'][name] = {
                'enabled': task.enabled,
                'is_running': task.is_running,
                'last_run': task.last_run.isoformat() if task.last_run else None,
                'next_run': task.next_run.isoformat() if task.next_run else None,
                'consecutive_failures': task.consecutive_failures
            }
            
        return status
        
    def run_daemon(self):
        """Run the scheduler as a daemon process."""
        self.logger.info("Starting NBA Betting Model Scheduler")
        self.logger.info(f"Loaded {len(self.tasks)} tasks")
        self.logger.info(f"Check interval: {self.check_interval} seconds")
        
        self.running = True
        
        # Show next run times
        for name, task in self.tasks.items():
            if task.enabled and task.next_run:
                self.logger.info(f"{name}: next run at {task.next_run}")
                
        while self.running:
            try:
                # Check each task
                for name, task in self.tasks.items():
                    if task.should_run():
                        # Run task in thread to avoid blocking
                        thread = threading.Thread(
                            target=task.run,
                            args=(self.logger,),
                            name=f"task_{name}"
                        )
                        thread.start()
                        
                # Sleep until next check
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"Scheduler error: {e}")
                time.sleep(self.check_interval)
                
        self.logger.info("Scheduler stopped")


def main():
    """Main entry point for the scheduler script."""
    parser = argparse.ArgumentParser(
        description="Task scheduler for NBA betting model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/scheduler.py                        # Run with default config
    python scripts/scheduler.py --daemon               # Run as background daemon
    python scripts/scheduler.py --run-once daily_update # Run single task
    python scripts/scheduler.py --status               # Show status and exit
    python scripts/scheduler.py --config custom.yml    # Use custom config
        """
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to schedule configuration file"
    )
    
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as background daemon"
    )
    
    parser.add_argument(
        "--run-once",
        metavar="TASK_NAME",
        help="Run a single task and exit"
    )
    
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show scheduler status and exit"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    try:
        # Create scheduler
        scheduler = TaskScheduler(config_file=args.config)
        
        # Set log level
        scheduler.logger.setLevel(args.log_level)
        for handler in scheduler.logger.handlers:
            handler.setLevel(args.log_level)
            
        # Handle different modes
        if args.status:
            status = scheduler.get_status()
            print("\nScheduler Status:")
            print(f"  Running: {status['scheduler_running']}")
            print(f"  Total tasks: {status['total_tasks']}")
            print(f"  Enabled tasks: {status['enabled_tasks']}")
            print("\nTask Details:")
            for name, task_status in status['tasks'].items():
                print(f"  {name}:")
                print(f"    Enabled: {task_status['enabled']}")
                print(f"    Running: {task_status['is_running']}")
                print(f"    Last run: {task_status['last_run'] or 'Never'}")
                print(f"    Next run: {task_status['next_run'] or 'Not scheduled'}")
                print(f"    Failures: {task_status['consecutive_failures']}")
            sys.exit(0)
            
        elif args.run_once:
            success = scheduler.run_single_task(args.run_once)
            sys.exit(0 if success else 1)
            
        else:
            # Run daemon
            scheduler.run_daemon()
            sys.exit(0)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
