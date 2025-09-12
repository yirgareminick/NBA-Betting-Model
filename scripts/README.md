# NBA Betting Model - Automation Scripts

This directory contains Python-based automation scripts for the NBA betting model data pipeline. These production-ready scripts provide robust, cross-platform automation for data ingestion, feature engineering, and pipeline orchestration.

## Quick Start

```bash
# Run daily update
python scripts/daily_update.py

# Run weekly refresh with backup
python scripts/weekly_refresh.py --backup --notify

# Manual update for specific years
python scripts/manual_update.py --years 2023 2024

# Start the scheduler daemon
python scripts/scheduler.py --daemon
```

## Script Overview

### Core Scripts

| Script | Purpose | Frequency | Key Features |
|--------|---------|-----------|--------------|
| `daily_update.py` | Daily data updates during NBA season | Daily | Season-aware, odds priority, team stats on Tue/Wed |
| `weekly_refresh.py` | Comprehensive weekly data refresh | Weekly | Full refresh, quality checks, backup support |
| `manual_update.py` | On-demand updates with flexible options | As needed | Selective updates, custom date ranges, dry-run |
| `scheduler.py` | Python-based task scheduler | Continuous | Replaces cron, retry logic, monitoring |

### Supporting Modules

- `automation_base.py` - Base class with common functionality (logging, error handling, utilities)
- `configs/schedule.yml` - Scheduler configuration file

## Key Features

- **Cross-Platform**: Works seamlessly on Windows, macOS, and Linux
- **Robust Error Handling**: Advanced exception handling with retry logic and detailed logging
- **Smart Scheduling**: Built-in scheduler with YAML configuration (no cron dependency)
- **Season Awareness**: Automatically adjusts behavior based on NBA calendar
- **Quality Monitoring**: Real-time data quality checks and validation
- **Flexible Configuration**: YAML-based configuration with environment variable support
- **Rich CLI**: Comprehensive command-line interface with help and validation
- **Testing Support**: Dry-run mode and unit testable components
- **Notification Integration**: Slack/Discord webhook support for monitoring
- **Automatic Cleanup**: Log rotation and backup management

## Detailed Usage

### Daily Update Script

```bash
# Basic daily update (auto-detects season)
python scripts/daily_update.py

# Force season mode (useful for testing)
python scripts/daily_update.py --force-season

# Force off-season mode
python scripts/daily_update.py --force-offseason

# Send notification on completion
python scripts/daily_update.py --notify

# Debug mode with verbose logging
python scripts/daily_update.py --log-level DEBUG
```

**Season Logic:**
- **In-Season (Oct-Jun)**: Updates odds, recent games, team stats (Tue/Wed), features
- **Off-Season (Jul-Sep)**: Minimal updates for roster changes and maintenance

### Weekly Refresh Script

```bash
# Basic weekly refresh
python scripts/weekly_refresh.py

# Create backup before refresh
python scripts/weekly_refresh.py --backup

# Quick mode (skip detailed quality checks)
python scripts/weekly_refresh.py --quick

# Full refresh with backup and notifications
python scripts/weekly_refresh.py --backup --notify
```

**Features:**
- Full data refresh for current and previous seasons
- Comprehensive quality checks
- Optional backup creation
- Old backup cleanup (14-day retention)

### Manual Update Script

```bash
# Update current season
python scripts/manual_update.py

# Update specific years
python scripts/manual_update.py --years 2023 2024

# Update only odds
python scripts/manual_update.py --odds-only

# Update only games for 2024
python scripts/manual_update.py --games-only --years 2024

# Rebuild features only
python scripts/manual_update.py --features-only

# Custom bookmakers
python scripts/manual_update.py --bookmakers draftkings fanduel bovada

# Dry run (show what would be done)
python scripts/manual_update.py --dry-run

# Custom lookback period for features
python scripts/manual_update.py --lookback 14
```

### Scheduler Script

```bash
# Start scheduler daemon
python scripts/scheduler.py --daemon

# Show current status
python scripts/scheduler.py --status

# Run single task
python scripts/scheduler.py --run-once daily_update

# Use custom config
python scripts/scheduler.py --config configs/custom_schedule.yml
```

**Scheduler Features:**
- Replaces cron with Python-based scheduling
- YAML configuration for easy management
- Built-in retry logic and timeout handling
- Concurrent task execution with limits
- Real-time status monitoring
- Graceful shutdown handling

## Configuration

### Environment Variables

```bash
# Required for data ingestion
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
export ODDS_API_KEY="your_odds_api_key"

# Optional for notifications
export SLACK_WEBHOOK_URL="https://hooks.slack.com/..."
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
```

### Schedule Configuration

Edit `configs/schedule.yml` to customize task scheduling:

```yaml
tasks:
  daily_update:
    script: "scripts/daily_update.py"
    schedule:
      time: "02:00"  # 2:00 AM
      days: [0, 1, 2, 3, 4, 5, 6]  # Every day
    enabled: true
    args: ["--notify"]
```

## Monitoring and Logging

### Log Files

All scripts create detailed log files in the `logs/` directory:

```
logs/
├── daily_update_20240706_140530.log
├── weekly_refresh_20240701_010015.log
├── manual_update_20240705_093022.log
└── scheduler_20240701_000001.log
```

### Log Features

- **Timestamped entries** with severity levels
- **Both file and console** output
- **Structured format** for easy parsing
- **Automatic cleanup** of old logs (30-day retention)
- **Debug mode** for troubleshooting

### Quality Metrics

Scripts automatically track and report:

- Feature record counts
- Latest game dates
- Data file sizes and ages
- Success/failure rates
- Execution times

##  Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure you're in the project root
   cd /path/to/NBA-Betting-Model
   python scripts/daily_update.py
   ```

2. **Permission Errors**
   ```bash
   # Make scripts executable
   chmod +x scripts/*.py
   ```

3. **Missing Dependencies**
   ```bash
   # Install required packages
   poetry install
   # or
   pip install -r requirements.txt
   ```

4. **Environment Variables**
   ```bash
   # Check environment variables
   python -c "import os; print(os.getenv('KAGGLE_USERNAME'))"
   ```

### Debug Mode

Use debug logging for troubleshooting:

```bash
python scripts/daily_update.py --log-level DEBUG
```

### Testing

Test scripts without making changes:

```bash
# Dry run for manual updates
python scripts/manual_update.py --dry-run

# Test scheduler status
python scripts/scheduler.py --status
```

## Future Enhancements

The automation system is designed for extensibility:

- **Database Integration**: Direct database connections for better data management
- **Machine Learning Pipeline**: Integration with model training and prediction
- **Advanced Monitoring**: Prometheus metrics, Grafana dashboards
- **Workflow Orchestration**: Integration with Airflow or Prefect
- **Cloud Deployment**: Docker containers, Kubernetes scheduling
- **API Integration**: REST API for remote control and monitoring

## Support

For issues with the automation scripts:

1. Check the log files in `logs/` directory
2. Run with `--log-level DEBUG` for detailed output
3. Verify environment variables are set correctly
4. Ensure all dependencies are installed

The automation system provides a robust, maintainable foundation for your NBA betting model data pipeline.
