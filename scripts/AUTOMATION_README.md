# Python Automation Scripts

This directory contains comprehensive Python-based automation scripts for the NBA betting model. All shell scripts have been replaced with cross-platform Python implementations.

## Available Scripts

### Core Automation Scripts

#### `daily_update.py`
- **Purpose**: Daily data updates during NBA season
- **Features**: Season-aware updates, selective team stats updates
- **Usage**: `python scripts/daily_update.py [--force-season] [--notify]`

#### `weekly_refresh.py` 
- **Purpose**: Comprehensive weekly data refresh
- **Features**: Full data refresh, backup creation, quality checks
- **Usage**: `python scripts/weekly_refresh.py [--backup] [--notify]`

#### `manual_update.py`
- **Purpose**: Flexible manual data updates
- **Features**: Selective updates, dry-run capability, custom date ranges
- **Usage**: `python scripts/manual_update.py [--odds-only] [--games-only] [--dry-run]`

#### `scheduler.py`
- **Purpose**: Python-based task scheduler (replaces cron)
- **Features**: YAML configuration, cross-platform, status monitoring
- **Usage**: `python scripts/scheduler.py [--daemon] [--status]`

### Enhanced Prediction & Betting Scripts

#### `daily_predictions.py`
- **Purpose**: Simple daily predictions with betting recommendations
- **Features**: Clean CLI interface, predictions-only mode, custom bankroll
- **Usage**: `python scripts/daily_predictions.py [--bankroll 10000] [--predictions-only]`

#### `daily_betting_pipeline.py` ⭐ **RECOMMENDED**
- **Purpose**: Complete production-ready daily pipeline
- **Features**: Advanced monitoring, automatic retraining, comprehensive logging
- **Usage**: `python scripts/daily_betting_pipeline.py [--bankroll 15000] [--retrain] [--dry-run]`

### Utility Scripts

#### `cleanup.py`
- **Purpose**: Clean up temporary files and old artifacts
- **Features**: Configurable cleanup rules, dry-run mode
- **Usage**: `python scripts/cleanup.py [--dry-run] [--deep]`

#### `automation_base.py`
- **Purpose**: Shared functionality base class
- **Features**: Common utilities, error handling, notifications
- **Note**: Not run directly, used by other scripts

## Key Features

### ✅ **Cross-Platform Compatibility**
- All scripts work on Windows, macOS, and Linux
- No shell script dependencies
- Python virtual environment support

### ✅ **Advanced Error Handling**
- Try/catch blocks with retry logic
- Detailed error logging and reporting
- Graceful failure handling

### ✅ **Professional Logging**
- Configurable log levels (DEBUG, INFO, WARNING, ERROR)
- File and console output
- Timestamped entries with execution context

### ✅ **Configuration Management**
- YAML-based configuration files
- Environment variable support
- Command-line argument overrides

### ✅ **Production Ready**
- Comprehensive argument validation
- Status monitoring and health checks
- Performance monitoring integration

## Migration from Shell Scripts

**COMPLETED**: All shell scripts have been removed and replaced with Python equivalents:

- ❌ `daily_update.sh` → ✅ `daily_update.py`
- ❌ `weekly_refresh.sh` → ✅ `weekly_refresh.py`  
- ❌ `manual_update.sh` → ✅ `manual_update.py`

## Recommended Usage

### Daily Operations
```bash
# Simple predictions (beginners)
python scripts/daily_predictions.py --bankroll 10000

# Complete pipeline (production)
python scripts/daily_betting_pipeline.py --bankroll 15000
```

### Weekly Maintenance
```bash
# Full refresh with backup
python scripts/weekly_refresh.py --backup --notify
```

### Setup Scheduler (Optional)
```bash
# Start Python scheduler (replaces cron)
python scripts/scheduler.py --daemon
```

## Dependencies

All scripts require the project dependencies installed via:
```bash
poetry install
# or
pip install -r requirements.txt
```

Optional advanced ML libraries:
```bash
pip install xgboost lightgbm
```

## Support

For issues or questions about the automation scripts, check:
1. Script help: `python scripts/<script_name>.py --help`
2. Logs directory: `logs/`
3. Configuration: `configs/model.yml`
4. Documentation: `README.md`
