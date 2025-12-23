NBA Moneyline Betting Pipeline
==============================

![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)
![Machine Learning](https://img.shields.io/badge/ML-Machine%20Learning-red.svg)
![GitHub last commit](https://img.shields.io/github/last-commit/yirgareminick/NBA-Betting-Model)

**A comprehensive machine learning pipeline designed to predict NBA moneyline outcomes, size bets using the Kelly criterion, and generate daily betting insights with advanced performance monitoring.**

This project processes historical NBA games (2020-2025), team statistics, and real-time betting odds to model and recommend daily bets. The pipeline includes automated data ingestion, feature engineering, advanced model training with ensemble methods, comprehensive performance tracking, and production-ready scheduling capabilities.

**Project Structure**
------------------------------

```
NBA-Betting-Model/
├── configs/                      # Configuration files
│   ├── model.yml                 # Model training parameters
│   ├── paths.yml                 # Data path configurations
│   └── schedule.yml              # Automation scheduling
├── data/                         # Data storage
│   ├── predictions/              # Daily prediction outputs
│   ├── processed/                # Feature-engineered datasets
│   ├── raw/                      # Raw data (SQLite, CSV, odds)
│   └── performance.db            # Performance tracking database
├── docs/                         # Documentation
├── logs/                         # Execution logs
├── models/                       # Trained ML models
├── notebooks/                    # Jupyter analysis notebooks
├── reports/                      # Generated reports
│   └── performance/              # Performance analytics
├── scripts/                      # Automation scripts
│   ├── daily_betting_pipeline.py # Enhanced pipeline with monitoring
│   ├── daily_update.py           # Data refresh automation
│   ├── weekly_refresh.py         # Weekly maintenance
│   └── scheduler.py              # Python-based task scheduler
├── src/                          # Source code
│   ├── features/                 # Feature engineering
│   │   └── build_features.py     # NBA feature pipeline
│   ├── ingest/                   # Data ingestion
│   │   ├── ingest_games_new.py   # Game data from Kaggle
│   │   ├── ingest_odds.py        # Betting odds from APIs
│   │   ├── ingest_team_stats.py  # Team statistics
│   │   └── live_data_fetcher.py  # Real-time NBA data
│   ├── models/                   # ML training and tracking
│   │   ├── train_model.py        # Basic model training
│   │   ├── advanced_trainer.py   # Ensemble methods
│   │   └── performance_tracker.py # Model monitoring
│   ├── predict/                  # Prediction system
│   │   ├── predict_games.py      # Game prediction engine
│   │   └── daily_report.py       # Report generation
│   ├── stake/                    # Betting strategy
│   │   └── kelly_criterion.py    # Optimal bet sizing
│   └── pipeline.py               # Prefect workflow orchestration
└── tests/                        # Test suite
    ├── unit/                     # Unit tests
    ├── integration/              # Integration tests
    └── notebooks/                # Notebook validation tests
```

**Enhanced Features**
------------------------------

### **Advanced Machine Learning**
- **Multiple Algorithms**: Random Forest, XGBoost, LightGBM, Neural Networks
- **Ensemble Methods**: Voting classifiers combining multiple models
- **Hyperparameter Tuning**: Automated optimization for best performance
- **Model Selection**: Automatic selection of best-performing algorithm

### **Comprehensive Prediction System**
- **Daily Game Predictions**: Automated prediction generation with confidence scores
- **Edge Calculation**: Sophisticated comparison of model probabilities vs market odds
- **Risk Assessment**: Monte Carlo simulations for betting outcome analysis
- **Multi-format Output**: CSV, JSON, and HTML reports

### 💰 **Advanced Betting Strategy**
- **Kelly Criterion Implementation**: Optimal bet sizing with fractional Kelly
- **Bankroll Management**: Configurable maximum bet sizes and risk limits
- **Portfolio Approach**: Simultaneous evaluation of multiple betting opportunities
- **Expected Value Optimization**: Focus on positive EV bets with minimum edge thresholds

### 📈 **Performance Monitoring**
- **Real-time Tracking**: SQLite database storing all predictions and outcomes
- **Accuracy Monitoring**: Daily, weekly, and monthly performance metrics
- **Model Drift Detection**: Automatic identification of performance degradation
- **ROI Analysis**: Comprehensive return on investment tracking and reporting

### **Production-Ready Automation**
- **Enhanced Daily Pipeline**: Complete automation from data ingestion to bet recommendations
- **Smart Retraining**: Automatic model retraining when performance degrades
- **Comprehensive Logging**: Detailed execution logs with error tracking
- **Flexible Scheduling**: Python-based scheduler replacing cron dependencies

### **Testing & Quality Assurance**
- **Unit Tests**: Comprehensive testing of individual components
- **Integration Tests**: End-to-end pipeline validation
- **Coverage Reporting**: Code coverage analysis and reporting
- **Continuous Validation**: Automated testing framework

**Quick Start**
------------------------------

### 1. **Environment Setup**
```bash
git clone https://github.com/yirgareminick/NBA-Betting-Model.git  
cd NBA-Betting-Model
poetry install

# Install optional advanced ML libraries
pip install xgboost lightgbm coverage
```

### 2. **Configuration**
```bash
export KAGGLE_USERNAME=your_username  
export KAGGLE_KEY=your_key  
export ODDS_API_KEY=your_oddsapi_key  
export SLACK_WEBHOOK_URL=your_slack_webhook  # Optional
```

### 3. **Initial Data Setup**
```bash
# Ingest historical data
python src/ingest/ingest_games_new.py 2024 2025
python src/ingest/ingest_team_stats.py --seasons 2024 2025
python src/ingest/ingest_odds.py --regions us

# Build features and train initial model
python src/features/build_features.py
python src/models/advanced_trainer.py
```

### 4. **Daily Operations**

#### Daily Operations

#### Complete Daily Pipeline (Recommended)
```bash
# Generate predictions and betting analysis
python scripts/daily_betting_pipeline.py --bankroll 10000

# Full betting analysis with specific date
python scripts/daily_betting_pipeline.py --bankroll 15000 --date 2025-12-23
```

#### Advanced Pipeline Options
```bash
# Force model retraining
python scripts/daily_betting_pipeline.py --retrain --bankroll 15000

# Dry run (no actual betting recommendations saved)
python scripts/daily_betting_pipeline.py --dry-run --notify --bankroll 10000

# Quiet mode with custom config
python scripts/daily_betting_pipeline.py --quiet --config configs/custom.yml
```

### 5. **Testing**
```bash
# Run all tests
python tests/run_tests.py

# Run with coverage analysis
python tests/run_tests.py --coverage

# Run specific test suites
python tests/run_tests.py --unit-only
python tests/run_tests.py --integration-only
```

**Advanced Configuration**
------------------------------

### Model Configuration (`configs/model.yml`)
```yaml
# Advanced model training settings
model:
  algorithms: ["random_forest", "xgboost", "lightgbm"]
  ensemble: true
  hyperparameter_tuning: false  # Set true for production training

# Betting strategy settings
betting:
  kelly_fraction: 0.5          # Use 50% of full Kelly
  min_edge: 0.02              # Minimum 2% edge required
  max_bet_size: 0.1           # Maximum 10% of bankroll per bet

# Performance monitoring
thresholds:
  min_accuracy: 0.55          # Retrain if accuracy falls below 55%
  retrain_threshold: 0.05     # Retrain if accuracy drops by 5%
```

**Sample Output**
------------------------------

### Daily Predictions Report
```
🏀 NBA DAILY PREDICTIONS - 2025-07-17
================================================================================
📊 Games Analyzed: 8
💰 Recommended Bets: 3
📈 Expected Value: $127.50
🎯 Probability of Profit: 64.2%

Top Recommendations:
• Lakers vs Celtics → Bet Lakers (-110) | Edge: 8.4% | Stake: $250
• Warriors vs Suns → Bet Warriors (+120) | Edge: 6.1% | Stake: $175  
• Heat vs Bulls → Bet Heat (-105) | Edge: 3.2% | Stake: $125

Risk Analysis:
• Expected Return: $89.50 ± $156.20
• 95% Confidence Interval: [-$185, +$364]
• Maximum Potential Loss: -$550
```

### Performance Dashboard
```
📈 30-Day Performance Summary
================================================================================
Overall Accuracy: 62.8% (188/300 predictions)
Betting Performance: +$1,247 profit on $8,450 wagered (14.7% ROI)
Recent Trend: 7-day accuracy 65.1% ↗️

Model Status: ✅ Performing well
Last Retrained: 2025-07-10 (7 days ago)
Recommendation: Continue current strategy
```

**Development Roadmap**
------------------------------

**Completed Features**
- Historical data ingestion and feature engineering
- Advanced multi-algorithm model training with ensembles
- Comprehensive prediction and betting recommendation system
- Performance monitoring and model drift detection
- Complete automation pipeline with error handling
- Comprehensive testing framework

**In Development**
- Real-time NBA API integration for live game data
- Web dashboard for visual analytics and monitoring
- Advanced notification system (Slack, email, SMS)
- Live betting integration with major sportsbooks

### 📋 **Future Enhancements**
- Player injury impact modeling
- Weather and venue-specific adjustments
- Advanced market analysis and line movement tracking
- Portfolio optimization across multiple bet types

**Contributing**
------------------------------

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests to ensure nothing breaks (`python tests/run_tests.py`)
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

**Disclaimer**
------------------------------

This software is for educational and research purposes only. Sports betting involves substantial risk and may not be legal in all jurisdictions. Please gamble responsibly and within your means. I am not responsible for any financial losses incurred through the use of my software.

* **Recommended Stake (Fractional Kelly):** 3.2% of bankroll  
* **Slack Message Preview:**  
  Value Pick: Thunder ML -150 (Implied: 60%, Model: 66.8%) – Bet 2.3 units based on post-loss bounce-back trend.


**Planned Improvements**
------------------------------

- Incorporate player-level data and injury reports  
- Build dashboard for model evaluation and backtesting  
- Explore alternative modeling techniques (LightGBM, ensemble methods)

