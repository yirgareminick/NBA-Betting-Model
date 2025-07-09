NBA Moneyline Betting Pipeline
==============================

**A machine learning pipeline designed to predict NBA moneyline outcomes, size bets using the Kelly criterion, and generate daily betting insights.**

This project processes historical NBA games (2020-2025), team statistics, and real-time betting odds to model and recommend daily bets. The pipeline includes automated data ingestion, feature engineering, and production-ready scheduling capabilities.

**Project Structure**
------------------------------

nba-betting/
├── data/                 # Data staging areas  
│   ├── raw/              # Raw data (SQLite, CSV, odds)  
│   ├── interim/          # Cleaned + aligned tables  
│   └── processed/        # Feature matrices for modeling  
├── src/  
│   ├── ingest/           # Game, team stats, odds ingestors  
│   ├── features/         # Feature engineering logic  
│   ├── models/           # Training & prediction  
│   ├── stake/            # Kelly sizing utils  
│   └── pipeline.py       # Prefect flow for full pipeline  
├── scripts/              # Automation scripts (daily, weekly, manual)
├── configs/              # YAML configuration files  
├── notebooks/            # EDA and prototyping  
├── tests/                # Unit tests  
├── logs/                 # Execution logs
├── docs/                 # Documentation  

**Key Features**
------------------------------

- **Historical game ingestion** via `kagglehub` (5K+ games, 2020-2025)  
- **Team-level metrics** from Basketball Reference  
- **Real-time odds ingestion** using TheOddsAPI  
- **Feature engineering** with 18 predictive features across 9,250+ records  
- **Model training** using Random Forest with cross-validation (62.2% accuracy)  
- **Kelly criterion stake sizing** for bankroll strategy  
- **Automated scheduling** with daily/weekly update scripts  
- **Production monitoring** with logging and error handling  
- **Season-aware automation** adapting to NBA calendar  

**Setup & Usage**
------------------------------

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourname/nba-betting.git  
   cd nba-betting
   ```

2. **Install dependencies:**
   ```bash
   poetry install
   ```

3. **Set required environment variables:**
   ```bash
   export KAGGLE_USERNAME=your_username  
   export KAGGLE_KEY=your_key  
   export ODDS_API_KEY=your_oddsapi_key  
   export SLACK_WEBHOOK_URL=your_slack_webhook
   ```

4. **Run data ingestion:**
   ```bash
   # Ingest games data
   python src/ingest/ingest_games_new.py 2024 2025
   
   # Ingest team stats
   python src/ingest/ingest_team_stats.py --seasons 2024 2025
   
   # Ingest odds
   python src/ingest/ingest_odds.py --regions us
   ```

5. **Build features and train model:**
   ```bash
   python src/features/build_features.py
   python src/models/train_model.py
   ```

6. **Use automation scripts:**
   ```bash
   # Daily update
   python scripts/daily_update.py
   
   # Weekly refresh
   python scripts/weekly_refresh.py --backup
   
   # Start scheduler
   python scripts/scheduler.py --daemon
   ```

**Current Status**
------------------------------

The pipeline is functional with the following components:

- ✅ **Data Ingestion**: Games, team stats, and odds data ingestion
- ✅ **Feature Engineering**: 18 predictive features with rolling averages
- ✅ **Model Training**: Random Forest achieving 62.2% accuracy
- ✅ **Automation**: Daily/weekly update scripts with scheduling
- ✅ **Monitoring**: Comprehensive logging and error handling
- 🔄 **Prediction Pipeline**: In development
- 🔄 **Bet Sizing**: Kelly criterion implementation in progress

**Documentation**
------------------------------

- **Data Ingestion Strategy**: See `docs/ingestion-strategy.md` for detailed data source information
- **Automation Scripts**: See `scripts/README.md` for comprehensive automation guide
- **Feature Engineering**: Rolling averages, team performance metrics, and market indicators
- **Model Configuration**: YAML-based configuration in `configs/` directory

**Next Steps**
------------------------------

- Complete prediction pipeline integration
- Implement live betting recommendations
- Add model performance tracking and backtesting
- Incorporate player-level data and injury reports  
- Build dashboard for model evaluation  
- Explore alternative modeling techniques (LightGBM, ensemble methods)
