NBA Moneyline Betting Pipeline
==============================

**A machine learning pipeline designed to predict NBA moneyline outcomes, size bets using the Kelly criterion, and eventually generate daily betting insights.**

This project is currently under development. It brings together historical NBA data, season-level metrics, and real-time betting odds to model and recommend daily bets. While the core components are being built and tested, the structure supports easy iteration and future expansion.

**Project Structure**
------------------------------

nba-betting/
├── data/                 # Data staging areas  
│   ├── raw/              # Raw pulled data (from Kaggle, etc.)  
│   ├── interim/          # Cleaned + aligned tables  
│   └── processed/        # Feature matrices for modeling  
├── src/  
│   ├── ingest/           # Game, team stats, odds ingestors  
│   ├── features/         # Feature engineering logic  
│   ├── models/           # Training & prediction  
│   ├── stake/            # Kelly sizing utils  
│   └── pipeline.py       # Prefect flow for full pipeline  
├── configs/              # YAML configuration files  
├── notebooks/            # EDA and prototyping  
├── tests/                # Unit tests  

**Key Features (In Progress)**
------------------------------

- **Historical game ingestion** via `kagglehub`  
- **Team-level metrics** from Basketball Reference  
- **Real-time odds ingestion** using TheOddsAPI  
- **Feature engineering** including rolling stats and market signals  
- **Model training** using XGBoost with cross-validation  
- **Kelly criterion stake sizing** for bankroll strategy  
- **Pipeline orchestration** with Prefect  
- **Slack integration** to notify live picks  

**Setup & Usage**
------------------------------

1. **Clone the repository:**
   git clone https://github.com/yourname/nba-betting.git  
   cd nba-betting

2. **Install dependencies:**
   poetry install

3. **Set required environment variables:**
   export KAGGLE_USERNAME=your_username  
   export KAGGLE_KEY=your_key  
   export ODDS_API_KEY=your_oddsapi_key  
   export SLACK_WEBHOOK_URL=your_slack_webhook

4. **Run the pipeline:**
   poetry run python src/pipeline.py --run-date $(date +%F)

**Sample Output (Development Preview)**
------------------------------

This output is representative of the intended functionality: 

* **Predicted Probabilities:** Thunder: 62.5%, Pacers: 37.5%  
* **Edge over Market:** +6.8%  
* **Recommended Stake (Fractional Kelly):** 3.2% of bankroll  
* **Slack Message Preview:**  
  Value Pick: Thunder ML -150 (Implied: 60%, Model: 66.8%) – Bet 2.3 units based on post-loss bounce-back trend.


**Planned Improvements**
------------------------------

- Incorporate player-level data and injury reports  
- Build dashboard for model evaluation and backtesting  
- Explore alternative modeling techniques (LightGBM, ensemble methods)
