# NBA Betting Model - Data Ingestion Strategy

## Overview
This document outlines the data ingestion strategy for the NBA betting model, including when and how to run each ingestion script for optimal data freshness and cost efficiency.

## Data Sources & Update Frequencies

### 1. Games Data (`ingest_games_new.py`)
- **Source**: Kaggle NBA dataset via SQLite database
- **Update Frequency**: Weekly during season, monthly in off-season
- **Optimal Timing**: After games complete (next day, 1-2 AM)
- **Cost**: Kaggle API calls (rate limited)
- **Command**: `python src/ingest/ingest_games_new.py <start_year> <end_year>`

### 2. Team Stats (`ingest_team_stats.py`)
- **Source**: Basketball Reference team ratings
- **Update Frequency**: Weekly during season (Tuesday/Wednesday)
- **Optimal Timing**: Mid-week when sites update ratings
- **Cost**: Web scraping (respect rate limits)
- **Command**: `python src/ingest/ingest_team_stats.py --seasons 2024 2025`

### 3. Odds Data (`ingest_odds.py`)
- **Source**: The Odds API
- **Update Frequency**: Multiple times daily during season
- **Optimal Timing**: 
  - Morning (8 AM) - Opening lines
  - Afternoon (2 PM) - Pre-game adjustments  
  - Evening (6 PM) - Final lines
- **Cost**: API calls ($$ - monitor usage)
- **Command**: `python src/ingest/ingest_odds.py --regions us --bookmakers draftkings,fanduel`

## NBA Season Calendar

### Regular Season (October - April)
```
Daily Schedule:
├── 6:00 AM  → Morning odds update
├── 10:00 AM → Team stats (Tuesday/Wednesday only)
├── 2:00 PM  → Afternoon odds refresh
├── 6:00 PM  → Evening odds for game day
├── 11:00 PM → Post-game data ingestion
└── 11:30 PM → Feature engineering + model training
```

### Playoffs (April - June)
```
Enhanced Schedule:
├── Every 4 hours → Odds updates (higher frequency)
├── Daily → Games ingestion
├── Weekly → Team stats updates
└── Real-time → Feature engineering for elimination games
```

### Off-Season (July - September)
```
Maintenance Schedule:
├── Monthly → Historical data validation
├── Weekly → Team stats for trades/signings
├── As needed → Model retraining with full dataset
└── Quarterly → Data pipeline optimization
```

## Implementation Phases

### Phase 1: Manual Daily (Development - Current)
**Purpose**: Testing and development
**Frequency**: Manual execution
**Commands**:
```bash
# Morning routine (development)
python src/ingest/ingest_odds.py --regions us
python src/features/build_features.py

# Evening routine (after games)
python src/ingest/ingest_games_new.py 2024 2024
python src/features/build_features.py
```

### Phase 2: Semi-Automated (Testing)
**Purpose**: Scheduled execution with monitoring
**Implementation**: Use provided automation scripts
**Schedule**: Daily cron jobs

### Phase 3: Fully Automated (Production)
**Purpose**: Production betting pipeline
**Implementation**: Prefect workflows + cloud scheduling
**Monitoring**: Data quality alerts, cost tracking

## Cost & Rate Limit Management

### Kaggle API (Games Data)
- **Limits**: Standard rate limits apply
- **Strategy**: Cache aggressively, update weekly max
- **Cost**: Free tier usually sufficient

### Basketball Reference (Team Stats)
- **Limits**: Respect robots.txt, 1 request/second max
- **Strategy**: Weekly updates, cache locally
- **Cost**: Free but respect rate limits

### The Odds API (Odds Data)
- **Limits**: API key dependent (monitor usage)
- **Strategy**: Strategic timing, essential bookmakers only
- **Cost**: $$ - Monitor monthly usage closely

## Data Dependency Chain

```
Games Data → Feature Engineering → Model Training → Predictions
Team Stats ↗                    ↗
Odds Data ↗                     → Bet Sizing → Final Recommendations
```

## Quality Assurance

### Data Validation Checks
- [ ] Date ranges are reasonable
- [ ] No duplicate games
- [ ] Team names are consistent
- [ ] Odds are within expected ranges
- [ ] All expected teams are present

### Monitoring Alerts
- [ ] Failed ingestion runs
- [ ] Data freshness exceeds thresholds
- [ ] API rate limit violations
- [ ] Unexpected data schema changes
- [ ] Cost thresholds exceeded

## Emergency Procedures

### If Kaggle API Fails
1. Use cached games data
2. Manual CSV download as backup
3. Reduce update frequency temporarily

### If Basketball Reference is Blocked
1. Use cached team stats
2. Implement alternative data source
3. Reduce scraping frequency

### If Odds API Exceeds Limits
1. Prioritize key bookmakers only
2. Reduce update frequency
3. Implement cost alerts

## Getting Started

1. **Set up environment variables**:
   ```bash
   export KAGGLE_USERNAME=your_username
   export KAGGLE_KEY=your_api_key
   export ODDS_API_KEY=your_odds_api_key
   ```

2. **Test individual scripts**:
   ```bash
   # Test each ingestion script
   python src/ingest/ingest_games_new.py 2024 2024
   python src/ingest/ingest_team_stats.py --seasons 2024
   python src/ingest/ingest_odds.py --regions us
   ```

3. **Use automation scripts**:
   ```bash
   # Daily update
   python scripts/daily_update.py
   
   # Weekly full refresh  
   python scripts/weekly_refresh.py
   
   # Emergency manual update
   python scripts/manual_update.py
   
   # Enhanced daily pipeline (recommended)
   python scripts/daily_betting_pipeline.py
   ```

4. **Monitor and optimize**:
   - Check logs in `logs/` directory
   - Monitor API usage and costs
   - Adjust frequencies based on model performance

## Next Steps

- [ ] Implement monitoring dashboard
- [ ] Set up automated alerting
- [ ] Optimize update frequencies based on model sensitivity
- [ ] Implement data quality tests
- [ ] Create backup data sources
- [ ] Set up production scheduling (Prefect/Airflow)

---

*Last updated: July 2025*
*Next review: Quarterly or when NBA schedule changes*
