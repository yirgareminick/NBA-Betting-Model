from datetime import date
from prefect import flow, task
from ingest.ingest_games_new import NBADataIngestion
from features.build_features import FeatureEngineer

# ── placeholder tasks flesh out later ─────────────────────────
@task
def ingest_raw(run_date: date):
    # Use recent years available in dataset (2020-2023)
    year_start = 2020
    year_end = 2023
    print(f"Ingesting NBA games data for {year_start}-{year_end}")
    ingestion = NBADataIngestion()
    result = ingestion.ingest_games_data(year_start, year_end)
    if result:
        df, output_file = result
        print(f"Successfully ingested {len(df)} games for {year_start}-{year_end}")
        return df
    else:
        raise Exception("Failed to ingest games data")
        
@task
def build_features(df): 
    print("Building features from ingested data...")
    feature_engineer = FeatureEngineer()
    features = feature_engineer.build_features()
    print(f"Features built: {len(features)} records with {len(features.columns)} columns")
    return features
    
@task
def train_model(feats): 
    print("Model training not implemented yet")
    return "model_placeholder"
    
@task
def size_bets(model_artifact): 
    print("Bet sizing not implemented yet")
    return "bets_placeholder"
    
@task
def push_picks(bets): 
    print("Push picks not implemented yet")
    return None

# ── master flow ──────────────────────────────────────────────────────
@flow(name="nba-betting-daily")
def main(run_date: date = date.today()):
    raw = ingest_raw(run_date)
    feats = build_features(raw)
    model_artifact = train_model(feats)
    bets = size_bets(model_artifact)
    push_picks(bets)

if __name__ == "__main__":
    main()