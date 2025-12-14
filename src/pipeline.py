from datetime import date
from prefect import flow, task
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from ingest.ingest_games_new import NBADataIngestion
from features.build_features import FeatureEngineer
from models.train_model import NBAModelTrainer
from predict.daily_report import generate_daily_report
from predict.predict_games import predict_daily_games
from stake.kelly_criterion import calculate_daily_bets

# Constants
DEFAULT_BANKROLL = 10000

# ── Data ingestion and model training tasks ─────────────────────────
@task
def ingest_raw(run_date: date):
    """Ingest NBA games data for recent years."""
    year_start = 2020
    year_end = 2023
    ingestion = NBADataIngestion()
    result = ingestion.ingest_games_data(year_start, year_end)
    if result:
        df, output_file = result
        print(f"✓ Ingested: {len(df)} games")
        return df
    else:
        raise Exception("Failed to ingest games data")

@task
def build_features(df):
    """Build features from ingested data."""
    feature_engineer = FeatureEngineer()
    features = feature_engineer.build_features()
    print(f"✓ Features: {len(features)} records")
    return features

@task
def train_model(feats):
    """Train NBA prediction model."""
    trainer = NBAModelTrainer()
    metrics = trainer.train_and_save()
    print(f"✓ Model: {metrics['test_accuracy']:.3f} accuracy")
    return metrics

# ── Prediction and betting tasks ───────────────────────────────────
@task
def generate_predictions(run_date: date):
    """Generate predictions for upcoming games."""
    print(f"Generating predictions for {run_date}")
    predictions = predict_daily_games(run_date)
    return predictions

@task
def calculate_bets(predictions, bankroll: float = DEFAULT_BANKROLL):
    """Calculate optimal bet sizes using Kelly criterion."""
    print("Calculating optimal bet sizes...")
    betting_recommendations, simulation_results = calculate_daily_bets(predictions, bankroll)
    return betting_recommendations, simulation_results

@task
def generate_report(run_date: date, bankroll: float = DEFAULT_BANKROLL):
    """Generate comprehensive daily betting report."""
    print("Generating daily betting report...")
    report = generate_daily_report(run_date, bankroll)
    return report

@task
def push_picks(report):
    """Push betting picks to external systems (placeholder)."""
    recommended_bets = report.get('recommended_bets', 0)
    if recommended_bets > 0:
        print(f"Would push {recommended_bets} betting recommendations to external systems")
        # Future: Implement notification/API calls (Slack, email, betting platforms)
    else:
        print("No betting recommendations to push")
    return report

# ── Master flows ──────────────────────────────────────────────────────
@flow(name="nba-betting-training")
def training_pipeline(run_date: date = date.today()):
    """Complete model training pipeline."""
    raw = ingest_raw(run_date)
    feats = build_features(raw)
    model_metrics = train_model(feats)
    return model_metrics

@flow(name="nba-betting-daily")
def daily_prediction_pipeline(run_date: date = date.today(), bankroll: float = DEFAULT_BANKROLL):
    """Daily prediction and betting pipeline."""
    predictions = generate_predictions(run_date)
    betting_data, simulation = calculate_bets(predictions, bankroll)
    report = generate_report(run_date, bankroll)
    push_picks(report)
    return report

@flow(name="nba-betting-full")
def full_pipeline(run_date: date = date.today(), bankroll: float = DEFAULT_BANKROLL, retrain: bool = False):
    """Complete pipeline including optional model retraining."""
    if retrain:
        print("Running full pipeline with model retraining...")
        model_metrics = training_pipeline(run_date)
        print(f"Model retrained with accuracy: {model_metrics['test_accuracy']:.3f}")

    # Always run daily predictions
    report = daily_prediction_pipeline(run_date, bankroll)
    return report

if __name__ == "__main__":
    # Default: run daily predictions only
    report = daily_prediction_pipeline()
    print(f"Pipeline completed: {report['recommended_bets']} betting recommendations generated")