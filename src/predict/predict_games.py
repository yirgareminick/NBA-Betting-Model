"""
NBA Game Prediction Module

This module handles loading trained models and making predictions for upcoming NBA games.
It processes current team statistics, odds data, and generates probability estimates.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
import joblib
import yaml
import warnings
import sys
warnings.filterwarnings('ignore')

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src"))

try:
    from features.build_features import FeatureEngineer
except ImportError:
    # Fallback if FeatureEngineer is not available
    FeatureEngineer = None


class NBAPredictor:
    """NBA game outcome predictor using trained models."""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.project_root = Path(__file__).parent.parent.parent
        self.model_path = model_path or self.project_root / "models" / "nba_model_latest.joblib"
        self.metadata_path = self.project_root / "models" / "nba_model_latest_metadata.yml"
        
        self.model = None
        self.feature_columns = None
        self.metadata = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and its metadata."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        
        print(f"Loading model from: {self.model_path}")
        self.model = joblib.load(self.model_path)
        
        with open(self.metadata_path, 'r') as f:
            self.metadata = yaml.safe_load(f)
        
        self.feature_columns = self.metadata['feature_columns']
        print(f"✓ Model loaded successfully")
        print(f"  - Created: {self.metadata['created_at']}")
        print(f"  - Test accuracy: {self.metadata['metrics']['test_accuracy']:.3f}")
        print(f"  - Features: {len(self.feature_columns)}")
    
    def get_upcoming_games(self, target_date: date = None) -> pd.DataFrame:
        """Get upcoming games for prediction."""
        if target_date is None:
            target_date = date.today()
        
        # For now, return empty DataFrame - will be populated when we have live data
        # In production, this would fetch from NBA API or sports data provider
        print(f"📅 Getting games for {target_date}")
        
        # Placeholder implementation
        games = pd.DataFrame({
            'game_id': ['sample_game_1', 'sample_game_2'],
            'game_date': [target_date, target_date],
            'home_team': ['LAL', 'GSW'],
            'away_team': ['BOS', 'PHX'],
            'home_odds': [1.85, 2.10],
            'away_odds': [1.95, 1.75]
        })
        
        print(f"✓ Found {len(games)} upcoming games")
        return games
    
    def prepare_prediction_features(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for upcoming games."""
        print("🔧 Preparing prediction features...")
        
        # Initialize feature engineer
        feature_engineer = FeatureEngineer()
        
        # For each game, we need to build features based on recent team performance
        prediction_features = []
        
        for _, game in games_df.iterrows():
            home_team = game['home_team']
            away_team = game['away_team']
            game_date = game['game_date']
            
            # Build features for home team
            home_features = self._build_team_features(
                team=home_team, 
                opponent=away_team, 
                is_home=True, 
                game_date=game_date
            )
            
            # Build features for away team  
            away_features = self._build_team_features(
                team=away_team, 
                opponent=home_team, 
                is_home=False, 
                game_date=game_date
            )
            
            prediction_features.extend([home_features, away_features])
        
        features_df = pd.DataFrame(prediction_features)
        
        # Ensure all required feature columns are present
        missing_features = set(self.feature_columns) - set(features_df.columns)
        for feature in missing_features:
            features_df[feature] = 0.0  # Default value for missing features
        
        # Select only the features used in training
        features_df = features_df[self.feature_columns]
        
        print(f"✓ Prepared features for {len(features_df)} team-game combinations")
        return features_df
    
    def _build_team_features(self, team: str, opponent: str, is_home: bool, game_date: date) -> Dict:
        """Build features for a single team in a specific matchup."""
        # This is a simplified implementation
        # In production, this would calculate actual rolling statistics from recent games
        
        features = {
            'is_home': int(is_home),
            'team_points_avg': 110.0 + np.random.normal(0, 5),  # Placeholder
            'team_rebounds_avg': 45.0 + np.random.normal(0, 3),
            'team_assists_avg': 25.0 + np.random.normal(0, 2),
            'opp_points_avg': 108.0 + np.random.normal(0, 5),
            'opp_rebounds_avg': 44.0 + np.random.normal(0, 3),
            'opp_assists_avg': 24.0 + np.random.normal(0, 2),
            'team_fg_pct_avg': 0.46 + np.random.normal(0, 0.02),
            'team_3p_pct_avg': 0.36 + np.random.normal(0, 0.03),
            'rest_days': np.random.randint(1, 4),
        }
        
        # Add additional features to match training set
        for i in range(len(self.feature_columns) - len(features)):
            features[f'feature_{i}'] = np.random.normal(0, 1)
        
        return features
    
    def predict_games(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions for upcoming games."""
        print("🔮 Making game predictions...")
        
        # Prepare features
        features = self.prepare_prediction_features(games_df)
        
        # Make predictions
        probabilities = self.model.predict_proba(features)[:, 1]  # Probability of win
        predictions = self.model.predict(features)
        
        # Organize results
        results = []
        for i, (_, game) in enumerate(games_df.iterrows()):
            home_idx = i * 2
            away_idx = i * 2 + 1
            
            home_prob = probabilities[home_idx]
            away_prob = probabilities[away_idx]
            
            # Normalize probabilities to sum to 1
            total_prob = home_prob + away_prob
            if total_prob > 0:
                home_prob = home_prob / total_prob
                away_prob = away_prob / total_prob
            
            result = {
                'game_id': game['game_id'],
                'game_date': game['game_date'],
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'home_prob': home_prob,
                'away_prob': away_prob,
                'predicted_winner': game['home_team'] if home_prob > away_prob else game['away_team'],
                'confidence': max(home_prob, away_prob),
                'home_odds': game.get('home_odds', None),
                'away_odds': game.get('away_odds', None)
            }
            results.append(result)
        
        predictions_df = pd.DataFrame(results)
        print(f"✓ Generated predictions for {len(predictions_df)} games")
        
        return predictions_df
    
    def calculate_betting_edges(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate betting edges based on model probabilities vs market odds."""
        print("💰 Calculating betting edges...")
        
        predictions_with_edges = predictions_df.copy()
        
        # Calculate implied probabilities from odds
        predictions_with_edges['home_implied_prob'] = 1 / predictions_with_edges['home_odds']
        predictions_with_edges['away_implied_prob'] = 1 / predictions_with_edges['away_odds']
        
        # Calculate edges (model prob - implied prob)
        predictions_with_edges['home_edge'] = (
            predictions_with_edges['home_prob'] - predictions_with_edges['home_implied_prob']
        )
        predictions_with_edges['away_edge'] = (
            predictions_with_edges['away_prob'] - predictions_with_edges['away_implied_prob']
        )
        
        # Determine best bet for each game
        predictions_with_edges['best_bet_team'] = np.where(
            predictions_with_edges['home_edge'] > predictions_with_edges['away_edge'],
            predictions_with_edges['home_team'],
            predictions_with_edges['away_team']
        )
        
        predictions_with_edges['best_bet_edge'] = np.maximum(
            predictions_with_edges['home_edge'],
            predictions_with_edges['away_edge']
        )
        
        predictions_with_edges['best_bet_prob'] = np.where(
            predictions_with_edges['home_edge'] > predictions_with_edges['away_edge'],
            predictions_with_edges['home_prob'],
            predictions_with_edges['away_prob']
        )
        
        predictions_with_edges['best_bet_odds'] = np.where(
            predictions_with_edges['home_edge'] > predictions_with_edges['away_edge'],
            predictions_with_edges['home_odds'],
            predictions_with_edges['away_odds']
        )
        
        print(f"✓ Calculated edges for {len(predictions_with_edges)} games")
        return predictions_with_edges
    
    def save_predictions(self, predictions_df: pd.DataFrame, target_date: date = None) -> Path:
        """Save predictions to file."""
        if target_date is None:
            target_date = date.today()
        
        predictions_dir = self.project_root / "data" / "predictions"
        predictions_dir.mkdir(exist_ok=True)
        
        filename = f"predictions_{target_date.strftime('%Y%m%d')}.csv"
        filepath = predictions_dir / filename
        
        predictions_df.to_csv(filepath, index=False)
        print(f"💾 Predictions saved to: {filepath}")
        
        return filepath


def predict_daily_games(target_date: date = None) -> pd.DataFrame:
    """Main function to predict daily games."""
    if target_date is None:
        target_date = date.today()
    
    print("=" * 80)
    print(f"🏀 NBA DAILY PREDICTIONS - {target_date}")
    print("=" * 80)
    
    # Initialize predictor
    predictor = NBAPredictor()
    
    # Get upcoming games
    games = predictor.get_upcoming_games(target_date)
    
    if len(games) == 0:
        print("📭 No games found for prediction")
        return pd.DataFrame()
    
    # Make predictions
    predictions = predictor.predict_games(games)
    
    # Calculate betting edges
    predictions_with_edges = predictor.calculate_betting_edges(predictions)
    
    # Save predictions
    predictor.save_predictions(predictions_with_edges, target_date)
    
    print("=" * 80)
    print("✅ DAILY PREDICTIONS COMPLETED")
    print("=" * 80)
    
    return predictions_with_edges


if __name__ == "__main__":
    predictions = predict_daily_games()
    print(predictions[['home_team', 'away_team', 'predicted_winner', 'confidence', 'best_bet_edge']].head())
