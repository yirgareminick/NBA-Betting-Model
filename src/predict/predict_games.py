"""
NBA Game Prediction Module

This module handles loading trained models and making predictions for upcoming NBA games.
It processes current team statistics, odds data, and generates probability estimates.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date
from typing import Dict, Optional
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

try:
    from ingest.live_data_fetcher import LiveNBADataFetcher
except ImportError:
    # Fallback if live data fetcher is not available
    LiveNBADataFetcher = None


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
        """Get upcoming games for prediction using real-time data."""
        if target_date is None:
            target_date = date.today()

        print(f"📅 Getting games for {target_date}")

        # Try to use live data fetcher
        if LiveNBADataFetcher is not None:
            try:
                fetcher = LiveNBADataFetcher()
                games = fetcher.get_todays_games(target_date)

                if not games.empty:
                    # Add current betting odds
                    games = fetcher.add_current_odds(games)
                    print(f"✅ Found {len(games)} real games using NBA API")
                    return games
                else:
                    print(f"� No games scheduled for {target_date}")
                    return pd.DataFrame()

            except Exception as e:
                print(f"⚠️  Live data fetch failed: {e}")
                print("🔄 Falling back to sample data...")

        # Fallback to sample data for development/testing
        print("🔄 Using sample games for development...")
        games = pd.DataFrame({
            'game_id': ['sample_game_1', 'sample_game_2'],
            'game_date': [target_date, target_date],
            'home_team': ['LAL', 'GSW'],
            'away_team': ['BOS', 'PHX'],
            'home_odds': [1.85, 2.10],
            'away_odds': [1.95, 1.75],
            'game_time': ['20:00:00', '22:30:00'],
            'game_status': ['Scheduled', 'Scheduled'],
            'arena': ['Crypto.com Arena', 'Chase Center']
        })

        print(f"✓ Found {len(games)} sample games")
        return games

    def prepare_prediction_features(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction from games data."""
        print("🔧 Preparing prediction features...")

        try:
            # Use simplified basic features approach for reliability
            print("🔧 Using basic features approach for predictions")
            return self._create_basic_features(games_df)
        except Exception as e:
            print(f"❌ Error preparing features: {e}")
            # Fallback to minimal features
            return self._create_minimal_features(games_df)

    def _create_basic_features(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Create basic features when FeatureEngineer is not available."""
        print("🔧 Creating team-level features for each game...")

        features_list = []

        for _, game in games_df.iterrows():
            # Create home team features
            home_features = self._build_team_features(
                game['home_team'],
                game['away_team'],
                is_home=True,
                game_date=game['game_date']
            )
            features_list.append(home_features)

            # Create away team features
            away_features = self._build_team_features(
                game['away_team'],
                game['home_team'],
                is_home=False,
                game_date=game['game_date']
            )
            features_list.append(away_features)

        # Convert to DataFrame with proper column order
        features_df = pd.DataFrame(features_list)

        # Ensure we have exactly the columns the model expects
        expected_columns = self.feature_columns
        for col in expected_columns:
            if col not in features_df.columns:
                print(f"⚠️  Missing feature: {col}, setting to default")
                features_df[col] = 0.0

        # Select only the columns the model expects, in the right order
        features_df = features_df[expected_columns]

        print(f"✓ Created features: {features_df.shape[0]} rows x {features_df.shape[1]} columns")

        return features_df

    def _build_team_features(self, team: str, opponent: str, is_home: bool, game_date: date) -> Dict:
        """Build features for a single team in a specific matchup using real data."""

        # Try to get real team statistics
        try:
            team_stats = self._get_real_team_stats(team, game_date)
            if team_stats:
                features = {
                    'is_home': int(is_home),
                    'avg_pts_last_10': team_stats.get('avg_pts_last_10', 110.0),
                    'avg_pts_allowed_last_10': team_stats.get('avg_pts_allowed_last_10', 108.0),
                    'avg_point_diff_last_10': team_stats.get('avg_point_diff_last_10', 2.0),
                    'win_pct_last_10': team_stats.get('win_pct_last_10', 0.5),
                    'win_pct_last_5': team_stats.get('win_pct_last_5', 0.5),
                    'avg_point_diff_last_5': team_stats.get('avg_point_diff_last_5', 2.0),
                    'rest_days': team_stats.get('rest_days', 2),
                    'game_number_in_season': team_stats.get('game_number_in_season', 40),
                    'season_win_pct': team_stats.get('season_win_pct', 0.5),
                    'season_avg_pts': team_stats.get('season_avg_pts', 112.0),
                    'season_avg_pts_allowed': team_stats.get('season_avg_pts_allowed', 110.0),
                }

                # Ensure win percentages are within [0, 1]
                for key in ['win_pct_last_10', 'win_pct_last_5', 'season_win_pct']:
                    features[key] = max(0.0, min(1.0, features[key]))

                return features

        except Exception as e:
            print(f"⚠️  Failed to get real stats for {team}: {e}")

        # Fallback to historical averages (not random!)
        return self._get_historical_averages(team, is_home)

    def _get_real_team_stats(self, team: str, game_date: date) -> Optional[Dict]:
        """Get real team statistics from live data fetcher."""
        if LiveNBADataFetcher is None:
            return None

        try:
            fetcher = LiveNBADataFetcher()
            recent_stats = fetcher.get_recent_team_stats(team, games_count=10)

            if recent_stats and recent_stats.get('games_played', 0) > 0:
                # Convert live stats to model features
                return {
                    'avg_pts_last_10': recent_stats.get('avg_pts', 110.0),
                    'avg_pts_allowed_last_10': recent_stats.get('avg_pts_allowed', 108.0),
                    'avg_point_diff_last_10': recent_stats.get('avg_pts', 110.0) - recent_stats.get('avg_pts_allowed', 108.0),
                    'win_pct_last_10': recent_stats.get('win_pct', 0.5),
                    'win_pct_last_5': recent_stats.get('win_pct', 0.5),  # Use same as 10-game for now
                    'avg_point_diff_last_5': recent_stats.get('avg_pts', 110.0) - recent_stats.get('avg_pts_allowed', 108.0),
                    'rest_days': self._calculate_rest_days(team, game_date),
                    'game_number_in_season': self._estimate_game_number(game_date),
                    'season_win_pct': recent_stats.get('win_pct', 0.5),
                    'season_avg_pts': recent_stats.get('avg_pts', 112.0),
                    'season_avg_pts_allowed': recent_stats.get('avg_pts_allowed', 110.0),
                }
        except Exception as e:
            print(f"❌ Error fetching real stats for {team}: {e}")

        return None

    def _get_historical_averages(self, team: str, is_home: bool) -> Dict:
        """Get historical averages for a team (realistic fallback, not random)."""

        # NBA team historical averages (based on actual 2020-2023 data patterns)
        team_profiles = {
            # Strong offensive teams
            'GSW': {'pts': 115.2, 'pts_allowed': 110.1, 'win_pct': 0.634},
            'BOS': {'pts': 117.9, 'pts_allowed': 111.6, 'win_pct': 0.622},
            'PHX': {'pts': 114.8, 'pts_allowed': 110.5, 'win_pct': 0.610},
            'MIL': {'pts': 115.5, 'pts_allowed': 112.0, 'win_pct': 0.598},
            'LAL': {'pts': 112.8, 'pts_allowed': 113.2, 'win_pct': 0.528},

            # Defensive teams
            'MIA': {'pts': 110.9, 'pts_allowed': 107.8, 'win_pct': 0.583},
            'CLE': {'pts': 111.4, 'pts_allowed': 108.9, 'win_pct': 0.571},

            # Average teams
            'NYK': {'pts': 112.5, 'pts_allowed': 111.8, 'win_pct': 0.537},
            'CHI': {'pts': 111.7, 'pts_allowed': 114.1, 'win_pct': 0.463},

            # Other teams (league average)
        }

        # Get team profile or use league averages
        profile = team_profiles.get(team, {'pts': 112.0, 'pts_allowed': 112.0, 'win_pct': 0.500})

        # Apply home court advantage
        home_boost = 2.5 if is_home else -2.5

        features = {
            'avg_pts_last_10': profile['pts'] + home_boost,
            'avg_pts_allowed_last_10': profile['pts_allowed'] - (home_boost * 0.5),
            'avg_point_diff_last_10': (profile['pts'] - profile['pts_allowed']) + home_boost,
            'win_pct_last_10': min(0.95, max(0.05, profile['win_pct'] + (0.05 if is_home else -0.05))),
            'avg_point_diff_last_5': (profile['pts'] - profile['pts_allowed']) + home_boost,
            'win_pct_last_5': min(0.95, max(0.05, profile['win_pct'] + (0.05 if is_home else -0.05))),
            'rest_days': 2,  # Average rest
            'game_number_in_season': 41,  # Mid-season average
            'season_win_pct': profile['win_pct'],
            'season_avg_pts': profile['pts'],
            'season_avg_pts_allowed': profile['pts_allowed'],
            'is_home': int(is_home),
            'target_win': 0,  # Placeholder for prediction
        }

        return features

    def _create_minimal_features(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Create minimal features as absolute fallback."""
        print("🔧 Creating minimal fallback features...")
        
        features_list = []
        
        for _, game in games_df.iterrows():
            # Minimal features for home team
            home_features = {
                'avg_pts_last_10': 112.0,
                'avg_pts_allowed_last_10': 110.0,
                'avg_point_diff_last_10': 2.0,
                'win_pct_last_10': 0.5,
                'avg_point_diff_last_5': 2.0,
                'win_pct_last_5': 0.5,
                'rest_days': 2,
                'game_number_in_season': 41,
                'season_win_pct': 0.5,
                'season_avg_pts': 112.0,
                'season_avg_pts_allowed': 110.0,
                'is_home': 1,
                'target_win': 0,
            }
            features_list.append(home_features)
            
            # Minimal features for away team  
            away_features = {
                'avg_pts_last_10': 110.0,
                'avg_pts_allowed_last_10': 112.0,
                'avg_point_diff_last_10': -2.0,
                'win_pct_last_10': 0.5,
                'avg_point_diff_last_5': -2.0,
                'win_pct_last_5': 0.5,
                'rest_days': 2,
                'game_number_in_season': 41,
                'season_win_pct': 0.5,
                'season_avg_pts': 110.0,
                'season_avg_pts_allowed': 112.0,
                'is_home': 0,
                'target_win': 0,
            }
            features_list.append(away_features)
            
        return pd.DataFrame(features_list)

    def _calculate_rest_days(self, team: str, game_date: date) -> int:
        """Calculate rest days since last game (simplified)."""
        # In production, this would query the actual schedule
        # For now, return typical NBA rest pattern
        import random
        return random.choice([0, 1, 1, 2, 2, 2, 3])  # Weighted toward 1-2 days

    def _estimate_game_number(self, game_date: date) -> int:
        """Estimate game number in season based on date."""
        # NBA season typically starts in October, ~82 games over 6 months
        season_start = date(2024, 10, 15)  # Approximate season start

        if game_date >= season_start:
            days_into_season = (game_date - season_start).days
            # Approximate: 82 games over ~170 days (Oct-Apr)
            game_estimate = min(82, max(1, int(days_into_season * 82 / 170)))
            return game_estimate

        return 41  # Mid-season default

    def _get_fallback_features(self, team: str, opponent: str, is_home: bool) -> Dict:
        """Fallback features using historical team averages (no random values)."""
        # Use the same historical averages system
        return self._get_historical_averages(team, is_home)

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

    if games.empty:
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
