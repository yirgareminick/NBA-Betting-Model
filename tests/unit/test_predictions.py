"""
Unit Tests for Prediction System

Tests for the NBA game prediction and model loading functionality.
"""

import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import date
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from predict.predict_games import NBAPredictor, predict_daily_games


class TestNBAPredictor(unittest.TestCase):
    """Test cases for NBA prediction functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.project_root = Path(__file__).parent.parent.parent
    
    @patch('predict.predict_games.joblib.load')
    @patch('predict.predict_games.yaml.safe_load')
    @patch('builtins.open')
    def test_predictor_initialization_with_mock_model(self, mock_open, mock_yaml, mock_joblib):
        """Test predictor initialization with mocked model."""
        # Mock the model and metadata
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1, 0])
        mock_model.predict_proba.return_value = np.array([[0.4, 0.6], [0.7, 0.3]])
        
        mock_metadata = {
            'created_at': '2024-01-01T00:00:00',
            'metrics': {'test_accuracy': 0.65},
            'feature_columns': ['feature1', 'feature2', 'is_home']
        }
        
        mock_joblib.return_value = mock_model
        mock_yaml.return_value = mock_metadata
        
        # Test initialization
        predictor = NBAPredictor()
        
        self.assertIsNotNone(predictor.model)
        self.assertEqual(predictor.feature_columns, ['feature1', 'feature2', 'is_home'])
    
    def test_get_upcoming_games(self):
        """Test upcoming games generation."""
        # Create predictor without loading actual model
        predictor = NBAPredictor.__new__(NBAPredictor)
        predictor.project_root = self.project_root
        
        games = predictor.get_upcoming_games()
        
        # Should return DataFrame with required columns
        self.assertIsInstance(games, pd.DataFrame)
        required_columns = ['game_id', 'game_date', 'home_team', 'away_team', 'home_odds', 'away_odds']
        for col in required_columns:
            self.assertIn(col, games.columns)
        
        # Should have at least some sample games
        self.assertGreater(len(games), 0)
    
    def test_build_team_features(self):
        """Test team feature building."""
        predictor = NBAPredictor.__new__(NBAPredictor)
        # Use realistic feature columns that match actual implementation
        predictor.feature_columns = [
            'is_home', 'avg_pts_last_10', 'avg_pts_allowed_last_10',
            'rest_days', 'season_win_pct'
        ]
        
        features = predictor._build_team_features('LAL', 'BOS', True, date.today())
        
        # Should return dictionary with features
        self.assertIsInstance(features, dict)
        self.assertIn('is_home', features)
        self.assertEqual(features['is_home'], 1)  # Home team should be 1
        
        # Away team test
        away_features = predictor._build_team_features('BOS', 'LAL', False, date.today())
        self.assertEqual(away_features['is_home'], 0)  # Away team should be 0
    
    @patch('predict.predict_games.NBAPredictor._load_model')
    def test_prepare_prediction_features(self, mock_load_model):
        """Test prediction feature preparation."""
        # Mock the model loading
        mock_load_model.return_value = None
        
        predictor = NBAPredictor()
        # Use feature columns that match the actual feature generation
        predictor.feature_columns = [
            'is_home', 'avg_pts_last_10', 'avg_pts_allowed_last_10',
            'avg_point_diff_last_10', 'win_pct_last_10', 'win_pct_last_5',
            'avg_point_diff_last_5', 'rest_days', 'game_number_in_season',
            'season_win_pct', 'season_avg_pts', 'season_avg_pts_allowed'
        ]
        
        # Sample games data
        games_df = pd.DataFrame({
            'game_id': ['game1'],
            'game_date': [date.today()],
            'home_team': ['LAL'],
            'away_team': ['BOS'],
            'home_odds': [1.8],
            'away_odds': [2.0]
        })
        
        features = predictor.prepare_prediction_features(games_df)
        
        # Should return DataFrame with correct structure
        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(len(features), 2)  # One row per team
        
        # Should have all required feature columns
        for col in predictor.feature_columns:
            self.assertIn(col, features.columns)


class TestPredictDailyGames(unittest.TestCase):
    """Test cases for daily game prediction function."""
    
    @patch('predict.predict_games.NBAPredictor')
    def test_predict_daily_games_no_games(self, mock_predictor_class):
        """Test prediction when no games are available."""
        mock_predictor = Mock()
        mock_predictor.get_upcoming_games.return_value = pd.DataFrame()
        mock_predictor_class.return_value = mock_predictor
        
        result = predict_daily_games()
        
        # Should return empty DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 0)
    
    @patch('predict.predict_games.NBAPredictor')
    def test_predict_daily_games_with_games(self, mock_predictor_class):
        """Test prediction with available games."""
        # Mock upcoming games
        mock_games = pd.DataFrame({
            'game_id': ['game1', 'game2'],
            'game_date': [date.today(), date.today()],
            'home_team': ['LAL', 'GSW'],
            'away_team': ['BOS', 'PHX'],
            'home_odds': [1.8, 2.1],
            'away_odds': [2.0, 1.75]
        })
        
        # Mock predictions
        mock_predictions = pd.DataFrame({
            'game_id': ['game1', 'game2'],
            'game_date': [date.today(), date.today()],
            'home_team': ['LAL', 'GSW'],
            'away_team': ['BOS', 'PHX'],
            'home_prob': [0.6, 0.55],
            'away_prob': [0.4, 0.45],
            'predicted_winner': ['LAL', 'GSW'],
            'confidence': [0.6, 0.55],
            'home_odds': [1.8, 2.1],
            'away_odds': [2.0, 1.75],
            'home_implied_prob': [0.556, 0.476],
            'away_implied_prob': [0.5, 0.571],
            'home_edge': [0.044, 0.074],
            'away_edge': [-0.1, -0.121],
            'best_bet_team': ['LAL', 'GSW'],
            'best_bet_edge': [0.044, 0.074],
            'best_bet_prob': [0.6, 0.55],
            'best_bet_odds': [1.8, 2.1]
        })
        
        mock_predictor = Mock()
        mock_predictor.get_upcoming_games.return_value = mock_games
        mock_predictor.predict_games.return_value = mock_predictions
        mock_predictor.calculate_betting_edges.return_value = mock_predictions
        mock_predictor.save_predictions.return_value = Path("/tmp/predictions.csv")
        mock_predictor_class.return_value = mock_predictor
        
        result = predict_daily_games()
        
        # Should return predictions DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        
        # Should have required columns
        required_columns = ['game_id', 'home_team', 'away_team', 'predicted_winner', 'confidence']
        for col in required_columns:
            self.assertIn(col, result.columns)


class TestIntegrationPredict(unittest.TestCase):
    """Integration tests for prediction components."""
    
    def test_feature_consistency(self):
        """Test that feature generation is consistent."""
        predictor = NBAPredictor.__new__(NBAPredictor)
        predictor.feature_columns = ['is_home', 'team_points_avg', 'feature_1']
        
        # Generate features multiple times
        features1 = predictor._build_team_features('LAL', 'BOS', True, date.today())
        features2 = predictor._build_team_features('LAL', 'BOS', True, date.today())
        
        # Should have same structure
        self.assertEqual(set(features1.keys()), set(features2.keys()))
        
        # is_home should be consistent
        self.assertEqual(features1['is_home'], features2['is_home'])
    
    def test_prediction_data_types(self):
        """Test that predictions have correct data types."""
        # This would test actual prediction output if we had a real model
        sample_prediction = {
            'game_id': 'test_game',
            'home_prob': 0.65,
            'away_prob': 0.35,
            'confidence': 0.65,
            'best_bet_edge': 0.05
        }
        
        # Test data type expectations
        self.assertIsInstance(sample_prediction['home_prob'], (int, float))
        self.assertIsInstance(sample_prediction['away_prob'], (int, float))
        self.assertIsInstance(sample_prediction['confidence'], (int, float))
        
        # Probabilities should sum to approximately 1
        prob_sum = sample_prediction['home_prob'] + sample_prediction['away_prob']
        self.assertAlmostEqual(prob_sum, 1.0, places=2)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
