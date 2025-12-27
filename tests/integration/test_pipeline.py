"""
Integration Tests for NBA Betting Model

End-to-end tests for the complete betting pipeline.
"""

import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import date
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from predict.daily_report import generate_daily_report
try:
    from models.performance_tracker import PerformanceTracker
except ImportError:
    # Skip performance tracker tests if module not available
    PerformanceTracker = None
from stake.kelly_criterion import calculate_daily_bets


class TestIntegrationPipeline(unittest.TestCase):
    """Integration tests for the complete betting pipeline."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.bankroll = 10000
        
        # Sample prediction data
        self.sample_predictions = pd.DataFrame({
            'game_id': ['game1', 'game2', 'game3'],
            'game_date': [date.today(), date.today(), date.today()],
            'home_team': ['LAL', 'GSW', 'BOS'],
            'away_team': ['BOS', 'PHX', 'MIA'],
            'home_prob': [0.65, 0.55, 0.45],
            'away_prob': [0.35, 0.45, 0.55],
            'predicted_winner': ['LAL', 'GSW', 'MIA'],
            'confidence': [0.65, 0.55, 0.55],
            'home_odds': [1.8, 2.2, 2.0],
            'away_odds': [2.0, 1.75, 1.85],
            'home_implied_prob': [0.556, 0.455, 0.5],
            'away_implied_prob': [0.5, 0.571, 0.541],
            'home_edge': [0.094, 0.095, -0.05],
            'away_edge': [-0.15, -0.121, 0.009],
            'best_bet_team': ['LAL', 'GSW', 'MIA'],
            'best_bet_edge': [0.094, 0.095, 0.009],
            'best_bet_prob': [0.65, 0.55, 0.55],
            'best_bet_odds': [1.8, 2.2, 1.85]
        })
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_end_to_end_betting_calculation(self):
        """Test complete betting calculation pipeline."""
        # Calculate betting recommendations
        betting_recommendations, simulation_results = calculate_daily_bets(
            self.sample_predictions, self.bankroll
        )
        
        # Verify structure
        self.assertIsInstance(betting_recommendations, pd.DataFrame)
        self.assertIsInstance(simulation_results, dict)
        
        # Check that high-edge bets are recommended
        high_edge_bets = betting_recommendations[betting_recommendations['best_bet_edge'] > 0.05]
        self.assertTrue(high_edge_bets['recommended_bet'].any())
        
        # Check simulation results
        required_sim_keys = ['mean_return', 'std_return', 'prob_profit']
        for key in required_sim_keys:
            self.assertIn(key, simulation_results)
    
    @patch('predict.daily_report.predict_daily_games')
    @patch('stake.kelly_criterion.calculate_daily_bets')
    def test_daily_report_generation(self, mock_calculate_bets, mock_predict_games):
        """Test daily report generation."""
        # Mock the dependencies
        mock_predict_games.return_value = self.sample_predictions
        
        mock_betting_recommendations = self.sample_predictions.copy()
        mock_betting_recommendations['recommended_bet'] = [True, True, False]
        mock_betting_recommendations['stake_amount'] = [500, 300, 0]
        mock_betting_recommendations['expected_value'] = [50, 30, 0]
        
        mock_simulation = {
            'mean_return': 25.0,
            'std_return': 150.0,
            'prob_profit': 0.6
        }
        
        mock_calculate_bets.return_value = (mock_betting_recommendations, mock_simulation)
        
        # Generate report
        report = generate_daily_report(date.today(), self.bankroll)
        
        # Verify report structure
        self.assertIsInstance(report, dict)
        required_keys = ['date', 'bankroll', 'total_games', 'recommended_bets']
        for key in required_keys:
            self.assertIn(key, report)
        
        # Check values
        self.assertEqual(report['total_games'], 3)
        self.assertEqual(report['recommended_bets'], 2)
    
    def test_performance_tracker_integration(self):
        """Test performance tracking integration."""
        if PerformanceTracker is None:
            self.skipTest("PerformanceTracker module not available")
            
        # Create temporary tracker
        temp_db = self.test_dir / "test_performance.db"
        tracker = PerformanceTracker(temp_db)
        
        # Record predictions
        tracker.record_predictions(self.sample_predictions)
        
        # Simulate some results
        game_results = [
            {'game_id': 'game1', 'winner': 'LAL', 'bet_result': 'win', 'bet_payout': 900},
            {'game_id': 'game2', 'winner': 'PHX', 'bet_result': 'loss', 'bet_payout': 0},
            {'game_id': 'game3', 'winner': 'MIA', 'bet_result': None, 'bet_payout': 0}
        ]
        
        tracker.update_actual_results(game_results)
        
        # Calculate daily performance
        daily_perf = tracker.calculate_daily_performance(date.today())
        
        self.assertIsNotNone(daily_perf)
        self.assertEqual(daily_perf.total_games, 3)
        self.assertEqual(daily_perf.correct_predictions, 2)  # LAL and MIA won
    
    def test_bankroll_management_integration(self):
        """Test that bankroll management works across components."""
        different_bankrolls = [5000, 10000, 20000]
        
        for bankroll in different_bankrolls:
            betting_recommendations, _ = calculate_daily_bets(
                self.sample_predictions, bankroll
            )
            
            total_stake = betting_recommendations['stake_amount'].sum()
            
            # Total stake should never exceed bankroll
            self.assertLessEqual(total_stake, bankroll)
            
            # Total stake should be reasonable (not 0, not too high)
            if betting_recommendations['recommended_bet'].any():
                self.assertGreater(total_stake, 0)
                self.assertLess(total_stake, bankroll * 0.5)  # Max 50% of bankroll
    
    def test_edge_calculation_consistency(self):
        """Test that edge calculations are consistent across components."""
        # Test with known odds and probabilities
        test_cases = [
            {'prob': 0.6, 'odds': 1.8, 'expected_edge': 0.044},  # 60% vs 55.6% implied
            {'prob': 0.5, 'odds': 2.0, 'expected_edge': 0.0},    # 50% vs 50% implied (fair)
            {'prob': 0.4, 'odds': 2.0, 'expected_edge': -0.1}    # 40% vs 50% implied (negative)
        ]
        
        for case in test_cases:
            implied_prob = 1 / case['odds']
            calculated_edge = case['prob'] - implied_prob
            
            # Should match expected edge within tolerance
            self.assertAlmostEqual(calculated_edge, case['expected_edge'], places=2)
    
    def test_data_flow_consistency(self):
        """Test that data flows consistently through the pipeline."""
        # Start with predictions
        predictions = self.sample_predictions.copy()
        
        # Calculate betting recommendations
        betting_recommendations, simulation_results = calculate_daily_bets(
            predictions, self.bankroll
        )
        
        # Check data consistency
        self.assertEqual(len(predictions), len(betting_recommendations))
        
        # Game IDs should be the same set (order may change due to sorting)
        orig_game_ids = set(predictions['game_id'])
        rec_game_ids = set(betting_recommendations['game_id'])
        self.assertEqual(orig_game_ids, rec_game_ids)
        
        # Teams should be the same set (order may change due to sorting)
        orig_home_teams = set(predictions['home_team'])
        rec_home_teams = set(betting_recommendations['home_team'])
        self.assertEqual(orig_home_teams, rec_home_teams)


class TestErrorHandling(unittest.TestCase):
    """Test error handling in various scenarios."""
    
    def test_empty_predictions_handling(self):
        """Test handling of empty predictions."""
        empty_predictions = pd.DataFrame()
        
        # Should handle gracefully
        with self.assertRaises(Exception):
            calculate_daily_bets(empty_predictions, 10000)
    
    def test_malformed_predictions_handling(self):
        """Test handling of malformed prediction data."""
        # Missing required columns
        malformed_predictions = pd.DataFrame({
            'game_id': ['game1'],
            'home_team': ['LAL']
            # Missing other required columns
        })
        
        with self.assertRaises((KeyError, AttributeError)):
            calculate_daily_bets(malformed_predictions, 10000)
    
    def test_zero_bankroll_handling(self):
        """Test handling of zero or negative bankroll."""
        predictions = pd.DataFrame({
            'game_id': ['game1'],
            'best_bet_prob': [0.6],
            'best_bet_odds': [1.8],
            'best_bet_edge': [0.05]
        })
        
        # Zero bankroll should result in zero stakes
        betting_recommendations, _ = calculate_daily_bets(predictions, 0)
        self.assertEqual(betting_recommendations['stake_amount'].sum(), 0)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
