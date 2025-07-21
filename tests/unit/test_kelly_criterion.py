"""
Unit Tests for Kelly Criterion Implementation

Tests for the betting strategy and bankroll management functions.
"""

import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from stake.kelly_criterion import KellyCriterion, calculate_daily_bets


class TestKellyCriterion(unittest.TestCase):
    """Test cases for Kelly Criterion calculations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.kelly = KellyCriterion({
            'kelly_fraction': 0.5,
            'min_edge': 0.02,
            'max_bet_size': 0.1
        })
        self.bankroll = 10000
    
    def test_kelly_fraction_calculation(self):
        """Test basic Kelly fraction calculation."""
        # Test case: 60% win prob, 2.0 odds
        p_win = 0.6
        odds = 2.0
        kelly_frac = self.kelly.calculate_kelly_fraction(p_win, odds)
        
        # Expected: (2.0-1)*0.6 - 0.4 / (2.0-1) = 0.2
        expected = 0.2
        self.assertAlmostEqual(kelly_frac, expected, places=3)
    
    def test_fractional_kelly(self):
        """Test fractional Kelly calculation."""
        p_win = 0.6
        odds = 2.0
        stake = self.kelly.fractional_kelly(p_win, odds, self.bankroll)
        
        # Expected: 0.2 * 0.5 * 10000 = 1000
        expected = 1000
        self.assertAlmostEqual(stake, expected, places=0)
    
    def test_max_bet_constraint(self):
        """Test maximum bet size constraint."""
        # Very high edge case
        p_win = 0.9
        odds = 2.0
        stake = self.kelly.fractional_kelly(p_win, odds, self.bankroll)
        
        # Should be capped at 10% of bankroll = 1000
        max_allowed = self.bankroll * self.kelly.max_bet_size
        self.assertLessEqual(stake, max_allowed)
    
    def test_negative_edge_handling(self):
        """Test that negative edges return 0 stake."""
        p_win = 0.4  # Low probability
        odds = 1.5   # Low odds
        stake = self.kelly.fractional_kelly(p_win, odds, self.bankroll)
        
        self.assertEqual(stake, 0.0)
    
    def test_bet_size_calculation_with_dataframe(self):
        """Test bet size calculation with DataFrame input."""
        sample_data = {
            'best_bet_prob': [0.65, 0.55, 0.45],
            'best_bet_odds': [1.8, 2.2, 2.0],
            'best_bet_edge': [0.08, 0.01, -0.05]
        }
        predictions_df = pd.DataFrame(sample_data)
        
        betting_results = self.kelly.calculate_bet_sizes(predictions_df, self.bankroll)
        
        # Should have 1 recommended bet (edge >= 0.02)
        recommended_bets = betting_results['recommended_bet'].sum()
        self.assertEqual(recommended_bets, 1)
        
        # Check that negative edge bet is not recommended
        negative_edge_row = betting_results[betting_results['best_bet_edge'] < 0]
        self.assertFalse(negative_edge_row['recommended_bet'].any())
    
    def test_simulation_outcomes(self):
        """Test betting outcome simulation."""
        sample_data = {
            'best_bet_prob': [0.65],
            'best_bet_odds': [1.8],
            'best_bet_edge': [0.08]
        }
        predictions_df = pd.DataFrame(sample_data)
        
        betting_results = self.kelly.calculate_bet_sizes(predictions_df, self.bankroll)
        simulation = self.kelly.simulate_betting_outcomes(betting_results, num_simulations=100)
        
        # Check simulation results structure
        required_keys = ['mean_return', 'std_return', 'prob_profit', 'percentile_5', 'percentile_95']
        for key in required_keys:
            self.assertIn(key, simulation)
        
        # Probability of profit should be reasonable for positive edge
        self.assertGreater(simulation['prob_profit'], 0.4)
        self.assertLess(simulation['prob_profit'], 0.9)


class TestDailyBetsCalculation(unittest.TestCase):
    """Test cases for daily betting calculations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_predictions = pd.DataFrame({
            'game_id': ['game1', 'game2', 'game3'],
            'home_team': ['LAL', 'GSW', 'BOS'],
            'away_team': ['BOS', 'PHX', 'MIA'],
            'best_bet_prob': [0.65, 0.55, 0.45],
            'best_bet_odds': [1.8, 2.2, 2.0],
            'best_bet_edge': [0.08, 0.01, -0.05],
            'best_bet_team': ['LAL', 'GSW', 'BOS']
        })
        self.bankroll = 10000
    
    def test_calculate_daily_bets(self):
        """Test complete daily betting calculation."""
        betting_recommendations, simulation_results = calculate_daily_bets(
            self.sample_predictions, self.bankroll
        )
        
        # Should return DataFrames/dicts
        self.assertIsInstance(betting_recommendations, pd.DataFrame)
        self.assertIsInstance(simulation_results, dict)
        
        # Should have same number of rows as input
        self.assertEqual(len(betting_recommendations), len(self.sample_predictions))
        
        # Should have recommended bet columns
        required_columns = ['recommended_bet', 'stake_amount', 'expected_value']
        for col in required_columns:
            self.assertIn(col, betting_recommendations.columns)
    
    def test_empty_predictions_handling(self):
        """Test handling of empty predictions DataFrame."""
        empty_df = pd.DataFrame()
        
        with self.assertRaises(Exception):
            calculate_daily_bets(empty_df, self.bankroll)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
