"""
Kelly Criterion Implementation for NBA Betting

This module provides betting strategy functions using the Kelly criterion
to determine optimal bet sizes based on edge and bankroll management.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class KellyCriterion:
    """Kelly criterion calculator for NBA betting."""

    def __init__(self, config: Dict = None):
        self.config = config or self._load_default_config()
        self.min_edge = self.config.get('min_edge', 0.02)
        self.max_bet_size = self.config.get('max_bet_size', 0.1)
        self.kelly_fraction = self.config.get('kelly_fraction', 0.5)

    def _load_default_config(self) -> Dict:
        """Load default betting configuration."""
        project_root = Path(__file__).parent.parent.parent
        config_file = project_root / "configs" / "model.yml"

        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                return config.get('betting', {})
        except FileNotFoundError:
            return {
                'kelly_fraction': 0.5,
                'min_edge': 0.02,
                'max_bet_size': 0.1
            }

    def calculate_kelly_fraction(self, p_win: float, odds: float) -> float:
        """Calculate the Kelly fraction for a single bet."""
        if p_win <= 0 or p_win >= 1 or odds <= 1:
            return 0.0

        b = odds - 1  # Net odds (profit per unit bet)
        q = 1 - p_win  # Probability of losing

        # Kelly formula: f = (bp - q) / b
        kelly_fraction = (b * p_win - q) / b

        return max(0, kelly_fraction)

    def fractional_kelly(self, p_win: float, odds: float, bankroll: float,
                        kelly_fraction: Optional[float] = None) -> float:
        """Calculate stake size using fractional Kelly."""
        if kelly_fraction is None:
            kelly_fraction = self.kelly_fraction

        full_kelly = self.calculate_kelly_fraction(p_win, odds)
        fractional_kelly = full_kelly * kelly_fraction

        # Apply maximum bet size constraint
        max_stake = bankroll * self.max_bet_size
        optimal_stake = bankroll * fractional_kelly

        return min(optimal_stake, max_stake)

    def calculate_bet_sizes(self, predictions_df: pd.DataFrame, bankroll: float) -> pd.DataFrame:
        """Calculate bet sizes for multiple games."""
        logger.info(f"Calculating bet sizes for bankroll: ${bankroll:,.2f}")

        betting_df = predictions_df.copy()

        # Only consider bets with positive edge above minimum threshold
        valid_bets = betting_df['best_bet_edge'] >= self.min_edge

        betting_df['recommended_bet'] = False
        betting_df['stake_amount'] = 0.0
        betting_df['expected_value'] = 0.0
        betting_df['kelly_fraction'] = 0.0

        if not valid_bets.any():
            print("⚠️  No bets meet minimum edge requirement")
            return betting_df

        # Calculate stakes for valid bets
        for idx in betting_df[valid_bets].index:
            row = betting_df.loc[idx]

            p_win = row['best_bet_prob']
            odds = row['best_bet_odds']
            edge = row['best_bet_edge']

            # Calculate Kelly fraction and stake
            kelly_frac = self.calculate_kelly_fraction(p_win, odds)
            stake = self.fractional_kelly(p_win, odds, bankroll)

            # Calculate expected value
            expected_value = stake * (p_win * (odds - 1) - (1 - p_win))

            betting_df.loc[idx, 'recommended_bet'] = True
            betting_df.loc[idx, 'stake_amount'] = stake
            betting_df.loc[idx, 'expected_value'] = expected_value
            betting_df.loc[idx, 'kelly_fraction'] = kelly_frac

        # Sort by expected value (highest first)
        betting_df = betting_df.sort_values('expected_value', ascending=False)

        total_stake = betting_df['stake_amount'].sum()
        total_ev = betting_df['expected_value'].sum()
        num_bets = betting_df['recommended_bet'].sum()

        print(f"✓ Recommended {num_bets} bets")
        print(f"  - Total stake: ${total_stake:,.2f} ({total_stake/bankroll:.1%} of bankroll)")
        print(f"  - Total expected value: ${total_ev:,.2f}")
        print(f"  - Average edge: {betting_df[betting_df['recommended_bet']]['best_bet_edge'].mean():.1%}")

        return betting_df

    def simulate_betting_outcomes(self, betting_df: pd.DataFrame,
                                 num_simulations: int = 1000) -> Dict:
        """Simulate betting outcomes to assess risk with optimized performance."""
        recommended_bets = betting_df[betting_df['recommended_bet']].copy()

        if len(recommended_bets) == 0:
            return {'mean_return': 0, 'std_return': 0, 'prob_profit': 0}

        # Extract bet data for vectorized operations
        stakes = recommended_bets['stake_amount'].values
        p_wins = recommended_bets['best_bet_prob'].values
        odds = recommended_bets['best_bet_odds'].values
        
        # Pre-calculate profit/loss amounts
        win_returns = stakes * (odds - 1)  # Profit on win
        loss_returns = -stakes              # Loss on loss
        
        # Vectorized simulation using numpy
        random_outcomes = np.random.random((num_simulations, len(stakes)))
        wins = random_outcomes < p_wins[np.newaxis, :]
        
        # Calculate returns for each simulation
        returns = np.where(wins, win_returns, loss_returns)
        simulations = returns.sum(axis=1)

        simulations = np.array(simulations)

        return {
            'mean_return': float(simulations.mean()),
            'std_return': float(simulations.std()),
            'prob_profit': float((simulations > 0).mean()),
            'percentile_5': float(np.percentile(simulations, 5)),
            'percentile_95': float(np.percentile(simulations, 95)),
            'max_loss': float(simulations.min()),
            'max_gain': float(simulations.max())
        }


def calculate_daily_bets(predictions_df: pd.DataFrame, bankroll: float,
                        config: Dict = None) -> Tuple[pd.DataFrame, Dict]:
    """Main function to calculate daily betting recommendations."""
    print("=" * 80)
    print("💰 DAILY BETTING ANALYSIS")
    print("=" * 80)

    kelly = KellyCriterion(config)

    # Calculate bet sizes
    betting_recommendations = kelly.calculate_bet_sizes(predictions_df, bankroll)

    # Simulate outcomes
    simulation_results = kelly.simulate_betting_outcomes(betting_recommendations)

    print("\n📊 SIMULATION RESULTS:")
    print(f"  - Expected return: ${simulation_results['mean_return']:,.2f}")
    print(f"  - Standard deviation: ${simulation_results['std_return']:,.2f}")
    print(f"  - Probability of profit: {simulation_results['prob_profit']:.1%}")
    print(f"  - 5th percentile: ${simulation_results['percentile_5']:,.2f}")
    print(f"  - 95th percentile: ${simulation_results['percentile_95']:,.2f}")

    return betting_recommendations, simulation_results


# Legacy function for backward compatibility
def fractional_kelly(p_win: float, odds: float, bankroll: float, k: float = 0.5) -> float:
    """Return stake size using fractional Kelly."""
    kelly = KellyCriterion({'kelly_fraction': k})
    return kelly.fractional_kelly(p_win, odds, bankroll)
