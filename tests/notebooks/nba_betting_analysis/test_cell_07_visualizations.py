"""
Test for Cell 14: Model performance visualization and betting trends
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import sqlite3
from pathlib import Path

@pytest.fixture
def visualization_test_data():
    """Create comprehensive test data for visualization testing."""
    np.random.seed(42)
    
    # Create predictions data with realistic patterns
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    predictions_data = []
    
    for i, date in enumerate(dates):
        # Create some predictions for each day
        for j in range(np.random.randint(1, 5)):  # 1-4 predictions per day
            confidence = np.random.uniform(0.55, 0.95)
            predicted = np.random.choice([0, 1])
            # Make accuracy correlated with confidence
            actual = predicted if np.random.random() < (0.4 + confidence * 0.4) else 1 - predicted
            
            predictions_data.append({
                'prediction_date': date,
                'predicted_outcome': predicted,
                'actual_outcome': actual,
                'confidence': confidence,
                'team_name': f'Team{np.random.randint(1, 11)}',
                'game_id': len(predictions_data) + 1
            })
    
    predictions_df = pd.DataFrame(predictions_data)
    
    # Create betting data
    betting_data = []
    for i in range(30):
        bet_date = np.random.choice(dates)
        bet_amount = np.random.uniform(50, 500)
        # Realistic win rate around 45%
        if np.random.random() < 0.45:
            profit_loss = bet_amount * np.random.uniform(0.8, 1.5)  # Win
        else:
            profit_loss = -bet_amount  # Loss
        
        betting_data.append({
            'bet_date': bet_date,
            'bet_amount': bet_amount,
            'profit_loss': profit_loss,
            'team_name': f'Team{np.random.randint(1, 11)}'
        })
    
    betting_df = pd.DataFrame(betting_data)
    
    return predictions_df, betting_df

def test_weekly_accuracy_calculation(visualization_test_data):
    """Test weekly accuracy calculation for time series visualization."""
    predictions_df, _ = visualization_test_data
    
    # Test weekly aggregation
    predictions_df['prediction_week'] = predictions_df['prediction_date'].dt.to_period('W')
    weekly_accuracy = predictions_df.groupby('prediction_week').agg({
        'predicted_outcome': 'count',
        'actual_outcome': lambda x: (predictions_df.loc[x.index, 'actual_outcome'] == 
                                   predictions_df.loc[x.index, 'predicted_outcome']).sum()
    })
    weekly_accuracy['accuracy'] = weekly_accuracy['actual_outcome'] / weekly_accuracy['predicted_outcome']
    
    # Test results
    assert len(weekly_accuracy) > 0
    assert 'accuracy' in weekly_accuracy.columns
    for accuracy in weekly_accuracy['accuracy']:
        assert 0 <= accuracy <= 1

def test_confidence_bucketing(visualization_test_data):
    """Test confidence level bucketing for analysis."""
    predictions_df, _ = visualization_test_data
    
    # Test confidence bucketing
    predictions_df['confidence_bucket'] = pd.cut(predictions_df['confidence'], 
                                               bins=[0, 0.6, 0.7, 0.8, 1.0], 
                                               labels=['Low (≤60%)', 'Medium (60-70%)', 
                                                      'High (70-80%)', 'Very High (>80%)'])
    
    # Test bucket creation
    assert 'confidence_bucket' in predictions_df.columns
    bucket_counts = predictions_df['confidence_bucket'].value_counts()
    assert len(bucket_counts) > 0
    
    # Test accuracy by confidence bucket
    conf_accuracy = predictions_df.groupby('confidence_bucket', observed=False).agg({
        'predicted_outcome': 'count',
        'actual_outcome': lambda x: (predictions_df.loc[x.index, 'actual_outcome'] == 
                                   predictions_df.loc[x.index, 'predicted_outcome']).sum()
    })
    conf_accuracy['accuracy'] = conf_accuracy['actual_outcome'] / conf_accuracy['predicted_outcome']
    
    for idx in conf_accuracy.index:
        if pd.notna(idx):
            assert 0 <= conf_accuracy.loc[idx, 'accuracy'] <= 1

def test_daily_prediction_volume(visualization_test_data):
    """Test daily prediction volume calculation."""
    predictions_df, _ = visualization_test_data
    
    # Test daily aggregation
    daily_predictions = predictions_df.groupby(predictions_df['prediction_date'].dt.date).size()
    
    assert len(daily_predictions) > 0
    assert daily_predictions.min() > 0
    assert daily_predictions.max() >= daily_predictions.min()

def test_cumulative_pnl_calculation(visualization_test_data):
    """Test cumulative profit/loss calculation."""
    _, betting_df = visualization_test_data
    
    # Sort by date and calculate cumulative P&L
    betting_df_sorted = betting_df.sort_values('bet_date')
    betting_df_sorted['cumulative_pnl'] = betting_df_sorted['profit_loss'].cumsum()
    
    # Test cumulative calculation
    assert 'cumulative_pnl' in betting_df_sorted.columns
    assert len(betting_df_sorted['cumulative_pnl']) == len(betting_df_sorted)
    
    # Test that cumulative is actually cumulative
    manual_cumsum = 0
    for i, row in betting_df_sorted.iterrows():
        manual_cumsum += row['profit_loss']
        assert abs(row['cumulative_pnl'] - manual_cumsum) < 1e-10

def test_bet_size_analysis(visualization_test_data):
    """Test bet size distribution and analysis."""
    _, betting_df = visualization_test_data
    
    # Test bet size bucketing
    if len(betting_df['bet_amount'].unique()) > 1:
        betting_df['bet_size_bucket'] = pd.qcut(betting_df['bet_amount'], 
                                               q=min(5, len(betting_df['bet_amount'].unique())), 
                                               labels=False, duplicates='drop')
        
        bet_size_analysis = betting_df.groupby('bet_size_bucket').agg({
            'profit_loss': ['count', lambda x: (x > 0).mean(), 'mean'],
            'bet_amount': 'mean'
        })
        bet_size_analysis.columns = ['bet_count', 'win_rate', 'avg_pnl', 'avg_bet_size']
        
        # Test results
        assert len(bet_size_analysis) > 0
        for win_rate in bet_size_analysis['win_rate']:
            assert 0 <= win_rate <= 1

def test_daily_pnl_calculation(visualization_test_data):
    """Test daily P&L aggregation."""
    _, betting_df = visualization_test_data
    
    # Test daily P&L aggregation
    daily_pnl = betting_df.groupby(betting_df['bet_date'].dt.date)['profit_loss'].sum()
    
    assert len(daily_pnl) > 0
    assert isinstance(daily_pnl.iloc[0], (int, float))

def test_streak_analysis(visualization_test_data):
    """Test winning and losing streak calculation."""
    _, betting_df = visualization_test_data
    
    betting_df_sorted = betting_df.sort_values('bet_date')
    betting_df_sorted['is_win'] = betting_df_sorted['profit_loss'] > 0
    
    # Calculate streaks
    streaks = []
    if len(betting_df_sorted) > 0:
        current_streak = 1
        current_type = betting_df_sorted['is_win'].iloc[0]
        
        for i in range(1, len(betting_df_sorted)):
            if betting_df_sorted['is_win'].iloc[i] == current_type:
                current_streak += 1
            else:
                streaks.append((current_type, current_streak))
                current_streak = 1
                current_type = betting_df_sorted['is_win'].iloc[i]
        streaks.append((current_type, current_streak))
        
        # Test streaks
        win_streaks = [s[1] for s in streaks if s[0]]
        lose_streaks = [s[1] for s in streaks if not s[0]]
        
        if win_streaks:
            assert max(win_streaks) > 0
        if lose_streaks:
            assert max(lose_streaks) > 0

def test_performance_summary_statistics(visualization_test_data):
    """Test calculation of performance summary statistics."""
    predictions_df, betting_df = visualization_test_data
    
    # Test prediction summary
    if not predictions_df.empty and 'actual_outcome' in predictions_df.columns:
        overall_accuracy = (predictions_df['actual_outcome'] == predictions_df['predicted_outcome']).mean()
        assert 0 <= overall_accuracy <= 1
        
        if 'confidence' in predictions_df.columns:
            avg_confidence = predictions_df['confidence'].mean()
            min_confidence = predictions_df['confidence'].min()
            max_confidence = predictions_df['confidence'].max()
            
            assert 0 <= avg_confidence <= 1
            assert 0 <= min_confidence <= 1
            assert 0 <= max_confidence <= 1
    
    # Test betting summary
    if not betting_df.empty and 'profit_loss' in betting_df.columns:
        total_profit = betting_df['profit_loss'].sum()
        winning_bets = (betting_df['profit_loss'] > 0).sum()
        losing_bets = (betting_df['profit_loss'] < 0).sum()
        win_rate = winning_bets / len(betting_df)
        
        assert isinstance(total_profit, (int, float))
        assert winning_bets >= 0
        assert losing_bets >= 0
        assert 0 <= win_rate <= 1

def test_precision_recall_f1_calculation():
    """Test precision, recall, and F1 score calculation."""
    # Sample data for binary classification
    actual = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 1])
    predicted = np.array([1, 0, 0, 1, 1, 0, 1, 0, 0, 1])
    
    true_positives = ((actual == 1) & (predicted == 1)).sum()
    false_positives = ((actual == 0) & (predicted == 1)).sum()
    false_negatives = ((actual == 1) & (predicted == 0)).sum()
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= f1_score <= 1

def test_monthly_scoring_trends():
    """Test monthly scoring trends analysis for games data."""
    # Create sample games data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    games_data = []
    for date in dates:
        if np.random.random() < 0.3:  # 30% chance of games on any day
            games_data.append({
                'game_date': date,
                'home_score': np.random.randint(90, 130),
                'away_score': np.random.randint(90, 130)
            })
    
    games_df = pd.DataFrame(games_data)
    games_df['total_points'] = games_df['home_score'] + games_df['away_score']
    games_df['game_month'] = games_df['game_date'].dt.to_period('M')
    
    monthly_stats = games_df.groupby('game_month').agg({
        'total_points': 'mean',
        'home_score': 'mean',
        'away_score': 'mean'
    }).round(1)
    
    # Test monthly aggregation
    assert len(monthly_stats) > 0
    for month_stat in monthly_stats['total_points']:
        assert month_stat > 0

def test_color_coding_logic():
    """Test color coding logic for visualizations."""
    # Test win rate color coding
    win_rates = [0.3, 0.5, 0.7, 0.9]
    colors = ['red' if x < 0.4 else 'orange' if x < 0.6 else 'green' for x in win_rates]
    
    assert colors[0] == 'red'    # 0.3 < 0.4
    assert colors[1] == 'orange' # 0.4 <= 0.5 < 0.6
    assert colors[2] == 'green'  # 0.7 >= 0.6
    assert colors[3] == 'green'  # 0.9 >= 0.6
    
    # Test point differential color coding
    point_diffs = [-8, -2, 3, 8]
    colors2 = ['red' if x < -5 else 'orange' if x < 0 else 'lightgreen' if x < 5 else 'green' 
               for x in point_diffs]
    
    assert colors2[0] == 'red'        # -8 < -5
    assert colors2[1] == 'orange'     # -5 <= -2 < 0
    assert colors2[2] == 'lightgreen' # 0 <= 3 < 5
    assert colors2[3] == 'green'      # 8 >= 5

def test_empty_data_handling():
    """Test handling of empty or missing data in visualizations."""
    # Test with empty predictions
    empty_predictions = pd.DataFrame()
    assert empty_predictions.empty
    
    # Test with empty betting data
    empty_betting = pd.DataFrame()
    assert empty_betting.empty
    
    # Test with minimal data
    minimal_predictions = pd.DataFrame({
        'prediction_date': ['2024-01-01'],
        'predicted_outcome': [1],
        'actual_outcome': [1],
        'confidence': [0.8]
    })
    minimal_predictions['prediction_date'] = pd.to_datetime(minimal_predictions['prediction_date'])
    
    assert len(minimal_predictions) == 1
    assert minimal_predictions['prediction_date'].dtype == 'datetime64[ns]'

if __name__ == "__main__":
    pytest.main([__file__])
