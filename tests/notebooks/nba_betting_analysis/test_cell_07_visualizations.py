"""
Test for Cell 7: Visualizations and Charts
"""
import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tempfile
import os

@pytest.fixture
def visualization_data():
    """Create sample data for visualization testing."""
    np.random.seed(42)
    
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(100)]
    
    data = []
    cumulative_profit = 0
    
    for i, date in enumerate(dates):
        win_prob = 0.5 + 0.1 * np.sin(i / 10)
        actual_win = np.random.random() < win_prob
        predicted_prob = win_prob + np.random.normal(0, 0.1)
        predicted_prob = np.clip(predicted_prob, 0.1, 0.9)
        
        bet_amount = np.random.uniform(10, 50) if predicted_prob > 0.55 else 0
        
        if actual_win and bet_amount > 0:
            profit = bet_amount * np.random.uniform(0.8, 1.2)
        elif bet_amount > 0:
            profit = -bet_amount
        else:
            profit = 0
            
        cumulative_profit += profit
        
        data.append({
            'game_date': date,
            'actual_outcome': int(actual_win),
            'predicted_prob': predicted_prob,
            'predicted_outcome': int(predicted_prob > 0.5),
            'bet_amount': bet_amount,
            'profit': profit,
            'cumulative_profit': cumulative_profit,
            'home_team': f'Team_{np.random.randint(1, 31):02d}',
            'away_team': f'Team_{np.random.randint(1, 31):02d}',
            'confidence': abs(predicted_prob - 0.5),
            'month': date.month,
            'week': date.isocalendar()[1]
        })
    
    return pd.DataFrame(data)

@pytest.fixture
def team_performance_data():
    """Create team performance data for visualization testing."""
    teams = ['Lakers', 'Warriors', 'Celtics', 'Heat', 'Nuggets', 'Suns', 'Knicks', 'Bulls']
    
    data = []
    for team in teams:
        data.append({
            'team_name': team,
            'win_rate': np.random.uniform(0.3, 0.8),
            'avg_pts_scored': np.random.normal(110, 8),
            'avg_pts_allowed': np.random.normal(108, 8),
            'point_differential': np.random.normal(0, 8),
            'recent_form': np.random.uniform(0.2, 0.9),
            'home_record': np.random.uniform(0.4, 0.9),
            'away_record': np.random.uniform(0.2, 0.7)
        })
    
    df = pd.DataFrame(data)
    df['point_differential'] = df['avg_pts_scored'] - df['avg_pts_allowed']
    return df

def test_profit_timeline_data_preparation(visualization_data):
    """Test data preparation for profit timeline visualization."""
    viz_df = visualization_data.copy()
    
    viz_df['game_date'] = pd.to_datetime(viz_df['game_date'])
    viz_df = viz_df.sort_values('game_date')
    
    assert 'cumulative_profit' in viz_df.columns
    assert 'game_date' in viz_df.columns
    assert len(viz_df) > 0
    
    if len(viz_df) > 1:
        total_change = viz_df['cumulative_profit'].iloc[-1] - viz_df['cumulative_profit'].iloc[0]
        assert isinstance(total_change, (int, float))

def test_accuracy_trend_data_preparation(visualization_data):
    """Test data preparation for accuracy trend visualization."""
    viz_df = visualization_data.copy()
    
    window_size = 20
    viz_df['correct_prediction'] = (viz_df['predicted_outcome'] == viz_df['actual_outcome']).astype(int)
    
    if len(viz_df) >= window_size:
        viz_df['rolling_accuracy'] = viz_df['correct_prediction'].rolling(
            window=window_size, min_periods=1
        ).mean()
        
        assert 'rolling_accuracy' in viz_df.columns
        rolling_data = viz_df['rolling_accuracy'].dropna()
        assert len(rolling_data) > 0
        assert (rolling_data >= 0).all()
        assert (rolling_data <= 1).all()

def test_monthly_performance_aggregation(visualization_data):
    """Test monthly performance aggregation for bar charts."""
    viz_df = visualization_data.copy()
    
    monthly_stats = viz_df.groupby('month').agg({
        'profit': ['sum', 'count', 'mean'],
        'bet_amount': 'sum',
        'actual_outcome': 'mean',
        'predicted_prob': 'mean'
    }).round(2)
    
    assert len(monthly_stats) <= 12
    assert len(monthly_stats) >= 1
    
    for month in monthly_stats.index:
        assert 1 <= month <= 12
    
    assert monthly_stats.columns.nlevels == 2

def test_confidence_distribution_analysis(visualization_data):
    """Test confidence distribution analysis for histograms."""
    viz_df = visualization_data.copy()
    
    confidence_stats = {
        'mean_confidence': viz_df['confidence'].mean(),
        'median_confidence': viz_df['confidence'].median(),
        'std_confidence': viz_df['confidence'].std(),
        'min_confidence': viz_df['confidence'].min(),
        'max_confidence': viz_df['confidence'].max()
    }
    
    # Test confidence statistics
    assert 0 <= confidence_stats['mean_confidence'] <= 0.5
    assert 0 <= confidence_stats['median_confidence'] <= 0.5
    assert confidence_stats['std_confidence'] >= 0
    assert 0 <= confidence_stats['min_confidence'] <= confidence_stats['max_confidence'] <= 0.5

def test_correlation_heatmap_data(visualization_data):
    """Test data preparation for correlation heatmap."""
    viz_df = visualization_data.copy()
    
    # Select numeric columns for correlation
    numeric_cols = ['predicted_prob', 'bet_amount', 'profit', 'confidence', 'actual_outcome']
    correlation_data = viz_df[numeric_cols]
    
    # Calculate correlation matrix
    corr_matrix = correlation_data.corr()
    
    # Test correlation matrix
    assert corr_matrix.shape[0] == corr_matrix.shape[1]  # Square matrix
    assert corr_matrix.shape[0] == len(numeric_cols)
    
    # Test diagonal is 1.0 (perfect self-correlation)
    for i in range(len(corr_matrix)):
        assert abs(corr_matrix.iloc[i, i] - 1.0) < 1e-10
    
    # Test matrix is symmetric
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            assert abs(corr_matrix.iloc[i, j] - corr_matrix.iloc[j, i]) < 1e-10

def test_team_performance_chart_data(team_performance_data):
    """Test team performance data for bar charts."""
    team_df = team_performance_data.copy()
    
    # Sort teams by win rate for visualization
    team_df_sorted = team_df.sort_values('win_rate')
    
    # Test sorting
    assert len(team_df_sorted) == len(team_df)
    if len(team_df_sorted) > 1:
        assert team_df_sorted['win_rate'].iloc[0] <= team_df_sorted['win_rate'].iloc[-1]
    
    # Test color coding logic
    colors = ['red' if x < 0.4 else 'orange' if x < 0.6 else 'green' 
              for x in team_df_sorted['win_rate']]
    assert len(colors) == len(team_df_sorted)
    
    # Test that all colors are valid
    valid_colors = {'red', 'orange', 'green'}
    assert all(color in valid_colors for color in colors)

def test_scatter_plot_data_preparation(visualization_data):
    """Test data preparation for scatter plots."""
    viz_df = visualization_data.copy()
    
    viz_df['correct'] = (viz_df['predicted_outcome'] == viz_df['actual_outcome']).astype(int)
    
    viz_df['confidence_bin'] = pd.cut(viz_df['confidence'], bins=5, labels=False)
    
    scatter_data = viz_df.groupby('confidence_bin').agg({
        'confidence': 'mean',
        'correct': 'mean',
        'game_date': 'count'
    }).reset_index()
    
    assert len(scatter_data) <= 5
    assert len(scatter_data) >= 1
    
    assert (scatter_data['correct'] >= 0).all()
    assert (scatter_data['correct'] <= 1).all()

def test_time_series_resampling(visualization_data):
    """Test time series resampling for different time periods."""
    viz_df = visualization_data.copy()
    viz_df['game_date'] = pd.to_datetime(viz_df['game_date'])
    viz_df = viz_df.set_index('game_date')
    
    weekly_data = viz_df.resample('W').agg({
        'profit': 'sum',
        'bet_amount': 'sum',
        'actual_outcome': 'mean'
    })
    
    assert len(weekly_data) <= len(viz_df)
    assert len(weekly_data) >= 1
    
    expected_columns = {'profit', 'bet_amount', 'actual_outcome'}
    assert set(weekly_data.columns) == expected_columns

def test_performance_metrics_visualization_data(visualization_data):
    """Test data preparation for performance metrics visualization."""
    viz_df = visualization_data.copy()
    
    total_games = len(viz_df)
    total_bets = len(viz_df[viz_df['bet_amount'] > 0])
    win_rate = viz_df['actual_outcome'].mean()
    prediction_accuracy = (viz_df['predicted_outcome'] == viz_df['actual_outcome']).mean()
    
    total_profit = viz_df['profit'].sum()
    total_wagered = viz_df['bet_amount'].sum()
    roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0
    
    metrics = {
        'Total Games': total_games,
        'Total Bets': total_bets,
        'Win Rate': f"{win_rate:.1%}",
        'Prediction Accuracy': f"{prediction_accuracy:.1%}",
        'Total Profit': f"${total_profit:.2f}",
        'ROI': f"{roi:.1f}%"
    }
    
    # Test metrics calculation
    assert metrics['Total Games'] > 0
    assert metrics['Total Bets'] >= 0
    assert 0 <= win_rate <= 1
    assert 0 <= prediction_accuracy <= 1

def test_visualization_edge_cases(visualization_data):
    """Test edge cases in visualization data preparation."""
    viz_df = visualization_data.copy()
    
    # Test with no bets
    no_bets_df = viz_df.copy()
    no_bets_df['bet_amount'] = 0
    no_bets_df['profit'] = 0
    
    total_wagered = no_bets_df['bet_amount'].sum()
    assert total_wagered == 0
    
    # Test with single data point
    single_point = viz_df.head(1)
    assert len(single_point) == 1
    
    monthly_single = single_point.groupby('month').agg({'profit': 'sum'})
    assert len(monthly_single) == 1
    
    # Test with all wins
    all_wins = viz_df.copy()
    all_wins['actual_outcome'] = 1
    all_wins['predicted_outcome'] = 1
    
    accuracy = (all_wins['predicted_outcome'] == all_wins['actual_outcome']).mean()
    assert accuracy == 1.0

def test_color_palette_generation():
    """Test color palette generation for visualizations."""
    colors_categorical = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    colors_sequential = ['#fee5d9', '#fcbba1', '#fc9272', '#fb6a4a', '#de2d26']
    colors_diverging = ['#ca0020', '#f4a582', '#ffffff', '#92c5de', '#0571b0']
    
    assert len(colors_categorical) == 5
    assert len(colors_sequential) == 5
    assert len(colors_diverging) == 5
    
    for color in colors_categorical + colors_sequential + colors_diverging:
        assert color.startswith('#')
        assert len(color) == 7

def test_chart_data_validation():
    """Test validation of chart data before plotting."""
    def validate_chart_data(data, required_columns):
        """Validate data for chart creation."""
        if data is None or len(data) == 0:
            return False, "No data provided"
        
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            return False, f"Missing columns: {missing_cols}"
        
        return True, "Valid"
    
    valid_data = pd.DataFrame({
        'x': [1, 2, 3],
        'y': [4, 5, 6]
    })
    
    is_valid, message = validate_chart_data(valid_data, ['x', 'y'])
    assert is_valid == True
    assert message == "Valid"
    
    is_valid, message = validate_chart_data(valid_data, ['x', 'y', 'z'])
    assert is_valid == False
    assert "Missing columns" in message
    
    empty_data = pd.DataFrame()
    is_valid, message = validate_chart_data(empty_data, ['x', 'y'])
    assert is_valid == False
    assert "No data provided" in message

if __name__ == "__main__":
    pytest.main([__file__])
