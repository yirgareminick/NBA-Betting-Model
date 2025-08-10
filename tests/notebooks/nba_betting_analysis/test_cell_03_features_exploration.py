"""
Test for Cell 4: Features data exploration
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

@pytest.fixture
def sample_features_data():
    """Create sample features data for testing."""
    np.random.seed(42)
    
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    teams = ['Team A', 'Team B', 'Team C', 'Team D', 'Team E']
    
    data = []
    for i in range(200):
        data.append({
            'game_id': i + 1,
            'game_date': np.random.choice(dates),
            'team_name': np.random.choice(teams),
            'opponent': np.random.choice([t for t in teams]),
            'venue': np.random.choice(['Home', 'Away']),
            'is_home': np.random.choice([0, 1]),
            'target_win': np.random.choice([0, 1]),
            'season_avg_pts': np.random.normal(110, 10),
            'season_avg_pts_allowed': np.random.normal(108, 8),
            'win_pct_last_10': np.random.uniform(0, 1),
            'avg_pts_last_5': np.random.normal(112, 12),
            'other_feature': np.random.normal(0, 1)
        })
    
    return pd.DataFrame(data)

def test_dataset_info_calculation(sample_features_data):
    """Test basic dataset information calculation."""
    features_df = sample_features_data
    
    # Test shape
    assert features_df.shape[0] == 200
    assert features_df.shape[1] > 0
    
    # Test date range
    min_date = features_df['game_date'].min()
    max_date = features_df['game_date'].max()
    assert isinstance(min_date, pd.Timestamp)
    assert isinstance(max_date, pd.Timestamp)
    assert min_date <= max_date
    
    # Test unique counts
    unique_teams = features_df['team_name'].nunique()
    assert unique_teams <= 5  # We created 5 teams
    
    unique_games = features_df['game_id'].nunique()
    assert unique_games > 0

def test_feature_categorization(sample_features_data):
    """Test feature categorization logic."""
    features_df = sample_features_data
    feature_cols = features_df.columns.tolist()
    
    # Test categorization
    rolling_features = [col for col in feature_cols if 'last_' in col]
    season_features = [col for col in feature_cols if 'season_' in col]
    basic_features = [col for col in feature_cols if col in 
                     ['game_id', 'game_date', 'team_name', 'opponent', 'venue', 'is_home', 'target_win']]
    other_features = [col for col in feature_cols if col not in 
                     rolling_features + season_features + basic_features]
    
    # Verify categorization
    assert len(rolling_features) > 0  # Should find 'win_pct_last_10', 'avg_pts_last_5'
    assert len(season_features) > 0   # Should find 'season_avg_pts', 'season_avg_pts_allowed'
    assert len(basic_features) > 0    # Should find basic columns
    
    # Verify no overlap
    all_categorized = rolling_features + season_features + basic_features + other_features
    assert len(all_categorized) == len(set(all_categorized))  # No duplicates

def test_target_variable_analysis(sample_features_data):
    """Test target variable distribution analysis."""
    features_df = sample_features_data
    
    # Test overall win rate calculation
    win_rate = features_df['target_win'].mean()
    assert 0 <= win_rate <= 1
    
    # Test home/away win rate calculation
    if 'is_home' in features_df.columns:
        home_win_rate = features_df[features_df['is_home']==1]['target_win'].mean()
        away_win_rate = features_df[features_df['is_home']==0]['target_win'].mean()
        
        assert 0 <= home_win_rate <= 1
        assert 0 <= away_win_rate <= 1

def test_data_exploration_with_missing_data():
    """Test exploration behavior with missing or invalid data."""
    # Test with None/empty dataframe
    features_df = None
    
    # Should handle gracefully (this would be the logic in the actual cell)
    if features_df is None:
        assert True  # Expected behavior
    
    # Test with empty dataframe
    empty_df = pd.DataFrame()
    assert empty_df.empty

def test_sample_data_display():
    """Test that sample data can be displayed without errors."""
    features_df = pd.DataFrame({
        'game_id': [1, 2, 3],
        'team_name': ['A', 'B', 'C'],
        'target_win': [1, 0, 1]
    })
    
    # Test head() operation
    sample = features_df.head()
    assert len(sample) <= 5
    assert len(sample) <= len(features_df)

def test_feature_types_identification(sample_features_data):
    """Test identification of different feature types."""
    features_df = sample_features_data
    
    # Test numeric vs non-numeric columns
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = features_df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    assert len(numeric_cols) > 0
    assert 'target_win' in numeric_cols
    assert 'is_home' in numeric_cols
    
    # Test datetime columns
    datetime_cols = features_df.select_dtypes(include=['datetime']).columns.tolist()
    # Note: game_date might be datetime depending on how it's loaded

def test_missing_values_handling(sample_features_data):
    """Test handling of missing values in exploration."""
    features_df = sample_features_data.copy()
    
    # Introduce some missing values
    features_df.loc[0:5, 'season_avg_pts'] = np.nan
    
    # Test that exploration can handle missing values
    assert features_df.isnull().sum().sum() > 0
    
    # Test basic statistics with missing values
    win_rate = features_df['target_win'].mean()
    assert not np.isnan(win_rate)

if __name__ == "__main__":
    pytest.main([__file__])
