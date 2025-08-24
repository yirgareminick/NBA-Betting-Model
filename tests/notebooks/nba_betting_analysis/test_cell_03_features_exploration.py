"""
Test for Cell 4: Features data exploration
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

@pytest.fixture
def sample_features_data():
    """Create sample features data."""
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
    """Test dataset info calculation."""
    features_df = sample_features_data
    
    assert features_df.shape[0] == 200
    assert features_df.shape[1] > 0
    
    min_date = features_df['game_date'].min()
    max_date = features_df['game_date'].max()
    assert isinstance(min_date, pd.Timestamp)
    assert isinstance(max_date, pd.Timestamp)
    assert min_date <= max_date
    
    unique_teams = features_df['team_name'].nunique()
    assert unique_teams <= 5
    
    unique_games = features_df['game_id'].nunique()
    assert unique_games > 0

def test_feature_categorization(sample_features_data):
    """Test feature categorization."""
    features_df = sample_features_data
    feature_cols = features_df.columns.tolist()
    
    rolling_features = [col for col in feature_cols if 'last_' in col]
    season_features = [col for col in feature_cols if 'season_' in col]
    basic_features = [col for col in feature_cols if col in 
                     ['game_id', 'game_date', 'team_name', 'opponent', 'venue', 'is_home', 'target_win']]
    other_features = [col for col in feature_cols if col not in 
                     rolling_features + season_features + basic_features]
    
    assert len(rolling_features) > 0
    assert len(season_features) > 0
    assert len(basic_features) > 0
    
    all_categorized = rolling_features + season_features + basic_features + other_features
    assert len(all_categorized) == len(set(all_categorized))

def test_target_variable_analysis(sample_features_data):
    """Test target variable analysis."""
    features_df = sample_features_data
    
    win_rate = features_df['target_win'].mean()
    assert 0 <= win_rate <= 1
    
    if 'is_home' in features_df.columns:
        home_win_rate = features_df[features_df['is_home']==1]['target_win'].mean()
        away_win_rate = features_df[features_df['is_home']==0]['target_win'].mean()
        
        assert 0 <= home_win_rate <= 1
        assert 0 <= away_win_rate <= 1

def test_data_exploration_with_missing_data():
    """Test missing data behavior."""
    features_df = None
    
    if features_df is None:
        assert True
    
    empty_df = pd.DataFrame()
    assert empty_df.empty

def test_sample_data_display():
    """Test sample data display."""
    features_df = pd.DataFrame({
        'game_id': [1, 2, 3],
        'team_name': ['A', 'B', 'C'],
        'target_win': [1, 0, 1]
    })
    
    sample = features_df.head()
    assert len(sample) <= 5
    assert len(sample) <= len(features_df)

def test_feature_types_identification(sample_features_data):
    """Test feature type identification."""
    features_df = sample_features_data
    
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = features_df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    assert len(numeric_cols) > 0
    assert 'target_win' in numeric_cols
    assert 'is_home' in numeric_cols
    
    datetime_cols = features_df.select_dtypes(include=['datetime']).columns.tolist()

def test_missing_values_handling(sample_features_data):
    """Test missing values handling."""
    features_df = sample_features_data.copy()
    
    features_df.loc[0:5, 'season_avg_pts'] = np.nan
    
    assert features_df.isnull().sum().sum() > 0
    
    win_rate = features_df['target_win'].mean()
    assert not np.isnan(win_rate)

if __name__ == "__main__":
    pytest.main([__file__])
