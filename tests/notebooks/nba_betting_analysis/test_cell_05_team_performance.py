"""
Test for Cell 5: Team Performance Analysis
"""
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def team_performance_data():
    """Create sample team performance data for testing."""
    np.random.seed(42)
    
    teams = ['Lakers', 'Warriors', 'Celtics', 'Heat', 'Nuggets', 'Suns']
    data = []
    
    for team in teams:
        # Generate multiple games per team
        for game in range(15):  # 15 games per team
            data.append({
                'team_name': team,
                'target_win': np.random.choice([0, 1], p=[0.45, 0.55]),  # Slight bias toward wins
                'season_win_pct': np.random.uniform(0.3, 0.8),
                'season_avg_pts': np.random.normal(110, 10),
                'season_avg_pts_allowed': np.random.normal(108, 8),
                'win_pct_last_10': np.random.uniform(0.2, 0.9),
                'is_home': np.random.choice([0, 1]),
                'game_id': len(data) + 1,
                'game_date': f'2024-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}'
            })
    
    return pd.DataFrame(data)

def test_team_aggregation_calculation(team_performance_data):
    """Test team-level aggregation calculations."""
    features_df = team_performance_data
    
    team_performance = features_df.groupby('team_name').agg({
        'target_win': ['count', 'mean'],
        'season_win_pct': 'mean',
        'season_avg_pts': 'mean',
        'season_avg_pts_allowed': 'mean',
        'win_pct_last_10': 'mean',
        'is_home': 'mean'
    }).round(3)
    
    # Test aggregation structure
    assert team_performance.shape[1] == 7  # 7 aggregated columns (target_win has count+mean)
    assert len(team_performance) == 6      # 6 teams
    
    # Test that all teams are present
    teams = ['Lakers', 'Warriors', 'Celtics', 'Heat', 'Nuggets', 'Suns']
    for team in teams:
        assert team in team_performance.index

def test_column_flattening_and_renaming(team_performance_data):
    """Test flattening of multi-level columns and renaming."""
    features_df = team_performance_data
    
    team_performance = features_df.groupby('team_name').agg({
        'target_win': ['count', 'mean'],
        'season_win_pct': 'mean',
        'season_avg_pts': 'mean',
        'season_avg_pts_allowed': 'mean',
        'win_pct_last_10': 'mean',
        'is_home': 'mean'
    }).round(3)
    
    # Flatten column names
    team_performance.columns = ['games_played', 'win_rate', 'season_win_pct', 
                               'avg_pts_scored', 'avg_pts_allowed', 'recent_win_pct', 'home_pct']
    team_performance = team_performance.reset_index()
    
    # Test column names
    expected_columns = ['team_name', 'games_played', 'win_rate', 'season_win_pct', 
                       'avg_pts_scored', 'avg_pts_allowed', 'recent_win_pct', 'home_pct']
    assert team_performance.columns.tolist() == expected_columns
    
    # Test data types and ranges
    assert team_performance['games_played'].min() > 0
    assert (team_performance['win_rate'] >= 0).all()
    assert (team_performance['win_rate'] <= 1).all()

def test_point_differential_calculation(team_performance_data):
    """Test point differential calculation."""
    features_df = team_performance_data
    
    team_performance = features_df.groupby('team_name').agg({
        'target_win': ['count', 'mean'],
        'season_win_pct': 'mean',
        'season_avg_pts': 'mean',
        'season_avg_pts_allowed': 'mean',
        'win_pct_last_10': 'mean',
        'is_home': 'mean'
    }).round(3)
    
    team_performance.columns = ['games_played', 'win_rate', 'season_win_pct', 
                               'avg_pts_scored', 'avg_pts_allowed', 'recent_win_pct', 'home_pct']
    team_performance = team_performance.reset_index()
    
    # Add point differential
    team_performance['point_differential'] = team_performance['avg_pts_scored'] - team_performance['avg_pts_allowed']
    
    # Test point differential calculation
    assert 'point_differential' in team_performance.columns
    
    # Test that point differential is calculated correctly
    for idx, row in team_performance.iterrows():
        expected_diff = row['avg_pts_scored'] - row['avg_pts_allowed']
        assert abs(row['point_differential'] - expected_diff) < 1e-10

def test_team_performance_insights(team_performance_data):
    """Test calculation of team performance insights."""
    features_df = team_performance_data
    
    team_perf = features_df.groupby('team_name').agg({
        'target_win': ['count', 'mean'],
        'season_win_pct': 'mean',
        'season_avg_pts': 'mean',
        'season_avg_pts_allowed': 'mean',
        'win_pct_last_10': 'mean',
        'is_home': 'mean'
    }).round(3)
    
    team_perf.columns = ['games_played', 'win_rate', 'season_win_pct', 
                        'avg_pts_scored', 'avg_pts_allowed', 'recent_win_pct', 'home_pct']
    team_perf = team_perf.reset_index()
    team_perf['point_differential'] = team_perf['avg_pts_scored'] - team_perf['avg_pts_allowed']
    
    # Test best/worst team identification
    best_team = team_perf.loc[team_perf['win_rate'].idxmax()]
    worst_team = team_perf.loc[team_perf['win_rate'].idxmin()]
    highest_scoring = team_perf.loc[team_perf['avg_pts_scored'].idxmax()]
    best_defense = team_perf.loc[team_perf['avg_pts_allowed'].idxmin()]
    
    assert best_team['win_rate'] >= worst_team['win_rate']
    assert highest_scoring['avg_pts_scored'] == team_perf['avg_pts_scored'].max()
    assert best_defense['avg_pts_allowed'] == team_perf['avg_pts_allowed'].min()

def test_form_analysis(team_performance_data):
    """Test analysis of team form (improving vs declining)."""
    features_df = team_performance_data
    
    team_perf = features_df.groupby('team_name').agg({
        'target_win': ['count', 'mean'],
        'season_win_pct': 'mean',
        'season_avg_pts': 'mean',
        'season_avg_pts_allowed': 'mean',
        'win_pct_last_10': 'mean',
        'is_home': 'mean'
    }).round(3)
    
    team_perf.columns = ['games_played', 'win_rate', 'season_win_pct', 
                        'avg_pts_scored', 'avg_pts_allowed', 'recent_win_pct', 'home_pct']
    team_perf = team_perf.reset_index()
    
    # Form analysis
    improving_teams = team_perf[team_perf['recent_win_pct'] > team_perf['season_win_pct'] + 0.1]
    declining_teams = team_perf[team_perf['recent_win_pct'] < team_perf['season_win_pct'] - 0.1]
    
    # Test form analysis
    assert len(improving_teams) >= 0
    assert len(declining_teams) >= 0
    assert len(improving_teams) + len(declining_teams) <= len(team_perf)
    
    # Test that improving teams have higher recent win pct
    for idx, team in improving_teams.iterrows():
        assert team['recent_win_pct'] > team['season_win_pct']
    
    # Test that declining teams have lower recent win pct
    for idx, team in declining_teams.iterrows():
        assert team['recent_win_pct'] < team['season_win_pct']

def test_visualization_data_preparation(team_performance_data):
    """Test data preparation for visualizations."""
    features_df = team_performance_data
    
    team_perf = features_df.groupby('team_name').agg({
        'target_win': ['count', 'mean'],
        'season_win_pct': 'mean',
        'season_avg_pts': 'mean',
        'season_avg_pts_allowed': 'mean',
        'win_pct_last_10': 'mean',
        'is_home': 'mean'
    }).round(3)
    
    team_perf.columns = ['games_played', 'win_rate', 'season_win_pct', 
                        'avg_pts_scored', 'avg_pts_allowed', 'recent_win_pct', 'home_pct']
    team_perf = team_perf.reset_index()
    team_perf['point_differential'] = team_perf['avg_pts_scored'] - team_perf['avg_pts_allowed']
    
    # Test sorting for visualizations
    team_perf_sorted = team_perf.sort_values('win_rate')
    assert team_perf_sorted['win_rate'].iloc[0] <= team_perf_sorted['win_rate'].iloc[-1]
    
    team_diff_sorted = team_perf.sort_values('point_differential')
    assert team_diff_sorted['point_differential'].iloc[0] <= team_diff_sorted['point_differential'].iloc[-1]
    
    # Test color coding logic
    colors = ['red' if x < 0.4 else 'orange' if x < 0.6 else 'green' 
              for x in team_perf_sorted['win_rate']]
    assert len(colors) == len(team_perf_sorted)
    
    colors2 = ['red' if x < -5 else 'orange' if x < 0 else 'lightgreen' if x < 5 else 'green' 
               for x in team_diff_sorted['point_differential']]
    assert len(colors2) == len(team_diff_sorted)

def test_statistical_calculations(team_performance_data):
    """Test various statistical calculations for team performance."""
    features_df = team_performance_data
    
    team_perf = features_df.groupby('team_name').agg({
        'target_win': ['count', 'mean'],
        'season_win_pct': 'mean',
        'season_avg_pts': 'mean',
        'season_avg_pts_allowed': 'mean',
        'win_pct_last_10': 'mean',
        'is_home': 'mean'
    }).round(3)
    
    team_perf.columns = ['games_played', 'win_rate', 'season_win_pct', 
                        'avg_pts_scored', 'avg_pts_allowed', 'recent_win_pct', 'home_pct']
    team_perf = team_perf.reset_index()
    team_perf['point_differential'] = team_perf['avg_pts_scored'] - team_perf['avg_pts_allowed']
    
    # Test league averages
    league_avg_diff = team_perf['point_differential'].mean()
    assert isinstance(league_avg_diff, (int, float))
    
    # Test most balanced team (closest to 0 point differential)
    most_balanced_idx = team_perf['point_differential'].abs().idxmin()
    most_balanced_team = team_perf.loc[most_balanced_idx, 'team_name']
    assert most_balanced_team in team_perf['team_name'].values

def test_edge_cases():
    """Test edge cases in team performance analysis."""
    # Test with single team
    single_team_data = pd.DataFrame({
        'team_name': ['SingleTeam'] * 5,
        'target_win': [1, 0, 1, 1, 0],
        'season_win_pct': [0.6] * 5,
        'season_avg_pts': [110] * 5,
        'season_avg_pts_allowed': [105] * 5,
        'win_pct_last_10': [0.7] * 5,
        'is_home': [1, 0, 1, 0, 1]
    })
    
    team_agg = single_team_data.groupby('team_name').agg({
        'target_win': ['count', 'mean'],
        'season_win_pct': 'mean'
    })
    
    assert len(team_agg) == 1
    assert team_agg.iloc[0, 0] == 5  # count
    assert 0 <= team_agg.iloc[0, 1] <= 1  # mean win rate

if __name__ == "__main__":
    pytest.main([__file__])
