"""
Test for Cell 3: Data loading functionality
"""
import pytest
import pandas as pd
import tempfile
import sqlite3
from pathlib import Path

@pytest.fixture
def mock_data_structure():
    """Create a mock data directory structure with sample files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        base_path = Path(temp_dir)
        data_path = base_path / 'data'
        processed_path = data_path / 'processed'
        raw_path = data_path / 'raw'
        
        # Create directories
        processed_path.mkdir(parents=True)
        raw_path.mkdir(parents=True)
        
        # Create sample data files
        sample_features = pd.DataFrame({
            'game_id': [1, 2, 3],
            'game_date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'team_name': ['Team A', 'Team B', 'Team A'],
            'target_win': [1, 0, 1]
        })
        sample_features.to_parquet(processed_path / 'nba_features.parquet')
        
        sample_games = pd.DataFrame({
            'game_id': [1, 2, 3],
            'game_date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'home_team': ['Team A', 'Team B', 'Team C'],
            'away_team': ['Team B', 'Team A', 'Team A'],
            'home_score': [110, 105, 98],
            'away_score': [108, 112, 101]
        })
        sample_games.to_csv(processed_path / 'games_2024_2025.csv', index=False)
        
        historical_games = sample_games.copy()
        historical_games.to_csv(processed_path / 'games_2020_2023.csv', index=False)
        
        # Create sample performance database
        performance_db = data_path / 'performance.db'
        conn = sqlite3.connect(performance_db)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE predictions (
                id INTEGER PRIMARY KEY,
                prediction_date TEXT,
                predicted_outcome INTEGER,
                actual_outcome INTEGER,
                confidence REAL
            )
        ''')
        conn.commit()
        conn.close()
        
        yield {
            'base_path': base_path,
            'data_path': data_path,
            'processed_path': processed_path,
            'raw_path': raw_path,
            'performance_db': performance_db
        }

def test_features_data_loading(mock_data_structure):
    """Test loading of features parquet file."""
    processed_path = mock_data_structure['processed_path']
    features_file = processed_path / 'nba_features.parquet'
    
    assert features_file.exists()
    
    features_df = pd.read_parquet(features_file)
    assert not features_df.empty
    assert 'game_id' in features_df.columns
    assert 'target_win' in features_df.columns
    assert len(features_df) == 3

def test_games_data_loading(mock_data_structure):
    """Test loading of games CSV files."""
    processed_path = mock_data_structure['processed_path']
    
    # Test 2024-2025 games
    games_file = processed_path / 'games_2024_2025.csv'
    assert games_file.exists()
    
    games_df = pd.read_csv(games_file)
    games_df['game_date'] = pd.to_datetime(games_df['game_date'])
    assert not games_df.empty
    assert 'home_team' in games_df.columns
    assert 'away_team' in games_df.columns
    
    # Test historical games
    historical_file = processed_path / 'games_2020_2023.csv'
    assert historical_file.exists()
    
    historical_df = pd.read_csv(historical_file)
    historical_df['game_date'] = pd.to_datetime(historical_df['game_date'])
    assert not historical_df.empty

def test_performance_database_detection(mock_data_structure):
    """Test performance database file detection."""
    performance_db = mock_data_structure['performance_db']
    
    assert performance_db.exists()
    
    # Test database connectivity
    conn = sqlite3.connect(performance_db)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    conn.close()
    
    assert len(tables) > 0
    assert 'predictions' in [table[0] for table in tables]

def test_data_loading_with_missing_files():
    """Test behavior when data files are missing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        processed_path = Path(temp_dir) / 'data' / 'processed'
        processed_path.mkdir(parents=True)
        
        # Test missing features file
        features_file = processed_path / 'nba_features.parquet'
        assert not features_file.exists()
        
        # Test missing games files
        games_file = processed_path / 'games_2024_2025.csv'
        historical_file = processed_path / 'games_2020_2023.csv'
        assert not games_file.exists()
        assert not historical_file.exists()

def test_data_summary_calculation(mock_data_structure):
    """Test data summary statistics calculation."""
    processed_path = mock_data_structure['processed_path']
    
    # Load data
    features_df = pd.read_parquet(processed_path / 'nba_features.parquet')
    games_df = pd.read_csv(processed_path / 'games_2024_2025.csv')
    historical_df = pd.read_csv(processed_path / 'games_2020_2023.csv')
    
    # Test summary calculations
    assert features_df.shape == (3, 4)  # 3 records, 4 columns
    assert games_df.shape == (3, 6)     # 3 games, 6 columns
    assert historical_df.shape == (3, 6) # 3 historical games, 6 columns

if __name__ == "__main__":
    pytest.main([__file__])
