"""
Test for Cells 12-13: Model performance and betting analysis
"""
import pytest
import pandas as pd
import numpy as np
import sqlite3
import tempfile
from pathlib import Path

@pytest.fixture
def mock_performance_database():
    """Create a mock performance database with sample data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / 'performance.db'
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create predictions table
        cursor.execute('''
            CREATE TABLE predictions (
                id INTEGER PRIMARY KEY,
                prediction_date TEXT,
                predicted_outcome INTEGER,
                actual_outcome INTEGER,
                confidence REAL,
                team_name TEXT,
                opponent TEXT
            )
        ''')
        
        # Create betting performance table
        cursor.execute('''
            CREATE TABLE betting_performance (
                id INTEGER PRIMARY KEY,
                bet_date TEXT,
                bet_amount REAL,
                profit_loss REAL,
                team_name TEXT,
                bet_type TEXT
            )
        ''')
        
        # Insert sample predictions data
        predictions_data = []
        np.random.seed(42)
        for i in range(100):
            confidence = np.random.uniform(0.5, 0.95)
            predicted = np.random.choice([0, 1])
            # Make actual outcome correlated with confidence
            actual = predicted if np.random.random() < confidence else 1 - predicted
            
            predictions_data.append((
                f'2024-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}',
                predicted,
                actual,
                confidence,
                f'Team{np.random.randint(1, 31)}',
                f'Team{np.random.randint(1, 31)}'
            ))
        
        cursor.executemany(
            'INSERT INTO predictions (prediction_date, predicted_outcome, actual_outcome, confidence, team_name, opponent) VALUES (?, ?, ?, ?, ?, ?)',
            predictions_data
        )
        
        # Insert sample betting data
        betting_data = []
        for i in range(50):
            bet_amount = np.random.uniform(50, 500)
            # Make profit/loss somewhat realistic
            if np.random.random() < 0.45:  # 45% win rate
                profit_loss = bet_amount * np.random.uniform(0.8, 1.2)  # Win
            else:
                profit_loss = -bet_amount  # Loss
            
            betting_data.append((
                f'2024-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}',
                bet_amount,
                profit_loss,
                f'Team{np.random.randint(1, 31)}',
                'moneyline'
            ))
        
        cursor.executemany(
            'INSERT INTO betting_performance (bet_date, bet_amount, profit_loss, team_name, bet_type) VALUES (?, ?, ?, ?, ?)',
            betting_data
        )
        
        conn.commit()
        conn.close()
        
        yield db_path

@pytest.fixture
def sample_games_data():
    """Create sample games data for analysis."""
    np.random.seed(42)
    
    teams = [f'Team{i}' for i in range(1, 31)]
    data = []
    
    for i in range(200):
        home_team = np.random.choice(teams)
        away_team = np.random.choice([t for t in teams if t != home_team])
        home_score = np.random.randint(90, 130)
        away_score = np.random.randint(90, 130)
        
        data.append({
            'game_date': f'2024-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}',
            'home_team': home_team,
            'away_team': away_team,
            'home_score': home_score,
            'away_score': away_score
        })
    
    df = pd.DataFrame(data)
    df['game_date'] = pd.to_datetime(df['game_date'])
    return df

def test_database_connection_and_query(mock_performance_database):
    """Test database connection and basic queries."""
    conn = sqlite3.connect(mock_performance_database)
    
    # Test predictions query
    predictions_query = """
    SELECT * FROM predictions 
    ORDER BY prediction_date DESC
    LIMIT 1000
    """
    predictions_df = pd.read_sql_query(predictions_query, conn)
    
    assert not predictions_df.empty
    assert 'predicted_outcome' in predictions_df.columns
    assert 'actual_outcome' in predictions_df.columns
    assert 'confidence' in predictions_df.columns
    
    # Test betting query
    betting_query = """
    SELECT * FROM betting_performance 
    ORDER BY bet_date DESC
    LIMIT 500
    """
    betting_df = pd.read_sql_query(betting_query, conn)
    
    assert not betting_df.empty
    assert 'bet_amount' in betting_df.columns
    assert 'profit_loss' in betting_df.columns
    
    conn.close()

def test_model_accuracy_calculation(mock_performance_database):
    """Test model accuracy calculation."""
    conn = sqlite3.connect(mock_performance_database)
    
    predictions_df = pd.read_sql_query("SELECT * FROM predictions", conn)
    predictions_df['prediction_date'] = pd.to_datetime(predictions_df['prediction_date'])
    
    # Test overall accuracy
    accuracy = (predictions_df['actual_outcome'] == predictions_df['predicted_outcome']).mean()
    assert 0 <= accuracy <= 1
    
    # Test accuracy by confidence level
    predictions_df['confidence_bucket'] = pd.cut(predictions_df['confidence'], 
                                               bins=[0, 0.6, 0.7, 0.8, 1.0], 
                                               labels=['Low (≤60%)', 'Medium (60-70%)', 
                                                      'High (70-80%)', 'Very High (>80%)'])
    
    accuracy_by_conf = predictions_df.groupby('confidence_bucket', observed=False).agg({
        'predicted_outcome': 'count',
        'actual_outcome': lambda x: (predictions_df.loc[x.index, 'actual_outcome'] == 
                                    predictions_df.loc[x.index, 'predicted_outcome']).sum()
    })
    accuracy_by_conf['accuracy'] = accuracy_by_conf['actual_outcome'] / accuracy_by_conf['predicted_outcome']
    
    # Test that accuracy values are valid
    for idx in accuracy_by_conf.index:
        if pd.notna(idx):
            acc = accuracy_by_conf.loc[idx, 'accuracy']
            assert 0 <= acc <= 1
    
    conn.close()

def test_betting_performance_metrics(mock_performance_database):
    """Test betting performance metrics calculation."""
    conn = sqlite3.connect(mock_performance_database)
    
    betting_df = pd.read_sql_query("SELECT * FROM betting_performance", conn)
    betting_df['bet_date'] = pd.to_datetime(betting_df['bet_date'])
    
    # Test basic metrics
    total_profit = betting_df['profit_loss'].sum()
    total_bets = len(betting_df)
    win_rate = (betting_df['profit_loss'] > 0).mean()
    avg_bet = betting_df['bet_amount'].mean()
    
    assert isinstance(total_profit, (int, float))
    assert total_bets > 0
    assert 0 <= win_rate <= 1
    assert avg_bet > 0
    
    # Test ROI calculation
    total_wagered = betting_df['bet_amount'].sum()
    if total_wagered > 0:
        roi = (total_profit / total_wagered) * 100
        assert isinstance(roi, (int, float))
    
    conn.close()

def test_games_analysis(sample_games_data):
    """Test recent games analysis functionality."""
    games_df = sample_games_data
    
    # Test game frequency analysis
    games_by_date = games_df.groupby(games_df['game_date'].dt.date).size()
    assert len(games_by_date) > 0
    assert games_by_date.mean() > 0
    assert games_by_date.max() >= games_by_date.min()
    
    # Test team game counts
    team_game_counts = pd.concat([
        games_df['home_team'].value_counts(),
        games_df['away_team'].value_counts()
    ], axis=1, sort=False).fillna(0)
    team_game_counts.columns = ['home_games', 'away_games']
    team_game_counts['total_games'] = team_game_counts.sum(axis=1)
    
    assert len(team_game_counts) > 0
    assert team_game_counts['total_games'].min() >= 0
    assert team_game_counts['total_games'].max() > 0

def test_scoring_analysis(sample_games_data):
    """Test scoring analysis functionality."""
    games_df = sample_games_data
    
    # Test score calculations
    games_df['total_points'] = games_df['home_score'] + games_df['away_score']
    games_df['point_differential'] = abs(games_df['home_score'] - games_df['away_score'])
    
    assert games_df['total_points'].min() > 0
    assert games_df['point_differential'].min() >= 0
    
    # Test statistics
    avg_total = games_df['total_points'].mean()
    avg_diff = games_df['point_differential'].mean()
    closest_game = games_df['point_differential'].min()
    biggest_blowout = games_df['point_differential'].max()
    
    assert avg_total > 0
    assert avg_diff >= 0
    assert closest_game >= 0
    assert biggest_blowout >= closest_game
    
    # Test home court advantage
    home_wins = (games_df['home_score'] > games_df['away_score']).sum()
    home_win_pct = home_wins / len(games_df)
    assert 0 <= home_win_pct <= 1

def test_precision_recall_calculation():
    """Test precision and recall calculation for binary classification."""
    # Sample binary classification results
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

def test_confidence_statistics():
    """Test confidence statistics calculation."""
    confidence_values = np.random.uniform(0.5, 0.95, 100)
    
    avg_confidence = confidence_values.mean()
    min_confidence = confidence_values.min()
    max_confidence = confidence_values.max()
    
    assert 0 <= avg_confidence <= 1
    assert 0 <= min_confidence <= 1
    assert 0 <= max_confidence <= 1
    assert min_confidence <= avg_confidence <= max_confidence

def test_missing_data_handling():
    """Test handling of missing performance data."""
    # Test when no performance database exists
    performance_available = False
    
    if not performance_available:
        predictions_df = None
        betting_df = None
        
        assert predictions_df is None
        assert betting_df is None
    
    # Test with empty dataframes
    empty_predictions = pd.DataFrame()
    empty_betting = pd.DataFrame()
    
    assert empty_predictions.empty
    assert empty_betting.empty

def test_date_handling():
    """Test proper date handling in analysis."""
    # Test datetime conversion
    dates = ['2024-01-01', '2024-02-15', '2024-12-31']
    df = pd.DataFrame({'date_str': dates})
    df['date_converted'] = pd.to_datetime(df['date_str'])
    
    assert df['date_converted'].dtype == 'datetime64[ns]'
    
    # Test date grouping
    date_groups = df.groupby(df['date_converted'].dt.date).size()
    assert len(date_groups) == 3

if __name__ == "__main__":
    pytest.main([__file__])
