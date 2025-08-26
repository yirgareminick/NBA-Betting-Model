"""
Test for Cell 6: Model Performance Tracking
"""
import pytest
import pandas as pd
import numpy as np
import sqlite3
import os
from datetime import datetime, date
import tempfile

@pytest.fixture
def mock_predictions_data():
    """Create sample predictions data."""
    np.random.seed(42)
    
    data = []
    for i in range(100):
        data.append({
            'game_id': f'game_{i:03d}',
            'game_date': f'2024-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}',
            'home_team': f'Team_{np.random.randint(1, 31):02d}',
            'away_team': f'Team_{np.random.randint(1, 31):02d}',
            'actual_outcome': np.random.choice([0, 1]),
            'predicted_prob': np.random.uniform(0.2, 0.8),
            'kelly_bet_amount': np.random.uniform(0, 50),
            'betting_odds': np.random.uniform(1.5, 3.0),
            'roi': np.random.uniform(-1, 2),
            'season': '2024-25'
        })
    
    return pd.DataFrame(data)

@pytest.fixture
def temp_db():
    """Create temporary database."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    temp_file.close()
    
    yield temp_file.name
    
    if os.path.exists(temp_file.name):
        os.unlink(temp_file.name)

def test_accuracy_calculation(mock_predictions_data):
    """Test accuracy calculation."""
    predictions_df = mock_predictions_data
    
    predictions_df['predicted_outcome'] = (predictions_df['predicted_prob'] > 0.5).astype(int)
    
    accuracy = (predictions_df['predicted_outcome'] == predictions_df['actual_outcome']).mean()
    
    assert 0 <= accuracy <= 1
    assert isinstance(accuracy, (float, np.floating))
    
    perfect_df = predictions_df.copy()
    perfect_df['predicted_outcome'] = perfect_df['actual_outcome']
    perfect_accuracy = (perfect_df['predicted_outcome'] == perfect_df['actual_outcome']).mean()
    assert perfect_accuracy == 1.0

def test_model_calibration_analysis(mock_predictions_data):
    """Test model calibration analysis."""
    predictions_df = mock_predictions_data.copy()
    
    predictions_df['prob_bin'] = pd.cut(predictions_df['predicted_prob'], 
                                      bins=5, 
                                      labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'])
    
    calibration = predictions_df.groupby('prob_bin', observed=False).agg({
        'predicted_prob': 'mean',
        'actual_outcome': ['count', 'mean']
    }).round(3)
    
    assert len(calibration) >= 1
    assert len(calibration) <= 5
    
    for bin_name in calibration.index.dropna():
        count = calibration.loc[bin_name, ('actual_outcome', 'count')]
        assert count >= 0

def test_streak_analysis(mock_predictions_data):
    """Test streak analysis functionality."""
    predictions_df = mock_predictions_data.copy()
    
    # Create streak groups that match the dataframe length exactly
    streak_options = ['short', 'medium', 'long']
    predictions_df['streak_group'] = [streak_options[i % len(streak_options)] for i in range(len(predictions_df))]
    
    streaks = predictions_df.groupby('streak_group', observed=False).agg({
        'predicted_prob': ['mean', 'std'],
        'actual_outcome': ['count', 'sum', 'mean']
    }).round(3)
    
    assert len(streaks) >= 1
    assert all(count >= 0 for count in streaks[('actual_outcome', 'count')])

def test_database_storage_operations(temp_db):
    """Test database storage operations."""
    db_path = temp_db
    
    # First create the table
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE predictions (
                id INTEGER PRIMARY KEY,
                game_id TEXT,
                predicted_prob REAL,
                actual_outcome INTEGER,
                roi REAL
            )
        ''')
        conn.commit()
        
        # Test table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'")
        table_exists = cursor.fetchone() is not None
        assert table_exists
        
        cursor.execute("SELECT COUNT(*) FROM predictions")
        row_count = cursor.fetchone()[0]
        assert row_count >= 0
        
        cursor.execute("PRAGMA table_info(predictions)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        expected_columns = ['game_id', 'predicted_prob', 'actual_outcome', 'roi']
        for col in expected_columns:
            assert col in column_names

def test_betting_performance_metrics(mock_predictions_data):
    """Test betting performance metrics."""
    predictions_df = mock_predictions_data
    
    total_bets = len(predictions_df[predictions_df['kelly_bet_amount'] > 0])
    total_wagered = predictions_df['kelly_bet_amount'].sum()
    total_return = (predictions_df['kelly_bet_amount'] * 
                   predictions_df['actual_outcome'] * 
                   predictions_df['betting_odds']).sum()
    
    net_profit = total_return - total_wagered
    roi = (net_profit / total_wagered * 100) if total_wagered > 0 else 0
    
    assert total_bets >= 0
    assert total_wagered >= 0
    assert isinstance(roi, (int, float))
    
    profitable_df = predictions_df.copy()
    profitable_df['actual_outcome'] = 1
    profitable_df['betting_odds'] = 2.0
    
    profit_return = (profitable_df['kelly_bet_amount'] * 
                    profitable_df['actual_outcome'] * 
                    profitable_df['betting_odds']).sum()
    profit_wagered = profitable_df['kelly_bet_amount'].sum()
    profit_roi = ((profit_return - profit_wagered) / profit_wagered * 100) if profit_wagered > 0 else 0
    
    assert profit_roi > 0

def test_monthly_performance_breakdown(mock_predictions_data):
    """Test monthly performance breakdown."""
    predictions_df = mock_predictions_data.copy()
    
    predictions_df['game_date'] = pd.to_datetime(predictions_df['game_date'])
    predictions_df['month'] = predictions_df['game_date'].dt.month
    
    monthly_performance = predictions_df.groupby('month', observed=False).agg({
        'game_id': 'count',
        'actual_outcome': 'mean',
        'kelly_bet_amount': ['sum', 'count'],
        'roi': 'mean'
    }).round(3)
    
    assert len(monthly_performance) >= 1
    assert len(monthly_performance) <= 12
    
    for month in monthly_performance.index:
        assert 1 <= month <= 12
    
    for month in monthly_performance.index:
        accuracy = monthly_performance.loc[month, ('actual_outcome', 'mean')]
        assert 0 <= accuracy <= 1

def test_confidence_vs_accuracy_analysis(mock_predictions_data):
    """Test confidence vs accuracy analysis."""
    predictions_df = mock_predictions_data.copy()
    
    predictions_df['confidence'] = abs(predictions_df['predicted_prob'] - 0.5)
    predictions_df['confidence_bin'] = pd.cut(predictions_df['confidence'], 
                                            bins=3, 
                                            labels=['Low', 'Medium', 'High'])
    
    confidence_analysis = predictions_df.groupby('confidence_bin', observed=False).agg({
        'actual_outcome': ['count', 'mean'],
        'predicted_prob': ['mean', 'std'],
        'confidence': 'mean'
    }).round(3)
    
    assert len(confidence_analysis) >= 1
    
    confidence_means = confidence_analysis[('confidence', 'mean')].dropna()
    if len(confidence_means) > 1:
        assert confidence_means.iloc[-1] >= confidence_means.iloc[0]

def test_database_operations(temp_db, mock_predictions_data):
    """Test database operations for performance tracking."""
    predictions_df = mock_predictions_data
    
    # Connect to temporary database
    conn = sqlite3.connect(temp_db)
    
    # Create performance table
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS model_performance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        accuracy REAL,
        total_bets INTEGER,
        total_wagered REAL,
        net_profit REAL,
        roi REAL,
        season TEXT
    )
    """
    conn.execute(create_table_sql)
    
    # Insert sample performance data
    performance_data = {
        'date': str(date.today()),
        'accuracy': 0.65,
        'total_bets': 50,
        'total_wagered': 1000.0,
        'net_profit': 150.0,
        'roi': 15.0,
        'season': '2024-25'
    }
    
    insert_sql = """
    INSERT INTO model_performance (date, accuracy, total_bets, total_wagered, net_profit, roi, season)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    
    conn.execute(insert_sql, (
        performance_data['date'],
        performance_data['accuracy'],
        performance_data['total_bets'],
        performance_data['total_wagered'],
        performance_data['net_profit'],
        performance_data['roi'],
        performance_data['season']
    ))
    conn.commit()
    
    # Query the data back
    query_sql = "SELECT * FROM model_performance WHERE season = '2024-25'"
    result_df = pd.read_sql_query(query_sql, conn)
    
    # Test database operations
    assert len(result_df) == 1
    assert result_df.iloc[0]['accuracy'] == 0.65
    assert result_df.iloc[0]['total_bets'] == 50
    assert result_df.iloc[0]['roi'] == 15.0
    
    conn.close()

def test_performance_trends(mock_predictions_data):
    """Test performance trend analysis."""
    predictions_df = mock_predictions_data.copy()
    
    # Convert to datetime and sort
    predictions_df['game_date'] = pd.to_datetime(predictions_df['game_date'])
    predictions_df = predictions_df.sort_values('game_date')
    
    # Create rolling windows
    predictions_df['game_number'] = range(len(predictions_df))
    window_size = 20
    
    if len(predictions_df) >= window_size:
        # Calculate rolling accuracy
        predictions_df['predicted_outcome'] = (predictions_df['predicted_prob'] > 0.5).astype(int)
        predictions_df['rolling_accuracy'] = (
            predictions_df['predicted_outcome'] == predictions_df['actual_outcome']
        ).rolling(window=window_size).mean()
        
        # Test rolling calculations
        rolling_data = predictions_df['rolling_accuracy'].dropna()
        assert len(rolling_data) == len(predictions_df) - window_size + 1
        assert (rolling_data >= 0).all()
        assert (rolling_data <= 1).all()

def test_outlier_detection(mock_predictions_data):
    """Test detection of outlier predictions."""
    predictions_df = mock_predictions_data.copy()
    
    # Calculate confidence
    predictions_df['confidence'] = abs(predictions_df['predicted_prob'] - 0.5)
    
    # Identify high-confidence wrong predictions
    predictions_df['predicted_outcome'] = (predictions_df['predicted_prob'] > 0.5).astype(int)
    predictions_df['correct'] = (predictions_df['predicted_outcome'] == predictions_df['actual_outcome'])
    
    # High confidence threshold
    high_confidence_threshold = 0.3  # Distance from 0.5
    high_confidence_wrong = predictions_df[
        (predictions_df['confidence'] > high_confidence_threshold) & 
        (~predictions_df['correct'])
    ]
    
    # Test outlier detection
    assert len(high_confidence_wrong) >= 0
    assert len(high_confidence_wrong) <= len(predictions_df)
    
    # Test that all outliers are indeed high confidence and wrong
    for idx, row in high_confidence_wrong.iterrows():
        assert row['confidence'] > high_confidence_threshold
        assert not row['correct']

def test_performance_summary_generation(mock_predictions_data):
    """Test generation of performance summary."""
    predictions_df = mock_predictions_data.copy()
    
    # Calculate key metrics
    predictions_df['predicted_outcome'] = (predictions_df['predicted_prob'] > 0.5).astype(int)
    
    summary = {
        'total_predictions': len(predictions_df),
        'accuracy': (predictions_df['predicted_outcome'] == predictions_df['actual_outcome']).mean(),
        'avg_confidence': abs(predictions_df['predicted_prob'] - 0.5).mean(),
        'total_bets': len(predictions_df[predictions_df['kelly_bet_amount'] > 0]),
        'avg_bet_size': predictions_df[predictions_df['kelly_bet_amount'] > 0]['kelly_bet_amount'].mean(),
        'profitable_bets': len(predictions_df[
            (predictions_df['kelly_bet_amount'] > 0) & 
            (predictions_df['actual_outcome'] == 1)
        ])
    }
    
    # Test summary metrics
    assert summary['total_predictions'] == len(predictions_df)
    assert 0 <= summary['accuracy'] <= 1
    assert summary['avg_confidence'] >= 0
    assert summary['total_bets'] >= 0
    assert summary['profitable_bets'] >= 0
    assert summary['profitable_bets'] <= summary['total_bets']
    
    if summary['total_bets'] > 0:
        assert summary['avg_bet_size'] > 0

if __name__ == "__main__":
    pytest.main([__file__])
