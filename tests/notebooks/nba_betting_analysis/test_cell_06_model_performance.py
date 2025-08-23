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
    """Create sample predictions data for testing."""
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
    """Create a temporary database for testing."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    temp_file.close()
    
    yield temp_file.name
    
    # Cleanup
    if os.path.exists(temp_file.name):
        os.unlink(temp_file.name)

def test_accuracy_calculation(mock_predictions_data):
    """Test model accuracy calculation."""
    predictions_df = mock_predictions_data
    
    # Binary predictions based on probability threshold
    predictions_df['predicted_outcome'] = (predictions_df['predicted_prob'] > 0.5).astype(int)
    
    # Calculate accuracy
    accuracy = (predictions_df['predicted_outcome'] == predictions_df['actual_outcome']).mean()
    
    # Test accuracy calculation
    assert 0 <= accuracy <= 1
    assert isinstance(accuracy, (float, np.floating))
    
    # Test with perfect predictions
    perfect_df = predictions_df.copy()
    perfect_df['predicted_outcome'] = perfect_df['actual_outcome']
    perfect_accuracy = (perfect_df['predicted_outcome'] == perfect_df['actual_outcome']).mean()
    assert perfect_accuracy == 1.0

def test_probability_calibration_analysis(mock_predictions_data):
    """Test probability calibration analysis."""
    predictions_df = mock_predictions_data
    
    # Create probability bins
    predictions_df['prob_bin'] = pd.cut(predictions_df['predicted_prob'], 
                                       bins=[0, 0.4, 0.6, 1.0], 
                                       labels=['Low (0-40%)', 'Medium (40-60%)', 'High (60%+)'])
    
    # Calculate calibration by bin
    calibration = predictions_df.groupby('prob_bin').agg({
        'actual_outcome': ['count', 'mean'],
        'predicted_prob': 'mean'
    }).round(3)
    
    # Test calibration structure
    assert len(calibration) >= 1  # At least one bin should have data
    assert calibration.columns.nlevels == 2  # MultiIndex columns
    
    # Test that probabilities and outcomes are in valid ranges
    for bin_name in calibration.index:
        if not pd.isna(bin_name):
            actual_rate = calibration.loc[bin_name, ('actual_outcome', 'mean')]
            avg_prob = calibration.loc[bin_name, ('predicted_prob', 'mean')]
            assert 0 <= actual_rate <= 1
            assert 0 <= avg_prob <= 1

def test_betting_performance_metrics(mock_predictions_data):
    """Test betting performance metrics calculation."""
    predictions_df = mock_predictions_data
    
    # Calculate betting metrics
    total_bets = len(predictions_df[predictions_df['kelly_bet_amount'] > 0])
    total_wagered = predictions_df['kelly_bet_amount'].sum()
    total_return = (predictions_df['kelly_bet_amount'] * 
                   predictions_df['actual_outcome'] * 
                   predictions_df['betting_odds']).sum()
    
    net_profit = total_return - total_wagered
    roi = (net_profit / total_wagered * 100) if total_wagered > 0 else 0
    
    # Test betting metrics
    assert total_bets >= 0
    assert total_wagered >= 0
    assert isinstance(roi, (int, float))
    
    # Test with profitable scenario
    profitable_df = predictions_df.copy()
    profitable_df['actual_outcome'] = 1  # All wins
    profitable_df['betting_odds'] = 2.0   # 2:1 odds
    
    profit_return = (profitable_df['kelly_bet_amount'] * 
                    profitable_df['actual_outcome'] * 
                    profitable_df['betting_odds']).sum()
    profit_wagered = profitable_df['kelly_bet_amount'].sum()
    profit_roi = ((profit_return - profit_wagered) / profit_wagered * 100) if profit_wagered > 0 else 0
    
    assert profit_roi > 0  # Should be profitable

def test_monthly_performance_breakdown(mock_predictions_data):
    """Test monthly performance breakdown."""
    predictions_df = mock_predictions_data.copy()
    
    # Convert game_date to datetime
    predictions_df['game_date'] = pd.to_datetime(predictions_df['game_date'])
    predictions_df['month'] = predictions_df['game_date'].dt.month
    
    # Monthly breakdown
    monthly_performance = predictions_df.groupby('month').agg({
        'game_id': 'count',
        'actual_outcome': 'mean',
        'kelly_bet_amount': ['sum', 'count'],
        'roi': 'mean'
    }).round(3)
    
    # Test monthly breakdown structure
    assert len(monthly_performance) >= 1
    assert len(monthly_performance) <= 12  # Max 12 months
    
    # Test that all months are valid
    for month in monthly_performance.index:
        assert 1 <= month <= 12
    
    # Test that accuracy values are valid
    for month in monthly_performance.index:
        accuracy = monthly_performance.loc[month, ('actual_outcome', 'mean')]
        assert 0 <= accuracy <= 1

def test_confidence_vs_accuracy_analysis(mock_predictions_data):
    """Test analysis of confidence vs accuracy relationship."""
    predictions_df = mock_predictions_data.copy()
    
    # Create confidence bins
    predictions_df['confidence'] = abs(predictions_df['predicted_prob'] - 0.5)
    predictions_df['confidence_bin'] = pd.cut(predictions_df['confidence'], 
                                            bins=3, 
                                            labels=['Low', 'Medium', 'High'])
    
    # Confidence analysis
    confidence_analysis = predictions_df.groupby('confidence_bin').agg({
        'actual_outcome': ['count', 'mean'],
        'predicted_prob': ['mean', 'std'],
        'confidence': 'mean'
    }).round(3)
    
    # Test confidence analysis
    assert len(confidence_analysis) >= 1
    
    # Test that confidence increases across bins
    confidence_means = confidence_analysis[('confidence', 'mean')].dropna()
    if len(confidence_means) > 1:
        # Check if generally increasing (allowing for some noise)
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
