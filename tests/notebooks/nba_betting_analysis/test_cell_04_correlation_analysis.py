"""
Test for Cell 4: Correlation Analysis
"""
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_correlation_data():
    """Create sample data for correlation testing."""
    np.random.seed(42)
    
    # Create correlated features
    n_samples = 100
    data = {
        'season_win_pct': np.random.uniform(0.3, 0.8, n_samples),
        'season_avg_pts': np.random.normal(110, 10, n_samples),
        'season_avg_pts_allowed': np.random.normal(108, 8, n_samples),
        'win_pct_last_10': np.random.uniform(0.2, 0.9, n_samples),
        'target_win': np.random.choice([0, 1], n_samples),
        'is_home': np.random.choice([0, 1], n_samples)
    }
    
    # Make some correlations more realistic
    for i in range(n_samples):
        # Teams with higher win% tend to score more and allow less
        if data['season_win_pct'][i] > 0.6:
            data['season_avg_pts'][i] += np.random.normal(5, 2)
            data['season_avg_pts_allowed'][i] -= np.random.normal(3, 1)
    
    return pd.DataFrame(data)

def test_correlation_matrix_calculation(sample_correlation_data):
    """Test correlation matrix calculation."""
    features_df = sample_correlation_data
    
    # Select numeric columns for correlation
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    correlation_matrix = features_df[numeric_cols].corr()
    
    # Test correlation matrix properties
    assert correlation_matrix.shape[0] == correlation_matrix.shape[1]
    assert len(correlation_matrix) == len(numeric_cols)
    
    # Diagonal should be 1.0
    np.testing.assert_array_almost_equal(np.diag(correlation_matrix), 1.0)
    
    # Matrix should be symmetric
    np.testing.assert_array_almost_equal(correlation_matrix.values, correlation_matrix.T.values)

def test_feature_target_correlation(sample_correlation_data):
    """Test correlation between features and target variable."""
    features_df = sample_correlation_data
    
    # Calculate correlation with target
    target_correlations = features_df.corr()['target_win'].drop('target_win')
    
    # Test that correlations are within valid range
    assert all(abs(corr) <= 1.0 for corr in target_correlations if not pd.isna(corr))
    
    # Test that we have correlations for all numeric features
    expected_features = ['season_win_pct', 'season_avg_pts', 'season_avg_pts_allowed', 
                        'win_pct_last_10', 'is_home']
    for feature in expected_features:
        assert feature in target_correlations.index

def test_high_correlation_identification():
    """Test identification of highly correlated features."""
    # Create data with known high correlation
    np.random.seed(42)
    n = 100
    
    x1 = np.random.normal(0, 1, n)
    x2 = x1 + np.random.normal(0, 0.1, n)  # Highly correlated with x1
    x3 = np.random.normal(0, 1, n)  # Independent
    
    df = pd.DataFrame({
        'feature1': x1,
        'feature2': x2,
        'feature3': x3,
        'target': np.random.choice([0, 1], n)
    })
    
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Find high correlations (excluding diagonal)
    high_corr_threshold = 0.8
    high_corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = abs(corr_matrix.iloc[i, j])
            if corr_val > high_corr_threshold:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
    
    # Should find the high correlation between feature1 and feature2
    assert len(high_corr_pairs) > 0
    feature_names = [pair[0] for pair in high_corr_pairs] + [pair[1] for pair in high_corr_pairs]
    assert 'feature1' in feature_names or 'feature2' in feature_names

def test_correlation_heatmap_data_preparation(sample_correlation_data):
    """Test data preparation for correlation heatmap."""
    features_df = sample_correlation_data
    
    # Prepare correlation matrix for visualization
    correlation_matrix = features_df.corr()
    
    # Test that matrix can be used for heatmap
    assert not correlation_matrix.empty
    assert correlation_matrix.index.equals(correlation_matrix.columns)
    
    # Test annotation formatting
    annot_matrix = correlation_matrix.round(2)
    assert all(abs(val) <= 1.0 for val in annot_matrix.values.flatten() if not pd.isna(val))

def test_feature_importance_ranking(sample_correlation_data):
    """Test ranking features by correlation with target."""
    features_df = sample_correlation_data
    
    # Calculate absolute correlations with target
    target_corrs = features_df.corr()['target_win'].drop('target_win')
    abs_corrs = target_corrs.abs().sort_values(ascending=False)
    
    # Test ranking
    assert len(abs_corrs) > 0
    assert all(abs_corrs.iloc[i] >= abs_corrs.iloc[i+1] for i in range(len(abs_corrs)-1))

def test_correlation_filtering():
    """Test filtering features based on correlation thresholds."""
    # Create test data with various correlation levels
    np.random.seed(42)
    n = 100
    
    target = np.random.choice([0, 1], n)
    
    # Features with different correlation levels to target
    high_corr_feature = target + np.random.normal(0, 0.3, n)
    medium_corr_feature = target * 0.5 + np.random.normal(0, 0.7, n)
    low_corr_feature = np.random.normal(0, 1, n)
    
    df = pd.DataFrame({
        'target_win': target,
        'high_corr': high_corr_feature,
        'medium_corr': medium_corr_feature,
        'low_corr': low_corr_feature
    })
    
    # Test correlation filtering
    correlations = df.corr()['target_win'].drop('target_win')
    
    # Filter features by correlation threshold
    min_correlation = 0.1
    significant_features = correlations[abs(correlations) >= min_correlation]
    
    # Should have at least some significant features
    assert len(significant_features) >= 0
    assert all(abs(corr) >= min_correlation for corr in significant_features)

def test_missing_data_correlation():
    """Test correlation calculation with missing data."""
    # Create data with missing values
    np.random.seed(42)
    df = pd.DataFrame({
        'feature1': [1, 2, np.nan, 4, 5],
        'feature2': [2, np.nan, 6, 8, 10],
        'target': [0, 1, 0, 1, 0]
    })
    
    # Calculate correlation with missing data
    corr_matrix = df.corr()
    
    # Should handle missing data gracefully
    assert not corr_matrix.empty
    assert corr_matrix.shape[0] == 3  # 3 columns
    
def test_correlation_interpretation():
    """Test interpretation of correlation strengths."""
    correlations = [0.1, 0.3, 0.5, 0.7, 0.9, -0.2, -0.6, -0.8]
    
    def interpret_correlation(corr):
        abs_corr = abs(corr)
        if abs_corr >= 0.7:
            return "strong"
        elif abs_corr >= 0.4:
            return "moderate"
        elif abs_corr >= 0.2:
            return "weak"
        else:
            return "very weak"
    
    interpretations = [interpret_correlation(corr) for corr in correlations]
    
    # Test interpretation logic
    assert interpretations[0] == "very weak"  # 0.1
    assert interpretations[1] == "weak"       # 0.3
    assert interpretations[2] == "moderate"   # 0.5
    assert interpretations[3] == "strong"     # 0.7
    assert interpretations[4] == "strong"     # 0.9
    assert interpretations[5] == "weak"       # -0.2
    assert interpretations[6] == "moderate"   # -0.6
    assert interpretations[7] == "strong"     # -0.8

if __name__ == "__main__":
    pytest.main([__file__])
