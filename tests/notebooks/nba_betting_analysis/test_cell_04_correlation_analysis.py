"""
Test for Cell 5: Feature correlation analysis
"""
import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

@pytest.fixture
def correlation_test_data():
    """Create sample data with known correlations for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    # Create features with known relationships
    feature1 = np.random.normal(0, 1, n_samples)
    feature2 = 0.8 * feature1 + 0.2 * np.random.normal(0, 1, n_samples)  # High correlation
    feature3 = -0.6 * feature1 + 0.4 * np.random.normal(0, 1, n_samples)  # Negative correlation
    feature4 = np.random.normal(0, 1, n_samples)  # No correlation
    
    # Target variable influenced by features
    target_win = (0.5 * feature1 + 0.3 * feature2 - 0.2 * feature3 + 
                  0.1 * np.random.normal(0, 1, n_samples) > 0).astype(int)
    
    data = {
        'game_id': range(1, n_samples + 1),
        'feature1': feature1,
        'feature2': feature2,
        'feature3': feature3,
        'feature4': feature4,
        'target_win': target_win,
        'categorical_feature': np.random.choice(['A', 'B', 'C'], n_samples)
    }
    
    return pd.DataFrame(data)

def test_numeric_feature_selection(correlation_test_data):
    """Test selection of numeric features for correlation analysis."""
    features_df = correlation_test_data
    
    # Select numeric features
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['game_id', 'target_win']
    numeric_features = [col for col in numeric_cols if col not in exclude_cols]
    
    assert 'feature1' in numeric_features
    assert 'feature2' in numeric_features
    assert 'feature3' in numeric_features
    assert 'feature4' in numeric_features
    assert 'game_id' not in numeric_features
    assert 'target_win' not in numeric_features
    assert 'categorical_feature' not in numeric_features

def test_correlation_matrix_calculation(correlation_test_data):
    """Test correlation matrix calculation."""
    features_df = correlation_test_data
    
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['game_id', 'target_win']
    numeric_features = [col for col in numeric_cols if col not in exclude_cols]
    
    # Calculate correlation matrix
    corr_matrix = features_df[numeric_features + ['target_win']].corr()
    
    # Test matrix properties
    assert corr_matrix.shape[0] == corr_matrix.shape[1]  # Square matrix
    assert len(corr_matrix) == len(numeric_features) + 1  # +1 for target_win
    
    # Test diagonal elements are 1
    for i in range(len(corr_matrix)):
        assert abs(corr_matrix.iloc[i, i] - 1.0) < 1e-10
    
    # Test symmetry
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            assert abs(corr_matrix.iloc[i, j] - corr_matrix.iloc[j, i]) < 1e-10

def test_target_correlations_ranking(correlation_test_data):
    """Test ranking of features by correlation with target."""
    features_df = correlation_test_data
    
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['game_id', 'target_win']
    numeric_features = [col for col in numeric_cols if col not in exclude_cols]
    
    corr_matrix = features_df[numeric_features + ['target_win']].corr()
    target_correlations = corr_matrix['target_win'].drop('target_win').sort_values(key=abs, ascending=False)
    
    # Test that we get correlations for all features
    assert len(target_correlations) == len(numeric_features)
    
    # Test that correlations are within valid range
    for corr in target_correlations:
        assert -1 <= corr <= 1
    
    # Test sorting by absolute value
    abs_correlations = target_correlations.abs()
    for i in range(len(abs_correlations) - 1):
        assert abs_correlations.iloc[i] >= abs_correlations.iloc[i + 1]

def test_correlation_insights_calculation(correlation_test_data):
    """Test calculation of correlation insights."""
    features_df = correlation_test_data
    
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['game_id', 'target_win']
    numeric_features = [col for col in numeric_cols if col not in exclude_cols]
    
    corr_matrix = features_df[numeric_features + ['target_win']].corr()
    target_correlations = corr_matrix['target_win'].drop('target_win')
    
    # Test strongest positive/negative predictors
    strongest_positive = target_correlations.idxmax()
    strongest_negative = target_correlations.idxmin()
    
    assert strongest_positive in numeric_features
    assert strongest_negative in numeric_features
    assert target_correlations[strongest_positive] >= target_correlations[strongest_negative]
    
    # Test feature counts by correlation threshold
    high_positive = len(target_correlations[target_correlations > 0.1])
    high_negative = len(target_correlations[target_correlations < -0.1])
    
    assert high_positive >= 0
    assert high_negative >= 0
    assert high_positive + high_negative <= len(numeric_features)

def test_correlation_heatmap_data_preparation(correlation_test_data):
    """Test preparation of data for correlation heatmap."""
    features_df = correlation_test_data
    
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['game_id', 'target_win']
    numeric_features = [col for col in numeric_cols if col not in exclude_cols]
    
    corr_matrix = features_df[numeric_features + ['target_win']].corr()
    target_correlations = corr_matrix['target_win'].drop('target_win').sort_values(key=abs, ascending=False)
    
    # Test top features selection for heatmap
    top_features = target_correlations.head(10).index.tolist() + ['target_win']
    top_corr_matrix = features_df[top_features].corr()
    
    assert 'target_win' in top_corr_matrix.columns
    assert 'target_win' in top_corr_matrix.index
    assert len(top_corr_matrix) <= 11  # Max 10 features + target

def test_empty_or_invalid_data_handling():
    """Test handling of edge cases in correlation analysis."""
    # Test with no numeric features
    df_no_numeric = pd.DataFrame({
        'categorical1': ['A', 'B', 'C'],
        'categorical2': ['X', 'Y', 'Z'],
        'target_win': [1, 0, 1]
    })
    
    numeric_cols = df_no_numeric.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['target_win']
    numeric_features = [col for col in numeric_cols if col not in exclude_cols]
    
    assert len(numeric_features) == 0
    
    # Test with single feature
    df_single = pd.DataFrame({
        'feature1': [1, 2, 3],
        'target_win': [1, 0, 1]
    })
    
    corr_matrix = df_single.corr()
    assert corr_matrix.shape == (2, 2)

def test_correlation_with_missing_values(correlation_test_data):
    """Test correlation calculation with missing values."""
    features_df = correlation_test_data.copy()
    
    # Introduce missing values
    features_df.loc[0:50, 'feature1'] = np.nan
    features_df.loc[100:150, 'feature2'] = np.nan
    
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['game_id', 'target_win']
    numeric_features = [col for col in numeric_cols if col not in exclude_cols]
    
    # Test that correlation can be calculated with missing values
    corr_matrix = features_df[numeric_features + ['target_win']].corr()
    
    # Some correlations might be NaN, but matrix should still be calculable
    assert corr_matrix.shape[0] == corr_matrix.shape[1]

def test_matplotlib_heatmap_compatibility():
    """Test that correlation matrix is compatible with matplotlib/seaborn heatmap."""
    # Create simple correlation matrix
    data = pd.DataFrame({
        'A': [1, 2, 3, 4],
        'B': [2, 3, 4, 5],
        'C': [4, 3, 2, 1]
    })
    
    corr_matrix = data.corr()
    
    # Test that matrix has the properties needed for heatmap
    assert not corr_matrix.isnull().all().all()  # Not all NaN
    assert corr_matrix.index.tolist() == corr_matrix.columns.tolist()  # Same index/columns

if __name__ == "__main__":
    pytest.main([__file__])
