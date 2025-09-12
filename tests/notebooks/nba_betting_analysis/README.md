# NBA Betting Analysis Notebook Tests

This directory contains comprehensive tests for the `nba_betting_analysis.ipynb` notebook functionality. Each test file corresponds to specific cells or groups of related cells in the notebook.

## Test Structure

### Test Files

1. **`test_cell_01_setup.py`** - Tests library imports and environment setup
   - Library import validation
   - Pandas configuration testing
   - Matplotlib/Seaborn style setup
   - Path configuration logic
   - Warnings filter setup

2. **`test_cell_02_data_loading.py`** - Tests data loading functionality
   - Features data loading from parquet
   - Games data loading from CSV files
   - Performance database detection
   - Missing file handling
   - Data summary calculations

3. **`test_cell_03_features_exploration.py`** - Tests features data exploration
   - Dataset information calculation
   - Feature categorization (rolling, season, basic, other)
   - Target variable analysis
   - Missing data handling
   - Sample data display functionality

4. **`test_cell_04_correlation_analysis.py`** - Tests correlation analysis
   - Numeric feature selection
   - Correlation matrix calculation
   - Target variable correlation ranking
   - Correlation insights calculation
   - Heatmap data preparation

5. **`test_cell_05_team_performance.py`** - Tests team performance analysis
   - Team-level aggregation calculations
   - Column flattening and renaming
   - Point differential calculations
   - Team performance insights
   - Form analysis (improving vs declining teams)

6. **`test_cell_06_model_performance.py`** - Tests model and betting performance
   - Database connectivity and queries
   - Model accuracy calculations
   - Betting performance metrics
   - Games analysis functionality
   - Precision/recall/F1 calculations

7. **`test_cell_07_visualizations.py`** - Tests visualization and advanced analytics
   - Weekly accuracy calculations
   - Confidence level bucketing
   - Cumulative P&L calculations
   - Bet size analysis
   - Streak analysis
   - Performance summary statistics

### Running Tests

#### Run All Tests
```bash
cd /Users/yirgareminick/Documents/NBA-Betting-Model/tests/notebooks/nba_betting_analysis
python run_all_tests.py
```

#### Run Individual Test Files
```bash
# Run setup tests
pytest test_cell_01_setup.py -v

# Run data loading tests
pytest test_cell_02_data_loading.py -v

# Run correlation analysis tests
pytest test_cell_04_correlation_analysis.py -v

# etc.
```

#### Run Specific Test Functions
```bash
# Test just the library imports
pytest test_cell_01_setup.py::test_required_libraries_import -v

# Test correlation matrix calculation
pytest test_cell_04_correlation_analysis.py::test_correlation_matrix_calculation -v
```

## Test Coverage

The tests cover:

### Functionality Testing
- ✅ Data loading and validation
- ✅ Statistical calculations and aggregations
- ✅ Correlation analysis and insights
- ✅ Team performance metrics
- ✅ Model performance evaluation
- ✅ Betting analysis and metrics
- ✅ Visualization data preparation

### Edge Cases
- ✅ Missing or empty data handling
- ✅ Single team/minimal data scenarios
- ✅ Invalid data type handling
- ✅ Database connectivity issues
- ✅ Missing file scenarios

### Data Validation
- ✅ Value range validation (0-1 for percentages, etc.)
- ✅ Data type consistency
- ✅ Mathematical operation correctness
- ✅ Aggregation logic verification

## Mock Data

The tests use realistic mock data that simulates:

- **NBA team performance data** with realistic statistics
- **Game results** with scores and team matchups
- **Model predictions** with confidence levels and outcomes
- **Betting records** with amounts, profits/losses, and dates
- **Performance databases** with structured SQL tables

## Dependencies

The tests require:
- `pytest` - Testing framework
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `sqlite3` - Database operations (built-in)
- `tempfile` - Temporary file/directory creation (built-in)
- `pathlib` - Path operations (built-in)

## Benefits

These tests provide:

1. **Confidence** - Verify notebook functionality works as expected
2. **Regression Prevention** - Catch issues when modifying notebook code
3. **Documentation** - Tests serve as executable documentation
4. **Validation** - Ensure data processing logic is correct
5. **Debugging** - Isolated testing helps identify specific issues

## Usage Notes

- Tests use mock data and don't require actual NBA data files
- Database tests create temporary SQLite databases
- All tests are isolated and don't affect each other
- Tests can be run individually or as a complete suite
- Failed tests provide detailed error messages for debugging

## Future Enhancements

Potential test additions:
- Integration tests with real data files
- Performance benchmarking tests
- Visualization output validation
- API integration tests
- End-to-end workflow tests
