"""
Test for Cell 2: Library imports and environment setup
"""
import pytest
import sys
from pathlib import Path
import tempfile
import os

def test_required_libraries_import():
    """Test that all required libraries can be imported successfully."""
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import warnings
        import sqlite3
        from datetime import datetime, date
        import yaml
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import required library: {e}")

def test_pandas_configuration():
    """Test pandas display configuration."""
    import pandas as pd
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    assert pd.get_option('display.max_columns') is None
    assert pd.get_option('display.width') == 1000

def test_matplotlib_style_setting():
    """Test matplotlib style configuration."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    try:
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        assert True
    except Exception as e:
        pytest.fail(f"Failed to set plotting styles: {e}")

def test_path_setup():
    """Test path configuration logic."""
    from pathlib import Path
    
    # Mock the notebook directory structure
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        notebooks_dir = temp_path / "notebooks"
        data_dir = temp_path / "data"
        processed_dir = data_dir / "processed"
        raw_dir = data_dir / "raw"
        
        # Create the directory structure
        notebooks_dir.mkdir()
        data_dir.mkdir()
        processed_dir.mkdir()
        raw_dir.mkdir()
        
        # Test path logic (simulating notebook execution from notebooks/ dir)
        os.chdir(notebooks_dir)
        PROJECT_ROOT = Path('.').absolute().parent
        DATA_PATH = PROJECT_ROOT / 'data'
        PROCESSED_PATH = DATA_PATH / 'processed'
        RAW_PATH = DATA_PATH / 'raw'
        
        assert DATA_PATH.exists()
        assert PROCESSED_PATH.exists()
        assert RAW_PATH.exists()

def test_warnings_filter():
    """Test warnings filter setup."""
    import warnings
    
    # Test that warnings filter can be set
    warnings.filterwarnings('ignore')
    assert True  # If no exception raised, test passes

if __name__ == "__main__":
    pytest.main([__file__])
