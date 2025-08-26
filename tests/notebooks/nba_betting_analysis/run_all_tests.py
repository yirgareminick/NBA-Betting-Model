"""
Test runner for all NBA betting analysis notebook tests
"""
import pytest
import sys
from pathlib import Path

def run_all_tests():
    """Run all tests for the NBA betting analysis notebook."""
    test_dir = Path(__file__).parent
    
    # List of test modules to run
    test_modules = [
        'test_cell_01_setup.py',
        'test_cell_02_data_loading.py', 
        'test_cell_03_features_exploration.py',
        'test_cell_04_correlation_analysis.py',
        'test_cell_05_team_performance.py',
        'test_cell_06_model_performance.py',
        'test_cell_07_visualizations.py'
    ]
    
    print("Running Analysis Notebook Tests")
    print("=" * 60)
    
    all_passed = True
    results = {}
    
    for test_module in test_modules:
        test_path = test_dir / test_module
        if test_path.exists():
            print(f"\n📊 Running {test_module}...")
            
            # Run pytest for this specific module
            result = pytest.main([str(test_path), '-v'])
            results[test_module] = result
            
            if result != 0:
                all_passed = False
                print(f"❌ {test_module} FAILED")
            else:
                print(f"✅ {test_module} PASSED")
        else:
            print(f"⚠️  {test_module} not found")
            all_passed = False
    
    print("\n" + "=" * 60)
    print("📈 FINAL TEST RESULTS")
    print("=" * 60)
    
    for test_module, result in results.items():
        status = "✅ PASSED" if result == 0 else "❌ FAILED"
        print(f"{test_module:<40} {status}")
    
    if all_passed:
        print("\n🎉 All tests passed! The notebook functionality is working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the output above for details.")
        return 1

if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
