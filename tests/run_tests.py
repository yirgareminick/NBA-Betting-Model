#!/usr/bin/env python3
"""
Test Runner for NBA Betting Model

Runs all unit tests and integration tests with coverage reporting.

Usage:
    python tests/run_tests.py
    python tests/run_tests.py --unit-only
    python tests/run_tests.py --integration-only
    python tests/run_tests.py --coverage
"""

import argparse
import logging
import sys
import unittest
from pathlib import Path
import importlib.util

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from scripts.logging_config import setup_basic_logging
setup_basic_logging()
logger = logging.getLogger(__name__)


def discover_tests(test_dir: Path, pattern: str = "test_*.py"):
    """Discover and load tests from directory."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for test_file in test_dir.glob(pattern):
        # Import the test module
        spec = importlib.util.spec_from_file_location(
            test_file.stem, test_file
        )
        test_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_module)
        
        # Add tests to suite
        module_suite = loader.loadTestsFromModule(test_module)
        suite.addTest(module_suite)
    
    return suite


def run_unit_tests():
    """Run unit tests."""
    logger.info("🧪 Running unit tests...")
    
    unit_dir = Path(__file__).parent / "unit"
    suite = discover_tests(unit_dir)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_integration_tests():
    """Run integration tests."""
    logger.info("🔗 Running integration tests...")
    
    integration_dir = Path(__file__).parent / "integration"
    suite = discover_tests(integration_dir)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_all_tests():
    """Run all tests."""
    logger.info("🏀 NBA test suite")
    
    unit_success = run_unit_tests()
    logger.info("")
    integration_success = run_integration_tests()
    
    logger.info(f"✅ Unit: {'PASSED' if unit_success else 'FAILED'}, Integration: {'PASSED' if integration_success else 'FAILED'}")
    
    overall_success = unit_success and integration_success
    
    return overall_success


def run_with_coverage():
    """Run tests with coverage reporting."""
    try:
        import coverage
    except ImportError:
        logger.error("❌ Coverage package not installed. Install with: pip install coverage")
        return False
    
    logger.info("📊 Running with coverage...")
    
    # Start coverage
    cov = coverage.Coverage(source=['src'])
    cov.start()
    
    try:
        # Run tests
        success = run_all_tests()
        
        # Stop coverage and generate report
        cov.stop()
        cov.save()
        
        logger.info("📊 Coverage Report:")
        cov.report()
        
        # Generate HTML report
        html_dir = Path(__file__).parent.parent / "reports" / "coverage"
        html_dir.mkdir(parents=True, exist_ok=True)
        cov.html_report(directory=str(html_dir))
        logger.info(f"📊 HTML coverage report generated: {html_dir}/index.html")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Coverage analysis failed: {e}")
        return False


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="NBA Betting Model Test Runner")
    parser.add_argument("--unit-only", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration-only", action="store_true", help="Run only integration tests")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage analysis")
    
    args = parser.parse_args()
    
    try:
        if args.coverage:
            success = run_with_coverage()
        elif args.unit_only:
            success = run_unit_tests()
        elif args.integration_only:
            success = run_integration_tests()
        else:
            success = run_all_tests()
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Test runner failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
