#!/usr/bin/env python3
"""
Test script for the enhanced comprehensive analysis report with real data.
"""

import sys
import os

# Add paths
sys.path.append("/home/callebalik/EnviDis/scripts/time_series_analysis")
sys.path.append("/home/callebalik/EnviDis/scripts/analysis")

from comprehensive_analysis_report import ComprehensiveAnalysisReport, main_real_data


def test_synthetic_data():
    """Test with synthetic data."""
    print("Testing with synthetic data...")

    try:
        # Create report with synthetic data
        report = ComprehensiveAnalysisReport(use_real_data=False)
        results = report.generate_report()

        print("✓ Synthetic data analysis completed successfully")
        return True

    except Exception as e:
        print(f"✗ Synthetic data analysis failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_real_data():
    """Test with real data."""
    print("Testing with real data...")

    # Check if real data file exists
    data_path = "/home/callebalik/EnviDis/data/raw/time-series/time_series.csv"

    if not os.path.exists(data_path):
        print(f"✗ Real data file not found: {data_path}")
        return False

    try:
        # Create report with real data
        report = ComprehensiveAnalysisReport(use_real_data=True, data_path=data_path)
        results = report.generate_report()

        print("✓ Real data analysis completed successfully")
        print(f"✓ Data shape: {report.data.shape}")
        print(f"✓ Date range: {report.data.index.min()} to {report.data.index.max()}")
        return True

    except Exception as e:
        print(f"✗ Real data analysis failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING ENHANCED COMPREHENSIVE ANALYSIS")
    print("=" * 60)

    tests_passed = 0
    total_tests = 2

    # Test 1: Synthetic data
    if test_synthetic_data():
        tests_passed += 1

    print("\n" + "-" * 40)

    # Test 2: Real data
    if test_real_data():
        tests_passed += 1

    print("\n" + "=" * 60)
    print(f"TESTING COMPLETED: {tests_passed}/{total_tests} tests passed")
    print("=" * 60)

    return tests_passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
