"""
Test Script for Modular Time Series Analysis
This script tests each component individually to ensure they work correctly.
"""

import sys

# Add the analysis directory to Python path
sys.path.append("/home/callebalik/EnviDis/scripts/analysis")


def test_data_generation():
    """Test data generation functionality."""
    print("Testing data generation...")
    try:
        from scripts.analysis.time_series_analysis.demo.demo_data_generation import (
            generate_sample_data,
        )

        data = generate_sample_data(start_year=2020, end_year=2024, seed=123)
        print(f"✓ Data generation successful. Shape: {data.shape}")
        print(f"✓ Columns: {list(data.columns)}")
        return True
    except Exception as e:
        print(f"✗ Data generation failed: {e}")
        return False


def test_basic_model_fitting():
    """Test basic model fitting functionality."""
    print("\nTesting basic model fitting...")
    try:
        from scripts.analysis.time_series_analysis.demo.demo_data_generation import (
            generate_sample_data,
        )
        from basic_model_fitting import BasicModelFitter

        data = generate_sample_data(start_year=2020, end_year=2024, seed=123)
        fitter = BasicModelFitter(data)

        # Test individual model fitting
        poisson_results = fitter.fit_poisson_model()
        nb_results = fitter.fit_negative_binomial_model()

        print(f"✓ Poisson model AIC: {poisson_results.aic:.2f}")
        print(f"✓ Negative Binomial model AIC: {nb_results.aic:.2f}")

        # Test overdispersion check
        overdispersion = fitter.check_overdispersion()
        print(f"✓ Overdispersion check: {overdispersion['interpretation']}")

        # Test model comparison
        comparison = fitter.compare_models()
        print(f"✓ Best model: {comparison['chosen_model']}")

        return True
    except Exception as e:
        print(f"✗ Basic model fitting failed: {e}")
        return False


def test_trend_modeling():
    """Test trend modeling functionality."""
    print("\nTesting trend modeling...")
    try:
        from scripts.analysis.time_series_analysis.demo.demo_data_generation import (
            generate_sample_data,
        )
        from trend_modeling import TrendModelFitter

        data = generate_sample_data(start_year=2020, end_year=2024, seed=123)
        trend_fitter = TrendModelFitter(data)

        # Test polynomial models
        linear_results = trend_fitter.fit_linear_model()
        quad_results = trend_fitter.fit_quadratic_model()
        cubic_results = trend_fitter.fit_cubic_model()

        print(f"✓ Linear model AIC: {linear_results.aic:.2f}")
        print(f"✓ Quadratic model AIC: {quad_results.aic:.2f}")
        print(f"✓ Cubic model AIC: {cubic_results.aic:.2f}")

        # Test spline model
        spline_results = trend_fitter.fit_spline_model(df=3)
        print(f"✓ Spline model AIC: {spline_results.aic:.2f}")

        # Test model comparison
        comparison = trend_fitter.compare_all_models()
        print(f"✓ Best trend model: {comparison['best_overall']}")

        return True
    except Exception as e:
        print(f"✗ Trend modeling failed: {e}")
        return False


def test_autocorrelation_analysis():
    """Test autocorrelation analysis functionality."""
    print("\nTesting autocorrelation analysis...")
    try:
        from scripts.analysis.time_series_analysis.demo.demo_data_generation import (
            generate_sample_data,
        )
        from trend_modeling import TrendModelFitter
        from autocorrelation_analysis import AutocorrelationAnalyzer

        data = generate_sample_data(start_year=2020, end_year=2024, seed=123)

        # Get a fitted model
        trend_fitter = TrendModelFitter(data)
        model_results = trend_fitter.fit_linear_model()

        # Test autocorrelation analysis
        analyzer = AutocorrelationAnalyzer(model_results)
        dw_results = analyzer.durbin_watson_test()
        acf_values = analyzer.calculate_acf(nlags=5)

        print(f"✓ Durbin-Watson statistic: {dw_results['dw_statistic']:.3f}")
        print(f"✓ Autocorrelation interpretation: {dw_results['interpretation']}")
        print(f"✓ ACF values calculated: {len(acf_values)} lags")

        return True
    except Exception as e:
        print(f"✗ Autocorrelation analysis failed: {e}")
        return False


def run_all_tests():
    """Run all component tests."""
    print("=" * 60)
    print("RUNNING MODULAR ANALYSIS COMPONENT TESTS")
    print("=" * 60)

    tests = [
        test_data_generation,
        test_basic_model_fitting,
        test_trend_modeling,
        test_autocorrelation_analysis,
    ]

    results = []
    for test in tests:
        results.append(test())

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("✓ All tests passed! The modular scripts are working correctly.")
    else:
        print("✗ Some tests failed. Please check the error messages above.")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
