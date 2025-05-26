"""
Demo Script: Using Individual Components
This script demonstrates how to use each modular component individually.
"""

import sys

# Add the analysis directory to Python path
sys.path.append("/home/callebalik/EnviDis/scripts/analysis")


def demo_data_generation():
    """Demonstrate data generation functionality."""
    print("=" * 50)
    print("DEMO: Data Generation")
    print("=" * 50)

    from scripts.analysis.time_series_analysis.demo.demo_data_generation import (
        generate_sample_data,
    )

    # Generate custom data
    data = generate_sample_data(start_year=2010, end_year=2020, seed=999)
    print(f"Generated data shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print("\nFirst 3 rows:")
    print(data.head(3))

    return data


def demo_basic_modeling(data):
    """Demonstrate basic model fitting."""
    print("\n" + "=" * 50)
    print("DEMO: Basic Model Fitting")
    print("=" * 50)

    from basic_model_fitting import BasicModelFitter

    fitter = BasicModelFitter(data)

    # Fit models
    poisson_results = fitter.fit_poisson_model()
    nb_results = fitter.fit_negative_binomial_model()

    print(f"Poisson AIC: {poisson_results.aic:.2f}")
    print(f"Negative Binomial AIC: {nb_results.aic:.2f}")

    # Check overdispersion
    overdispersion = fitter.check_overdispersion()
    print(f"Overdispersion: {overdispersion['interpretation']}")

    # Compare models
    comparison = fitter.compare_models()
    print(f"Best model: {comparison['chosen_model']}")

    return fitter.chosen_model


def demo_trend_modeling(data):
    """Demonstrate trend modeling."""
    print("\n" + "=" * 50)
    print("DEMO: Trend Modeling")
    print("=" * 50)

    from trend_modeling import TrendModelFitter

    trend_fitter = TrendModelFitter(data)

    # Fit different models
    linear = trend_fitter.fit_linear_model()
    quadratic = trend_fitter.fit_quadratic_model()
    cubic = trend_fitter.fit_cubic_model()

    print(f"Linear AIC: {linear.aic:.2f}")
    print(f"Quadratic AIC: {quadratic.aic:.2f}")
    print(f"Cubic AIC: {cubic.aic:.2f}")

    # Compare all models
    comparison = trend_fitter.compare_all_models()
    print(f"Best trend model: {comparison['best_overall']}")

    return comparison["best_model"]


def demo_autocorrelation_analysis(best_model, data):
    """Demonstrate autocorrelation analysis."""
    print("\n" + "=" * 50)
    print("DEMO: Autocorrelation Analysis")
    print("=" * 50)

    from autocorrelation_analysis import AutocorrelationAnalyzer, LaggedModelFitter

    # Analyze autocorrelation
    analyzer = AutocorrelationAnalyzer(best_model)
    dw_results = analyzer.durbin_watson_test()

    print(f"Durbin-Watson statistic: {dw_results['dw_statistic']:.3f}")
    print(f"Interpretation: {dw_results['interpretation']}")

    # If autocorrelation detected, fit lagged model
    if dw_results["significant_autocorrelation"]:
        print("\nFitting lagged model...")
        lagged_fitter = LaggedModelFitter(data, best_model)
        lagged_results = lagged_fitter.fit_lagged_model()

        # Check if autocorrelation was resolved
        lagged_analyzer = lagged_fitter.analyze_lagged_residuals()
        print("Lagged model fitted successfully!")


def demo_custom_analysis():
    """Demonstrate custom analysis workflow."""
    print("\n" + "=" * 50)
    print("DEMO: Custom Analysis Workflow")
    print("=" * 50)

    # Step 1: Generate custom data
    data = demo_data_generation()

    # Step 2: Basic modeling
    chosen_model = demo_basic_modeling(data)

    # Step 3: Trend modeling
    best_trend_model = demo_trend_modeling(data)

    # Step 4: Autocorrelation analysis
    demo_autocorrelation_analysis(best_trend_model, data)

    print("\n" + "=" * 50)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 50)


if __name__ == "__main__":
    demo_custom_analysis()
