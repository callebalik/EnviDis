#!/usr/bin/env python3
"""Demo Script: Showcasing Model Plotting Functionality
This script demonstrates the new plotting features in autocorrelation_analysis.py.
"""

import os
import sys

# Add the analysis directory to Python path
sys.path.append("/home/callebalik/EnviDis/scripts/analysis")

from envidis.time.analysis.autocorrelation_analysis import (
    AutocorrelationAnalyzer,
    LaggedModelFitter,
)
from envidis.time.analysis.basic_model_fitting import BasicModelFitter
from envidis.time.demo.data_generation import generate_sample_data


def demo_model_plotting() -> None:
    """Demonstrate the new plotting functionality."""
    print("=" * 60)
    print("DEMO: Model Plotting Functionality")
    print("=" * 60)

    # Generate sample data
    print("Generating sample data...")
    data = generate_sample_data(start_year=2000, end_year=2024, seed=42)
    print(f"Data shape: {data.shape}")

    # Fit a basic model
    print("\nFitting basic model...")
    fitter = BasicModelFitter(data)
    model_results = fitter.fit_negative_binomial_model()  # Use NB model
    print(f"Model AIC: {model_results.aic:.2f}")

    # Create analyzer and demonstrate plotting
    print("\nCreating diagnostic plots...")
    analyzer = AutocorrelationAnalyzer(model_results)

    # Individual plots
    print("1. Plotting fitted vs actual data...")
    analyzer.plot_fitted_vs_actual(data)

    print("2. Plotting residuals vs fitted values...")
    analyzer.plot_residuals_vs_fitted()

    print("3. Plotting ACF of residuals...")
    analyzer.plot_acf_residuals()

    # Check for autocorrelation
    dw_results = analyzer.durbin_watson_test()
    print(f"\nDurbin-Watson test: {dw_results['interpretation']}")

    if dw_results["significant_autocorrelation"]:
        print("\nFitting lagged model...")
        lagged_fitter = LaggedModelFitter(data, model_results)
        lagged_results = lagged_fitter.fit_lagged_model(lag_periods=1)
        print(f"Lagged model AIC: {lagged_results.aic:.2f}")

        print("4. Plotting lagged model fit...")
        lagged_fitter.plot_lagged_model_fit()

        print("5. Comparing base vs lagged models...")
        lagged_fitter.compare_models_plot()
    else:
        print("No significant autocorrelation detected - skipping lagged model.")

    print("\n" + "=" * 60)
    print("Demo completed! All plots should have been displayed.")
    print("=" * 60)


def demo_batch_plotting() -> None:
    """Demonstrate batch plotting with output directory."""
    print("\n" + "=" * 60)
    print("DEMO: Batch Plotting to Directory")
    print("=" * 60)

    # Create output directory
    output_dir = "/home/callebalik/EnviDis/results/analysis/demo_plots"
    os.makedirs(output_dir, exist_ok=True)

    # Generate data and fit model
    data = generate_sample_data(start_year=1990, end_year=2024, seed=123)
    fitter = BasicModelFitter(data)
    model_results = fitter.fit_negative_binomial_model()

    # Create analyzer and save plots
    analyzer = AutocorrelationAnalyzer(model_results)
    analyzer.create_diagnostic_plots(data, output_dir)

    print(f"\nPlots saved to: {output_dir}")
    print("Files created:")
    for file in os.listdir(output_dir):
        if file.endswith(".png"):
            print(f"  - {file}")


if __name__ == "__main__":
    try:
        # Demo interactive plotting
        demo_model_plotting()

        # Demo batch plotting
        demo_batch_plotting()

    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback

        traceback.print_exc()
