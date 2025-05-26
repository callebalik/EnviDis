#!/usr/bin/env python3
"""
Enhanced Trend Visualization Demo
This script demonstrates the new enhanced trend plotting features with document histograms and normalization.
"""

import sys
import os

# Add the analysis directory to Python path
sys.path.append("/home/callebalik/EnviDis/scripts/analysis")

from envidis.time.demo.data_generation import generate_sample_data
from envidis.time.plots.data_visualization import (
    plot_entities_with_documents_histogram,
    plot_entities_normalized_and_absolute,
    create_all_plots,
)
from trend_modeling import TrendModelFitter


def demo_enhanced_visualizations():
    """Demonstrate the enhanced visualization capabilities."""
    print("=" * 70)
    print("ENHANCED TREND VISUALIZATION DEMO")
    print("=" * 70)

    # Generate sample data
    print("Generating sample data with complex trends...")
    data = generate_sample_data(
        start_year=1990, end_year=2024, seed=42, generation_type="non_constant_upward"
    )
    print(f"Generated data shape: {data.shape}")
    print(f"Year range: {data.index.min()} - {data.index.max()}")

    # Show basic statistics
    print("\nData Summary:")
    print(
        f"  Entities range: {data['ObservedEntities'].min()} - {data['ObservedEntities'].max()}"
    )
    print(
        f"  Documents range: {data['TotalDocuments'].min()} - {data['TotalDocuments'].max()}"
    )
    print(
        f"  Mean entities per 1000 docs: {(data['ObservedEntities'] / data['TotalDocuments'] * 1000).mean():.2f}"
    )

    # Fit trend models (always on absolute counts)
    print("\n" + "=" * 50)
    print("FITTING MODELS (on absolute counts)")
    print("=" * 50)

    print("Fitting trend models...")
    trend_fitter = TrendModelFitter(data)
    trend_fitter.fit_all_models()

    comparison = trend_fitter.compare_all_models()
    print(f"Best model: {comparison['best_overall']}")
    print(f"Model AIC: {comparison['best_model'].aic:.2f}")

    # Demonstrate both visualization approaches
    print("\n" + "=" * 50)
    print("VISUALIZATION APPROACH 1: Post-Fitting Normalization")
    print("=" * 50)
    print("This normalizes the fitted trend for visualization while keeping")
    print("the statistical model based on absolute counts.")

    print("\nCreating normalized visualization...")
    normalized_viz = trend_fitter.plot_enhanced_trends_with_documents(
        normalize_visualization=True
    )

    print("\n" + "=" * 50)
    print("VISUALIZATION APPROACH 2: Absolute Values")
    print("=" * 50)
    print("This shows the raw fitted trend without normalization.")

    print("\nCreating absolute visualization...")
    absolute_viz = trend_fitter.plot_enhanced_trends_with_documents(
        normalize_visualization=False
    )

    # Compare the approaches
    print("\n" + "=" * 50)
    print("COMPARISON OF APPROACHES")
    print("=" * 50)

    print("Benefits of Post-Fitting Normalization:")
    print("✓ Maintains statistical validity (model fitted on proper counts)")
    print("✓ Easier visual interpretation (accounts for document volume)")
    print("✓ Shows entity density trends rather than absolute volume")
    print("✓ No impact on model selection or statistical inference")

    print("\nBenefits of Absolute Visualization:")
    print("✓ Direct representation of model predictions")
    print("✓ Shows actual entity count trends")
    print("✓ Useful for understanding total impact/volume")

    # Show the difference in trend patterns
    abs_trend_range = (
        absolute_viz["model_results"].fittedvalues.max()
        - absolute_viz["model_results"].fittedvalues.min()
    )
    norm_trend_range = (
        normalized_viz["normalized_fitted"].max()
        - normalized_viz["normalized_fitted"].min()
    )

    print("\nTrend Magnitude Comparison:")
    print(f"  Absolute trend range: {abs_trend_range:.1f} entities")
    print(f"  Normalized trend range: {norm_trend_range:.2f} entities/1000 docs")

    # Show enhanced visualizations
    print("\n" + "=" * 50)
    print("ENHANCED VISUALIZATIONS")
    print("=" * 50)

    print("Creating entities plot with document histogram...")
    plot_entities_with_documents_histogram(data)

    print("Creating normalized vs absolute comparison plots...")
    normalized_data = plot_entities_normalized_and_absolute(data)

    # Fit trend models and create enhanced trend plots
    print("\n" + "=" * 50)
    print("2. ENHANCED TREND MODELING PLOTS")
    print("=" * 50)

    print("Fitting trend models...")
    trend_fitter = TrendModelFitter(data)
    trend_fitter.fit_all_models()

    print("Creating comprehensive trend analysis plots...")
    normalized_results = trend_fitter.plot_enhanced_trends_with_documents()

    # Print summary
    print("\n" + "=" * 50)
    print("3. ANALYSIS SUMMARY")
    print("=" * 50)

    comparison = trend_fitter.compare_all_models()
    print(f"Best overall model: {comparison['best_overall']}")
    print(f"Best model AIC: {comparison['best_model'].aic:.2f}")

    if normalized_results:
        print(f"Normalized model AIC: {normalized_results.aic:.2f}")
        # If normalized model has better AIC:
        if normalized_results.aic < comparison["best_model"].aic:
            print("✓ Document normalization improves model fit!")
            # This suggests the temporal trend is better explained as a rate change
        else:
            print("✓ Absolute counts provide better fit than normalized data")
            # This suggests the temporal trend is about absolute volume changes

    print("\nModel Performance (AIC):")
    if trend_fitter.linear_results:
        print(f"  Linear: {trend_fitter.linear_results.aic:.2f}")
    if trend_fitter.poly2_results:
        print(f"  Quadratic: {trend_fitter.poly2_results.aic:.2f}")
    if trend_fitter.poly3_results:
        print(f"  Cubic: {trend_fitter.poly3_results.aic:.2f}")
    if trend_fitter.spline_results:
        print(f"  Spline: {trend_fitter.spline_results.aic:.2f}")


def demo_batch_output():
    """Demonstrate batch output to directory."""
    print("\n" + "=" * 70)
    print("BATCH OUTPUT DEMO")
    print("=" * 70)

    # Create output directory
    output_dir = "/home/callebalik/EnviDis/results/analysis/enhanced_demo_plots"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Generate different data scenarios
    scenarios = [
        {"name": "linear_trend", "seed": 123, "type": "constant_upward"},
        {"name": "complex_trend", "seed": 456, "type": "non_constant_upward"},
    ]

    for scenario in scenarios:
        print(f"\nProcessing scenario: {scenario['name']}")

        # Generate data
        data = generate_sample_data(
            start_year=1985,
            end_year=2024,
            seed=scenario["seed"],
            generation_type=scenario["type"],
        )

        # Create scenario-specific output directory
        scenario_dir = f"{output_dir}/{scenario['name']}"
        os.makedirs(scenario_dir, exist_ok=True)

        # Generate all basic plots
        create_all_plots(data, scenario_dir)

        # Generate enhanced trend plots
        trend_fitter = TrendModelFitter(data)
        trend_fitter.fit_all_models()
        trend_fitter.plot_enhanced_trends_with_documents(
            f"{scenario_dir}/enhanced_trend_analysis.png"
        )

        print(f"  Saved plots to: {scenario_dir}")

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    try:
        # Run main demo
        demo_enhanced_visualizations()

        # Run batch output demo
        demo_batch_output()

        print("\n" + "=" * 70)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("Features demonstrated:")
        print("✓ Joint entity and document histogram plots")
        print("✓ Document-normalized trend analysis")
        print("✓ Side-by-side comparison of absolute vs normalized trends")
        print("✓ Enhanced model comparison visualizations")
        print("✓ Batch output to organized directories")
        print("✓ Multiple data generation scenarios")

    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback

        traceback.print_exc()
