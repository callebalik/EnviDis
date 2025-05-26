#!/usr/bin/env python3
"""Demonstration script for the complete modular time series analysis framework.

This script shows how to use all components of the time series analysis package
for both synthetic and real data analysis.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add the time series analysis package to the path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

# Import all components of the time series analysis package
from time_series_analysis.analysis.diagnostics import perform_comprehensive_diagnostics
from time_series_analysis.analysis.model_selection import (
    calculate_akaike_weights,
    compare_models,
    generate_model_comparison_report,
    rank_models,
)
from time_series_analysis.analysis.temporal_analysis_new import (
    perform_comprehensive_temporal_analysis,
)
from time_series_analysis.demo.demo_data_generation import (
    generate_constant_trend_data,
    generate_non_constant_trend_data,
)
from time_series_analysis.modeling.base_models import (
    fit_hurdle_model,
    fit_negative_binomial_glm,
    fit_poisson_glm,
    fit_zero_inflated_model,
)
from time_series_analysis.modeling.trend_models import (
    fit_polynomial_trends,
    fit_spline_trends,
)
from time_series_analysis.plotting.comprehensive_plots import (
    create_comprehensive_analysis_plots,
)
from time_series_analysis.plotting.diagnostic_plots import create_diagnostic_plots
from time_series_analysis.plotting.trend_plots import create_trend_plots
from time_series_analysis.utils.data_preparation import (
    load_and_prepare_data,
    preprocess_data,
    validate_data_structure,
)
from time_series_analysis.utils.modeling_utils import (
    calculate_model_diagnostics,
    create_offset_variable,
)


def run_synthetic_data_demo() -> None:
    """Demonstrate the framework with synthetic data."""
    print("=" * 60)
    print("SYNTHETIC DATA DEMONSTRATION")
    print("=" * 60)

    # Create output directory for synthetic data results
    output_dir = Path("demo_results/synthetic_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate synthetic data
    print("\n1. Generating synthetic data...")

    # Generate constant trend data
    data_constant = generate_constant_trend_data(
        n_entities=20,
        n_time_points=24,
        base_intercept=2.0,
        trend_slope=0.05,
        noise_level=0.1,
        random_seed=42,
    )

    # Generate non-constant trend data
    data_non_constant = generate_non_constant_trend_data(
        n_entities=20,
        n_time_points=24,
        base_intercept=2.0,
        trend_coefficients=[0.05, -0.001],
        noise_level=0.15,
        random_seed=42,
    )

    print(f"Generated constant trend data: {data_constant.shape}")
    print(f"Generated non-constant trend data: {data_non_constant.shape}")

    # Use constant trend data for main demo
    demo_data = data_constant.copy()

    # 2. Data preparation and validation
    print("\n2. Preparing and validating data...")

    # Validate data structure
    validation_result = validate_data_structure(demo_data)
    print(f"Data validation passed: {validation_result['is_valid']}")
    if not validation_result["is_valid"]:
        print(f"Validation issues: {validation_result['issues']}")
        return

    # Preprocess data
    processed_data = preprocess_data(demo_data)
    print(f"Processed data shape: {processed_data.shape}")

    # Create offset variable
    processed_data = create_offset_variable(processed_data, "TotalDocuments")

    # 3. Temporal analysis
    print("\n3. Performing temporal analysis...")

    temporal_results = perform_comprehensive_temporal_analysis(
        processed_data,
        time_col="time",
        target_col="AffectedEntities",
    )

    print("Temporal analysis completed.")
    print(
        f"Detected trend: {temporal_results.get('trend_detection', {}).get('trend_type', 'Unknown')}",
    )

    # 4. Model fitting
    print("\n4. Fitting models...")

    model_results = {}

    # Base GLM models
    try:
        model_results["Poisson"] = fit_poisson_glm(
            processed_data,
            "AffectedEntities ~ time",
            "log_offset",
        )
        print("✓ Poisson GLM fitted")
    except Exception as e:
        print(f"✗ Poisson GLM failed: {e}")

    try:
        model_results["NegBinomial"] = fit_negative_binomial_glm(
            processed_data,
            "AffectedEntities ~ time",
            "log_offset",
        )
        print("✓ Negative Binomial GLM fitted")
    except Exception as e:
        print(f"✗ Negative Binomial GLM failed: {e}")

    # Trend models
    try:
        poly_results = fit_polynomial_trends(
            processed_data,
            time_col="time",
            target_col="AffectedEntities",
            max_degree=3,
        )
        for degree, result in poly_results.items():
            model_results[f"Polynomial_{degree}"] = result
        print("✓ Polynomial trends fitted (degrees 1-3)")
    except Exception as e:
        print(f"✗ Polynomial trends failed: {e}")

    # 5. Model comparison
    print("\n5. Comparing models...")

    if model_results:
        comparison_results = compare_models(model_results)
        akaike_weights = calculate_akaike_weights(model_results)
        model_ranking = rank_models(model_results)

        print("Model comparison completed.")
        print(f"Best model: {model_ranking[0]['model']}")

        # Generate comparison report
        report = generate_model_comparison_report(
            model_results,
            comparison_results,
            akaike_weights,
            model_ranking,
        )

        # Save report
        report_path = output_dir / "model_comparison_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        print(f"Model comparison report saved to: {report_path}")

    # 6. Model diagnostics
    print("\n6. Performing model diagnostics...")

    if model_results:
        best_model_name = model_ranking[0]["model"]
        best_model_result = model_results[best_model_name]

        diagnostic_results = perform_comprehensive_diagnostics(
            best_model_result["model"],
            processed_data,
            "AffectedEntities",
        )

        print(f"Diagnostics completed for {best_model_name}")
        print(
            f"Overall quality score: {diagnostic_results.get('overall_score', 'N/A'):.3f}",
        )

    # 7. Generate comprehensive plots
    print("\n7. Creating visualizations...")

    try:
        # Comprehensive analysis plots
        comp_figures = create_comprehensive_analysis_plots(
            processed_data,
            model_results,
            temporal_results,
            time_col="time",
            target_col="AffectedEntities",
            save_dir=str(output_dir),
        )
        print("✓ Comprehensive analysis plots created")

        # Diagnostic plots for best model
        if model_results and "best_model_result" in locals():
            diag_figures = create_diagnostic_plots(
                processed_data,
                best_model_result,
                save_dir=str(output_dir),
            )
            print("✓ Diagnostic plots created")

        # Trend plots
        trend_figures = create_trend_plots(
            processed_data,
            {"trend_analysis": temporal_results, "model_results": model_results},
            time_col="time",
            target_col="AffectedEntities",
            save_dir=str(output_dir),
        )
        print("✓ Trend plots created")

    except Exception as e:
        print(f"✗ Plotting failed: {e}")

    print(f"\nSynthetic data demo completed. Results saved to: {output_dir}")


def run_real_data_demo(data_path: str) -> None:
    """Demonstrate the framework with real data."""
    print("=" * 60)
    print("REAL DATA DEMONSTRATION")
    print("=" * 60)

    # Create output directory for real data results
    output_dir = Path("demo_results/real_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and prepare real data
    print("\n1. Loading and preparing real data...")

    try:
        real_data = load_and_prepare_data(data_path, data_type="real")
        print(f"Loaded real data: {real_data.shape}")

        # Validate data structure
        validation_result = validate_data_structure(real_data)
        print(f"Data validation passed: {validation_result['is_valid']}")

        if not validation_result["is_valid"]:
            print(f"Validation issues: {validation_result['issues']}")
            return

        # Preprocess data
        processed_data = preprocess_data(real_data)
        processed_data = create_offset_variable(processed_data, "TotalDocuments")

    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        print("Skipping real data demo...")
        return

    # 2. Run similar analysis as synthetic data
    print("\n2. Performing temporal analysis...")

    temporal_results = perform_comprehensive_temporal_analysis(
        processed_data,
        time_col="time",
        target_col="AffectedEntities",
    )

    # 3. Model fitting (same as synthetic)
    print("\n3. Fitting models...")

    model_results = {}

    # Fit available models
    for model_name, model_func in [
        (
            "Poisson",
            lambda: fit_poisson_glm(
                processed_data,
                "AffectedEntities ~ time",
                "log_offset",
            ),
        ),
        (
            "NegBinomial",
            lambda: fit_negative_binomial_glm(
                processed_data,
                "AffectedEntities ~ time",
                "log_offset",
            ),
        ),
    ]:
        try:
            model_results[model_name] = model_func()
            print(f"✓ {model_name} fitted")
        except Exception as e:
            print(f"✗ {model_name} failed: {e}")

    # 4. Model comparison and diagnostics
    if model_results:
        print("\n4. Model comparison and diagnostics...")

        comparison_results = compare_models(model_results)
        model_ranking = rank_models(model_results)

        best_model_name = model_ranking[0]["model"]
        best_model_result = model_results[best_model_name]

        diagnostic_results = perform_comprehensive_diagnostics(
            best_model_result["model"],
            processed_data,
            "AffectedEntities",
        )

        print(f"Best model: {best_model_name}")
        print(
            f"Overall quality score: {diagnostic_results.get('overall_score', 'N/A'):.3f}",
        )

    # 5. Generate plots
    print("\n5. Creating visualizations...")

    try:
        comp_figures = create_comprehensive_analysis_plots(
            processed_data,
            model_results,
            temporal_results,
            time_col="time",
            target_col="AffectedEntities",
            save_dir=str(output_dir),
        )
        print("✓ Real data analysis plots created")

    except Exception as e:
        print(f"✗ Plotting failed: {e}")

    print(f"\nReal data demo completed. Results saved to: {output_dir}")


def run_comparison_demo() -> None:
    """Compare synthetic and real data analysis results."""
    print("=" * 60)
    print("COMPARISON DEMONSTRATION")
    print("=" * 60)

    # Create output directory for comparison
    output_dir = Path("demo_results/comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n1. Generating comparison data...")

    # Generate different types of synthetic data
    data_types = {
        "Linear": generate_constant_trend_data(20, 24, 2.0, 0.05, 0.1, 42),
        "Quadratic": generate_non_constant_trend_data(
            20,
            24,
            2.0,
            [0.05, -0.001],
            0.15,
            42,
        ),
        "High_Noise": generate_constant_trend_data(20, 24, 2.0, 0.05, 0.3, 123),
    }

    comparison_results = {}

    for data_name, data in data_types.items():
        print(f"\n2. Analyzing {data_name} data...")

        try:
            # Prepare data
            processed_data = preprocess_data(data)
            processed_data = create_offset_variable(processed_data, "TotalDocuments")

            # Fit models
            model_results = {}

            try:
                model_results["Poisson"] = fit_poisson_glm(
                    processed_data,
                    "AffectedEntities ~ time",
                    "log_offset",
                )
                model_results["NegBinomial"] = fit_negative_binomial_glm(
                    processed_data,
                    "AffectedEntities ~ time",
                    "log_offset",
                )
            except Exception as e:
                print(f"Model fitting failed for {data_name}: {e}")
                continue

            # Compare models
            comparison = compare_models(model_results)
            ranking = rank_models(model_results)

            comparison_results[data_name] = {
                "best_model": ranking[0]["model"],
                "best_aic": ranking[0]["aic"],
                "model_count": len(model_results),
                "data_shape": processed_data.shape,
            }

            print(
                f"✓ {data_name}: Best model = {ranking[0]['model']}, AIC = {ranking[0]['aic']:.2f}",
            )

        except Exception as e:
            print(f"✗ Analysis failed for {data_name}: {e}")

    # 3. Generate comparison summary
    print("\n3. Generating comparison summary...")

    summary_text = "COMPARISON SUMMARY\n" + "=" * 50 + "\n\n"

    for data_name, results in comparison_results.items():
        summary_text += f"{data_name} Data:\n"
        summary_text += f"  Best Model: {results['best_model']}\n"
        summary_text += f"  Best AIC: {results['best_aic']:.2f}\n"
        summary_text += f"  Models Fitted: {results['model_count']}\n"
        summary_text += f"  Data Shape: {results['data_shape']}\n\n"

    # Save comparison summary
    summary_path = output_dir / "comparison_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text)

    print(f"Comparison summary saved to: {summary_path}")
    print(f"\nComparison demo completed. Results saved to: {output_dir}")


def main() -> None:
    """Main demonstration function."""
    print("MODULAR TIME SERIES ANALYSIS FRAMEWORK DEMONSTRATION")
    print("=" * 60)

    # Ensure output directory exists
    Path("demo_results").mkdir(exist_ok=True)

    # Run synthetic data demonstration
    try:
        run_synthetic_data_demo()
    except Exception as e:
        print(f"Synthetic data demo failed: {e}")

    print("\n" + "=" * 60)

    # Run real data demonstration if data file exists
    real_data_path = "data/environmental_time_series.csv"  # Adjust path as needed
    if Path(real_data_path).exists():
        try:
            run_real_data_demo(real_data_path)
        except Exception as e:
            print(f"Real data demo failed: {e}")
    else:
        print("Real data file not found. Skipping real data demo.")
        print(f"Expected path: {real_data_path}")

    print("\n" + "=" * 60)

    # Run comparison demonstration
    try:
        run_comparison_demo()
    except Exception as e:
        print(f"Comparison demo failed: {e}")

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETED")
    print("=" * 60)
    print("\nResults have been saved to the 'demo_results' directory.")
    print("Check the following subdirectories:")
    print("  - demo_results/synthetic_data/")
    print("  - demo_results/real_data/")
    print("  - demo_results/comparison/")


if __name__ == "__main__":
    main()
