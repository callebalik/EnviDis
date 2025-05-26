#!/usr/bin/env python3
"""
Example script demonstrating the modular time series analysis framework.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from time_series_analysis import (
    DataPreparator,
    GLMModelFitter,
    TrendModelFitter,
    ComprehensivePlotter,
    TemporalAnalyzer,
    ModelSelector,
    ModelDiagnostics
)

def main():
    """Demonstrate the modular framework using real data."""

    print("=" * 60)
    print("MODULAR TIME SERIES ANALYSIS FRAMEWORK DEMO")
    print("=" * 60)

    # 1. Data Preparation
    print("\n1. LOADING AND PREPARING DATA")
    print("-" * 40)

    data_path = "/home/callebalik/EnviDis/data/raw/time-series/time_series.csv"
    preparator = DataPreparator()

    # Load and prepare data
    df = preparator.load_time_series_data(data_path)
    analysis_data = preparator.prepare_for_modeling(
        df,
        target_col='document_count',
        offset_col='co_compiled_count',
        create_time_features=True
    )

    print(f"✓ Loaded {len(analysis_data)} observations")
    print(f"✓ Date range: {analysis_data['date'].min()} to {analysis_data['date'].max()}")
    print(f"✓ Zero proportion: {(analysis_data['document_count'] == 0).mean():.1%}")

    # 2. Temporal Analysis
    print("\n2. TEMPORAL PATTERN ANALYSIS")
    print("-" * 40)

    temporal_analyzer = TemporalAnalyzer()
    temporal_results = temporal_analyzer.analyze_temporal_patterns(
        analysis_data,
        date_col='date',
        target_col='document_count'
    )

    # Print temporal summary
    temporal_summary = temporal_analyzer.generate_temporal_summary(temporal_results)
    print(temporal_summary)

    # 3. Model Fitting
    print("\n3. STATISTICAL MODEL FITTING")
    print("-" * 40)

    # Basic GLM models
    glm_fitter = GLMModelFitter(
        analysis_data,
        response_col='document_count',
        offset_col='co_compiled_count'
    )

    glm_models = glm_fitter.fit_all_models(['time_numeric'])

    # Trend models
    trend_fitter = TrendModelFitter(
        analysis_data,
        response_col='document_count',
        offset_col='co_compiled_count',
        time_var='time_numeric'
    )

    trend_models = trend_fitter.fit_all_trend_models()

    # Combine all models
    all_models = {**glm_models, **trend_models}

    # 4. Model Selection and Comparison
    print("\n4. MODEL SELECTION AND COMPARISON")
    print("-" * 40)

    selector = ModelSelector()
    comparison_results = selector.compare_models(all_models)

    # Print comparison report
    comparison_report = selector.generate_model_comparison_report(comparison_results)
    print(comparison_report)

    # Get best model
    best_name, best_model = selector.select_best_model(all_models, criterion='aic')
    print(f"\n✓ Best model: {best_name}")

    # 5. Model Diagnostics
    print("\n5. MODEL DIAGNOSTICS")
    print("-" * 40)

    diagnostics_analyzer = ModelDiagnostics()
    diagnostics = diagnostics_analyzer.run_comprehensive_diagnostics(
        best_model,
        analysis_data,
        'document_count'
    )

    # Print diagnostics report
    diagnostics_report = diagnostics_analyzer.generate_diagnostics_report(diagnostics)
    print(diagnostics_report)

    # 6. Visualization
    print("\n6. COMPREHENSIVE VISUALIZATION")
    print("-" * 40)

    plotter = ComprehensivePlotter()

    # Create output directory
    output_dir = "/home/callebalik/EnviDis/results/analysis/modular_demo"
    os.makedirs(output_dir, exist_ok=True)

    # Generate comprehensive plots
    figures = plotter.create_comprehensive_report(
        analysis_data,
        all_models,
        date_col='date',
        target_col='document_count',
        output_dir=output_dir
    )

    print(f"✓ Generated {len(figures)} visualization plots")
    print(f"✓ Plots saved to: {output_dir}")

    # 7. Summary Report
    print("\n7. ANALYSIS SUMMARY")
    print("-" * 40)

    print("FRAMEWORK COMPONENTS USED:")
    print("• DataPreparator - Data loading and preparation")
    print("• TemporalAnalyzer - Temporal pattern analysis")
    print("• GLMModelFitter - Basic GLM model fitting")
    print("• TrendModelFitter - Polynomial and spline trends")
    print("• ModelSelector - Model comparison and selection")
    print("• ModelDiagnostics - Model validation and diagnostics")
    print("• ComprehensivePlotter - Visualization and reporting")

    print("\nANALYSIS RESULTS:")
    print(f"• Best Model: {best_name}")
    print(f"• Model Quality: {diagnostics.get('overall_assessment', {}).get('overall_quality', 'Unknown')}")
    print(f"• Data Span: {temporal_results['basic_stats']['date_range_years']:.1f} years")
    print(f"• Total Observations: {temporal_results['basic_stats']['total_days']:,}")

    print("\n" + "=" * 60)
    print("MODULAR FRAMEWORK DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
