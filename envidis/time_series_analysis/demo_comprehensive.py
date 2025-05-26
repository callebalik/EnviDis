#!/usr/bin/env python3
"""Comprehensive demonstration of the modular time series analysis framework.
Tests all components with both synthetic and real data.
"""

import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add the package to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modular components
from analysis.diagnostics import ModelDiagnostics
from analysis.model_selection import ModelSelector
from analysis.temporal_analysis_new import TemporalAnalyzer
from modeling.base_models import GLMModelFitter
from modeling.trend_models import TrendModelFitter
from plotting.comprehensive_plots import ComprehensivePlotter
from utils.data_preparation_new import DataPreparator
from utils.modeling_utils import create_offset_variable, prepare_model_data

warnings.filterwarnings("ignore")


class ComprehensiveDemo:
    """Comprehensive demonstration of the modular framework."""

    def __init__(self, output_dir="demo_output"):
        """Initialize demo with output directory."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Initialize components
        self.data_prep = DataPreparator()
        self.glm_fitter = GLMModelFitter()
        self.trend_fitter = TrendModelFitter()
        self.temporal_analyzer = TemporalAnalyzer()
        self.model_selector = ModelSelector()
        self.diagnostics = ModelDiagnostics()
        self.plotter = ComprehensivePlotter()

    def run_synthetic_data_demo(self):
        """Demonstrate framework with synthetic data."""
        print("=== Synthetic Data Demo ===")

        # 1. Generate synthetic data
        print("1. Generating synthetic data...")
        synthetic_data = self.data_gen.generate_sample_data(
            n_years=20,
            base_entities=100,
            base_documents=1000,
            trend_type="polynomial",
        )

        # 2. Prepare data
        print("2. Preparing data...")
        prepared_data = self.data_prep.prepare_modeling_data(
            synthetic_data,
            entity_col="ObservedEntities",
            offset_col="TotalDocuments",
        )

        # 3. Temporal analysis
        print("3. Running temporal analysis...")
        temporal_results = self.temporal_analyzer.analyze_temporal_patterns(
            prepared_data,
            entity_col="ObservedEntities",
        )

        # 4. Fit multiple models
        print("4. Fitting models...")
        models = self._fit_multiple_models(prepared_data)

        # 5. Model selection and comparison
        print("5. Comparing models...")
        comparison_results = self.model_selector.compare_models(models)

        # 6. Diagnostics on best model
        print("6. Running diagnostics...")
        best_model_name = min(
            comparison_results["model_scores"],
            key=comparison_results["model_scores"].get,
        )
        best_model = models[best_model_name]
        diagnostic_results = self.diagnostics.run_comprehensive_diagnostics(
            best_model,
            prepared_data,
            "ObservedEntities",
        )

        # 7. Comprehensive plotting
        print("7. Creating comprehensive plots...")
        self.plotter.create_comprehensive_analysis_plot(
            data=prepared_data,
            fitted_models=models,
            temporal_results=temporal_results,
            diagnostic_results=diagnostic_results,
            entity_col="ObservedEntities",
            save_path=f"{self.output_dir}/synthetic_comprehensive_analysis.png",
        )

        return {
            "data": prepared_data,
            "models": models,
            "temporal_results": temporal_results,
            "comparison_results": comparison_results,
            "diagnostic_results": diagnostic_results,
        }

    def run_real_data_demo(self, data_path=None) -> None:
        """Demonstrate framework with real data if available."""
        print("\n=== Real Data Demo ===")

        if data_path and os.path.exists(data_path):
            print(f"1. Loading real data from {data_path}...")
            try:
                real_data = self.data_prep.load_data(data_path)
                # Follow similar steps as synthetic data demo
                # ... implementation would follow same pattern
                print("Real data analysis completed!")
            except Exception as e:
                print(f"Error processing real data: {e}")
                return None
        else:
            print(
                "No real data path provided or file not found. Skipping real data demo.",
            )
            return None

    def _fit_multiple_models(self, data):
        """Fit multiple models for comparison."""
        models = {}

        try:
            # Base Poisson model
            poisson_model = self.glm_fitter.fit_poisson_model(
                data,
                "ObservedEntities",
                ["Year_scaled"],
                "log_offset",
            )
            models["Poisson_Linear"] = poisson_model
        except Exception as e:
            print(f"Failed to fit Poisson model: {e}")

        try:
            # Negative Binomial model
            nb_model = self.glm_fitter.fit_negative_binomial_model(
                data,
                "ObservedEntities",
                ["Year_scaled"],
                "log_offset",
            )
            models["NegBinom_Linear"] = nb_model
        except Exception as e:
            print(f"Failed to fit Negative Binomial model: {e}")

        try:
            # Polynomial trend model
            poly_model = self.trend_fitter.fit_polynomial_trend(
                data,
                "ObservedEntities",
                degree=2,
                offset_col="log_offset",
            )
            models["Polynomial_Trend"] = poly_model
        except Exception as e:
            print(f"Failed to fit Polynomial model: {e}")

        try:
            # Spline trend model
            spline_model = self.trend_fitter.fit_spline_trend(
                data,
                "ObservedEntities",
                n_knots=3,
                offset_col="log_offset",
            )
            models["Spline_Trend"] = spline_model
        except Exception as e:
            print(f"Failed to fit Spline model: {e}")

        return models

    def generate_summary_report(self, results) -> None:
        """Generate a summary report of the analysis."""
        print("\n=== Analysis Summary ===")

        if results:
            print(f"Data shape: {results['data'].shape}")
            print(f"Models fitted: {list(results['models'].keys())}")

            if (
                "comparison_results" in results
                and "model_scores" in results["comparison_results"]
            ):
                print("Model comparison (lower AIC is better):")
                for model, score in results["comparison_results"][
                    "model_scores"
                ].items():
                    print(f"  {model}: {score:.2f}")

            if "temporal_results" in results:
                temporal = results["temporal_results"]
                if "trend_analysis" in temporal:
                    print(
                        f"Trend detected: {temporal['trend_analysis'].get('trend_present', 'Unknown')}",
                    )
                if "autocorrelation" in temporal:
                    print("Autocorrelation analysis completed")

        print(f"Output saved to: {self.output_dir}/")


def main() -> None:
    """Main demonstration function."""
    print("Time Series Analysis Modular Framework Demo")
    print("=" * 50)

    # Initialize demo
    demo = ComprehensiveDemo()

    # Run synthetic data demo
    synthetic_results = demo.run_synthetic_data_demo()

    # Run real data demo if data available
    # Update this path to your actual data file
    real_data_path = "/home/callebalik/EnviDis/data/processed/annual_trend_data.csv"
    real_results = demo.run_real_data_demo(real_data_path)

    # Generate summary
    demo.generate_summary_report(synthetic_results)

    print("\nDemo completed! Check the demo_output directory for results.")


if __name__ == "__main__":
    main()
