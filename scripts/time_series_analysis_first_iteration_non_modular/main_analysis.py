"""
Main Analysis Pipeline Script
This script orchestrates the complete time series analysis pipeline.
"""

import pandas as pd
import os
from scripts.analysis.time_series_analysis.demo.demo_data_generation import (
    generate_sample_data,
    save_sample_data,
)
from data_visualization import create_all_plots
from basic_model_fitting import BasicModelFitter
from trend_modeling import TrendModelFitter
from autocorrelation_analysis import AutocorrelationAnalyzer, LaggedModelFitter


class TimeSeriesAnalysisPipeline:
    """Complete time series analysis pipeline."""

    def __init__(self, data_path=None, output_dir=None):
        """
        Initialize the analysis pipeline.

        Parameters:
        -----------
        data_path : str, optional
            Path to existing data file. If None, generates sample data.
        output_dir : str, optional
            Directory for saving outputs
        """
        self.data_path = data_path
        self.output_dir = output_dir or "/home/callebalik/EnviDis/results/analysis"
        self.data = None
        self.basic_fitter = None
        self.trend_fitter = None
        self.best_model = None

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/plots", exist_ok=True)

    def load_or_generate_data(self):
        """Load existing data or generate sample data."""
        if self.data_path and os.path.exists(self.data_path):
            print(f"Loading data from {self.data_path}")
            self.data = pd.read_csv(self.data_path, index_col="Year")
        else:
            print("Generating sample data...")
            self.data = generate_sample_data()

            # Save generated data
            data_save_path = f"{self.output_dir}/sample_time_series_data.csv"
            save_sample_data(self.data, data_save_path)
            self.data_path = data_save_path

        print("Data summary:")
        print(self.data.head())
        print(f"Shape: {self.data.shape}")

    def create_visualizations(self):
        """Create all visualization plots."""
        print("\nCreating visualizations...")
        create_all_plots(self.data, f"{self.output_dir}/plots")

    def fit_basic_models(self):
        """Fit basic Poisson and Negative Binomial models."""
        print("\nFitting basic models...")
        self.basic_fitter = BasicModelFitter(self.data)
        self.basic_fitter.fit_poisson_model()
        self.basic_fitter.fit_negative_binomial_model()
        self.basic_fitter.print_results()

    def fit_trend_models(self):
        """Fit non-monotonic trend models."""
        print("\nFitting trend models...")
        self.trend_fitter = TrendModelFitter(self.data)
        self.trend_fitter.fit_all_models()
        self.trend_fitter.print_results()

        # Get best model
        comparison = self.trend_fitter.compare_all_models()
        self.best_model = comparison["best_model"]
        print(f"\nBest model selected: {comparison['best_overall']}")

    def analyze_autocorrelation(self):
        """Analyze autocorrelation in residuals."""
        if self.best_model is None:
            raise ValueError("Must fit trend models first")

        print("\nAnalyzing autocorrelation...")
        analyzer = AutocorrelationAnalyzer(self.best_model)

        # Create comprehensive diagnostic plots
        print("Creating model diagnostic plots...")
        analyzer.create_diagnostic_plots(self.data, f"{self.output_dir}/plots")

        dw_results = analyzer.durbin_watson_test()
        analyzer.print_diagnostics()

        return dw_results["significant_autocorrelation"]

    def fit_lagged_model(self):
        """Fit model with lagged dependent variables if needed."""
        has_autocorrelation = self.analyze_autocorrelation()

        if has_autocorrelation:
            print("\nFitting lagged model to address autocorrelation...")
            lagged_fitter = LaggedModelFitter(self.data, self.best_model)
            lagged_fitter.fit_lagged_model(lag_periods=1)
            lagged_fitter.print_results()

            # Create lagged model plots
            print("Creating lagged model plots...")
            lagged_fitter.plot_lagged_model_fit(
                save_path=f"{self.output_dir}/plots/lagged_model_fit.png"
            )
            lagged_fitter.compare_models_plot(
                save_path=f"{self.output_dir}/plots/model_comparison.png"
            )

            # Analyze lagged model residuals
            print("Analyzing lagged model residuals...")
            lagged_fitter.analyze_lagged_residuals(
                save_path=f"{self.output_dir}/plots/acf_lagged_residuals.png"
            )
        else:
            print("No significant autocorrelation detected. Lagged model not needed.")

    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("=" * 60)
        print("STARTING COMPLETE TIME SERIES ANALYSIS PIPELINE")
        print("=" * 60)

        try:
            # Step 1: Data
            self.load_or_generate_data()

            # Step 2: Visualization
            self.create_visualizations()

            # Step 3: Basic models
            self.fit_basic_models()

            # Step 4: Trend models
            self.fit_trend_models()

            # Step 5: Autocorrelation and lagged models
            self.fit_lagged_model()

            print("\n" + "=" * 60)
            print("ANALYSIS PIPELINE COMPLETED SUCCESSFULLY")
            print(f"Results saved to: {self.output_dir}")
            print("=" * 60)

        except Exception as e:
            print(f"\nError in analysis pipeline: {e}")
            raise

    def save_summary_report(self):
        """Save a summary report of the analysis."""
        report_path = f"{self.output_dir}/analysis_summary.txt"

        with open(report_path, "w") as f:
            f.write("TIME SERIES ANALYSIS SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Data shape: {self.data.shape}\n")
            f.write(
                f"Year range: {self.data.index.min()} - {self.data.index.max()}\n\n"
            )

            if self.basic_fitter:
                comparison = self.basic_fitter.compare_models()
                f.write(f"Best basic model: {comparison['chosen_model']}\n")
                f.write(f"Poisson AIC: {comparison['poisson_aic']:.2f}\n")
                f.write(f"Negative Binomial AIC: {comparison['nb_aic']:.2f}\n\n")

            if self.trend_fitter:
                trend_comparison = self.trend_fitter.compare_all_models()
                f.write(f"Best trend model: {trend_comparison['best_overall']}\n")
                f.write(
                    f"Best model formula: {trend_comparison['best_model'].model.formula}\n\n"
                )

            if self.best_model:
                analyzer = AutocorrelationAnalyzer(self.best_model)
                dw_results = analyzer.durbin_watson_test()
                f.write(f"Durbin-Watson statistic: {dw_results['dw_statistic']:.3f}\n")
                f.write(f"Autocorrelation: {dw_results['interpretation']}\n")

        print(f"Summary report saved to: {report_path}")


def main():
    """Main function to run the analysis."""
    # Option 1: Use existing data
    # pipeline = TimeSeriesAnalysisPipeline(
    #     data_path='/path/to/your/data.csv'
    # )

    # Option 2: Generate sample data (default)
    pipeline = TimeSeriesAnalysisPipeline()

    # Run complete analysis
    pipeline.run_complete_analysis()

    # Save summary report
    pipeline.save_summary_report()


if __name__ == "__main__":
    main()
