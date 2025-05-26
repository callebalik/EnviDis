"""Main Analysis Pipeline Script
This script orchestrates the complete time series analysis pipeline.
"""

import os
from typing import Any

import pandas as pd

from envidis.time.analysis.autocorrelation_analysis import (
    AutocorrelationAnalyzer,
    LaggedModelFitter,
)
from envidis.time.analysis.basic_model_fitting import BasicModelFitter
from envidis.time.analysis.trend_modeling import TrendModelFitter
from envidis.time.demo.data_generation import (
    generate_daily_sample_data,
    generate_sample_data,
    save_sample_data,
)
from envidis.time.plots.data_visualization import create_all_plots

# Column name constants
OBSERVED_ENTITIES_COL = "co_count"
COOC_COL = "co_count"
PUBDATE_COL = "date"
DAYS_SINCE_START_COL = "DaysSinceStart"
DAYS_SINCE_START_SCALED_COL = "DaysSinceStart_scaled"


class TimeSeriesAnalysisPipeline:
    """Complete time series analysis pipeline."""

    def __init__(self, data_path: str | None = None, output_dir: str | None = None):
        """Initialize the analysis pipeline."""
        self.data_path = data_path
        self.output_dir = output_dir or "/home/callebalik/EnviDis/results/analysis"
        self.data: pd.DataFrame = pd.DataFrame()
        self.basic_fitter: BasicModelFitter | None = None
        self.trend_fitter: TrendModelFitter | None = None
        self.best_model: Any | None = None

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/plots", exist_ok=True)

    def load_or_generate_data(self) -> None:
        """Load existing data or generate sample data with daily resolution."""
        if self.data_path and os.path.exists(self.data_path):
            print(f"Loading data from {self.data_path}")
            self.data = pd.read_csv(
                self.data_path,
                index_col=PUBDATE_COL,
                parse_dates=True,
            )
        else:
            print("Generating daily sample data...")
            self.data = generate_daily_sample_data()

            # Save generated data
            data_save_path = f"{self.output_dir}/daily_time_series_data.csv"
            self.data.to_csv(data_save_path)
            self.data_path = data_save_path

        print("Data summary:")
        print(self.data.head())
        print(f"Shape: {self.data.shape}")
        print(f"Date range: {self.data.index.min()} to {self.data.index.max()}")

    def create_visualizations(self) -> None:
        """Create visualizations for daily data."""
        print("\nCreating visualizations...")

        # For daily data, you might want to create both daily and aggregated views
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt

        # Create daily plot (sampled for readability)
        sample_data = self.data.iloc[::30]  # Sample every 30 days

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Daily time series (sampled)
        ax1.plot(sample_data.index, sample_data[OBSERVED_ENTITIES_COL], "b-", alpha=0.7)
        ax1.set_title("Daily Observed Entities (Every 30 Days)")
        ax1.set_ylabel("Observed Entities")
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax1.xaxis.set_major_locator(mdates.YearLocator())

        # Monthly aggregation for cleaner view
        monthly_data = self.data.resample("M").sum()
        ax2.plot(
            monthly_data.index,
            monthly_data[OBSERVED_ENTITIES_COL],
            "r-",
            linewidth=2,
        )
        ax2.set_title("Monthly Aggregated Observed Entities")
        ax2.set_ylabel("Monthly Total")
        ax2.set_xlabel("Date")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax2.xaxis.set_major_locator(mdates.YearLocator())

        plt.tight_layout()
        plt.savefig(
            f"{self.output_dir}/plots/daily_time_series.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Create additional plots as needed
        create_all_plots(self.data, f"{self.output_dir}/plots")

    def fit_basic_models(self) -> None:
        """Fit basic Poisson and Negative Binomial models with daily data."""
        print("\nFitting basic models...")

        # For large daily datasets, you might want to sample or aggregate
        modeling_data = self.data.copy()

        # Add time variables for daily modeling
        modeling_data[DAYS_SINCE_START_COL] = (
            modeling_data.index - modeling_data.index.min()
        ).days
        modeling_data[DAYS_SINCE_START_SCALED_COL] = (
            modeling_data[DAYS_SINCE_START_COL]
            - modeling_data[DAYS_SINCE_START_COL].mean()
        ) / modeling_data[DAYS_SINCE_START_COL].std()

        self.basic_fitter = BasicModelFitter(modeling_data)
        self.basic_fitter.fit_poisson_model()
        self.basic_fitter.fit_negative_binomial_model()
        self.basic_fitter.print_results()

    def fit_trend_models(self) -> None:
        """Fit trend models with daily resolution."""
        print("\nFitting trend models...")

        # Prepare data for modeling
        modeling_data = self.data.copy()
        modeling_data[DAYS_SINCE_START_COL] = (
            modeling_data.index - modeling_data.index.min()
        ).days
        modeling_data[DAYS_SINCE_START_SCALED_COL] = (
            modeling_data[DAYS_SINCE_START_COL]
            - modeling_data[DAYS_SINCE_START_COL].mean()
        ) / modeling_data[DAYS_SINCE_START_COL].std()

        # For very large datasets, consider sampling
        if len(modeling_data) > 10000:
            print(
                f"Large dataset ({len(modeling_data)} days). Sampling for modeling...",
            )
            modeling_data = modeling_data.sample(n=5000, random_state=42)

        self.trend_fitter = TrendModelFitter(modeling_data)
        self.trend_fitter.fit_all_models()
        self.trend_fitter.print_results()

        # Get best model
        comparison = self.trend_fitter.compare_all_models()
        self.best_model = comparison["best_model"]
        print(f"\nBest model selected: {comparison['best_overall']}")

    def analyze_autocorrelation(self):
        """Analyze autocorrelation in residuals."""
        if self.best_model is None:
            msg = "Must fit trend models first"
            raise ValueError(msg)

        print("\nAnalyzing autocorrelation...")
        analyzer = AutocorrelationAnalyzer(self.best_model)

        # Create comprehensive diagnostic plots
        print("Creating model diagnostic plots...")
        analyzer.create_diagnostic_plots(self.data, f"{self.output_dir}/plots")

        dw_results = analyzer.durbin_watson_test()
        analyzer.print_diagnostics()

        return dw_results["significant_autocorrelation"]

    def fit_lagged_model(self) -> None:
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
                save_path=f"{self.output_dir}/plots/lagged_model_fit.png",
            )
            lagged_fitter.compare_models_plot(
                save_path=f"{self.output_dir}/plots/model_comparison.png",
            )

            # Analyze lagged model residuals
            print("Analyzing lagged model residuals...")
            lagged_fitter.analyze_lagged_residuals(
                save_path=f"{self.output_dir}/plots/acf_lagged_residuals.png",
            )
        else:
            print("No significant autocorrelation detected. Lagged model not needed.")

    def run_complete_analysis(self) -> None:
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

    def save_summary_report(self) -> None:
        """Save a summary report of the analysis."""
        report_path = f"{self.output_dir}/analysis_summary.txt"

        with open(report_path, "w") as f:
            f.write("TIME SERIES ANALYSIS SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Data shape: {self.data.shape}\n")
            f.write(
                f"Year range: {self.data.index.min()} - {self.data.index.max()}\n\n",
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
                    f"Best model formula: {trend_comparison['best_model'].model.formula}\n\n",
                )

            if self.best_model:
                analyzer = AutocorrelationAnalyzer(self.best_model)
                dw_results = analyzer.durbin_watson_test()
                f.write(f"Durbin-Watson statistic: {dw_results['dw_statistic']:.3f}\n")
                f.write(f"Autocorrelation: {dw_results['interpretation']}\n")

        print(f"Summary report saved to: {report_path}")


def main() -> None:
    """Main function to run the analysis."""
    # Option 1: Use existing data
    pipeline = TimeSeriesAnalysisPipeline(
        data_path="data/raw/time-series/time_series.csv",
    )

    # Option 2: Generate sample data (default)
    # pipeline = TimeSeriesAnalysisPipeline()

    # Run complete analysis
    pipeline.run_complete_analysis()

    # Save summary report
    pipeline.save_summary_report()


if __name__ == "__main__":
    main()
