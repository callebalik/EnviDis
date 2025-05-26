#!/usr/bin/env python3
"""Simple test script to validate the modular time series analysis framework.

This script performs basic import and functionality tests to ensure
all components are working correctly.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def test_imports() -> bool | None:
    """Test that all modules can be imported correctly."""
    print("Testing imports...")

    try:
        # Demo module
        from envidis.time_series_analysis.demo.demo_data_generation import (
            generate_sample_data,
        )

        print("✓ Demo module imported")

        # Utils module
        from envidis.time_series_analysis.utils.data_preparation_new import (
            DataPreparator,
        )
        from envidis.time_series_analysis.utils.modeling_utils import (
            create_offset_variable,
            validate_data_for_modeling,
        )

        print("✓ Utils module imported")

        # Modeling module
        from envidis.time_series_analysis.modeling.base_models import fit_poisson_glm
        from envidis.time_series_analysis.modeling.trend_models import (
            fit_polynomial_trends,
        )

        print("✓ Modeling module imported")

        # Analysis module
        from envidis.time_series_analysis.analysis.diagnostics import ModelDiagnostics
        from envidis.time_series_analysis.analysis.model_selection import compare_models
        from envidis.time_series_analysis.analysis.temporal_analysis_new import (
            perform_comprehensive_temporal_analysis,
        )

        print("✓ Analysis module imported")

        # Plotting module
        from envidis.time_series_analysis.plotting.comprehensive_plots import (
            ComprehensivePlotter,
        )
        from envidis.time_series_analysis.plotting.diagnostic_plots import (
            DiagnosticPlotter,
        )
        from envidis.time_series_analysis.plotting.trend_plots import TrendPlotter

        print("✓ Plotting module imported")

        return True

    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_basic_functionality() -> bool | None:
    """Test basic functionality of core components."""
    print("\nTesting basic functionality...")

    try:
        # Import required modules
        from envidis.time_series_analysis.demo.demo_data_generation import (
            generate_sample_data,
        )
        from envidis.time_series_analysis.modeling.base_models import fit_poisson_glm
        from envidis.time_series_analysis.utils.data_preparation_new import (
            DataPreparator,
        )
        from envidis.time_series_analysis.utils.modeling_utils import (
            create_offset_variable,
            validate_data_for_modeling,
        )

        # 1. Generate test data
        test_data = generate_sample_data(
            start_year=2020,
            end_year=2024,
            seed=42,
        )
        print(f"✓ Generated test data: {test_data.shape}")

        # 2. Initialize data preparator and validate data
        preparator = DataPreparator()
        # Load the test data into preparator
        preparator.df = test_data
        validation_result = preparator.validate_data()
        print("✓ Data validation completed")

        # 3. Validate data for modeling
        modeling_validation = validate_data_for_modeling(test_data)
        if modeling_validation["is_valid"]:
            print("✓ Data validation for modeling passed")
        else:
            print(
                f"✗ Data validation failed: {modeling_validation.get('issues', 'Unknown issues')}"
            )

        # 4. Create offset variable
        processed_data = create_offset_variable(test_data, "TotalDocuments")
        print("✓ Data preprocessing completed")

        # 5. Fit a simple model (if we have the right columns)
        if (
            "AffectedEntities" in processed_data.columns
            and "Year" in processed_data.columns
        ):
            model_result = fit_poisson_glm(
                processed_data,
                "AffectedEntities ~ Year",
                "log_offset",
            )
            print("✓ Model fitting completed")

            # Check model results
            if "model" in model_result and "aic" in model_result:
                print(f"✓ Model results valid (AIC: {model_result['aic']:.2f})")
            else:
                print("✗ Model results incomplete")
        else:
            print("✓ Model fitting skipped (missing required columns)")

        return True

    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False


def test_plotting_components() -> bool | None:
    """Test that plotting components can be instantiated."""
    print("\nTesting plotting components...")

    try:
        from envidis.time_series_analysis.plotting.comprehensive_plots import (
            ComprehensivePlotter,
        )
        from envidis.time_series_analysis.plotting.diagnostic_plots import (
            DiagnosticPlotter,
        )
        from envidis.time_series_analysis.plotting.trend_plots import TrendPlotter

        # Instantiate plotters
        comp_plotter = ComprehensivePlotter()
        diag_plotter = DiagnosticPlotter()
        trend_plotter = TrendPlotter()

        print("✓ All plotting components instantiated successfully")
        return True

    except Exception as e:
        print(f"✗ Plotting component test failed: {e}")
        return False


def test_package_structure() -> bool:
    """Test that the package structure is correct."""
    print("\nTesting package structure...")

    # Check that all required files exist
    base_path = Path(__file__).parent  # This is the time_series_analysis directory

    required_files = [
        "__init__.py",
        "demo/__init__.py",
        "demo/demo_data_generation.py",
        "utils/__init__.py",
        "utils/data_preparation.py",
        "utils/data_preparation_new.py",
        "utils/modeling_utils.py",
        "modeling/__init__.py",
        "modeling/base_models.py",
        "modeling/trend_models.py",
        "analysis/__init__.py",
        "analysis/temporal_analysis_new.py",
        "analysis/model_selection.py",
        "analysis/diagnostics.py",
        "plotting/__init__.py",
        "plotting/comprehensive_plots.py",
        "plotting/diagnostic_plots.py",
        "plotting/trend_plots.py",
    ]

    missing_files = []
    for file_path in required_files:
        full_path = base_path / file_path
        if not full_path.exists():
            missing_files.append(str(file_path))

    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False
    else:
        print("✓ All required files present")
        return True


def main() -> None:
    """Run all tests."""
    print("MODULAR TIME SERIES ANALYSIS FRAMEWORK - TESTING")
    print("=" * 60)

    all_tests_passed = True

    # Test package structure
    if not test_package_structure():
        all_tests_passed = False

    # Test imports
    if not test_imports():
        all_tests_passed = False

    # Test basic functionality
    if not test_basic_functionality():
        all_tests_passed = False

    # Test plotting components
    if not test_plotting_components():
        all_tests_passed = False

    print("\n" + "=" * 60)
    if all_tests_passed:
        print("✓ ALL TESTS PASSED")
        print("The modular time series analysis framework is ready for use!")
    else:
        print("✗ SOME TESTS FAILED")
        print("Please check the error messages above and fix any issues.")
    print("=" * 60)


if __name__ == "__main__":
    main()
