#!/usr/bin/env python3
"""
Test script for comprehensive analysis
"""

import sys

sys.path.append("/home/callebalik/EnviDis/scripts/analysis")

print("Testing comprehensive analysis...")

try:
    print("1. Importing modules...")
    from scripts.time_series_analysis.demo.demo_data_generation import (
        generate_sample_data,
    )
    from comprehensive_analysis_report import ComprehensiveAnalysisReport

    print("   ✓ Imports successful")

    print("2. Generating sample data...")
    data = generate_sample_data()
    print(f"   ✓ Data generated: {len(data)} rows")

    print("3. Creating report generator...")
    report_generator = ComprehensiveAnalysisReport(data)
    print("   ✓ Report generator created")

    print("4. Running analyses...")
    report_generator.run_all_analyses()
    print("   ✓ All analyses completed")

    print("5. Creating model summary...")
    model_summary = report_generator.create_model_summary_table()
    print(f"   ✓ Model summary created: {len(model_summary)} models")
    print(model_summary)

    print("6. Test completed successfully!")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback

    traceback.print_exc()
