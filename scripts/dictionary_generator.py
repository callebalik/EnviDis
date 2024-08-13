import os
import sys
from pathlib import Path

import pandas as pd

# Add the project root to the system path
from config_loader import config

# Get dictionary file path
excel_file = (
    Path(config.DATA_DIR) / "dictionary" / "environmentalphenomena_dictionary_v1.xlsx"
)

print(f"Excel file path: {excel_file}")
# Read the Excel file
envPhenomenaDF = pd.read_excel(excel_file)

# Convert the DataFrame to CSV
envPhenomenaDF.to_csv(csv_file, index=False)
