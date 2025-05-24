# Time Series Analysis Scripts

This directory contains a modular time series analysis framework that has been split from the original monolithic script `1.py`. The framework provides comprehensive tools for analyzing count data with potential non-monotonic trends and autocorrelation.

## Scripts Overview

### 1. `data_generation.py`
**Purpose**: Generate simulated time series data for testing and analysis.

**Features**:
- Creates realistic simulated data with non-monotonic trends
- Generates both observed entity counts and total document counts
- Includes preprocessing (centering, scaling)
- Configurable time periods and random seeds

**Usage**:
```bash
python data_generation.py
```

### 2. `data_visualization.py`
**Purpose**: Create comprehensive visualizations of time series data.

**Features**:
- Time series plots for entities and documents
- Scatter plots showing relationships
- Configurable output paths for saving plots
- Modern, publication-ready visualizations

**Usage**:
```bash
python data_visualization.py
```

### 3. `basic_model_fitting.py`
**Purpose**: Fit and compare basic regression models (Poisson vs Negative Binomial).

**Features**:
- Poisson regression modeling
- Negative Binomial regression modeling
- Overdispersion testing
- Model comparison using AIC/BIC
- Automatic model selection

**Usage**:
```bash
python basic_model_fitting.py
```

### 4. `trend_modeling.py`
**Purpose**: Handle non-monotonic trend modeling using polynomials and splines.

**Features**:
- Linear, quadratic, and cubic polynomial trends
- Cubic spline regression with configurable degrees of freedom
- Comprehensive model comparison
- Automatic best model selection

**Usage**:
```bash
python trend_modeling.py
```

### 5. `autocorrelation_analysis.py`
**Purpose**: Analyze autocorrelation in residuals and fit lagged models.

**Features**:
- Autocorrelation Function (ACF) plotting
- Durbin-Watson test for first-order autocorrelation
- Lagged dependent variable modeling
- Residual diagnostics

**Usage**:
```bash
python autocorrelation_analysis.py
```

### 6. `main_analysis.py`
**Purpose**: Orchestrate the complete analysis pipeline.

**Features**:
- Complete end-to-end analysis workflow
- Automatic output directory management
- Summary report generation
- Error handling and progress reporting

**Usage**:
```bash
python main_analysis.py
```

## Understanding Key Data Preprocessing Features

### Data Columns Explained

When data is generated or loaded, several preprocessing columns are created:

#### `const` Column
```python
data['const'] = 1
```
- **Purpose**: A constant column of all 1s for explicit intercept terms
- **Usage**: Generally unused in this framework since statsmodels formulas automatically include intercepts
- **Note**: Legacy column maintained for compatibility

#### `Year_centered` Column
```python
data['Year_centered'] = data.index.values - data.index.values.mean()
```
- **Purpose**: Centers year values around zero by subtracting the mean year
- **Example**: For data spanning 1970-2024:
  - Mean year = 1997
  - Year 1970 becomes: -27
  - Year 1997 becomes: 0
  - Year 2024 becomes: 27

#### `Year_scaled` Column
```python
data['Year_scaled'] = data['Year_centered'] / data['Year_centered'].std()
```
- **Purpose**: Standardizes centered year values by dividing by standard deviation (creates z-scores)
- **Example transformation**:
  ```
  Original years: [1970, 1980, 1990, 2000, 2010, 2020]
  Year_centered:  [-27,  -17,   -7,    3,   13,   23]
  Year_scaled:    [-1.46, -0.92, -0.38, 0.16, 0.70, 1.24]
  ```

#### Benefits of Year Centering and Scaling
1. **Numerical Stability**: Prevents issues with large year values (like 2024) in regression algorithms
2. **Coefficient Interpretation**: The intercept represents the expected value at the mean year
3. **Convergence**: Helps optimization algorithms converge faster and more reliably
4. **Polynomial Terms**: Makes higher-order polynomial terms (Year², Year³) numerically stable
5. **Standardized Effects**: Year coefficients represent change per standard deviation rather than per year

**Model Usage**: The regression models use `Year_scaled` in their formulas, making coefficients interpretable as "change in outcome per standard deviation change in year."

## Dependencies

Make sure you have the following packages installed:

```bash
pip install pandas numpy matplotlib seaborn statsmodels
```

Or add to your `requirements.txt`:
```
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.3.0
seaborn>=0.11.0
statsmodels>=0.12.0
```

## Quick Start

### Option 1: Run Complete Pipeline
```bash
cd /home/callebalik/EnviDis/scripts/analysis
python main_analysis.py
```

### Option 2: Run Individual Components
```bash
# Step 1: Generate data
python data_generation.py

# Step 2: Create visualizations
python data_visualization.py

# Step 3: Fit basic models
python basic_model_fitting.py

# Step 4: Fit trend models
python trend_modeling.py

# Step 5: Analyze autocorrelation
python autocorrelation_analysis.py
```

## Output Structure

When you run the analysis, outputs will be saved to:
```
/home/callebalik/EnviDis/results/analysis/
├── plots/
│   ├── entities_over_time.png
│   ├── documents_over_time.png
│   ├── entities_vs_documents.png
│   ├── acf_residuals.png
│   └── acf_lagged_residuals.png
├── sample_time_series_data.csv
└── analysis_summary.txt
```

## Customization

### Using Your Own Data
To use your own data instead of simulated data, modify `main_analysis.py`:

```python
pipeline = TimeSeriesAnalysisPipeline(
    data_path='/path/to/your/data.csv'
)
```

Your data should have columns:
- `Year` (as index)
- `ObservedEntities`
- `TotalDocuments`

### Modifying Parameters
Each script has configurable parameters:

- **Spline degrees of freedom**: Modify `df` parameter in `trend_modeling.py`
- **Lag periods**: Modify `lag_periods` in `autocorrelation_analysis.py`
- **Visualization settings**: Modify plot parameters in `data_visualization.py`

## Class-Based Architecture

The scripts use object-oriented design for better modularity:

- `BasicModelFitter`: Handles Poisson/NB model fitting
- `TrendModelFitter`: Handles polynomial/spline modeling
- `AutocorrelationAnalyzer`: Handles residual analysis
- `LaggedModelFitter`: Handles lagged variable models
- `TimeSeriesAnalysisPipeline`: Orchestrates complete workflow

## Error Handling

The scripts include comprehensive error handling:
- Missing data file warnings
- Import dependency checks
- Model fitting validation
- Output directory creation

## Contributing

To add new modeling approaches:

1. Create a new script following the established pattern
2. Use class-based design with clear method separation
3. Include comprehensive docstrings
4. Add error handling and validation
5. Update this README with usage instructions

## Migration from Original Script

The original `1.py` script has been split as follows:

| Original Section | New Script |
|------------------|------------|
| Data generation | `data_generation.py` |
| Plotting | `data_visualization.py` |
| Basic GLM fitting | `basic_model_fitting.py` |
| Polynomial/spline trends | `trend_modeling.py` |
| Autocorrelation analysis | `autocorrelation_analysis.py` |
| Complete workflow | `main_analysis.py` |

This modular approach provides:
- **Better maintainability**: Each script has a single responsibility
- **Reusability**: Components can be used independently
- **Testability**: Each module can be tested separately
- **Extensibility**: New modeling approaches can be added easily
- **Collaboration**: Multiple developers can work on different components

##