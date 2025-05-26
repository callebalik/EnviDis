import pandas as pd


class DataPreparator:
    """Class for preparing data for time series analysis."""

    def __init__(
        self,
        data_path=None,
        date_col="pubdate_resolved",
        response_col="co_count",
        document_col="document_count",
    ):
        """Initialize data preparator.

        Parameters
        ----------
        data_path : str, optional
            Path to CSV file
        date_col : str
            Name of date column
        response_col : str
            Name of response variable column
        document_col : str
            Name of document count column

        """
        self.data_path = data_path
        self.date_col = date_col
        self.response_col = response_col
        self.document_col = document_col
        self.data = None

    def load_real_data(self, data_path=None):
        """Load real time series data from CSV."""
        path = data_path or self.data_path
        if path is None:
            msg = "Must provide data_path"
            raise ValueError(msg)

        print(f"Loading data from: {path}")
        df = pd.read_csv(path)

        # Convert date column
        df["Date"] = pd.to_datetime(df[self.date_col])

        # Standardize column names
        df["ObservedEntities"] = df[self.response_col]
        df["TotalDocuments"] = df[self.document_col]

        # Create time variables
        df = self._create_time_variables(df)

        # Set date as index
        df = df.set_index("Date").sort_index()

        self.data = df
        print(
            f"Data loaded: {df.shape[0]:,} observations from {df.index.min().date()} to {df.index.max().date()}",
        )

        return df

    def load_synthetic_data(self, start_year=1990, end_year=2024, seed=42):
        """Generate synthetic data for testing."""
        from ..demo.demo_data_generation import generate_sample_data

        print(f"Generating synthetic data ({start_year}-{end_year})")
        df = generate_sample_data(start_year=start_year, end_year=end_year, seed=seed)

        # Create time variables
        df = self._create_time_variables(df)

        self.data = df
        print(f"Synthetic data generated: {len(df)} observations")

        return df

    def _create_time_variables(self, df):
        """Create time variables for modeling."""
        # Extract time components
        if "Date" in df.columns:
            df["Year"] = df["Date"].dt.year
            df["Month"] = df["Date"].dt.month
            df["DayOfYear"] = df["Date"].dt.dayofyear

            # Create scaled time variables
            start_date = df["Date"].min()
            df["DaysSinceStart"] = (df["Date"] - start_date).dt.days
            df["DaysSinceStart_scaled"] = (
                df["DaysSinceStart"] - df["DaysSinceStart"].mean()
            ) / df["DaysSinceStart"].std()
        else:
            # Handle yearly data
            df["Year"] = df["Year"] if "Year" in df.columns else df.index

        # Year-based scaling
        df["Year_scaled"] = (df["Year"] - df["Year"].mean()) / df["Year"].std()

        return df

    def get_modeling_subset(self, sample_size=None, filter_percentile=0.95):
        """Get a subset of data suitable for modeling."""
        if self.data is None:
            msg = "Must load data first"
            raise ValueError(msg)

        modeling_data = self.data.copy()

        # Filter extreme values for model stability
        threshold = modeling_data["ObservedEntities"].quantile(filter_percentile)
        modeling_data = modeling_data[modeling_data["ObservedEntities"] <= threshold]

        # Sample if requested
        if sample_size and len(modeling_data) > sample_size:
            modeling_data = modeling_data.sample(n=sample_size, random_state=42)

        print(
            f"Modeling subset: {len(modeling_data):,} observations (filtered at {filter_percentile:.0%})",
        )

        return modeling_data

    def get_temporal_aggregations(self):
        """Get monthly and yearly aggregations."""
        if self.data is None:
            msg = "Must load data first"
            raise ValueError(msg)

        # Monthly aggregation
        monthly_data = self.data.groupby(
            [self.data.index.year, self.data.index.month],
        ).agg({"ObservedEntities": "sum", "TotalDocuments": "sum"})
        monthly_data.index.names = ["Year", "Month"]
        monthly_data = monthly_data.reset_index()

        # Yearly aggregation
        yearly_data = self.data.groupby(self.data.index.year).agg(
            {"ObservedEntities": "sum", "TotalDocuments": "sum"},
        )
        yearly_data.index.name = "Year"
        yearly_data = yearly_data.reset_index(drop=False)

        return monthly_data, yearly_data

    def get_data_summary(self):
        """Get comprehensive data summary."""
        if self.data is None:
            msg = "Must load data first"
            raise ValueError(msg)

        summary = {
            "total_observations": len(self.data),
            "date_range_start": (
                str(self.data.index.min().date())
                if hasattr(self.data.index.min(), "date")
                else str(self.data.index.min())
            ),
            "date_range_end": (
                str(self.data.index.max().date())
                if hasattr(self.data.index.max(), "date")
                else str(self.data.index.max())
            ),
            "years_covered": (
                self.data.index.year.nunique()
                if hasattr(self.data.index, "year")
                else self.data["Year"].nunique()
            ),
            "mean_response": self.data["ObservedEntities"].mean(),
            "median_response": self.data["ObservedEntities"].median(),
            "max_response": self.data["ObservedEntities"].max(),
            "zero_percentage": (self.data["ObservedEntities"] == 0).mean() * 100,
            "mean_documents": self.data["TotalDocuments"].mean(),
            "max_documents": self.data["TotalDocuments"].max(),
        }

        return summary
