import polars as pl
import yfinance as yf

from stock_market_analytics.data_collection.models.collection_plans import (
    YFinanceCollectionPlan,
)

from .base import FinancialDataCollector


class YFinanceCollector(FinancialDataCollector):
    """
    Implementation of the FinancialDataCollector protocol using yfinance.
    """

    def __init__(self, collection_plan: YFinanceCollectionPlan | None = None, **kwargs):
        self.collection_plan = collection_plan
        self.kwargs = kwargs
        # If collection_plan is not provided, create one from kwargs
        if collection_plan is None:
            self.collection_plan = YFinanceCollectionPlan(**kwargs)

        # Convert collection plan to yfinance parameters
        self.yf_params = self.collection_plan.to_yfinance_params()
        self.collection_successful = False
        self.collected_empty_data = False
        self.errors_during_collection = False

    def get_historical_data(self) -> pl.DataFrame:
        """
        Get historical data using a YFinanceCollectionPlan.

        Args:
            collection_plan: Validated YFinanceCollectionPlan instance

        Returns:
            pl.DataFrame: Historical price data
        """

        # Get the data from yfinance
        try:
            ticker = yf.Ticker(self.collection_plan.symbol)

            # in yfinance, this is a pandas dataframe
            data = ticker.history(**self.yf_params)

            # Convert to DataFrame and rename columns
            if not data.empty:
                # Reset index to make Date a column
                data = data.reset_index()

                # Rename columns to match expected format
                column_mapping = {
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
                data = data.rename(columns=column_mapping)

                # Convert to Polars DataFrame
                data["symbol"] = self.collection_plan.symbol

                output_df = pl.from_pandas(
                    data[["date", "symbol", "open", "high", "low", "close", "volume"]]
                )

                # apply correct schema
                output_df = output_df.with_columns(
                    pl.col("date").dt.strftime("%Y-%m-%d").cast(pl.Date),
                    pl.col("symbol").cast(pl.Utf8),
                    pl.col("open").cast(pl.Float64),
                    pl.col("high").cast(pl.Float64),
                    pl.col("low").cast(pl.Float64),
                    pl.col("close").cast(pl.Float64),
                    pl.col("volume").cast(pl.Int64),
                )
                self.collection_successful = True
                return output_df
            else:
                self.collected_empty_data = True
                # Return empty DataFrame with correct schema
                return pl.DataFrame(
                    schema={
                        "date": pl.Date,
                        "symbol": pl.Utf8,
                        "open": pl.Float64,
                        "high": pl.Float64,
                        "low": pl.Float64,
                        "close": pl.Float64,
                        "volume": pl.Int64,
                    }
                )

        except Exception as e:
            self.errors_during_collection = True
            self.error_message = str(e)
            # Return empty DataFrame with correct schema on error
            return pl.DataFrame(
                schema={
                    "date": pl.Date,
                    "symbol": pl.Utf8,
                    "open": pl.Float64,
                    "high": pl.Float64,
                    "low": pl.Float64,
                    "close": pl.Float64,
                    "volume": pl.Int64,
                }
            )
