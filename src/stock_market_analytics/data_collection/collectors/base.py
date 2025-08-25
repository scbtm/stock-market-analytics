from typing import Protocol

import polars as pl


class FinancialDataCollector(Protocol):
    """
    Protocol defining the interface for financial data collectors.
    This is a structural type that defines the expected behavior
    of any financial data collector implementation.
    """

    def get_historical_data(self) -> pl.DataFrame:
        """
        Fetch historical price data for the configured symbol.

        Returns:
            pl.DataFrame containing historical price data with the following schema:
            - date: pl.Date - The date of the data point
            - symbol: pl.Utf8 - The stock symbol
            - open: pl.Float64 - Opening price
            - high: pl.Float64 - Highest price during the period
            - low: pl.Float64 - Lowest price during the period
            - close: pl.Float64 - Closing price
            - volume: pl.Int64 - Trading volume

        Note:
            - Returns an empty DataFrame with the correct schema if no data is available
            - Returns an empty DataFrame with the correct schema if an error occurs
            - Implementations should handle errors gracefully and not raise exceptions
        """
        ...
