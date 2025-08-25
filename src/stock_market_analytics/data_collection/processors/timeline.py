import polars as pl


class ContinuousTimelineProcessor:
    """
    Processor that yields a continuous timeline of data for a given symbol, keeping null values.
    """

    def __init__(self, symbol: str, data: pl.DataFrame):
        self.symbol = symbol
        self.data = data
        self.data_is_valid = False
        self.schema = pl.Schema(
            {
                "date": pl.Date,
                "symbol": pl.Utf8,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Int64,
            }
        )

        self.processing_successful = False

    def _ensure_schema(self) -> bool:
        """
        Ensure the schema of the data is correct.

        Rules: date column must be of type pl.Date, symbol column must be of type pl.Utf8, volume must be of type pl.Int64, and all other columns must be of type pl.Float64.
        """

        data = self.data

        # check if the data is empty
        if data.is_empty():
            self.data_is_valid = False
            return False

        # check if the data has the correct schema
        if data.schema != self.schema:
            try:
                # columns must be in order for the schemas to be comparable
                data = data.with_columns(pl.col("date").cast(pl.Date))
                data = data.with_columns(pl.col("symbol").cast(pl.Utf8))
                data = data.with_columns(pl.col("open").cast(pl.Float64))
                data = data.with_columns(pl.col("high").cast(pl.Float64))
                data = data.with_columns(pl.col("low").cast(pl.Float64))
                data = data.with_columns(pl.col("close").cast(pl.Float64))
                data = data.with_columns(pl.col("volume").cast(pl.Int64))

                self.data_is_valid = True
                self.data = data
                return True

            except Exception:
                self.data_is_valid = False
                return False

        else:
            self.data_is_valid = True
            return True

    def _ensure_data_is_continuous(self) -> bool:
        """
        Ensure the data is continuous.
        This means that there is a date row for every day between the first and last date in the data.

        This should only run after ensuring schema is correct.
        """
        data = self.data

        if not self.data_is_valid:
            return False

        try:
            # get the first and last date in the data
            first_date = data.select(pl.min("date"))["date"][0]
            last_date = data.select(pl.max("date"))["date"][0]

            if (first_date == last_date) & (len(data) > 1):
                self.data_is_valid = False
                return False

            # create a date range between the first and last date
            date_range = pl.date_range(
                start=first_date, end=last_date, interval="1d", eager=True
            )

            # create a dataframe with the date range
            date_range_df = pl.DataFrame(date_range, schema={"date": pl.Date})

            # join the date range dataframe with the data dataframe, keep missing values
            data = date_range_df.join(data, on="date", how="left")

            # fill missing values in symbol column with the symbol
            data = data.with_columns(pl.col("symbol").fill_null(self.symbol))

            # ensure there is a row for every date in the date range
            if len(data) != len(date_range_df):
                return False

            # ensure data is sorted by date, ascending
            data = data.sort("date", descending=False)

            self.data_is_valid = True
            self.data = data
            return True

        except Exception:
            self.data_is_valid = False
            return False

    def process(self) -> pl.DataFrame | None:
        """
        Process the data to ensure it is valid.
        """

        if not self._ensure_schema():
            self.processing_successful = False
            return None

        if not self._ensure_data_is_continuous():
            self.processing_successful = False
            return None

        self.processing_successful = True
        return self.data
