import polars as pl


def base(raw_df: pl.DataFrame) -> pl.DataFrame:
    """Sort DataFrame by symbol and date to ensure consistent ordering for window operations.

    This function provides the foundational sorting required before applying any window
    operations or time-series analysis on stock market data. It ensures that data
    is ordered first by symbol (alphabetically) and then by date (chronologically)
    within each symbol group.

    Args:
        raw_df: Input DataFrame containing stock market data with at minimum
                'symbol' and 'date' columns

    Returns:
        DataFrame sorted by ['symbol', 'date'] with all original columns preserved

    Example:
        >>> import polars as pl
        >>> from datetime import date
        >>> df = pl.DataFrame(
        ...     {
        ...         "symbol": ["AAPL", "GOOGL", "AAPL"],
        ...         "date": [date(2023, 1, 2), date(2023, 1, 1), date(2023, 1, 1)],
        ...         "price": [150.0, 100.0, 149.0],
        ...     }
        ... )
        >>> sorted_df = base(df)
        >>> print(sorted_df)
        ┌────────┬────────────┬───────┐
        │ symbol ┆ date       ┆ price │
        │ ---    ┆ ---        ┆ ---   │
        │ str    ┆ date       ┆ f64   │
        ╞════════╪════════════╪═══════╡
        │ AAPL   ┆ 2023-01-01 ┆ 149.0 │
        │ AAPL   ┆ 2023-01-02 ┆ 150.0 │
        │ GOOGL  ┆ 2023-01-01 ┆ 100.0 │
        └────────┴────────────┴───────┘
    """
    return raw_df.sort(["symbol", "date"])


def interpolated(base: pl.DataFrame) -> pl.DataFrame:
    """Interpolate missing values across all columns grouped by symbol.

    This function performs linear interpolation to fill missing (null) values
    in all numeric columns. Interpolation is performed independently within
    each symbol group, ensuring that values from one stock don't influence
    the interpolation of another stock's missing data.

    The interpolation works by:
    1. Grouping data by symbol
    2. For each numeric column, filling nulls using linear interpolation
    3. Preserving non-null values unchanged
    4. **Important**: Only interpolates nulls that have non-null values on both sides
       within the same symbol group. Boundary nulls (at start/end) remain null.

    Args:
        base: Input DataFrame that should be pre-sorted by symbol and date.
              Typically the output of the base() function.

    Returns:
        DataFrame with interpolated values, maintaining the same structure
        and column order as the input

    Note:
        - Non-numeric columns (like strings) may have different interpolation behavior
        - Interpolation only works within symbol boundaries
        - Boundary nulls (first/last values) remain null
        - Only nulls with non-null values on both sides get interpolated
        - Input should be sorted by symbol and date for best results

    Example:
        >>> import polars as pl
        >>> from datetime import date
        >>> df = pl.DataFrame(
        ...     {
        ...         "symbol": ["AAPL", "AAPL", "AAPL"],
        ...         "date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
        ...         "price": [100.0, None, 104.0],
        ...     }
        ... )
        >>> interpolated_df = dfp(df)
        >>> print(interpolated_df)
        ┌────────┬────────────┬───────┐
        │ symbol ┆ date       ┆ price │
        │ ---    ┆ ---        ┆ ---   │
        │ str    ┆ date       ┆ f64   │
        ╞════════╪════════════╪═══════╡
        │ AAPL   ┆ 2023-01-01 ┆ 100.0 │
        │ AAPL   ┆ 2023-01-02 ┆ 102.0 │
        │ AAPL   ┆ 2023-01-03 ┆ 104.0 │
        └────────┴────────────┴───────┘

        >>> # Example with boundary nulls (these remain null)
        >>> df_boundary = pl.DataFrame(
        ...     {
        ...         "symbol": ["AAPL", "AAPL"],
        ...         "price": [100.0, None],  # boundary null stays null
        ...     }
        ... )
        >>> result = dfp(df_boundary)
        >>> print(result["price"].to_list())
        [100.0, None]
    """
    return base.with_columns(pl.col("*").interpolate().over("symbol"))


def dfp(interpolated: pl.DataFrame) -> pl.DataFrame:
    """Remove rows with any null values after interpolation, preserving complete records only.

    This function removes rows that contain null values in any column. It's designed to be
    used after the dfp() function to eliminate boundary nulls that couldn't be interpolated,
    leaving only complete records with all values filled.

    The cleaning works by:
    1. Identifying rows with any null values across all columns
    2. Filtering out these rows to keep only complete records
    3. Maintaining the original column structure and data types
    4. Preserving the sorted order from previous preprocessing steps

    Args:
        interpolated: Input DataFrame that has been interpolated, typically the output
                     of the dfp() function. Should be pre-sorted by symbol and date.

    Returns:
        DataFrame with all rows containing null values removed, maintaining the same
        column structure as the input

    Note:
        - This function removes entire rows if ANY column contains a null value
        - It's most effective when used after interpolation to remove boundary nulls
        - May result in uneven time series lengths across different symbols
        - Use with caution as it may remove valid data points if nulls exist in non-critical columns

    Example:
        >>> import polars as pl
        >>> from datetime import date
        >>> # Data with boundary nulls after interpolation
        >>> df = pl.DataFrame(
        ...     {
        ...         "symbol": ["AAPL", "AAPL", "AAPL", "AAPL"],
        ...         "date": [
        ...             date(2023, 1, 1),
        ...             date(2023, 1, 2),
        ...             date(2023, 1, 3),
        ...             date(2023, 1, 4),
        ...         ],
        ...         "price": [None, 100.0, 102.0, None],  # boundary nulls
        ...         "volume": [None, 1000, 1100, None],
        ...     }
        ... )
        >>> cleaned_df = clean(df)
        >>> print(cleaned_df)
        ┌────────┬────────────┬───────┬────────┐
        │ symbol ┆ date       ┆ price ┆ volume │
        │ ---    ┆ ---        ┆ ---   ┆ ---    │
        │ str    ┆ date       ┆ f64   ┆ i64    │
        ╞════════╪════════════╪═══════╪════════╡
        │ AAPL   ┆ 2023-01-02 ┆ 100.0 ┆ 1000   │
        │ AAPL   ┆ 2023-01-03 ┆ 102.0 ┆ 1100   │
        └────────┴────────────┴───────┴────────┘
    """
    return interpolated.drop_nulls()


def log_returns_d(dfp: pl.DataFrame) -> pl.DataFrame:
    """Calculate daily log returns from closing prices, grouped by symbol.

    Log returns are the natural logarithm of the ratio between consecutive prices.
    They have several advantages over simple returns: they are additive over time,
    approximately symmetric around zero for small changes, and handle compounding
    more naturally. This function computes log returns as the difference between
    consecutive log prices within each symbol group.

    The calculation works by:
    1. Taking the natural logarithm of the 'close' column
    2. Computing the first difference (current - previous) within each symbol group
    3. The first observation for each symbol will have a null value (no previous price)

    Formula: log_return_t = log(price_t) - log(price_{t-1}) = log(price_t / price_{t-1})

    Args:
        dfp: Input DataFrame containing stock data with at minimum 'symbol' and 'close'
             columns. Should be pre-sorted by ['symbol', 'date'] for correct calculation.

    Returns:
        DataFrame with all original columns plus a new 'log_returns_d' column containing
        the daily log returns. The first observation for each symbol will be null.

    Note:
        - Log returns are calculated independently within each symbol group
        - First observation per symbol will be null (no previous price to compare)
        - Input should be sorted by symbol and date for meaningful results
        - Close prices should be positive values (log of zero or negative values is undefined)

    Example:
        >>> import polars as pl
        >>> from datetime import date
        >>> df = pl.DataFrame(
        ...     {
        ...         "symbol": ["AAPL", "AAPL", "AAPL"],
        ...         "date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
        ...         "close": [100.0, 105.0, 102.0],
        ...     }
        ... )
        >>> result = log_returns_d(df)
        >>> print(result)
        ┌────────┬────────────┬───────┬──────────────┐
        │ symbol ┆ date       ┆ close ┆ log_returns_d │
        │ ---    ┆ ---        ┆ ---   ┆ ---          │
        │ str    ┆ date       ┆ f64   ┆ f64          │
        ╞════════╪════════════╪═══════╪══════════════╡
        │ AAPL   ┆ 2023-01-01 ┆ 100.0 ┆ null         │
        │ AAPL   ┆ 2023-01-02 ┆ 105.0 ┆ 0.048790     │
        │ AAPL   ┆ 2023-01-03 ┆ 102.0 ┆ -0.029559    │
        └────────┴────────────┴───────┴──────────────┘
    """
    return dfp.with_columns(
        pl.col("close").log().diff().over("symbol").alias("log_returns_d")
    )


def dollar_volume(dfp: pl.DataFrame) -> pl.DataFrame:
    """Calculate dollar volume by multiplying closing price with trading volume.

    Dollar volume represents the total monetary value of shares traded for each
    observation. It provides a measure of liquidity and trading activity that
    combines both price and volume information. High dollar volume indicates
    significant trading activity and good liquidity, while low dollar volume
    may suggest thin trading or low interest.

    This metric is particularly useful for:
    - Assessing market liquidity and trading activity
    - Comparing trading intensity across different price levels
    - Risk management (higher dollar volume often correlates with lower bid-ask spreads)
    - Market impact analysis for institutional trading

    Formula: dollar_volume = close_price x volume

    Args:
        dfp: Input DataFrame containing stock data with at minimum 'close' and
             'volume' columns. All original columns are preserved.

    Returns:
        DataFrame with all original columns plus a new 'dollar_volume' column containing
        the product of close price and volume for each observation.

    Note:
        - Calculation is performed row-wise, no grouping or time-series operations
        - Both close and volume should be non-negative values for meaningful results
        - Zero values in either close or volume will result in zero dollar volume
        - Missing values (nulls) in either input column will result in null dollar volume

    Example:
        >>> import polars as pl
        >>> from datetime import date
        >>> df = pl.DataFrame(
        ...     {
        ...         "symbol": ["AAPL", "AAPL", "GOOGL"],
        ...         "date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 1)],
        ...         "close": [150.0, 155.0, 2800.0],
        ...         "volume": [1000000, 1500000, 500000],
        ...     }
        ... )
        >>> result = dollar_volume(df)
        >>> print(result)
        ┌────────┬────────────┬───────┬─────────┬──────────────┐
        │ symbol ┆ date       ┆ close ┆ volume  ┆ dollar_volume │
        │ ---    ┆ ---        ┆ ---   ┆ ---     ┆ ---          │
        │ str    ┆ date       ┆ f64   ┆ i64     ┆ f64          │
        ╞════════╪════════════╪═══════╪═════════╪══════════════╡
        │ AAPL   ┆ 2023-01-01 ┆ 150.0 ┆ 1000000 ┆ 1.5000e8     │
        │ AAPL   ┆ 2023-01-02 ┆ 155.0 ┆ 1500000 ┆ 2.3250e8     │
        │ GOOGL  ┆ 2023-01-01 ┆ 2800.0┆ 500000  ┆ 1.4000e9     │
        └────────┴────────────┴───────┴─────────┴──────────────┘
    """
    return dfp.with_columns((pl.col("close") * pl.col("volume")).alias("dollar_volume"))


def y_log_returns(dfp: pl.DataFrame, horizon: int) -> pl.DataFrame:
    """Calculate forward-looking cumulative log returns over a specified horizon.

    This function computes forward cumulative log returns by calculating the difference
    between the natural logarithm of the closing price at time t+horizon and the
    natural logarithm of the closing price at time t. This creates a target variable
    for predictive modeling, representing the log return that will be realized over
    the next `horizon` periods.

    The calculation works by:
    1. Taking the natural logarithm of closing prices
    2. Shifting prices forward by `horizon` periods within each symbol group
    3. Computing the difference: ln(price_{t+horizon}) - ln(price_t)
    4. Results in null values for the last `horizon` observations per symbol

    Formula: y_log_returns = ln(close_{t+horizon}) - ln(close_t)

    This is particularly useful for:
    - Creating prediction targets for machine learning models
    - Forward-looking return analysis and forecasting
    - Risk modeling and portfolio optimization
    - Multi-period return analysis

    Args:
        dfp: Input DataFrame containing stock data with at minimum 'symbol' and 'close'
             columns. Should be pre-sorted by ['symbol', 'date'] for meaningful results.
        horizon: Number of periods to look forward for calculating returns. Must be a
                positive integer. Higher values create longer-term return targets.

    Returns:
        DataFrame with all original columns plus a new 'y_log_returns' column containing
        the forward cumulative log returns. The last `horizon` observations for each
        symbol will be null (no future data available).

    Note:
        - Calculations are performed independently within each symbol group
        - Last `horizon` observations per symbol will be null (no future prices)
        - Input should be sorted by symbol and date for meaningful time series results
        - Close prices should be positive values (log of zero or negative values is undefined)
        - Forward-looking nature makes this suitable as a prediction target

    Example:
        >>> import polars as pl
        >>> from datetime import date
        >>> df = pl.DataFrame(
        ...     {
        ...         "symbol": ["AAPL", "AAPL", "AAPL", "AAPL"],
        ...         "date": [
        ...             date(2023, 1, 1),
        ...             date(2023, 1, 2),
        ...             date(2023, 1, 3),
        ...             date(2023, 1, 4),
        ...         ],
        ...         "close": [100.0, 105.0, 102.0, 110.0],
        ...     }
        ... )
        >>> result = y_log_returns(df, horizon=2)
        >>> print(result)
        ┌────────┬────────────┬───────┬───────────────┐
        │ symbol ┆ date       ┆ close ┆ y_log_returns │
        │ ---    ┆ ---        ┆ ---   ┆ ---           │
        │ str    ┆ date       ┆ f64   ┆ f64           │
        ╞════════╪════════════╪═══════╪═══════════════╡
        │ AAPL   ┆ 2023-01-01 ┆ 100.0 ┆ 0.019803      │
        │ AAPL   ┆ 2023-01-02 ┆ 105.0 ┆ 0.047000      │
        │ AAPL   ┆ 2023-01-03 ┆ 102.0 ┆ null          │
        │ AAPL   ┆ 2023-01-04 ┆ 110.0 ┆ null          │
        └────────┴────────────┴───────┴───────────────┘

        >>> # With horizon=1 for single-period forward returns
        >>> result_1 = y_log_returns(df, horizon=1)
        >>> print(result_1)
        ┌────────┬────────────┬───────┬───────────────┐
        │ symbol ┆ date       ┆ close ┆ y_log_returns │
        │ ---    ┆ ---        ┆ ---   ┆ ---           │
        │ str    ┆ date       ┆ f64   ┆ f64           │
        ╞════════╪════════════╪═══════╪═══════════════╡
        │ AAPL   ┆ 2023-01-01 ┆ 100.0 ┆ 0.048790      │
        │ AAPL   ┆ 2023-01-02 ┆ 105.0 ┆ -0.029559     │
        │ AAPL   ┆ 2023-01-03 ┆ 102.0 ┆ 0.075154      │
        │ AAPL   ┆ 2023-01-04 ┆ 110.0 ┆ null          │
        └────────┴────────────┴───────┴───────────────┘
    """
    return dfp.with_columns(
        (
            pl.col("close").log().shift(-horizon).over("symbol") - pl.col("close").log()
        ).alias("y_log_returns")
    )


def dff(
    log_returns_d: pl.DataFrame,
    y_log_returns: pl.DataFrame,
    dollar_volume: pl.DataFrame,
) -> pl.DataFrame:
    """Merge multiple feature DataFrames into a single consolidated DataFrame.

    This function combines three input DataFrames containing different feature sets
    by performing inner joins on 'date' and 'symbol' columns. It creates a unified
    feature DataFrame that contains daily log returns, forward-looking log returns,
    dollar volume, and OHLC price data, making it suitable for machine learning
    model training.

    The merging works by:
    1. Extracting relevant columns from each input DataFrame
    2. Performing sequential inner joins on ['date', 'symbol']
    3. Maintaining data consistency by ensuring only matching date-symbol pairs remain
    4. Sorting the final result by symbol and date for consistent ordering

    Args:
        log_returns_d: DataFrame containing daily log returns data with at minimum
                      'date', 'symbol', and 'log_returns_d' columns
        y_log_returns: DataFrame containing forward-looking log returns data with at
                      minimum 'date', 'symbol', and 'y_log_returns' columns
        dollar_volume: DataFrame containing dollar volume and OHLC data with at minimum
                      'date', 'symbol', 'dollar_volume', 'open', 'high', 'low',
                      'close', and 'volume' columns

    Returns:
        DataFrame with all features merged, containing columns: 'date', 'symbol',
        'log_returns_d', 'y_log_returns', 'dollar_volume', 'open', 'high', 'low',
        'close', 'volume'. Sorted by ['symbol', 'date'].

    Note:
        - Uses inner joins, so only date-symbol pairs present in ALL input DataFrames
          will be included in the result
        - Input DataFrames should be pre-processed and cleaned before merging
        - The function expects specific column names to be present in each input
        - Final result maintains chronological ordering within each symbol group

    Example:
        >>> import polars as pl
        >>> from datetime import date
        >>> # Create sample input DataFrames
        >>> log_returns_df = pl.DataFrame(
        ...     {
        ...         "date": [date(2023, 1, 1), date(2023, 1, 2)],
        ...         "symbol": ["AAPL", "AAPL"],
        ...         "log_returns_d": [0.01, 0.02],
        ...     }
        ... )
        >>> y_returns_df = pl.DataFrame(
        ...     {
        ...         "date": [date(2023, 1, 1), date(2023, 1, 2)],
        ...         "symbol": ["AAPL", "AAPL"],
        ...         "y_log_returns": [0.05, 0.03],
        ...     }
        ... )
        >>> dollar_vol_df = pl.DataFrame(
        ...     {
        ...         "date": [date(2023, 1, 1), date(2023, 1, 2)],
        ...         "symbol": ["AAPL", "AAPL"],
        ...         "dollar_volume": [1000000.0, 1100000.0],
        ...         "open": [100.0, 101.0],
        ...         "high": [102.0, 103.0],
        ...         "low": [99.0, 100.0],
        ...         "close": [101.0, 102.0],
        ...         "volume": [10000, 11000],
        ...     }
        ... )
        >>> merged = dff(log_returns_df, y_returns_df, dollar_vol_df)
        >>> print(merged)
        ┌────────────┬────────┬──────────────┬───────────────┬──────────────┬──────┬──────┬─────┬───────┬────────┐
        │ date       ┆ symbol ┆ log_returns_d ┆ y_log_returns ┆ dollar_volume ┆ open ┆ high ┆ low ┆ close ┆ volume │
        │ ---        ┆ ---    ┆ ---          ┆ ---           ┆ ---          ┆ ---  ┆ ---  ┆ --- ┆ ---   ┆ ---    │
        │ date       ┆ str    ┆ f64          ┆ f64           ┆ f64          ┆ f64  ┆ f64  ┆ f64 ┆ f64   ┆ i64    │
        ╞════════════╪════════╪══════════════╪═══════════════╪══════════════╪══════╪══════╪═════╪═══════╪════════╡
        │ 2023-01-01 ┆ AAPL   ┆ 0.01         ┆ 0.05          ┆ 1000000.0    ┆ 100.0┆ 102.0┆ 99.0┆ 101.0 ┆ 10000  │
        │ 2023-01-02 ┆ AAPL   ┆ 0.02         ┆ 0.03          ┆ 1100000.0    ┆ 101.0┆ 103.0┆ 100.0┆102.0 ┆ 11000  │
        └────────────┴────────┴──────────────┴───────────────┴──────────────┴──────┴──────┴─────┴───────┴────────┘
    """

    # Join the dataframes based on date and symbol

    log_returns_d = log_returns_d.select(
        pl.col("date"), pl.col("symbol"), pl.col("log_returns_d")
    )

    y_log_returns = y_log_returns.select(
        pl.col("date"), pl.col("symbol"), pl.col("y_log_returns")
    )

    dollar_volume = dollar_volume.select(
        pl.col("date"),
        pl.col("symbol"),
        pl.col("dollar_volume"),
        pl.col("open"),
        pl.col("high"),
        pl.col("low"),
        pl.col("close"),
        pl.col("volume"),
    )

    merged = log_returns_d.join(y_log_returns, on=["date", "symbol"], how="inner")
    merged = merged.join(dollar_volume, on=["date", "symbol"], how="inner")

    return merged.sort(["symbol", "date"])
