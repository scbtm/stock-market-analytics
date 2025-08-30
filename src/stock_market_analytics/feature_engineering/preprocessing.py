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
        >>> df = pl.DataFrame({
        ...     "symbol": ["AAPL", "GOOGL", "AAPL"], 
        ...     "date": [date(2023, 1, 2), date(2023, 1, 1), date(2023, 1, 1)],
        ...     "price": [150.0, 100.0, 149.0]
        ... })
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


def dfp(base: pl.DataFrame) -> pl.DataFrame:
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
        >>> df = pl.DataFrame({
        ...     "symbol": ["AAPL", "AAPL", "AAPL"], 
        ...     "date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
        ...     "price": [100.0, None, 104.0]
        ... })
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
        >>> df_boundary = pl.DataFrame({
        ...     "symbol": ["AAPL", "AAPL"], 
        ...     "price": [100.0, None]  # boundary null stays null
        ... })
        >>> result = dfp(df_boundary)
        >>> print(result["price"].to_list())
        [100.0, None]
    """
    return base.with_columns(
        pl.col("*").interpolate().over("symbol")
    )


def clean(interpolated: pl.DataFrame) -> pl.DataFrame:
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
        >>> df = pl.DataFrame({
        ...     "symbol": ["AAPL", "AAPL", "AAPL", "AAPL"],
        ...     "date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3), date(2023, 1, 4)],
        ...     "price": [None, 100.0, 102.0, None],  # boundary nulls
        ...     "volume": [None, 1000, 1100, None]
        ... })
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
