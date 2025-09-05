import polars as pl

def sorted_df(raw_df: pl.DataFrame) -> pl.DataFrame:
    """
    Sorts the input DataFrame by 'symbol' and 'timestamp' columns.

    Args:
        raw_df (pl.DataFrame): Input DataFrame containing stock market data.

    Returns:
        pl.DataFrame: Sorted DataFrame.
    """
    return raw_df.sort(["symbol", "date"])

def interpolated_df(sorted_df: pl.DataFrame) -> pl.DataFrame:
    """
    Interpolates missing values in the sorted DataFrame using forward fill method.

    Args:
        sorted_df (pl.DataFrame): Sorted DataFrame containing stock market data.

    Returns:
        pl.DataFrame: DataFrame with interpolated missing values.
    """
    return sorted_df.with_columns(pl.col("*").interpolate().over("symbol")).drop_nulls()  # Drop rows with null values created by shifts

def dff(interpolated_df: pl.DataFrame, horizon: int) -> pl.DataFrame:
    """
    Computes basic features such as log returns and rolling volatility.

    Args:
        interpolated_df (pl.DataFrame): DataFrame with interpolated missing values.

    Returns:
        pl.DataFrame: DataFrame with additional basic features.
    """
    year_expr = pl.col("date").dt.year().alias("year")

    month_expr = pl.col("date").dt.month().alias("month")

    day_of_week_expr = pl.col("date").dt.weekday().alias("day_of_week")

    day_of_year_expr = pl.col("date").dt.ordinal_day().alias("day_of_year")

    dollar_volume_expr = (pl.col("close") * pl.col("volume")).shift(1).over("symbol").alias("dollar_volume")

    log_returns_d_expr = pl.col("close").log().diff().shift(1).over("symbol").alias("log_returns_d")

    y_log_returns_expr = (pl.col("close").log().shift(-horizon).over("symbol") - pl.col("close").log()).alias("y_log_returns")

    output_df = interpolated_df.with_columns([
        year_expr,
        month_expr,
        day_of_week_expr,
        day_of_year_expr,
        dollar_volume_expr,
        log_returns_d_expr,
        y_log_returns_expr,
    ])

    return output_df