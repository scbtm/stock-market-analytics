import polars as pl


def kurtosis_long_short(dff: pl.DataFrame, long_window: int, short_window: int) -> pl.DataFrame:
    """
    Compute the kurtosis of log returns for each symbol over specified rolling windows.
    """
    return dff.with_columns([
        pl.col('log_returns_d').rolling_kurtosis(window_size=long_window).over('symbol').alias('long_kurtosis'),
        pl.col('log_returns_d').rolling_kurtosis(window_size=short_window).over('symbol').alias('short_kurtosis'),
    ])

def skewness_long_short(dff: pl.DataFrame, long_window: int, short_window: int) -> pl.DataFrame:
    """
    Compute the skewness of log returns for each symbol over specified rolling windows.
    """
    return dff.with_columns([
        pl.col('log_returns_d').rolling_skew(window_size=long_window).over('symbol').alias('long_skewness'),
        pl.col('log_returns_d').rolling_skew(window_size=short_window).over('symbol').alias('short_skewness'),
    ])

def mean_long_short(dff: pl.DataFrame, long_window: int, short_window: int) -> pl.DataFrame:
    """
    Compute the mean of log returns for each symbol over specified rolling windows.
    """
    return dff.with_columns([
        pl.col('log_returns_d').rolling_mean(window_size=long_window).over('symbol').alias('long_mean'),
        pl.col('log_returns_d').rolling_mean(window_size=short_window).over('symbol').alias('short_mean'),
    ])

def mean_diff(mean_long_short: pl.DataFrame) -> pl.DataFrame:
    """
    Compute the difference between long and short mean log returns.
    """
    return mean_long_short.with_columns([
        (pl.col('long_mean') - pl.col('short_mean')).alias('mean_diff')
    ])

def absolute_diff(dff: pl.DataFrame, long_window: int, short_window: int) -> pl.DataFrame:
    """
    Compute the rolling difference between long and short mean log returns.
    """
    return dff.with_columns([
        (pl.col('log_returns_d').rolling_max(window_size=long_window).over('symbol') -
         pl.col('log_returns_d').rolling_min(window_size=long_window).over('symbol')).alias('long_diff'),
        (pl.col('log_returns_d').rolling_max(window_size=short_window).over('symbol') -
         pl.col('log_returns_d').rolling_min(window_size=short_window).over('symbol')).alias('short_diff')
    ])

def long_short_momentum(dff: pl.DataFrame, long_window: int = 252, short_window: int = 21) -> pl.DataFrame:
    """
    Compute the long-short momentum for each symbol.
    """
    return dff.with_columns(
        (pl.col("log_returns_d").rolling_sum(long_window).over("symbol") -
         pl.col("log_returns_d").rolling_sum(short_window).over("symbol")).alias("long_short_momentum")
    )

def pct_from_high_long(dff: pl.DataFrame, long_window: int) -> pl.DataFrame:
    """
    Compute the percentage distance from the highest close price over a long window.
    """
    return dff.with_columns(
        (pl.col("close") / pl.col("close").rolling_max(long_window).over("symbol") - 1)
        .alias("pct_from_high_long")
    )

def pct_from_high_short(dff: pl.DataFrame, short_window: int) -> pl.DataFrame:
    """
    Compute the percentage distance from the highest close price over a short window.
    """
    return dff.with_columns(
        (pl.col("close") / pl.col("close").rolling_max(short_window).over("symbol") - 1)
        .alias("pct_from_high_short")
    )

def chronological_features(dff: pl.DataFrame) -> pl.DataFrame:
    """
    Extract chronological features from the date column.
    """
    return dff.with_columns(
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.weekday().alias("day_of_week"),
        pl.col("date").dt.ordinal_day().alias("day_of_year")
    )

def df_features(
    dff: pl.DataFrame,
    kurtosis_long_short: pl.DataFrame,
    skewness_long_short: pl.DataFrame,
    mean_long_short: pl.DataFrame,
    mean_diff: pl.DataFrame,
    absolute_diff: pl.DataFrame,
    long_short_momentum: pl.DataFrame,
    pct_from_high_long: pl.DataFrame,
    pct_from_high_short: pl.DataFrame,
    chronological_features: pl.DataFrame
) -> pl.DataFrame:

    """
    Combine all statistical features into a single DataFrame.
    """
    dff = dff.select(
        pl.col('symbol'),
        pl.col('date'),
        pl.col('y_log_returns'),
        pl.col('dollar_volume')
    )

    kurtosis_long_short = kurtosis_long_short.select(
        pl.col('symbol'),
        pl.col('date'),
        pl.col('long_kurtosis'),
        pl.col('short_kurtosis')
    )
    skewness_long_short = skewness_long_short.select(
        pl.col('symbol'),
        pl.col('date'),
        pl.col('long_skewness'),
        pl.col('short_skewness')
    )
    mean_long_short = mean_long_short.select(
        pl.col('symbol'),
        pl.col('date'),
        pl.col('long_mean'),
        pl.col('short_mean')
    )
    mean_diff = mean_diff.select(
        pl.col('symbol'),
        pl.col('date'),
        pl.col('mean_diff')
    )
    absolute_diff = absolute_diff.select(
        pl.col('symbol'),
        pl.col('date'),
        pl.col('long_diff'),
        pl.col('short_diff')
    )

    long_short_momentum = long_short_momentum.select(
        pl.col('symbol'),
        pl.col('date'),
        pl.col('long_short_momentum')
    )

    pct_from_high_long = pct_from_high_long.select(
        pl.col('symbol'),
        pl.col('date'),
        pl.col('pct_from_high_long')
    )

    pct_from_high_short = pct_from_high_short.select(
        pl.col('symbol'),
        pl.col('date'),
        pl.col('pct_from_high_short')
    )

    chronological_features = chronological_features.select(
        pl.col('symbol'),
        pl.col('date'),
        pl.col('year'),
        pl.col('month'),
        pl.col('day_of_week'),
        pl.col('day_of_year')
    )

    final_df = dff.join(
        kurtosis_long_short,
        on=['symbol', 'date'],
        how='inner'
    ).join(
        skewness_long_short,
        on=['symbol', 'date'],
        how='inner'
    ).join(
        mean_long_short,
        on=['symbol', 'date'],
        how='inner'
    ).join(
        mean_diff,
        on=['symbol', 'date'],
        how='inner'
    ).join(
        absolute_diff,
        on=['symbol', 'date'],
        how='inner'
    ).join(
        long_short_momentum,
        on=['symbol', 'date'],
        how='inner'
    ).join(
        pct_from_high_long,
        on=['symbol', 'date'],
        how='inner'
    ).join(
        pct_from_high_short,
        on=['symbol', 'date'],
        how='inner'
    ).join(
        chronological_features,
        on=['symbol', 'date'],
        how='inner'
    )

    return final_df