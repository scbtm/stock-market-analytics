import polars as pl

def sorted_df(raw_df: pl.DataFrame) -> pl.DataFrame:
    """
    Sorts the input DataFrame by 'symbol' and 'timestamp' columns in ascending order.

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

    # Forward fill interpolation by symbol to prevent leakage across different stocks and dates
    sorted_df = sorted_df.with_columns(pl.col("*").forward_fill().over("symbol")).drop_nulls()

    year_expr = pl.col("date").dt.year().alias("year")

    month_expr = pl.col("date").dt.month().alias("month")

    day_of_week_expr = pl.col("date").dt.weekday().alias("day_of_week")

    day_of_year_expr = pl.col("date").dt.ordinal_day().alias("day_of_year")

    return  sorted_df.with_columns([
        year_expr,
        month_expr,
        day_of_week_expr,
        day_of_year_expr,
    ])

def basic_indicators_df(
        interpolated_df: pl.DataFrame,
        horizon: int,
        short_window: int,
        long_window: int
        ) -> pl.DataFrame:
    """
    Computes basic features such as log returns and rolling volatility.

    Args:
        interpolated_df (pl.DataFrame): DataFrame with interpolated missing values.

    Returns:
        pl.DataFrame: DataFrame with additional basic features.
    """

    window = int((short_window + long_window)/2)

    dollar_volume_expr = (pl.col("close") * pl.col("volume")).over("symbol").alias("dollar_volume")

    log_returns_d_expr = pl.col("close").log().diff().over("symbol").alias("log_returns_d")

    y_log_returns_expr = (pl.col("close").log().shift(-horizon).over("symbol") - pl.col("close").log()).alias("y_log_returns")

    short_log_returns_expr = pl.col("close").log().shift(short_window).over("symbol").alias("short_log_returns")
    long_log_returns_expr = pl.col("close").log().shift(long_window).over("symbol").alias("long_log_returns")

    log_returns_ratio_expr = (short_log_returns_expr - long_log_returns_expr).alias("log_returns_ratio")

    interpolated_df = interpolated_df.with_columns([
        dollar_volume_expr,
        log_returns_d_expr,
        log_returns_ratio_expr,
        y_log_returns_expr
    ])



    # Expression for an exponentially weighted RSI.
    up = pl.when(pl.col('log_returns_d') > 0).then(pl.col('log_returns_d')).otherwise(0.0)
    dn = pl.when(pl.col('log_returns_d') < 0).then(-pl.col('log_returns_d')).otherwise(0.0)
    rs = up.ewm_mean(span=window).over("symbol") / (
        dn.ewm_mean(span=window).over("symbol") + 1e-8
    )

    rsi_ewm_expr = (100 - (100 / (1 + rs))).alias("rsi_ewm")

    # Expression for the Sortino ratio over a short rolling window.
    log_ret = pl.col("log_returns_d")
    downside_returns = pl.when(log_ret < 0).then(log_ret).otherwise(0.0)
    downside_std = downside_returns.rolling_std(window).over("symbol")
    mean_return = log_ret.rolling_mean(window).over("symbol")
    sortino_ratio = (mean_return / (downside_std + 1e-8)).alias("sortino_ratio")

    # Expression for the Sharpe ratio over a rolling window.
    total_returns = log_ret.rolling_sum(window).over("symbol")
    sharpe_ratio_proxy = (total_returns / (pl.col("log_returns_d").rolling_std(window).over("symbol") + 1e-8)).alias("sharpe_ratio_proxy")

    return interpolated_df.with_columns([
        rsi_ewm_expr,
        sortino_ratio,
        sharpe_ratio_proxy
    ])

def volatility_indicators_df(basic_indicators_df: pl.DataFrame, long_window: int, short_window: int) -> pl.DataFrame:
    """
    Computes volatility indicators such as long-term and short-term rolling volatility.
    Args:
        basic_indicators_df (pl.DataFrame): DataFrame with basic features.
        long_window (int): Window size for long-term volatility.
        short_window (int): Window size for short-term volatility.
    Returns:
        pl.DataFrame: DataFrame with additional volatility indicators.
    """

    long_vol_expr = pl.col('log_returns_d').ewm_std(span=long_window).over("symbol").shift(1).alias("long_vol_ewm")
    short_vol_expr = pl.col('log_returns_d').ewm_std(span=short_window).over("symbol").shift(1).alias("short_vol_ewm")
    vol_ratio_expr = (long_vol_expr / short_vol_expr + 1e-8).alias("vol_ratio")

    return basic_indicators_df.with_columns([
        long_vol_expr,
        short_vol_expr,
        vol_ratio_expr
    ]).select(["symbol", "date", "vol_ratio", "long_vol_ewm", "short_vol_ewm"])

def volatility_features_df(volatility_indicators_df: pl.DataFrame, long_window: int) -> pl.DataFrame:

    long_vol_ewm_expr = pl.col("long_vol_ewm")
    vol_of_vol_expr = pl.col("short_vol_ewm").rolling_std(long_window).over("symbol").alias("vol_of_vol_ewm")

    vol_expansion_expr = (long_vol_ewm_expr / (long_vol_ewm_expr.shift(long_window).over("symbol") + 1e-8)).alias("vol_expansion")

    return volatility_indicators_df.with_columns([vol_of_vol_expr, vol_expansion_expr]).select([
        "symbol",
        "date",
        # "vol_ratio",
        "vol_of_vol_ewm",
        # "vol_expansion"
    ])

def momentum_indicators_df(basic_indicators_df: pl.DataFrame, long_window: int, short_window: int) -> pl.DataFrame:
    """
    Computes momentum indicators such as long-term and short-term momentum.
    Args:
        basic_indicators_df (pl.DataFrame): DataFrame with basic features.
        long_window (int): Window size for long-term momentum.
        short_window (int): Window size for short-term momentum.
    Returns:
        pl.DataFrame: DataFrame with additional momentum indicators.
    """

    long_momentum_expr = pl.col('log_returns_d').rolling_sum(long_window).over("symbol").alias("long_momentum")
    short_momentum_expr = pl.col('log_returns_d').rolling_sum(short_window).over("symbol").alias("short_momentum")
    long_short_momentum_expr = (long_momentum_expr - short_momentum_expr).alias("long_short_momentum")

    log_ret = pl.col("log_returns_d")
    sum_up = pl.when(log_ret > 0).then(log_ret).otherwise(0.0).rolling_sum(long_window).over("symbol")
    sum_down = pl.when(log_ret < 0).then(-log_ret).otherwise(0.0).rolling_sum(long_window).over("symbol")
    cmo = ((sum_up - sum_down) / (sum_up + sum_down + 1e-8) * 100).alias("cmo")

    return basic_indicators_df.with_columns([
        long_short_momentum_expr,
        cmo
    ]).select(["symbol", "date", "long_short_momentum", "cmo"])


def momentum_features_df(momentum_indicators_df: pl.DataFrame, volatility_indicators_df: pl.DataFrame) -> pl.DataFrame:

    momentum_indicators_df = momentum_indicators_df.join(
        volatility_indicators_df,
        on=["symbol", "date"],
        how="inner"
    )

    long_short_momentum_expr = pl.col("long_short_momentum")
    long_vol_ewm_expr = pl.col("long_vol_ewm")

    risk_adj_momentum_expr = (long_short_momentum_expr / (long_vol_ewm_expr + 1e-8)).alias("risk_adj_momentum")

    return momentum_indicators_df.with_columns([risk_adj_momentum_expr]).select([
        "symbol",
        "date",
        # "long_short_momentum",
        "cmo",
        "risk_adj_momentum"
    ])


def liquidity_indicators_df(
        basic_indicators_df: pl.DataFrame,
        short_window: int,
        long_window: int) -> pl.DataFrame:
    """
    Computes liquidity indicators such as average dollar volume.

    Args:
        basic_indicators_df (pl.DataFrame): DataFrame with basic features.
        volume_window (int): Window size for average dollar volume.

    Returns:
        pl.DataFrame: DataFrame with additional liquidity indicators.
    """

    abs_return = pl.col("log_returns_d").abs()

    amihud_illiq_expr = (
        (abs_return / (pl.col("dollar_volume") + 1e-8))
        .rolling_mean(short_window)
        .over("symbol")
    ).alias("amihud_illiq")

    avg_dollar_volume = pl.col("dollar_volume").rolling_mean(long_window).over("symbol")
    turnover_proxy = (pl.col("dollar_volume") / (avg_dollar_volume + 1e-12)).alias("turnover_proxy")

    return basic_indicators_df.with_columns([
        amihud_illiq_expr,
        turnover_proxy
    ]).select(["symbol", "date", "amihud_illiq", "turnover_proxy"])

def statistical_indicators_df(
        basic_indicators_df: pl.DataFrame,
        long_window: int,
        short_window: int) -> pl.DataFrame:
    """
    Computes statistical indicators such as rolling z-score of log returns.

    Args:
        basic_indicators_df (pl.DataFrame): DataFrame with basic features.
        zscore_window (int): Window size for rolling z-score.

    Returns:
        pl.DataFrame: DataFrame with additional statistical indicators.
    """

    long_kurtosis_expr = pl.col("log_returns_d").rolling_kurtosis(window_size=long_window).over("symbol").alias("long_kurtosis")
    short_kurtosis_expr = pl.col("log_returns_d").rolling_kurtosis(window_size=short_window).over("symbol").alias("short_kurtosis")
    kurtosis_ratio_expr = (long_kurtosis_expr / (short_kurtosis_expr + 1e-8)).alias("kurtosis_ratio")

    long_skew_expr = pl.col("log_returns_d").rolling_skew(window_size=long_window).over("symbol").alias("long_skew")
    short_skew_expr = pl.col("log_returns_d").rolling_skew(window_size=short_window).over("symbol").alias("short_skew")
    skew_ratio_expr = (long_skew_expr / (short_skew_expr + 1e-8)).alias("skew_ratio")
    
    long_rolling_mean = pl.col("log_returns_d").rolling_mean(long_window).over("symbol")
    long_rolling_std = pl.col("log_returns_d").rolling_std(long_window).over("symbol")

    short_rolling_mean = pl.col("log_returns_d").rolling_mean(short_window).over("symbol")
    short_rolling_std = pl.col("log_returns_d").rolling_std(short_window).over("symbol")

    long_zscore_expr = ((pl.col("log_returns_d") - long_rolling_mean) / (long_rolling_std + 1e-8)).alias("long_zscore")
    short_zscore_expr = ((pl.col("log_returns_d") - short_rolling_mean) / (short_rolling_std + 1e-8)).alias("short_zscore")
    zscore_ratio_expr = (long_zscore_expr / (short_zscore_expr + 1e-8)).alias("zscore_ratio")

    return basic_indicators_df.with_columns([
        kurtosis_ratio_expr,
        skew_ratio_expr,
        zscore_ratio_expr,
    ]).select(["symbol", "date", "kurtosis_ratio", "skew_ratio", "zscore_ratio", "log_returns_d"])

def statistical_features_df(statistical_indicators_df: pl.DataFrame, horizon: int) -> pl.DataFrame:

    # Expression for the autocorrelation of squared returns.
    r2 = pl.col("log_returns_d").pow(2)
    statistical_indicators_df = statistical_indicators_df.with_columns(r2.alias("r2"))
    statistical_indicators_df = statistical_indicators_df.with_columns([pl.col("r2").shift(horizon).over("symbol").alias("r2_shifted")])

    autocorr_expr = pl.rolling_corr(a = pl.col("r2"), b = pl.col("r2_shifted"), window_size=horizon).over("symbol")

    y_hat_expr = pl.when(autocorr_expr.is_finite()).then(autocorr_expr.clip(-1.0, 1.0)).otherwise(0).alias("autocorr_r2")

    p90 = pl.col("log_returns_d").rolling_quantile(quantile = 0.90, interpolation = "nearest", window_size = horizon*4).over("symbol")
    p10 = pl.col("log_returns_d").rolling_quantile(quantile = 0.10, interpolation = "nearest", window_size = horizon*4).over("symbol")

    iqr_vol_expr = (p90 - p10).alias("iqr_vol")

    return statistical_indicators_df.with_columns([y_hat_expr, iqr_vol_expr]).select([
        "symbol",
        "date",
        "kurtosis_ratio",
        "skew_ratio",
        "zscore_ratio",
        "autocorr_r2",
        "iqr_vol"
    ]).sort(["symbol", "date"])

def ichimoku_indicators_df(
        basic_indicators_df: pl.DataFrame,
        ichimoku_params: dict
        ) -> pl.DataFrame:
    
    p1 = ichimoku_params["p1"]
    p2 = ichimoku_params["p2"]
    p3 = ichimoku_params["p3"]
    atr_n = ichimoku_params["atr_n"]
    
    high = pl.col("high"); low = pl.col("low"); close = pl.col("close")

    # --- Core lines (aligned to "now" — no forward shift to avoid leaks) ---
    # by symbol to avoid leakage across different stocks
    tenkan = ((high.rolling_max(p1) + low.rolling_min(p1)) / 2).over("symbol")
    kijun  = ((high.rolling_max(p2) + low.rolling_min(p2)) / 2).over("symbol")
    span_a_now = ((tenkan + kijun) / 2)                              # A (no shift)
    span_b_now = ((high.rolling_max(p3) + low.rolling_min(p3)) / 2).over("symbol")  # B (no shift)

    cloud_top_now = pl.max_horizontal(span_a_now, span_b_now)
    cloud_bot_now = pl.min_horizontal(span_a_now, span_b_now)

    prev_close = close.shift(1).over("symbol")

    basic_indicators_df = basic_indicators_df.with_columns([
        tenkan.alias("tenkan"),
        kijun.alias("kijun"),
        span_a_now.alias("span_a_now"),
        span_b_now.alias("span_b_now"),
        cloud_top_now.alias("cloud_top_now"),
        cloud_bot_now.alias("cloud_bot_now"),
        prev_close.alias("prev_close")
    ])

    # --- “Lead” cloud as on charts (computed in past, plotted ahead). Safe to use if BOTH are shifted. ---
    span_a_now = pl.col("span_a_now")
    span_b_now = pl.col("span_b_now")
    # Shifted by p2 periods into the future (lead)
    span_a_lead = span_a_now.shift(p2).over("symbol")
    span_b_lead = span_b_now.shift(p2).over("symbol")
    lead_top = pl.max_horizontal(span_a_lead, span_b_lead)
    lead_bot = pl.min_horizontal(span_a_lead, span_b_lead)

    # --- ATR for normalization (comparability across symbols/time) ---
    prev_close = pl.col("prev_close")
    tr = pl.max_horizontal(
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    )

    atr = tr.rolling_mean(window_size = atr_n).over("symbol").alias("atr") + 1e-8  # avoid div by zero

    # --- Distances / spreads (ATR-normalized) ---
    price_above_cloud_atr = ((close - cloud_top_now) / atr ).alias("price_above_cloud_atr")
    price_below_cloud_atr = ((cloud_bot_now - close) / atr).alias("price_below_cloud_atr")
    tenkan_kijun_spread_atr = ((tenkan - kijun) / atr).alias("tenkan_kijun_spread_atr")
    cloud_thickness_atr = ((span_a_now - span_b_now).abs() / atr).alias("cloud_thickness_atr")
    price_vs_lead_top_atr = ((close - lead_top) / atr).alias("price_vs_lead_top_atr")
    price_vs_lead_bot_atr = ((close - lead_bot) / atr).alias("price_vs_lead_bot_atr")

    return basic_indicators_df.with_columns([
        atr,
        price_above_cloud_atr,
        price_below_cloud_atr,
        tenkan_kijun_spread_atr,
        cloud_thickness_atr,
        price_vs_lead_top_atr,
        price_vs_lead_bot_atr
    ]).select([
        "symbol",
        "date",
        "close",
        "tenkan",
        "kijun",
        "span_a_now",
        "span_b_now",
        "cloud_top_now",
        "cloud_bot_now",
        "atr",
        "price_above_cloud_atr",
        "price_below_cloud_atr",
        "tenkan_kijun_spread_atr",
        "cloud_thickness_atr",
        "price_vs_lead_top_atr",
        "price_vs_lead_bot_atr"
    ])


def ichimoku_features_df(
        ichimoku_indicators_df: pl.DataFrame,
        ichimoku_params: dict
        ) -> pl.DataFrame:
    
    """
    Computes advanced Ichimoku-based features such as persistence and slope.
    Args:
        ichimoku_indicators_df (pl.DataFrame): DataFrame with Ichimoku indicators.
        persist_window (int): Window size for persistence calculation.
        slope_window (int): Window size for slope calculation.
    Returns:
        pl.DataFrame: DataFrame with additional Ichimoku-based features.
    """

    slope_window = ichimoku_params["slope_window"]

    # --- Slopes (trend strength) ---
    def _slope(column: str):
        # A column must exist before calling this function
        return ((pl.col(column) - pl.col(column).shift(slope_window).over("symbol")) / slope_window)

    features = {
        "tenkan_slope": _slope("tenkan").alias("tenkan_slope"),
        "kijun_slope": _slope("kijun").alias("kijun_slope"),
        "span_a_slope": _slope("span_a_now").alias("span_a_slope"),
        "span_b_slope": _slope("span_b_now").alias("span_b_slope"),
        "cloud_top_slope": _slope("cloud_top_now").alias("cloud_top_slope"),
        "cloud_bot_slope": _slope("cloud_bot_now").alias("cloud_bot_slope")
    }

    # --- Regime + persistence (state & time-in-state) ---
    close = pl.col("close")
    cloud_top_now = pl.col("cloud_top_now")
    cloud_bot_now = pl.col("cloud_bot_now")
    persist_window = ichimoku_params["persist_window"]

    tenkan = pl.col("tenkan")
    kijun = pl.col("kijun")

    above_cloud = (close > cloud_top_now).cast(pl.Int8).alias("above_cloud")
    between_cloud = ((close >= cloud_bot_now) & (close <= cloud_top_now)).cast(pl.Int8).alias("between_cloud")
    below_cloud = (close < cloud_bot_now).cast(pl.Int8).alias("below_cloud")
    features |= {
        "above_cloud": above_cloud,
        "between_cloud": between_cloud,
        "below_cloud": below_cloud,
        "above_cloud_persist": above_cloud.rolling_mean(persist_window).over("symbol").alias("above_cloud_persist"),
        "below_cloud_persist": below_cloud.rolling_mean(persist_window).over("symbol").alias("below_cloud_persist"),
    }


     # --- Cross / breakout events (binary) ---
    features |= {
        "tenkan_cross_up": ((tenkan > kijun) & (tenkan.shift(1).over("symbol") <= kijun.shift(1).over("symbol"))).cast(pl.Int8).alias("tenkan_cross_up"),
        "tenkan_cross_dn": ((tenkan < kijun) & (tenkan.shift(1).over("symbol") >= kijun.shift(1).over("symbol"))).cast(pl.Int8).alias("tenkan_cross_dn"),
        "price_break_up": ((close > cloud_top_now) & (close.shift(1).over("symbol") <= cloud_top_now.shift(1).over("symbol"))).cast(pl.Int8).alias("price_break_up"),
        "price_break_dn": ((close < cloud_bot_now) & (close.shift(1).over("symbol") >= cloud_bot_now.shift(1).over("symbol"))).cast(pl.Int8).alias("price_break_dn"),
    }

    ichimoku_indicators_df = ichimoku_indicators_df.with_columns(features.values())


    # aux features for twist detection
    span_a_now = pl.col("span_a_now")
    span_b_now = pl.col("span_b_now")

    span_diff = (span_a_now - span_b_now)

    features |= {"span_diff": span_diff.alias("span_diff")}

    ichimoku_indicators_df = ichimoku_indicators_df.with_columns(features.values())

    ichimoku_indicators_df = ichimoku_indicators_df.with_columns(pl.col("span_diff").shift(1).over("symbol").alias("span_diff_shifted"))
    #empty dict to collect features

    features = {}
    span_diff = pl.col("span_diff")
    span_diff_shifted = pl.col("span_diff_shifted")

    # --- Cloud twist (A crosses B) & recency of twists ---
    features |= {
        "twist_event": ((span_diff * span_diff_shifted) < 0).cast(pl.Int8).alias("twist_event"),
    }

    ichimoku_indicators_df = ichimoku_indicators_df.with_columns(features.values())

    features = {}

    # Recency of twist events (how long since last twist)
    twist_event = pl.col("twist_event")
    features |= {
        "twist_recent": twist_event.rolling_sum(persist_window).over("symbol").alias("twist_recent")
    }

    ichimoku_indicators_df = ichimoku_indicators_df.with_columns(features.values())

    features = {}

    # --- Interaction signals (strength conditional on regime) ---
    tenkan_kijun_spread_atr = pl.col("tenkan_kijun_spread_atr")
    # price_above_cloud_atr = pl.col("price_above_cloud_atr")
    # price_below_cloud_atr = pl.col("price_below_cloud_atr")
    # cloud_thickness_atr = pl.col("cloud_thickness_atr")
    # price_vs_lead_top_atr = pl.col("price_vs_lead_top_atr")
    # price_vs_lead_bot_atr = pl.col("price_vs_lead_bot_atr")
    atr = pl.col("atr")
    features |= {
        "bull_strength": (tenkan_kijun_spread_atr * above_cloud).alias("bull_strength"),
        "bear_strength": ((kijun - tenkan) / atr * below_cloud).alias("bear_strength"),
    }

    return ichimoku_indicators_df.with_columns(features.values()).select([
        "symbol",
        "date",
        # "tenkan_slope",
        # "kijun_slope",
        # "span_a_slope",
        # "span_b_slope",
        "cloud_top_slope",
        "cloud_bot_slope",
        "above_cloud",
        # "between_cloud",
        # "below_cloud",
        "above_cloud_persist",
        # "below_cloud_persist",
        "tenkan_cross_up",
        # "tenkan_cross_dn",
        "price_break_up",
        # "price_break_dn",
        # "twist_event",
        "twist_recent",
        "bull_strength",
        # "bear_strength",
        "tenkan_kijun_spread_atr",
        # "price_above_cloud_atr",
        # "price_below_cloud_atr",
        "cloud_thickness_atr",
        "price_vs_lead_top_atr",
        # "price_vs_lead_bot_atr",
        # "atr"
    ]).sort(["symbol", "date"])


def df_features(
        basic_indicators_df: pl.DataFrame,
        volatility_features_df: pl.DataFrame,
        momentum_features_df: pl.DataFrame,
        liquidity_indicators_df: pl.DataFrame,
        statistical_features_df: pl.DataFrame,
        ichimoku_features_df: pl.DataFrame
        ) -> pl.DataFrame:

    basic_indicators_df = basic_indicators_df.drop(["dollar_volume"])
    
    basic_indicators_df = basic_indicators_df.join(
        liquidity_indicators_df,
        on=["symbol", "date"],
        how="inner"
    ).join(
        statistical_features_df,
        on=["symbol", "date"],
        how="inner"
    ).join(
        momentum_features_df,
        on=["symbol", "date"],
        how="inner"
    ).join(
        volatility_features_df,
        on=["symbol", "date"],
        how="inner"
    ).join(
        ichimoku_features_df,
        on=["symbol", "date"],
        how="inner"
    )

    return basic_indicators_df