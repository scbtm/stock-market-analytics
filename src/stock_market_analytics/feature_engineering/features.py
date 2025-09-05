import polars as pl

# This file demonstrates how to structure a Polars feature engineering pipeline
# to be compatible with the Hamilton framework. The key idea is to have functions
# define and return Polars *expressions* (pl.Expr) rather than fully computed
# DataFrames. A final function then gathers all expressions and executes them
# in a single, optimized `with_columns` call.

# This pattern gives you the best of both worlds:
# 1. Hamilton's DAG visualization and modularity.
# 2. Polars' high-performance query optimization.
# 3. Easy unit testing for each individual feature expression.

# --- Base Expresions (Inputs to nodes in the DAG) ---

def long_vol_ewm_expr(long_window: int) -> pl.Expr:
    """Expression for long-term exponentially weighted volatility."""
    return pl.col('log_returns_d').ewm_std(span=long_window).over("symbol").shift(1).alias("long_vol_ewm")

def short_vol_ewm_expr(short_window: int) -> pl.Expr:
    """Expression for short-term exponentially weighted volatility."""
    return pl.col('log_returns_d').ewm_std(span=short_window).over("symbol").shift(1).alias("short_vol_ewm")

def long_momentum_expr(long_window: int) -> pl.Expr:
    """Expression for long-term rolling sum of returns (momentum)."""
    return pl.col('log_returns_d').rolling_sum(long_window).over("symbol").shift(1).alias("long_momentum")

def short_momentum_expr(short_window: int) -> pl.Expr:
    """Expression for short-term rolling sum of returns (momentum)."""
    return pl.col('log_returns_d').rolling_sum(short_window).over("symbol").shift(1).alias("short_momentum")

#Input information is already shifted by 1 in the above functions
def long_short_momentum_expr(long_momentum_expr: pl.Expr, short_momentum_expr: pl.Expr) -> pl.Expr:
    """Expression for standard long-short momentum."""
    return (long_momentum_expr - short_momentum_expr).alias("long_short_momentum")

def ichimoku_components_expr(
        ichimoku_p1: int,
        ichimoku_p2: int,
        ichimoku_p3: int
        ) -> dict[str, pl.Expr]:
    """
    Expressions for Ichimoku Cloud-based features.
    Returns a dictionary of expressions for various Ichimoku-based features.
    """
    # Standard Ichimoku periods
    p1, p2, p3 = ichimoku_p1, ichimoku_p2, ichimoku_p3

    high = pl.col("high")
    low = pl.col("low")
    close = pl.col("close")

    # Component Lines
    tenkan_sen = ((high.rolling_max(p1) + low.rolling_min(p1)) / 2).over("symbol")
    kijun_sen = ((high.rolling_max(p2) + low.rolling_min(p2)) / 2).over("symbol")
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    senkou_span_b = ((high.rolling_max(p3) + low.rolling_min(p3)) / 2).shift(p2).over("symbol")

    # --- Feature Engineering from Ichimoku Components ---

    # 1. Price position relative to the cloud (trend)
    price_vs_cloud_top = (close - pl.max_horizontal(senkou_span_a, senkou_span_b))
    # 2. Tenkan-Kijun cross (momentum signal)
    tenkan_kijun_cross = (tenkan_sen - kijun_sen)
    # 3. Cloud thickness (volatility / strength of support/resistance)
    cloud_thickness = (senkou_span_a - senkou_span_b).abs()
    return {
        "price_vs_cloud_top": price_vs_cloud_top,
        "tenkan_kijun_cross": tenkan_kijun_cross,
        "cloud_thickness": cloud_thickness
        }

# input_expressions = [
#     long_vol_ewm_expr,
#     short_vol_ewm_expr,
#     long_momentum_expr,
#     short_momentum_expr,
#     ichimoku_components_expr
# ]


# --- Feature Expressions (Nodes in the DAG) ---

def amihud_illiq_expr(short_window: int) -> pl.Expr:
    """Expression for the Amihud illiquidity measure over a rolling window."""
    abs_return = pl.col("log_returns_d").abs()
    illiq = (
        (abs_return / (pl.col("dollar_volume") + 1e-12))
        .rolling_mean(short_window)
        .over("symbol")
    )
    return illiq.shift(1).alias("amihud_illiq")


def turnover_proxy_expr(long_window: int) -> pl.Expr:
    """Expression for turnover proxy over a rolling window."""
    avg_dollar_volume = pl.col("dollar_volume").rolling_mean(long_window).over("symbol")
    turnover_proxy = (pl.col("dollar_volume") / (avg_dollar_volume + 1e-12))
    return turnover_proxy.shift(1).alias("turnover_proxy")

def drawdown_expr(long_window: int) -> pl.Expr:
    """Expression for maximum drawdown over a rolling window."""
    rolling_max = pl.col("close").rolling_max(long_window).over("symbol")
    drawdown = (pl.col("close") / rolling_max - 1)
    return drawdown.shift(1).alias("max_drawdown")

def price_vs_cloud_top_expr(ichimoku_components_expr: dict[str, pl.Expr]) -> pl.Expr:
    """Price position relative to the cloud (trend)"""
    return ichimoku_components_expr["price_vs_cloud_top"].shift(1).alias("price_vs_cloud_top")

def tenkan_kijun_cross_expr(ichimoku_components_expr: dict[str, pl.Expr]) -> pl.Expr:
    """Tenkan-Kijun cross (momentum signal)"""
    return ichimoku_components_expr["tenkan_kijun_cross"].shift(1).alias("tenkan_kijun_cross")

def cloud_thickness_expr(ichimoku_components_expr: dict[str, pl.Expr]) -> pl.Expr:
    """Cloud thickness (volatility / strength of support/resistance)"""
    return ichimoku_components_expr["cloud_thickness"].shift(1).alias("cloud_thickness")

# Input expressions are already shifted by 1 in the above functions
def risk_adj_momentum_ewm_expr(long_short_momentum_expr: pl.Expr, long_vol_ewm_expr: pl.Expr) -> pl.Expr:
    """Expression for volatility-adjusted momentum."""
    return (long_short_momentum_expr / (long_vol_ewm_expr + 1e-12)).alias("risk_adj_momentum_ewm")

def cmo_expr(long_window: int) -> pl.Expr:
    """Expression for the Chande Momentum Oscillator (CMO)."""
    log_ret = pl.col("log_returns_d")

    sum_up = pl.when(log_ret > 0).then(log_ret).otherwise(0.0).rolling_sum(long_window).over("symbol")
    sum_down = pl.when(log_ret < 0).then(-log_ret).otherwise(0.0).rolling_sum(long_window).over("symbol")

    cmo = ((sum_up - sum_down) / (sum_up + sum_down + 1e-12) * 100)

    return cmo.shift(1).alias("cmo")

def rsi_ewm_expr(long_window: int) -> pl.Expr:
    """Expression for an exponentially weighted RSI."""
    up = pl.when(pl.col('log_returns_d') > 0).then(pl.col('log_returns_d')).otherwise(0.0)
    dn = pl.when(pl.col('log_returns_d') < 0).then(-pl.col('log_returns_d')).otherwise(0.0)
    rs = up.ewm_mean(span=long_window).over("symbol") / (
        dn.ewm_mean(span=long_window).over("symbol") + 1e-12
    )
    return (100 - 100 / (1 + rs)).shift(1).alias("rsi_ewm")

def sortino_ratio_expr(short_window: int) -> pl.Expr:
    """Expression for the Sortino ratio over a rolling window."""
    log_ret = pl.col("log_returns_d")
    downside_returns = pl.when(log_ret < 0).then(log_ret).otherwise(0.0)
    downside_std = downside_returns.rolling_std(short_window).over("symbol")
    mean_return = log_ret.rolling_mean(short_window).over("symbol")
    sortino_ratio = (mean_return / (downside_std + 1e-12))
    return sortino_ratio.shift(1).alias("sortino_ratio")

# Input expression is already shifted by 1 in the above function
def volatility_of_volatility_expr(long_window: int) -> pl.Expr:
    """Expression for the volatility of volatility."""
    return pl.col("short_vol_ewm").rolling_std(long_window).over("symbol").alias("volatility_of_volatility")

def autocorr_sq_returns_expr(short_window: int, horizon: int) -> pl.Expr:
    """Expression for the autocorrelation of squared returns."""
    r2 = pl.col("log_returns_d").pow(2)
    autocorr_sq_returns = (
        pl.rolling_corr(a = r2, b = r2.shift(horizon), window_size=short_window)
        .over("symbol")
        )
    output = pl.when(autocorr_sq_returns.is_finite()).then(autocorr_sq_returns.clip(-1.0, 1.0)).otherwise(0)
    return output.shift(1).alias("autocorr_sq_returns")

def iqr_vol_expr(short_window: int) -> pl.Expr:
    """
    Rolling inter-quantile range (q90 - q10) of returns as a robust vol measure.
    """
    p90 = pl.col("log_returns_d").rolling_quantile(quantile = 0.90, interpolation = "nearest", window_size = short_window).over("symbol")
    p10 = pl.col("log_returns_d").rolling_quantile(quantile = 0.10, interpolation = "nearest", window_size = short_window).over("symbol")
    return (p90 - p10).shift(1).alias("iqr_vol")

def pct_from_high_long_expr(long_window: int) -> pl.Expr:
    """
    Compute the percentage distance from the highest close price over a long window.
    """
    pct_fh = (
        pl.col("close") / pl.col("close").rolling_max(long_window).over("symbol")
        - 1
    )
    return pct_fh.shift(1).alias("pct_from_high_long")

def pct_from_high_short_expr(short_window: int) -> pl.Expr:
    """
    Compute the percentage distance from the highest close price over a short window.
    """
    pct_fh = (
        pl.col("close") / pl.col("close").rolling_max(short_window).over("symbol")
        - 1
    )
    return pct_fh.shift(1).alias("pct_from_high_short")

# dag_nodes [
#     amihud_illiq_expr,
#     turnover_proxy_expr,
#     drawdown_expr,
#     price_vs_cloud_top_expr,
#     tenkan_kijun_cross_expr,
#     cloud_thickness_expr,
#     risk_adj_momentum_ewm_expr,
#     cmo_expr,
#     rsi_ewm_expr,
#     sortino_ratio_expr,
#     volatility_of_volatility_expr,
#     autocorr_sq_returns_expr,
#     iqr_vol_expr,
#     pct_from_high_long_expr,
#     pct_from_high_short_expr,
#     ]

# --- Final Aggregator Function (Terminal Node in the DAG) ---

def df_features(
    dff: pl.DataFrame,
    amihud_illiq_expr: pl.Expr,
    turnover_proxy_expr: pl.Expr,
    drawdown_expr: pl.Expr,
    price_vs_cloud_top_expr: pl.Expr,
    tenkan_kijun_cross_expr: pl.Expr,
    cloud_thickness_expr: pl.Expr,
    risk_adj_momentum_ewm_expr: pl.Expr,
    cmo_expr: pl.Expr,
    rsi_ewm_expr: pl.Expr,
    sortino_ratio_expr: pl.Expr,
    autocorr_sq_returns_expr: pl.Expr,
    # volatility_of_volatility_expr: pl.Expr,
    iqr_vol_expr: pl.Expr,
    pct_from_high_long_expr: pl.Expr,
    pct_from_high_short_expr: pl.Expr,
) -> pl.DataFrame:
    """
    Collects all feature expressions from the Hamilton DAG and executes them
    in a single, optimized Polars query.
    """
    # Create a list of all the expressions to compute
    all_feature_expressions = [
        amihud_illiq_expr,
        turnover_proxy_expr,
        drawdown_expr,
        risk_adj_momentum_ewm_expr,
        cmo_expr,
        rsi_ewm_expr,
        autocorr_sq_returns_expr,
        sortino_ratio_expr,
        price_vs_cloud_top_expr,
        tenkan_kijun_cross_expr,
        cloud_thickness_expr,
        iqr_vol_expr,
        # volatility_of_volatility_expr,
        pct_from_high_long_expr,
        pct_from_high_short_expr,
    ]

    final_df = dff.with_columns(all_feature_expressions)

    return final_df