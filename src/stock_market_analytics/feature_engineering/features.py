import polars as pl


def log_returns_d(dfp: pl.DataFrame) -> pl.DataFrame:
    return dfp.with_columns(
        pl.col("close").log().diff().over("symbol").alias("log_returns_d")
    )

def dollar_volume(dfp: pl.DataFrame) -> pl.DataFrame:
    return dfp.with_columns((pl.col("close") * pl.col("volume")).alias("dollar_volume"))

# 12–1 momentum (e.g., past=252, recent=21)
def momentum_excl_recent(log_returns_d: pl.DataFrame, past: int, recent: int) -> pl.DataFrame:
    return log_returns_d.with_columns(
        (pl.col("log_returns_d").rolling_sum(past, by="symbol") -
         pl.col("log_returns_d").rolling_sum(recent, by="symbol")).alias("mom_12_1")
    )

# Short-term reversal (1m)
def reversal_1m(log_returns_d: pl.DataFrame, window: int = 21) -> pl.DataFrame:
    return log_returns_d.with_columns(
        pl.col("log_returns_d").rolling_sum(window, by="symbol").alias("rev_1m")
    )

# Distance to 52w high
def pct_from_52w_high(dfp: pl.DataFrame, window: int = 252) -> pl.DataFrame:
    return dfp.with_columns(
        (pl.col("close") / pl.col("close").rolling_max(window, by="symbol") - 1)
        .alias("pct_from_52w_high")
    )

# Amihud illiquidity (use SIMPLE return)
def amihud_1m(log_returns_d: pl.DataFrame, dollar_volume: pl.DataFrame, window: int = 21) -> pl.DataFrame:
    x = log_returns_d.join(
        dollar_volume.select(["date","symbol","dollar_volume"]),
        on=["date","symbol"], how="left"
    ).with_columns(
        ((pl.col("log_returns_d").exp() - 1).abs() / pl.col("dollar_volume"))
        .alias("_amihud_inst")
    )
    return x.with_columns(
        pl.col("_amihud_inst").rolling_mean(window, by="symbol").alias("amihud_1m")
    ).drop("_amihud_inst")

# MAX effect (1m)
def maxret_1m(log_returns_d: pl.DataFrame, window: int = 21) -> pl.DataFrame:
    return log_returns_d.with_columns(
        (pl.col("log_returns_d").exp() - 1).rolling_max(window, by="symbol").alias("maxret_1m")
    )

# Realized vol (generic)
def volatility(log_returns_d: pl.DataFrame, window: int) -> pl.DataFrame:
    return log_returns_d.with_columns(
        pl.col("log_returns_d").rolling_std(window, by="symbol").alias(f"vol_{window}")
    )

# Downside vol (3m)
def downside_vol_3m(log_returns_d: pl.DataFrame, window: int = 63) -> pl.DataFrame:
    neg = pl.when(pl.col("log_returns_d") < 0).then(pl.col("log_returns_d")).otherwise(None)
    return log_returns_d.with_columns(
        neg.rolling_std(window, by="symbol").alias("downside_vol_3m")
    )

# Overnight & intraday
def ret_overnight(dfp: pl.DataFrame) -> pl.DataFrame:
    return dfp.with_columns(
        (pl.col("open").log() - pl.col("close").shift(1).over("symbol").log()).alias("ret_overnight")
    )

def ret_intraday(dfp: pl.DataFrame) -> pl.DataFrame:
    return dfp.with_columns(
        (pl.col("close").log() - pl.col("open").log()).alias("ret_intraday")
    )

def overnight_share_3m(ret_overnight: pl.DataFrame, ret_intraday: pl.DataFrame, window: int = 63) -> pl.DataFrame:
    df = ret_overnight.join(ret_intraday.select(["date","symbol","ret_intraday"]), on=["date","symbol"])
    num = pl.col("ret_overnight").abs().rolling_sum(window, by="symbol")
    den = (pl.col("ret_overnight").abs() + pl.col("ret_intraday").abs()).rolling_sum(window, by="symbol")
    return df.with_columns((num/den).alias("overnight_share_3m"))

# Beta (1y) & Idiosyncratic vol (3m) — requires market return in the frame as 'mkt_ret'
def beta_1y(log_returns_d: pl.DataFrame, dfp: pl.DataFrame, window: int = 252) -> pl.DataFrame:
    df = log_returns_d.join(dfp.select(["date","symbol","mkt_ret"]), on=["date","symbol"], how="left")
    x  = pl.col("log_returns_d"); y = pl.col("mkt_ret")
    ex = x.rolling_mean(window, by="symbol"); ey = y.rolling_mean(window, by="symbol")
    exy = (x*y).rolling_mean(window, by="symbol")
    eyy = (y*y).rolling_mean(window, by="symbol")
    cov_xy = exy - ex*ey
    var_y  = eyy - ey*ey
    return df.with_columns((cov_xy / var_y).alias("beta_1y"))

def idio_vol_3m(beta_1y: pl.DataFrame, dfp: pl.DataFrame, log_returns_d: pl.DataFrame, window: int = 63) -> pl.DataFrame:
    df = beta_1y.join(dfp.select(["date","symbol","mkt_ret"]), on=["date","symbol"]) \
                .join(log_returns_d.select(["date","symbol","log_returns_d"]), on=["date","symbol"])
    resid = pl.col("log_returns_d") - pl.col("beta_1y") * pl.col("mkt_ret")
    return df.with_columns(resid.rolling_std(window, by="symbol").alias("idio_vol_3m"))

# Cross-sectional z-score utility (by date)
def cs_z(df: pl.DataFrame, col: str) -> pl.DataFrame:
    mean = pl.col(col).group_by("date").mean()
    std  = pl.col(col).group_by("date").std()
    return df.with_columns(((pl.col(col) - mean) / std).alias(f"{col}_cs_z"))
