FEATURES_FILE = "stock_history_features.parquet"
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
TIMEOUT_MINS = 10
N_TRIALS = 200
STUDY_NAME = "catboost_hyperparameter_optimization_dummy"

FEATURES = [
    'month',
    'day_of_week',
    'day_of_year',
    'log_returns_d',
    'log_returns_ratio',
    'rsi_ewm',
    'sortino_ratio',
    'sharpe_ratio_proxy',
    'amihud_illiq',
    'turnover_proxy',
    'kurtosis_ratio',
    'skew_ratio',
    'zscore_ratio',
    'autocorr_r2',
    'iqr_vol',
    'long_short_momentum',
    'cmo',
    'risk_adj_momentum',
    'vol_ratio',
    'vol_of_vol_ewm',
    'vol_expansion',
    'tenkan_slope',
    'kijun_slope',
    'span_a_slope',
    'span_b_slope',
    'cloud_top_slope',
    'cloud_bot_slope',
    'above_cloud',
    'between_cloud',
    'below_cloud',
    'above_cloud_persist',
    'below_cloud_persist',
    'tenkan_cross_up',
    'tenkan_cross_dn',
    'price_break_up',
    'price_break_dn',
    'twist_event',
    'twist_recent',
    'bull_strength',
    'bear_strength',
    'tenkan_kijun_spread_atr',
    'price_above_cloud_atr',
    'price_below_cloud_atr',
    'cloud_thickness_atr',
    'price_vs_lead_top_atr',
    'price_vs_lead_bot_atr',
    'atr'
    ]

TIME_COLUMNS = [
    'month',
    'day_of_week',
    'day_of_year',
]

FINANCIAL_FEATURES = [
    'log_returns_d',
    'log_returns_ratio',
    'rsi_ewm',
    'sortino_ratio',
    'sharpe_ratio_proxy',
]

LIQUIDITY_FEATURES = [
    'amihud_illiq',
    'turnover_proxy',
]

STATISTICAL_FEATURES = [
    'kurtosis_ratio',
    'skew_ratio',
    'zscore_ratio',
    'autocorr_r2',
    'iqr_vol',
]

MOMENTUM_INDICATORS_FEATURES = [
    'long_short_momentum',
    'cmo',
    'risk_adj_momentum',
]

VOLATILITY_MEASURES_FEATURES = [
    'vol_ratio',
    'vol_of_vol_ewm',
    'vol_expansion',
]

ICHIMOKU_SLOPE_FEATURES = [
    'tenkan_slope',
    'kijun_slope',
    'span_a_slope',
    'span_b_slope',
    'cloud_top_slope',
    'cloud_bot_slope',
]

ICHIMOKU_POSITIONAL_FEATURES = [
    'above_cloud',
    'between_cloud',
    'below_cloud',
    'above_cloud_persist',
    'below_cloud_persist',
]

ICHIMOKU_CROSSOVER_FEATURES = [
    'tenkan_cross_up',
    'tenkan_cross_dn',
    'price_break_up',
    'price_break_dn',
    'twist_event',
    'twist_recent',
]

ICHIMOKU_STRENGTH_FEATURES = [
    'bull_strength',
    'bear_strength',
]

ICHIMOKU_ATR_FEATURES = [
    'tenkan_kijun_spread_atr',
    'price_above_cloud_atr',
    'price_below_cloud_atr',
    'cloud_thickness_atr',
    'price_vs_lead_top_atr',
    'price_vs_lead_bot_atr',
]

TARGET_COVERAGE = 0.8
LOW, MID, HIGH = 0, 2, 4  # indices in Q for 0.10, 0.50, 0.90

TARGET = "y_log_returns"

TIME_SPAN = 7 * 28 # 7 * n = weeks of historical data to use for validation and testing


modeling_config = {
    "FEATURES_FILE": FEATURES_FILE,
    "QUANTILES": QUANTILES,
    "TIMEOUT_MINS": TIMEOUT_MINS,
    "N_TRIALS": N_TRIALS,
    "STUDY_NAME": STUDY_NAME,
    "FEATURES": FEATURES,
    "TARGET_COVERAGE": TARGET_COVERAGE,
    "LOW": LOW,
    "MID": MID,
    "HIGH": HIGH,
    "TARGET": TARGET,
    "TIME_SPAN": TIME_SPAN,
    "FEATURE_GROUPS": {
        "TIME_COLUMNS": TIME_COLUMNS,
        "FINANCIAL_FEATURES": FINANCIAL_FEATURES,
        "LIQUIDITY_FEATURES": LIQUIDITY_FEATURES,
        "STATISTICAL_FEATURES": STATISTICAL_FEATURES,
        "MOMENTUM_INDICATORS_FEATURES": MOMENTUM_INDICATORS_FEATURES,
        "VOLATILITY_MEASURES_FEATURES": VOLATILITY_MEASURES_FEATURES,
        "ICHIMOKU_SLOPE_FEATURES": ICHIMOKU_SLOPE_FEATURES,
        "ICHIMOKU_POSITIONAL_FEATURES": ICHIMOKU_POSITIONAL_FEATURES,
        "ICHIMOKU_CROSSOVER_FEATURES": ICHIMOKU_CROSSOVER_FEATURES,
        "ICHIMOKU_STRENGTH_FEATURES": ICHIMOKU_STRENGTH_FEATURES,
        "ICHIMOKU_ATR_FEATURES": ICHIMOKU_ATR_FEATURES,
    }
}
