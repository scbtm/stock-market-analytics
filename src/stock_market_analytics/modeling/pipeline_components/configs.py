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
    # 'rsi_ewm',
    'sortino_ratio',
    # 'sharpe_ratio_proxy',
    # 'amihud_illiq',
    'turnover_proxy',
    'kurtosis_ratio',
    'skew_ratio',
    'zscore_ratio',
    # 'autocorr_r2',
    'iqr_vol',
    'cmo',
    # 'risk_adj_momentum',
    # 'vol_of_vol_ewm',
    'cloud_top_slope',
    # 'cloud_bot_slope',
    # 'above_cloud_persist',
    'tenkan_cross_up',
    'price_break_up',
    'twist_recent',
    'bull_strength',
    'tenkan_kijun_spread_atr',
    # 'cloud_thickness_atr',
    'price_vs_lead_top_atr',
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
}
