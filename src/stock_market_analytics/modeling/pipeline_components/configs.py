FEATURES_FILE = "stock_history_features.parquet"
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
TIMEOUT_MINS = 10
N_TRIALS = 200
STUDY_NAME = "catboost_hyperparameter_optimization_dummy"
# FEATURES = [
#     # "dollar_volume",
#     "amihud_illiq",
#     # "long_kurtosis",
#     # "short_kurtosis",
#     # "long_skewness",
#     # "short_skewness",
#     "kurtosis_diff",
#     "skewness_diff",
#     "long_mean",
#     "short_mean",
#     "mean_diff",
#     "long_diff",
#     "short_diff",
#     "rsi",
#     "long_short_momentum",
#     "risk_adj_momentum",
#     "pct_from_high_long",
#     "pct_from_high_short",
#     "iqr_vol",
#     #"year",
#     "month",
#     "day_of_week",
#     "day_of_year",
# ]

FEATURES = [
    "amihud_illiq",
    "turnover_proxy",
    "max_drawdown",
    "risk_adj_momentum_ewm",
    "cmo",
    "rsi_ewm",
    "sortino_ratio",
    "autocorr_sq_returns",
    "price_vs_cloud_top",
    "tenkan_kijun_cross",
    "cloud_thickness",
    "iqr_vol",
    #"volatility_of_volatility",
    # "year",
    "pct_from_high_long",
    "pct_from_high_short",
    "month",
    "day_of_week",
    "day_of_year",
    ]


TARGET_COVERAGE = 0.8
LOW, MID, HIGH = 0, 2, 4  # indices in Q for 0.10, 0.50, 0.90

TARGET = "y_log_returns"

TIME_SPAN = 70


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
