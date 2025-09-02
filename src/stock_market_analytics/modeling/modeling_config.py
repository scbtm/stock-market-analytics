FEATURES_FILE = "stock_history_features.parquet"
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
TIMEOUT_MINS = 10
N_TRIALS = 200
STUDY_NAME = "catboost_hyperparameter_optimization_dummy"
FEATURES = [
    'dollar_volume',
    'long_kurtosis',
    'short_kurtosis',
    'long_skewness',
    'short_skewness',
    'long_mean',
    'short_mean',
    'mean_diff',
    'long_diff',
    'short_diff',
    'long_short_momentum',
    'pct_from_high_long',
    'pct_from_high_short',
    'year',
    'month',
    'day_of_week',
    'day_of_year'
]

modeling_config = dict({
    "FEATURES_FILE": FEATURES_FILE,
    "QUANTILES": QUANTILES,
    "TIMEOUT_MINS": TIMEOUT_MINS,
    "N_TRIALS": N_TRIALS,
    "STUDY_NAME": STUDY_NAME,
    "FEATURES": FEATURES
})