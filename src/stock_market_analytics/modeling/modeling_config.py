FEATURES_FILE = "stock_history_features.parquet"
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
TIMEOUT_MINS = 10
N_TRIALS = 200
STUDY_NAME = "catboost_hyperparameter_optimization_dummy"
FEATURES = [
    "dollar_volume",
    "long_kurtosis",
    "short_kurtosis",
    "long_skewness",
    "short_skewness",
    "long_mean",
    "short_mean",
    "mean_diff",
    "long_diff",
    "short_diff",
    "long_short_momentum",
    "pct_from_high_long",
    "pct_from_high_short",
    "year",
    "month",
    "day_of_week",
    "day_of_year",
]

# Starter params. Some of the params might be modified during training flow.
alpha_str = ",".join([str(q) for q in QUANTILES])
        
PARAMS = {
    "loss_function": f"MultiQuantile:alpha={alpha_str}",
    "num_boost_round": 1_000,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3,
    "grow_policy": "SymmetricTree",
    "border_count": 32,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 1.0,
    "min_data_in_leaf": 100,
    "colsample_bylevel": 0.7,
    "random_state": 1,
    "verbose": False,
}

modeling_config = {
    "FEATURES_FILE": FEATURES_FILE,
    "QUANTILES": QUANTILES,
    "TIMEOUT_MINS": TIMEOUT_MINS,
    "N_TRIALS": N_TRIALS,
    "STUDY_NAME": STUDY_NAME,
    "FEATURES": FEATURES,
    "PARAMS": PARAMS,
}
