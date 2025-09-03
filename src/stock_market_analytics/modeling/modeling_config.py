FEATURES_FILE = "stock_history_features.parquet"
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
TIMEOUT_MINS = 10
N_TRIALS = 200
STUDY_NAME = "catboost_hyperparameter_optimization_dummy"
FEATURES = [
    "dollar_volume",
    "amihud_illiq",
    #"long_kurtosis",
    #"short_kurtosis",
    #"long_skewness",
    #"short_skewness",
    "long_mean",
    "short_mean",
    "mean_diff",
    "long_diff",
    "short_diff",
    "long_short_momentum",
    "risk_adj_momentum",
    "pct_from_high_long",
    "pct_from_high_short",
    #"year",
    "month",
    "day_of_week",
    "day_of_year",
]

# Starter params. Some of the params might be modified during training flow.
alpha_str = ",".join([str(q) for q in QUANTILES])
        
PARAMS = {
    "loss_function": f"MultiQuantile:alpha={alpha_str}",
    "num_boost_round": 1_000,
    "learning_rate": 0.01,
    "depth": 4,
    "l2_leaf_reg": 10,
    "grow_policy": "SymmetricTree",
    "border_count": 32,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 0.5,
    "min_data_in_leaf": 200,
    "colsample_bylevel": 0.7,
    "random_state": 1,
    "verbose": False,
}

TARGET_COVERAGE = 0.8
LOW, MID, HIGH = 0, 2, 4  # indices in Q for 0.10, 0.50, 0.90

TARGET = "y_log_returns"


modeling_config = {
    "FEATURES_FILE": FEATURES_FILE,
    "QUANTILES": QUANTILES,
    "TIMEOUT_MINS": TIMEOUT_MINS,
    "N_TRIALS": N_TRIALS,
    "STUDY_NAME": STUDY_NAME,
    "FEATURES": FEATURES,
    "PARAMS": PARAMS,
    "TARGET_COVERAGE": TARGET_COVERAGE,
    "LOW": LOW,
    "MID": MID,
    "HIGH": HIGH,
    "TARGET": TARGET,
}
