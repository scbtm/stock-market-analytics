# Configuration for CatBoost model (quantile regression)

from stock_market_analytics.modeling.pipeline_components.configs import modeling_config

QUANTILES = modeling_config["QUANTILES"]

alpha_str = ",".join([str(q) for q in QUANTILES])


cb_model_params = {
    "loss_function": f"MultiQuantile:alpha={alpha_str}",
    "num_boost_round": 1_000,
    "learning_rate": 0.01,
    "depth": 3,
    "l2_leaf_reg": 10,
    "grow_policy": "SymmetricTree",
    "border_count": 8,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 0.8,
    # "min_data_in_leaf": 200,
    # "colsample_bylevel": 0.7,
    "random_state": 1,
    "verbose": False,
}

early_stopping_rounds = int(cb_model_params["num_boost_round"] * 0.08)


cb_fit_params = {
    "early_stopping_rounds": early_stopping_rounds,
    "verbose": int(early_stopping_rounds / 2),
    "plot": False,
}

pca_params = {
    "n_components": 0.95,
    "svd_solver": "full",
    "random_state": 1,
}
