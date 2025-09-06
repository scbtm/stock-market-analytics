# Configuration for CatBoost model (quantile regression)

from stock_market_analytics.modeling.pipeline_components.configs import modeling_config

QUANTILES = modeling_config["QUANTILES"]

alpha_str = ",".join([str(q) for q in QUANTILES])


cb_model_params = {
    "loss_function": f"MultiQuantile:alpha={alpha_str}",
    "num_boost_round": 1_000,
    "learning_rate": 0.07,
    "depth": 5,
    "l2_leaf_reg": 10,
    "grow_policy": "SymmetricTree",
    "border_count": 128,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 0.5,
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
    "n_components": 0.8, # retain 97% of variance
    "svd_solver": "full",
    "random_state": 1,
}

pca_group_params = {
    # "TIME_COLUMNS": {
    #     "n_components": 1,  
    #     "random_state": 1,
    # },
    "FINANCIAL_FEATURES": {
        "n_components": 3,
        "random_state": 1,
    },
    "LIQUIDITY_FEATURES": {
        "n_components": 1,  
        "random_state": 1,
    },
    "STATISTICAL_FEATURES": {
        "n_components": 4,  
        "random_state": 1,
    },
    "MOMENTUM_INDICATORS_FEATURES": {
        "n_components": 1,  
        "random_state": 1,
    },
    "VOLATILITY_MEASURES_FEATURES": {
        "n_components": 1,  
        "random_state": 1,
    },
    "ICHIMOKU_SLOPE_FEATURES": {
        "n_components": 4,  
        "random_state": 1,
    },
    "ICHIMOKU_POSITIONAL_FEATURES": {
        "n_components": 3,  
        "random_state": 1,
    },
    "ICHIMOKU_CROSSOVER_FEATURES": {
        "n_components": 3,  
        "random_state": 1,
    },
    "ICHIMOKU_STRENGTH_FEATURES": {
        "n_components": 1,  
        "random_state": 1,
    },
    "ICHIMOKU_ATR_FEATURES": {
        "n_components": 2,  
        "random_state": 1,
    },

}
