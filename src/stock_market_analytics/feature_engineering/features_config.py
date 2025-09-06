features_config = {}

horizon = 5  # n days ahead to predict
past_horizon = 7 * 210  # 7 * n weeks of lookback window for training

features_config["horizon"] = horizon
features_config["short_window"] = horizon * 3
features_config["long_window"] = horizon * 5
features_config["past_horizon"] = past_horizon

# features_config["ichimoku_p1"] = horizon
# features_config["ichimoku_p2"] = horizon * 2
# features_config["ichimoku_p3"] = horizon * 4

ichimoku_params = {
    "p1": horizon * 2,
    "p2": horizon * 3,
    "p3": horizon * 4,
    "atr_n": horizon * 2,
    "slope_window": horizon * 2,
    "persist_window": horizon * 3
}

features_config["ichimoku_params"] = ichimoku_params
