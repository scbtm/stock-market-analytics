features_config = {}

horizon = 5  # n days ahead to predict
past_horizon = 7 * 280  # 7 * n weeks of lookback window for training

features_config["horizon"] = horizon
features_config["short_window"] = horizon * 3
features_config["long_window"] = horizon * 5
features_config["past_horizon"] = past_horizon

ichimoku_params = {
    "p1": horizon * 2,
    "p2": horizon * 3,
    "p3": horizon * 4,
    "atr_n": horizon * 2,
    "slope_window": horizon * 2,
    "persist_window": horizon * 3
}

features_config["ichimoku_params"] = ichimoku_params
