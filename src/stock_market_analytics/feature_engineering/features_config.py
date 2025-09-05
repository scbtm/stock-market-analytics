features_config = {}

horizon = 7  # n days ahead to predict
past_horizon = 7 * 400  # 7 * n weeks of lookback window for training

features_config["short_window"] = horizon * 2
features_config["long_window"] = horizon * 4
features_config["horizon"] = horizon
features_config["past_horizon"] = past_horizon

features_config["ichimoku_p1"] = horizon
features_config["ichimoku_p2"] = horizon * 2
features_config["ichimoku_p3"] = horizon * 4
