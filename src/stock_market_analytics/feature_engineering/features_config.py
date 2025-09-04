features_config = {}

features_config["long_window"] = 28
features_config["short_window"] = 14
features_config["horizon"] = 14
features_config["past_horizon"] = 364 * 3  # 364 * n years of lookback window for training
