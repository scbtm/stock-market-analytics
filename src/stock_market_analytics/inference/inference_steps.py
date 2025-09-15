import os

import joblib
import pandas as pd

import wandb
from stock_market_analytics.config import config
from stock_market_analytics.inference.inference_functions import (
    collect_inference_data,
    generate_inference_features,
)


def download_artifacts() -> tuple[str, str]:
    """
    Download model artifacts from Weights & Biases.
    """
    api = wandb.Api(api_key=config.wandb_key)
    model_name: str = os.environ.get("MODEL_NAME") or "pipeline:latest"
    model_version: str = model_name.split(":")[1] if ":" in model_name else "latest"
    print(f"Downloading model '{model_name}' version '{model_version}' from W&B...")
    # Reference the artifact by entity/project/name:version or :latest
    model = api.artifact(f"san-cbtm/stock-market-analytics/{model_name}", type="model")

    # Download to a specific folder (defaults to a temp dir if omitted)
    model_dir = model.download()

    return model_dir, model_name


def load_model(model_dir: str, model_name: str) -> object:
    """
    Load the model from the downloaded artifact directory.
    """
    file_name = model_name.split(":")[0] + ".pkl"
    model = joblib.load(f"{model_dir}/{file_name}")
    return model


def download_and_load_model() -> object:
    """
    Combined function to download and load the model.
    """
    model_dir, model_name = download_artifacts()
    model = load_model(model_dir, model_name)
    print(f"Model '{model_name}' loaded successfully.")
    return model


def get_inference_data(symbol: str) -> pd.DataFrame:
    """
    Complete pipeline to get inference-ready features for a single ticker.

    This is the main function combining data collection and feature engineering
    in a simple, clean interface that leverages existing infrastructure.

    Args:
        symbol: Stock ticker symbol (e.g., "AAPL", "MSFT")

    Returns:
        DataFrame with all engineered features, including latest data points

    Example:
        >>> features = get_inference_data("AAPL")
        >>> latest = features.sort("date").tail(1)  # Get most recent features
    """
    print(f"🚀 Starting inference pipeline for {symbol.upper()}...")

    try:
        # Step 1: Collect data using existing infrastructure
        raw_data = collect_inference_data(symbol)

        # Step 2: Generate features using existing pipeline
        features = generate_inference_features(raw_data)

        print(f"🏁 Inference pipeline completed for {symbol.upper()}")
        print(
            f"📊 Final dataset: {features.shape[0]} rows × {features.shape[1]} columns"
        )

        return features.to_pandas()

    except Exception as e:
        print(f"❌ Inference pipeline failed for {symbol}: {str(e)}")
        raise


def make_prediction_intervals(model: object, data: pd.DataFrame) -> pd.DataFrame:
    """
    Make prediction intervals (higher and lower calibrated quantiles) using the loaded model and engineered features.
    """
    print("🤖 Making predictions...")
    predictions = model.predict(data[config.modeling.features])
    low_quantile, high_quantile = predictions[:, 0], predictions[:, 1]
    data["pred_low_quantile"] = low_quantile
    data["pred_high_quantile"] = high_quantile
    return data


def predict_quantiles(model: object, data: pd.DataFrame) -> pd.DataFrame:
    """
    Full inference workflow to get all quantile predictions.

    Args:
        model: Loaded model object
        data: DataFrame with features for prediction

    Returns:
        DataFrame with prediction quantiles added as new columns
    """
    quantile_cols = []
    for q in config.modeling.quantiles:
        quantile_col = f"pred_Q_{q * 100}"
        quantile_cols.append(quantile_col)
        data[quantile_col] = None  # Initialize columns

    # Step 1: initial transforms
    transforms = model.named_steps["transformations"]
    x_inf_transformed = transforms.transform(data[config.modeling.features])

    regressor = model.named_steps["model"]
    preds = regressor.predict(x_inf_transformed, return_full_quantiles=True)
    for i, q in enumerate(config.modeling.quantiles):
        quantile_col = f"pred_Q_{int(q * 100)}"
        print(quantile_col)
        data[quantile_col] = preds[:, i]

    print("✅ Predictions completed")
    return data
