import wandb
import os
import joblib
import pandas as pd
from stock_market_analytics.config import config

from stock_market_analytics.inference.inference_functions import collect_inference_data, generate_inference_features

def download_artifacts() -> tuple[str, str]:
    """
    Download model artifacts from Weights & Biases.
    """
    api = wandb.Api(api_key=os.environ.get("WANDB_API_KEY"))
    model_name: str = os.environ.get("MODEL_NAME") or "pipeline:latest"
    model_version: str = model_name.split(':')[1] if ':' in model_name else 'latest'
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
    file_name = model_name.split(':')[0] + '.pkl'
    model = joblib.load(f'{model_dir}/{file_name}')
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
        print(f"📊 Final dataset: {features.shape[0]} rows × {features.shape[1]} columns")

        return features.to_pandas()

    except Exception as e:
        print(f"❌ Inference pipeline failed for {symbol}: {str(e)}")
        raise

def make_predictions(model: object, data: pd.DataFrame) -> pd.DataFrame:
    """
    Make predictions using the loaded model and engineered features.
    """
    print("🤖 Making predictions...")
    predictions = model.predict(data[config.modeling.features])
    low_quantile, high_quantile = predictions[:, 0], predictions[:, 1]
    data['pred_low_quantile'] = low_quantile
    data['pred_high_quantile'] = high_quantile
    return data