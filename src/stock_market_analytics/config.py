"""
Centralized configuration management for the stock market analytics application.

This module provides type-safe configuration classes using Pydantic for all components
of the system: data collection, feature engineering, and modeling.
"""

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, validator


class DataCollectionConfig(BaseModel):
    """Configuration for data collection pipeline."""

    tickers_file: str = "tickers.csv"
    metadata_file: str = "metadata.csv"
    stocks_history_file: str = "stocks_history.parquet"

    required_ticker_columns: list[str] = Field(
        default=["Symbol", "Name", "Country", "IPO Year", "Sector", "Industry"]
    )

    required_metadata_columns: list[str] = Field(
        default=["symbol", "last_ingestion", "max_date_recorded", "status"]
    )

    @property
    def ticker_column_mapping(self) -> dict[str, str]:
        """Generate column mapping for tickers file."""
        return {
            col: col.replace(" ", "_").lower() for col in self.required_ticker_columns
        }


class FeatureEngineeringConfig(BaseModel):
    """Configuration for feature engineering pipeline."""

    horizon: int = Field(default=5, description="Days ahead to predict")
    past_horizon: int = Field(
        default=7 * 280, description="Weeks of lookback for training"
    )

    @property
    def short_window(self) -> int:
        """Short rolling window size."""
        return self.horizon * 3

    @property
    def long_window(self) -> int:
        """Long rolling window size."""
        return self.horizon * 5

    @property
    def ichimoku_params(self) -> dict[str, int]:
        """Ichimoku cloud parameters."""
        return {
            "p1": self.horizon * 2,
            "p2": self.horizon * 3,
            "p3": self.horizon * 4,
            "atr_n": self.horizon * 2,
            "slope_window": self.horizon * 2,
            "persist_window": self.horizon * 3,
        }


class ModelingConfig(BaseModel):
    """Configuration for modeling pipeline."""

    features_file: str = "stock_history_features.parquet"
    quantiles: list[float] = Field(default=[0.1, 0.25, 0.5, 0.75, 0.9])
    target: str = "y_log_returns"
    target_coverage: float = 0.8
    time_span: int = Field(
        default=7 * 28, description="Weeks of historical data for validation/testing"
    )

    # Feature groups
    features: list[str] = Field(
        default=[
            "month",
            "day_of_week",
            "day_of_year",
            "log_returns_d",
            "log_returns_ratio",
            "rsi_ewm",
            "sortino_ratio",
            "sharpe_ratio_proxy",
            "amihud_illiq",
            "turnover_proxy",
            "kurtosis_ratio",
            "skew_ratio",
            "zscore_ratio",
            "autocorr_r2",
            "iqr_vol",
            "long_short_momentum",
            "cmo",
            "risk_adj_momentum",
            "vol_ratio",
            "vol_of_vol_ewm",
            "vol_expansion",
            "tenkan_slope",
            "kijun_slope",
            "span_a_slope",
            "span_b_slope",
            "cloud_top_slope",
            "cloud_bot_slope",
            "above_cloud",
            "between_cloud",
            "below_cloud",
            "above_cloud_persist",
            "below_cloud_persist",
            "tenkan_cross_up",
            "tenkan_cross_dn",
            "price_break_up",
            "price_break_dn",
            "twist_event",
            "twist_recent",
            "bull_strength",
            "bear_strength",
            "tenkan_kijun_spread_atr",
            "price_above_cloud_atr",
            "price_below_cloud_atr",
            "cloud_thickness_atr",
            "price_vs_lead_top_atr",
            "price_vs_lead_bot_atr",
            "atr",
        ]
    )

    @property
    def quantile_indices(self) -> dict[str, int]:
        """Get indices for key quantiles."""
        return {
            "LOW": 0,  # 0.1 quantile
            "MID": 2,  # 0.5 quantile
            "HIGH": 4,  # 0.9 quantile
        }

    @property
    def feature_groups(self) -> dict[str, list[str]]:
        """Organized feature groups for pipeline processing."""
        return {
            "FINANCIAL_FEATURES": [
                "log_returns_d",
                "log_returns_ratio",
                "rsi_ewm",
                "sortino_ratio",
                "sharpe_ratio_proxy",
            ],
            "LIQUIDITY_FEATURES": ["amihud_illiq", "turnover_proxy"],
            "STATISTICAL_FEATURES": [
                "kurtosis_ratio",
                "skew_ratio",
                "zscore_ratio",
                "autocorr_r2",
                "iqr_vol",
            ],
            "MOMENTUM_INDICATORS_FEATURES": [
                "long_short_momentum",
                "cmo",
                "risk_adj_momentum",
            ],
            "VOLATILITY_MEASURES_FEATURES": [
                "vol_ratio",
                "vol_of_vol_ewm",
                "vol_expansion",
            ],
            "ICHIMOKU_SLOPE_FEATURES": [
                "tenkan_slope",
                "kijun_slope",
                "span_a_slope",
                "span_b_slope",
                "cloud_top_slope",
                "cloud_bot_slope",
            ],
            "ICHIMOKU_POSITIONAL_FEATURES": [
                "above_cloud",
                "between_cloud",
                "below_cloud",
                "above_cloud_persist",
                "below_cloud_persist",
            ],
            "ICHIMOKU_CROSSOVER_FEATURES": [
                "tenkan_cross_up",
                "tenkan_cross_dn",
                "price_break_up",
                "price_break_dn",
                "twist_event",
                "twist_recent",
            ],
            "ICHIMOKU_STRENGTH_FEATURES": ["bull_strength", "bear_strength"],
            "ICHIMOKU_ATR_FEATURES": [
                "tenkan_kijun_spread_atr",
                "price_above_cloud_atr",
                "price_below_cloud_atr",
                "cloud_thickness_atr",
                "price_vs_lead_top_atr",
                "price_vs_lead_bot_atr",
            ],
        }


class AppConfig(BaseModel):
    """Main application configuration."""

    base_data_path: Path | None = Field(
        default=None, description="Base path for data files"
    )
    wandb_key: str | None = Field(
        default=None, description="Weights & Biases API key"
    )

    data_collection: DataCollectionConfig = Field(default_factory=DataCollectionConfig)
    feature_engineering: FeatureEngineeringConfig = Field(
        default_factory=FeatureEngineeringConfig
    )
    modeling: ModelingConfig = Field(default_factory=ModelingConfig)

    @validator("base_data_path", pre=True, always=True)
    def validate_base_data_path(cls, v: Any) -> Any:
        """Load base data path from environment if not provided."""
        if v is None:
            env_path = os.environ.get("BASE_DATA_PATH")
            if env_path:
                return Path(env_path)
        return v

    @validator("wandb_key", pre=True, always=True)
    def validate_wandb_key(cls, v: Any) -> Any:
        """Load WANDB key from environment if not provided."""
        if v is None:
            return os.environ.get("WANDB_KEY")
        return v


# Global configuration instance
config = AppConfig()


# Backward compatibility - maintain existing interfaces
def get_data_config() -> dict[str, Any]:
    """Get data collection configuration as dictionary for backward compatibility."""
    data_cfg = config.data_collection
    return {
        "TICKERS_FILE": data_cfg.tickers_file,
        "METADATA_FILE": data_cfg.metadata_file,
        "STOCKS_HISTORY_FILE": data_cfg.stocks_history_file,
        "REQUIRED_TICKER_COLUMNS": data_cfg.required_ticker_columns,
        "TICKER_COLUMN_MAPPING": data_cfg.ticker_column_mapping,
        "REQUIRED_METADATA_COLUMNS": data_cfg.required_metadata_columns,
    }


def get_features_config() -> dict[str, Any]:
    """Get feature engineering configuration as dictionary for backward compatibility."""
    feat_cfg = config.feature_engineering
    return {
        "horizon": feat_cfg.horizon,
        "short_window": feat_cfg.short_window,
        "long_window": feat_cfg.long_window,
        "past_horizon": feat_cfg.past_horizon,
        "ichimoku_params": feat_cfg.ichimoku_params,
    }


def get_modeling_config() -> dict[str, Any]:
    """Get modeling configuration as dictionary for backward compatibility."""
    model_cfg = config.modeling
    return {
        "FEATURES_FILE": model_cfg.features_file,
        "QUANTILES": model_cfg.quantiles,
        "FEATURES": model_cfg.features,
        "TARGET_COVERAGE": model_cfg.target_coverage,
        "TARGET": model_cfg.target,
        "TIME_SPAN": model_cfg.time_span,
        "FEATURE_GROUPS": model_cfg.feature_groups,
        **model_cfg.quantile_indices,
    }
