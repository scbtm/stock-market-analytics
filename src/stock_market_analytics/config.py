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

    tickers_file: str = "top_200_tickers.csv"
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

    horizon: int = Field(default=3, description="Days ahead to predict")
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

    @property
    def as_dict(self) -> dict[str, Any]:
        """Get all feature engineering configuration as a dictionary for easy unpacking."""
        return {
            "horizon": self.horizon,
            "past_horizon": self.past_horizon,
            "short_window": self.short_window,
            "long_window": self.long_window,
            "ichimoku_params": self.ichimoku_params,
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
    fractions: tuple[float, float, float, float] = Field(
        default=(0.6, 0.1, 0.2, 0.1), description="Train/val/cal/test split fractions"
    )

    # Deprecated tuning parameters (no longer used since tuning flow was removed)
    timeout_mins: int = 10
    n_trials: int = 200
    study_name: str = "catboost_hyperparameter_optimization_dummy"

    # Feature groups
    features: list[str] = Field(
        default=[
            "symbol",
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

    @property
    def cb_model_params(self) -> dict[str, Any]:
        """CatBoost model parameters."""
        alpha_str = ",".join([str(q) for q in self.quantiles])
        return {
            "loss_function": f"MultiQuantile:alpha={alpha_str}",
            "num_boost_round": 1_000,
            "learning_rate": 0.02,
            "depth": 5,
            "l2_leaf_reg": 20,
            "grow_policy": "SymmetricTree",
            "border_count": 256,
            "bootstrap_type": "Bayesian",
            "bagging_temperature": 0.5,
            "random_state": 1,
            "verbose": False,
        }

    @property
    def early_stopping_rounds(self) -> int:
        """Early stopping rounds for CatBoost."""
        return int(self.cb_model_params["num_boost_round"] * 0.08)

    @property
    def cb_fit_params(self) -> dict[str, Any]:
        """CatBoost fit parameters."""
        early_stopping = self.early_stopping_rounds
        return {
            "early_stopping_rounds": early_stopping,
            "verbose": int(early_stopping / 2),
            "plot": False,
        }

    @property
    def pca_params(self) -> dict[str, Any]:
        """PCA parameters."""
        return {
            "n_components": 0.8,  # retain 80% of variance
            "svd_solver": "full",
            "random_state": 1,
        }

    @property
    def pca_group_params(self) -> dict[str, dict[str, Any]]:
        """PCA parameters for each feature group."""
        return {
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


class AppConfig(BaseModel):
    """Main application configuration."""

    base_data_path: Path | None = Field(
        default=None, description="Base path for data files"
    )
    wandb_key: str | None = Field(default=None, description="Weights & Biases API key")

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
