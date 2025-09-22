"""
Centralized configuration management for the stock market analytics application.

This module provides type-safe configuration classes using Pydantic for all components
of the system: data collection, feature engineering, and modeling.
"""

import os
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class DataCollectionConfig(BaseModel):
    """Configuration for data collection pipeline."""

    # Use environment variable for files:
    tickers_file: str = os.getenv("TICKERS_FILE", "top_200_tickers.csv")
    metadata_file: str = os.getenv("METADATA_FILE", "metadata.csv")
    stocks_history_file: str = os.getenv(
        "STOCKS_HISTORY_FILE", "stocks_history.parquet"
    )

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

    features_file: str = os.getenv("FEATURES_FILE", "stock_history_features.parquet")
    quantiles: list[float] = Field(default=[0.1, 0.25, 0.5, 0.75, 0.9])
    target: str = "y_log_returns"
    target_coverage: float = 0.8
    time_span: int = Field(
        default=7 * 28, description="Weeks of historical data for validation/testing"
    )
    fractions: tuple[float, float, float, float] = Field(
        default=(0.6, 0.1, 0.2, 0.1), description="Train/val/cal/test split fractions"
    )

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

    base_data_path: str | None = Field(
        default=None, description="Base path for data files (local path or cloud URL)"
    )
    wandb_key: str | None = Field(default=None, description="Weights & Biases API key")

    data_collection: DataCollectionConfig = Field(default_factory=DataCollectionConfig)
    feature_engineering: FeatureEngineeringConfig = Field(
        default_factory=FeatureEngineeringConfig
    )
    modeling: ModelingConfig = Field(default_factory=ModelingConfig)

    model_name: str | None = Field(
        default=None, description="Model name in Weights & Biases"
    )

    tickers_path: str | None = Field(default=None, description="Path for tickers file")
    metadata_path: str | None = Field(
        default=None, description="Path for metadata file"
    )
    stocks_history_path: str | None = Field(
        default=None, description="Path for stocks history file"
    )
    features_path: str | None = Field(
        default=None, description="Path for features file"
    )

    def _join_path(self, base: str, filename: str) -> str:
        """
        URL-aware path joining that works for both local paths and cloud URLs.

        Args:
            base: Base path (local or URL)
            filename: Filename to append

        Returns:
            Properly joined path/URL
        """
        if not base:
            return filename

        # For URLs, always use forward slashes
        if base.startswith(("http://", "https://", "gs://", "s3://", "gcs://")):
            return f"{base.rstrip('/')}/{filename}"

        # For local paths, use os.path.join for OS compatibility
        # but normalize to forward slashes for consistency
        import os.path

        joined = os.path.join(base, filename)
        # Convert to forward slashes for consistency (works on all platforms)
        return joined.replace("\\", "/")

    @field_validator("base_data_path", mode="before")
    @classmethod
    def validate_base_data_path(cls, v: Any) -> str | None:
        """Load base data path from environment if not provided."""
        # In Pydantic v2, None may not be passed if it's the default
        if v is None or v == "":
            env_val = os.environ.get("BASE_DATA_PATH")
            return env_val if env_val else None
        return str(v) if v is not None else None

    @field_validator("wandb_key", mode="before")
    @classmethod
    def validate_wandb_key(cls, v: Any) -> str | None:
        """Load WANDB key from environment if not provided."""
        if v is None or v == "":
            env_val = os.environ.get("WANDB_KEY")
            return env_val if env_val else None
        return str(v) if v is not None else None

    @field_validator("model_name", mode="before")
    @classmethod
    def validate_model_name(cls, v: Any) -> str | None:
        """Load model name from environment if not provided."""
        if v is None or v == "":
            env_val = os.environ.get("MODEL_NAME")
            return env_val if env_val else None
        return str(v) if v is not None else None

    @model_validator(mode="after")
    def validate_paths(self) -> "AppConfig":
        """Handle environment variables and construct file paths."""
        # Load from environment if fields are None
        if self.base_data_path is None:
            env_base = os.environ.get("BASE_DATA_PATH")
            if env_base:
                self.base_data_path = env_base

        if self.wandb_key is None:
            env_wandb = os.environ.get("WANDB_KEY")
            if env_wandb:
                self.wandb_key = env_wandb

        if self.model_name is None:
            env_model = os.environ.get("MODEL_NAME")
            if env_model:
                self.model_name = env_model

        # Construct file paths if base_data_path is available
        if self.base_data_path:
            # Set tickers_path if not provided
            if not self.tickers_path:
                self.tickers_path = self._join_path(
                    self.base_data_path, self.data_collection.tickers_file
                )

            # Set metadata_path if not provided
            if not self.metadata_path:
                self.metadata_path = self._join_path(
                    self.base_data_path, self.data_collection.metadata_file
                )

            # Set stocks_history_path if not provided
            if not self.stocks_history_path:
                self.stocks_history_path = self._join_path(
                    self.base_data_path, self.data_collection.stocks_history_file
                )

            # Set features_path if not provided
            if not self.features_path:
                self.features_path = self._join_path(
                    self.base_data_path, self.modeling.features_file
                )

        return self


# Global configuration instance
config = AppConfig()
