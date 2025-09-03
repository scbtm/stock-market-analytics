"""
Model configuration classes.

This module defines configuration classes for CatBoost model parameters,
replacing the hardcoded configuration in modeling_config.py.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator


class CatBoostConfig(BaseModel):
    """Configuration for CatBoost multi-quantile model."""
    
    # Quantile settings
    quantiles: List[float] = Field(
        default=[0.1, 0.25, 0.5, 0.75, 0.9],
        description="Quantiles to predict"
    )
    
    # Core model parameters
    num_boost_round: int = Field(default=1000, description="Maximum number of boosting rounds")
    learning_rate: float = Field(default=0.01, description="Learning rate")
    depth: int = Field(default=3, description="Tree depth")
    l2_leaf_reg: int = Field(default=10, description="L2 regularization")
    
    # Training parameters  
    grow_policy: str = Field(default="SymmetricTree", description="Tree growing policy")
    border_count: int = Field(default=8, description="Number of splits for numerical features")
    bootstrap_type: str = Field(default="Bayesian", description="Bootstrap type")
    bagging_temperature: float = Field(default=0.8, description="Bagging temperature")
    
    # Reproducibility
    random_state: int = Field(default=42, description="Random seed")
    verbose: bool = Field(default=False, description="Training verbosity")
    use_best_model: bool = Field(default=True, description="Use best model from validation")
    
    @validator("quantiles")
    def validate_quantiles(cls, v):
        """Validate quantiles are in (0, 1) and sorted."""
        if not all(0 < q < 1 for q in v):
            raise ValueError("All quantiles must be between 0 and 1")
        if v != sorted(v):
            raise ValueError("Quantiles must be sorted")
        return v
        
    def to_catboost_params(self) -> Dict[str, Any]:
        """Convert to CatBoost parameters dictionary."""
        # Build loss function string
        alpha_str = ",".join([str(q) for q in self.quantiles])
        
        params = self.dict(exclude={"quantiles"})
        params["loss_function"] = f"MultiQuantile:alpha={alpha_str}"
        
        return params


class HyperparameterTuningConfig(BaseModel):
    """Configuration for hyperparameter tuning."""
    
    timeout_mins: int = Field(default=10, description="Tuning timeout in minutes")
    n_trials: int = Field(default=200, description="Number of trials")
    study_name: str = Field(default="catboost_tuning", description="Optuna study name")
    
    # Search space bounds
    learning_rate_range: tuple[float, float] = Field(
        default=(0.05, 0.25), 
        description="Learning rate search range"
    )
    depth_range: tuple[int, int] = Field(
        default=(4, 7), 
        description="Tree depth search range"
    )
    l2_leaf_reg_range: tuple[int, int] = Field(
        default=(1, 15), 
        description="L2 regularization search range"
    )
    grow_policies: List[str] = Field(
        default=["SymmetricTree", "Depthwise"],
        description="Grow policies to try"
    )
    bootstrap_types: List[str] = Field(
        default=["Bayesian", "Bernoulli"],
        description="Bootstrap types to try"
    )
    bagging_temperature_range: tuple[float, float] = Field(
        default=(0.5, 8.0),
        description="Bagging temperature range for Bayesian bootstrap"
    )
    subsample_range: tuple[float, float] = Field(
        default=(0.6, 0.95),
        description="Subsample range for Bernoulli bootstrap"
    )
    colsample_range: tuple[float, float] = Field(
        default=(0.6, 1.0),
        description="Column sampling range"
    )
    border_count_range: tuple[int, int] = Field(
        default=(128, 254),
        description="Border count range"
    )
    min_data_in_leaf: int = Field(
        default=100,
        description="Minimum data in leaf"
    )


class ModelConfig(BaseModel):
    """Complete model configuration."""
    
    catboost: CatBoostConfig = Field(default_factory=CatBoostConfig)
    tuning: HyperparameterTuningConfig = Field(default_factory=HyperparameterTuningConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        """Create config from dictionary."""
        return cls(**config_dict)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.dict()