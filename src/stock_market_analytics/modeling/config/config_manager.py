"""
Configuration manager for the modeling pipeline.

This module provides utilities for loading, validating, and managing
configurations across the modeling components.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .model_config import ModelConfig
from .data_config import DataConfig
from .evaluation_config import EvaluationConfig


class ConfigManager:
    """
    Centralized configuration management for the modeling pipeline.
    
    This class handles loading configurations from various sources,
    validation, and providing unified access to all configuration settings.
    """
    
    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        data_config: Optional[DataConfig] = None,
        evaluation_config: Optional[EvaluationConfig] = None
    ):
        """
        Initialize configuration manager.
        
        Args:
            model_config: Model configuration (uses defaults if None)
            data_config: Data configuration (uses defaults if None) 
            evaluation_config: Evaluation configuration (uses defaults if None)
        """
        self.model = model_config or ModelConfig()
        self.data = data_config or DataConfig()
        self.evaluation = evaluation_config or EvaluationConfig()
        
        # Update evaluation quantile indices based on model quantiles
        self.evaluation.update_quantile_indices(self.model.catboost.quantiles)
        
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "ConfigManager":
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            ConfigManager instance with loaded configuration
        """
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
            
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        return cls.from_dict(config_dict)
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ConfigManager":
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            ConfigManager instance
        """
        model_config = None
        if "model" in config_dict:
            model_config = ModelConfig.from_dict(config_dict["model"])
            
        data_config = None  
        if "data" in config_dict:
            data_config = DataConfig(**config_dict["data"])
            
        evaluation_config = None
        if "evaluation" in config_dict:
            evaluation_config = EvaluationConfig(**config_dict["evaluation"])
            
        return cls(model_config, data_config, evaluation_config)
        
    @classmethod
    def from_environment(cls, env_prefix: str = "MODELING_") -> "ConfigManager":
        """
        Load configuration from environment variables.
        
        Args:
            env_prefix: Prefix for environment variables
            
        Returns:
            ConfigManager instance with environment-based configuration
        """
        config_dict = {}
        
        # Parse environment variables
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                # Convert MODELING_MODEL_LEARNING_RATE -> model.learning_rate
                config_key = key[len(env_prefix):].lower()
                parts = config_key.split("_", 1)
                
                if len(parts) == 2:
                    section, param = parts
                    if section not in config_dict:
                        config_dict[section] = {}
                    
                    # Try to convert to appropriate type
                    try:
                        if value.lower() in ("true", "false"):
                            config_dict[section][param] = value.lower() == "true"
                        elif "." in value:
                            config_dict[section][param] = float(value)
                        else:
                            config_dict[section][param] = int(value)
                    except ValueError:
                        config_dict[section][param] = value
                        
        return cls.from_dict(config_dict) if config_dict else cls()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model": self.model.to_dict(),
            "data": self.data.dict(),
            "evaluation": self.evaluation.dict()
        }
        
    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            yaml_path: Path where to save the configuration
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
            
    def validate(self) -> None:
        """Validate the complete configuration."""
        # Validate that quantile indices are consistent
        n_quantiles = len(self.model.catboost.quantiles)
        
        if self.evaluation.high_idx >= n_quantiles:
            raise ValueError(f"high_idx ({self.evaluation.high_idx}) >= n_quantiles ({n_quantiles})")
            
        if self.evaluation.mid_idx >= n_quantiles:
            raise ValueError(f"mid_idx ({self.evaluation.mid_idx}) >= n_quantiles ({n_quantiles})")
            
        # Validate coverage interval matches quantiles
        interval_low, interval_high = self.evaluation.coverage_interval
        quantiles = self.model.catboost.quantiles
        
        if interval_low not in quantiles or interval_high not in quantiles:
            raise ValueError(
                f"Coverage interval {self.evaluation.coverage_interval} "
                f"not found in model quantiles {quantiles}"
            )
            
    def get_legacy_config_dict(self) -> Dict[str, Any]:
        """
        Get configuration in the legacy format for backward compatibility.
        
        This returns a dictionary matching the old modeling_config format
        to ease migration of existing code.
        """
        return {
            "FEATURES_FILE": self.data.features_file,
            "QUANTILES": self.model.catboost.quantiles,
            "TIMEOUT_MINS": self.model.tuning.timeout_mins,
            "N_TRIALS": self.model.tuning.n_trials,
            "STUDY_NAME": self.model.tuning.study_name,
            "FEATURES": self.data.features,
            "PARAMS": self.model.catboost.to_catboost_params(),
            "TARGET_COVERAGE": self.evaluation.target_coverage,
            "LOW": self.evaluation.low_idx,
            "MID": self.evaluation.mid_idx, 
            "HIGH": self.evaluation.high_idx,
            "TARGET": self.data.target,
            "TIME_SPAN": self.data.time_span,
        }
        
    def __repr__(self) -> str:
        """String representation of the configuration."""
        return f"ConfigManager(quantiles={self.model.catboost.quantiles}, features={len(self.data.features)})"