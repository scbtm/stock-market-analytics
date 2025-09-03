"""
Model loading utilities for inference.

This module provides utilities for loading trained models and associated
artifacts from various sources (local files, cloud storage, etc.).
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
import joblib
import json

from ..components import CatBoostMultiQuantileModel
from ..config import ConfigManager
from .predictor import ProductionPredictor

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Utility class for loading trained models and creating production predictors.
    
    This class handles loading models from different formats and sources,
    with support for various artifacts (model, config, conformal adjustments, etc.).
    """
    
    @staticmethod
    def load_predictor(predictor_path: Union[str, Path]) -> ProductionPredictor:
        """
        Load a complete production predictor.
        
        Args:
            predictor_path: Path to saved predictor
            
        Returns:
            Loaded ProductionPredictor instance
        """
        return ProductionPredictor.load(predictor_path)
        
    @staticmethod
    def load_model_artifacts(
        artifacts_dir: Union[str, Path],
        model_filename: str = "model.cbm",
        config_filename: str = "config.json",
        conformal_filename: str = "conformal.json",
        metadata_filename: str = "metadata.json"
    ) -> ProductionPredictor:
        """
        Load model from separate artifact files.
        
        Args:
            artifacts_dir: Directory containing model artifacts
            model_filename: CatBoost model file name
            config_filename: Configuration file name
            conformal_filename: Conformal adjustment file name
            metadata_filename: Training metadata file name
            
        Returns:
            ProductionPredictor instance
        """
        artifacts_dir = Path(artifacts_dir)
        
        if not artifacts_dir.exists():
            raise FileNotFoundError(f"Artifacts directory not found: {artifacts_dir}")
            
        # Load configuration
        config_path = artifacts_dir / config_filename
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            config = ConfigManager.from_dict(config_dict)
        else:
            logger.warning(f"Config file not found at {config_path}, using defaults")
            config = ConfigManager()
            
        # Load model
        model_path = artifacts_dir / model_filename
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        model = CatBoostMultiQuantileModel(
            quantiles=config.model.catboost.quantiles,
            **config.model.catboost.to_catboost_params()
        )
        model.load_model(str(model_path))
        
        # Load conformal adjustment if available
        conformal_adjustment = None
        conformal_path = artifacts_dir / conformal_filename
        if conformal_path.exists():
            with open(conformal_path, 'r') as f:
                conformal_data = json.load(f)
            conformal_adjustment = conformal_data.get("adjustment")
            
        # Load metadata if available
        model_metadata = {}
        metadata_path = artifacts_dir / metadata_filename
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)
                
        return ProductionPredictor(
            model=model,
            config=config,
            conformal_adjustment=conformal_adjustment,
            model_metadata=model_metadata
        )
        
    @staticmethod
    def create_predictor_from_training(
        trained_model: CatBoostMultiQuantileModel,
        config: ConfigManager,
        conformal_adjustment: Optional[float] = None,
        training_metrics: Optional[Dict[str, Any]] = None,
        training_params: Optional[Dict[str, Any]] = None
    ) -> ProductionPredictor:
        """
        Create a production predictor from training artifacts.
        
        Args:
            trained_model: Trained CatBoost model
            config: Configuration used for training
            conformal_adjustment: Conformal prediction adjustment
            training_metrics: Metrics from training
            training_params: Parameters used for training
            
        Returns:
            ProductionPredictor instance ready for inference
        """
        # Prepare metadata
        model_metadata = {}
        if training_metrics:
            model_metadata["training_metrics"] = training_metrics
        if training_params:
            model_metadata["training_params"] = training_params
            
        return ProductionPredictor(
            model=trained_model,
            config=config,
            conformal_adjustment=conformal_adjustment,
            model_metadata=model_metadata
        )
        
    @staticmethod
    def save_model_artifacts(
        predictor: ProductionPredictor,
        artifacts_dir: Union[str, Path],
        model_filename: str = "model.cbm",
        config_filename: str = "config.json",
        conformal_filename: str = "conformal.json",
        metadata_filename: str = "metadata.json"
    ) -> None:
        """
        Save model artifacts to separate files.
        
        Args:
            predictor: ProductionPredictor to save
            artifacts_dir: Directory to save artifacts
            model_filename: CatBoost model file name
            config_filename: Configuration file name
            conformal_filename: Conformal adjustment file name
            metadata_filename: Training metadata file name
        """
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Save CatBoost model
        model_path = artifacts_dir / model_filename
        predictor.model.save_model(str(model_path))
        
        # Save configuration
        config_path = artifacts_dir / config_filename
        with open(config_path, 'w') as f:
            json.dump(predictor.config.to_dict(), f, indent=2, default=str)
            
        # Save conformal adjustment if available
        if predictor.conformal_adjustment is not None:
            conformal_path = artifacts_dir / conformal_filename
            conformal_data = {
                "adjustment": predictor.conformal_adjustment,
                "target_coverage": predictor.config.evaluation.target_coverage
            }
            with open(conformal_path, 'w') as f:
                json.dump(conformal_data, f, indent=2)
                
        # Save metadata if available
        if predictor.model_metadata:
            metadata_path = artifacts_dir / metadata_filename
            with open(metadata_path, 'w') as f:
                json.dump(predictor.model_metadata, f, indent=2, default=str)
                
        logger.info(f"Model artifacts saved to {artifacts_dir}")
        
    @staticmethod
    def validate_model_artifacts(artifacts_dir: Union[str, Path]) -> Dict[str, bool]:
        """
        Validate that model artifacts exist and are loadable.
        
        Args:
            artifacts_dir: Directory containing model artifacts
            
        Returns:
            Dictionary with validation results for each artifact
        """
        artifacts_dir = Path(artifacts_dir)
        
        validation_results = {
            "artifacts_dir_exists": artifacts_dir.exists(),
            "model_file_exists": False,
            "config_file_exists": False,
            "conformal_file_exists": False,
            "metadata_file_exists": False,
            "model_loadable": False,
            "config_loadable": False
        }
        
        if not artifacts_dir.exists():
            return validation_results
            
        # Check file existence
        model_path = artifacts_dir / "model.cbm"
        config_path = artifacts_dir / "config.json"
        conformal_path = artifacts_dir / "conformal.json"
        metadata_path = artifacts_dir / "metadata.json"
        
        validation_results["model_file_exists"] = model_path.exists()
        validation_results["config_file_exists"] = config_path.exists()
        validation_results["conformal_file_exists"] = conformal_path.exists()
        validation_results["metadata_file_exists"] = metadata_path.exists()
        
        # Try loading configuration
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    json.load(f)
                validation_results["config_loadable"] = True
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                
        # Try loading model (basic check)
        if model_path.exists():
            try:
                # Just check file can be read, don't load full model
                with open(model_path, 'rb') as f:
                    f.read(100)  # Read first 100 bytes
                validation_results["model_loadable"] = True
            except Exception as e:
                logger.error(f"Error reading model file: {e}")
                
        return validation_results