"""
Simple functions that coordinate core modeling components and can be 
reused across different flows and scenarios.
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from stock_market_analytics.config import config
from stock_market_analytics.modeling.pipeline_components.calibrators import (
    PipelineWithCalibrator,
)
from stock_market_analytics.modeling.pipeline_components.evaluators import (
    ModelEvaluator,
)
from stock_market_analytics.modeling.pipeline_components.pipeline_factory import (
    get_baseline_pipeline,
    get_pipeline,
)


def split_data(df: pd.DataFrame, time_span: int) -> pd.DataFrame:
    """Split DataFrame chronologically into training, validation, and testing sets."""
    # Test with the most recent time_span days of data
    df["fold"] = "train"  # Initialize fold column
    df.loc[df["date"] >= df["date"].max() - pd.Timedelta(days=time_span), "fold"] = "test"

    min_test_date = df[df["fold"] == "test"]["date"].min()
    min_val_date = min_test_date - pd.Timedelta(days=time_span)

    # Validation is time_span days prior to test
    df.loc[(df["date"] < min_test_date) & (df["date"] >= min_val_date), "fold"] = "validation"

    return df


def get_data_metadata(split_data: pd.DataFrame) -> Dict[str, Any]:
    """Get metadata for the training, validation, and test sets."""
    training_start, training_end = (
        split_data[split_data["fold"] == "train"]["date"].min(),
        split_data[split_data["fold"] == "train"]["date"].max(),
    )
    validation_start, validation_end = (
        split_data[split_data["fold"] == "validation"]["date"].min(),
        split_data[split_data["fold"] == "validation"]["date"].max(),
    )
    test_start, test_end = (
        split_data[split_data["fold"] == "test"]["date"].min(),
        split_data[split_data["fold"] == "test"]["date"].max(),
    )

    training_n_rows = split_data[split_data["fold"] == "train"].shape[0]
    validation_n_rows = split_data[split_data["fold"] == "validation"].shape[0]
    test_n_rows = split_data[split_data["fold"] == "test"].shape[0]

    return {
        "date_of_run": pd.Timestamp.now(),
        "training_start": training_start,
        "training_end": training_end,
        "training_n_rows": training_n_rows,
        "validation_start": validation_start,
        "validation_end": validation_end,
        "validation_n_rows": validation_n_rows,
        "test_start": test_start,
        "test_end": test_end,
        "test_n_rows": test_n_rows,
        "columns": split_data.columns.tolist(),
    }


def create_modeling_datasets(
    split_data: pd.DataFrame,
    features: list[str],
    target: str = None,
) -> Dict[str, Any]:
    """Prepare modeling datasets from split data."""
    if target is None:
        target = config.modeling.target

    xtrain, ytrain = (
        split_data[split_data["fold"] == "train"][features],
        split_data[split_data["fold"] == "train"][target],
    )
    xval, yval = (
        split_data[split_data["fold"] == "validation"][features],
        split_data[split_data["fold"] == "validation"][target],
    )
    xtest, ytest = (
        split_data[split_data["fold"] == "test"][features],
        split_data[split_data["fold"] == "test"][target],
    )

    return {
        "xtrain": xtrain,
        "ytrain": ytrain,
        "xval": xval,
        "yval": yval,
        "xtest": xtest,
        "ytest": ytest,
    }


def load_features(base_data_path: Path, features_file: str = None) -> pd.DataFrame:
    """Load and validate features data from Parquet file."""
    if features_file is None:
        features_file = config.modeling.features_file
        
    features_path = base_data_path / features_file

    if not features_path.exists():
        raise FileNotFoundError(
            f"Features file not found at {features_path}. "
            "Features data must be provided."
        )

    try:
        return pd.read_parquet(features_path)
    except Exception as e:
        raise ValueError(f"Error loading features file: {str(e)}") from e


def prepare_training_data(
    data: pd.DataFrame, 
    time_span: int = None,
    features: list[str] = None,
    target: str = None
) -> Dict[str, Any]:
    """Prepare and split data for model training."""
    if time_span is None:
        time_span = config.modeling.time_span
    if features is None:
        features = config.modeling.features
    if target is None:
        target = config.modeling.target
        
    # Split data chronologically
    data_with_folds = split_data(df=data, time_span=time_span)
    
    # Create modeling datasets
    modeling_datasets = create_modeling_datasets(
        split_data=data_with_folds,
        features=features,
        target=target,
    )
    
    return {
        "split_data": data_with_folds,
        "modeling_datasets": modeling_datasets
    }


def train_catboost_model(modeling_datasets: Dict[str, Any]) -> Tuple[Any, int]:
    """Train CatBoost model with early stopping."""
    xtrain, ytrain = modeling_datasets["xtrain"], modeling_datasets["ytrain"]
    xval, yval = modeling_datasets["xval"], modeling_datasets["yval"]

    pipeline = get_pipeline()

    # Handle transformation pipeline for early stopping
    if pipeline.named_steps.get("transformations") is not None:
        transformations = pipeline[0]
        transformations.fit(xtrain)  # Fit PCA on training data only
        _xval = transformations.transform(xval)
    else:
        _xval = xval

    # Set up early stopping parameters
    fit_params = config.modeling.cb_fit_params.copy()
    fit_params["eval_set"] = (_xval, yval)
    fit_params = {f"quantile_regressor__{k}": v for k, v in fit_params.items()}

    # Train the pipeline
    pipeline.fit(xtrain, ytrain, **fit_params)

    # Extract final iteration count
    quantile_regressor = pipeline.named_steps["quantile_regressor"]
    final_iterations = quantile_regressor.best_iteration_

    return pipeline, final_iterations


def analyze_feature_importance(pipeline: Any, xtrain: Any) -> pd.DataFrame:
    """Analyze and return feature importance from trained model."""
    quantile_regressor = pipeline.named_steps["quantile_regressor"]
    feature_importance_df = quantile_regressor._model.get_feature_importance(prettified=True)

    # Map feature indices to names
    if pipeline.named_steps.get("transformations") is not None:
        transformations = pipeline[0]
        indx_to_col_name = {
            f"{i}": col
            for i, col in enumerate(transformations.get_feature_names_out())
        }
    else:
        indx_to_col_name = {f"{i}": col for i, col in enumerate(xtrain.columns)}

    # Map feature indices to names
    feature_importance_df = feature_importance_df.copy()
    feature_importance_df.loc[:, "Feature Id"] = feature_importance_df.loc[
        :, "Feature Id"
    ].map(indx_to_col_name)
    feature_importance_df = feature_importance_df.rename(
        columns={"Feature Id": "Feature", "Importances": "Importance"}
    )
    feature_importance_df = feature_importance_df.sort_values(
        by="Importance", ascending=False
    ).reset_index(drop=True)

    return feature_importance_df


def evaluate_model(pipeline: Any, modeling_datasets: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """Evaluate trained model on validation data."""
    xval, yval = modeling_datasets["xval"], modeling_datasets["yval"]
    
    evaluator = ModelEvaluator()
    loss, metrics = evaluator.evaluate_training(pipeline, xval, yval)
    
    return loss, metrics


def train_baseline_models(modeling_datasets: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Train baseline quantile regressors for comparison."""
    xtrain, ytrain = modeling_datasets["xtrain"], modeling_datasets["ytrain"]
    xval, yval = modeling_datasets["xval"], modeling_datasets["yval"]

    evaluator = ModelEvaluator()
    baselines = ["historical"]
    baseline_results = {}

    for baseline_name in baselines:
        # Get baseline pipeline
        baseline_pipeline = get_baseline_pipeline(baseline_name)
        
        # Train the baseline
        baseline_pipeline.fit(xtrain, ytrain)
        
        # Evaluate on validation set
        loss, metrics = evaluator.evaluate_training(baseline_pipeline, xval, yval)
        
        baseline_results[baseline_name] = {
            "pipeline": baseline_pipeline,
            "loss": loss,
            "metrics": metrics,
        }

    return baseline_results


def create_calibrated_pipeline(
    base_pipeline: Any, 
    modeling_datasets: Dict[str, Any]
) -> Tuple[Any, Any]:
    """Create calibrated pipeline for production use."""
    xcal, ycal = modeling_datasets["xval"], modeling_datasets["yval"]
    
    calibrated_pipeline, calibrator = (
        PipelineWithCalibrator.create_calibrated_pipeline(
            base_pipeline=base_pipeline, X_cal=xcal, y_cal=ycal
        )
    )
    
    return calibrated_pipeline, calibrator


def evaluate_calibrated_predictions(
    calibrated_pipeline: Any,
    base_pipeline: Any, 
    modeling_datasets: Dict[str, Any]
) -> Dict[str, Any]:
    """Evaluate calibrated predictions on test set."""
    xtest, ytest = modeling_datasets["xtest"], modeling_datasets["ytest"]
    
    evaluator = ModelEvaluator()
    
    # Get calibrated bounds
    calibrated_bounds = calibrated_pipeline.predict(xtest)
    
    # Get median predictions for pinball loss
    raw_predictions = base_pipeline.predict(xtest)
    mid_idx = config.modeling.quantile_indices["MID"]
    median_predictions = raw_predictions[:, mid_idx]
    
    conformal_results = evaluator.evaluate_calibrated_predictions(
        calibrated_bounds, ytest, median_predictions
    )
    
    return {
        "coverage": [conformal_results["coverage"]],
        "mean_width": [conformal_results["mean_width"]],
        "pinball_loss": [conformal_results["pinball_loss"]],
    }


def complete_training_workflow(base_data_path: Path) -> Dict[str, Any]:
    """Execute complete model training workflow."""
    # Load features
    data = load_features(base_data_path)
    data = data.dropna()  # Remove nulls for training
    
    # Prepare training data
    data_prep_result = prepare_training_data(data)
    modeling_datasets = data_prep_result["modeling_datasets"]
    
    # Train main model
    pipeline, final_iterations = train_catboost_model(modeling_datasets)
    
    # Evaluate main model
    loss, metrics = evaluate_model(pipeline, modeling_datasets)
    
    # Train baselines
    baseline_results = train_baseline_models(modeling_datasets)
    
    # Create calibrated pipeline
    calibrated_pipeline, calibrator = create_calibrated_pipeline(pipeline, modeling_datasets)
    
    # Evaluate conformal predictions
    final_metrics = evaluate_calibrated_predictions(
        calibrated_pipeline, pipeline, modeling_datasets
    )
    
    return {
        "pipeline": pipeline,
        "calibrated_pipeline": calibrated_pipeline,
        "calibrator": calibrator,
        "final_iterations": final_iterations,
        "training_loss": loss,
        "training_metrics": metrics,
        "baseline_results": baseline_results,
        "final_metrics": final_metrics,
        "data": data_prep_result["split_data"],
        "modeling_datasets": modeling_datasets
    }