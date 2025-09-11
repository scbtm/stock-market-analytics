"""Unit tests for modeling step functions."""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import date, datetime

from stock_market_analytics.modeling.modeling_steps import (
    load_features,
    prepare_training_data,
    create_modeling_datasets,
    train_catboost_model,
    analyze_feature_importance,
    evaluate_model,
    train_baseline_models,
    create_calibrated_pipeline,
    evaluate_calibrated_predictions,
    complete_training_workflow,
    _create_feature_name_mapping,
    _format_feature_importance_dataframe,
    _train_and_evaluate_main_model,
    _create_and_evaluate_calibrated_model,
)


class TestLoadFeatures:
    """Test suite for load_features function."""

    def test_load_features_success(self, tmp_path):
        """Test successful features loading."""
        features_file = tmp_path / "stock_history_features.parquet"
        test_data = pd.DataFrame(
            {
                "date": [date(2023, 1, 1), date(2023, 1, 2)],
                "symbol": ["AAPL", "AAPL"],
                "feature1": [0.5, 0.6],
                "feature2": [1.2, 1.3],
                "y_log_returns": [0.01, -0.02],
            }
        )
        test_data.to_parquet(features_file)

        result = load_features(tmp_path)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "feature1" in result.columns
        assert "y_log_returns" in result.columns

    def test_load_features_custom_filename(self, tmp_path):
        """Test loading with custom filename."""
        custom_file = tmp_path / "custom_features.parquet"
        test_data = pd.DataFrame(
            {"symbol": ["AAPL"], "feature1": [0.5], "y_log_returns": [0.01]}
        )
        test_data.to_parquet(custom_file)

        result = load_features(tmp_path, "custom_features.parquet")

        assert len(result) == 1
        assert result["symbol"].iloc[0] == "AAPL"

    def test_load_features_file_not_found(self, tmp_path):
        """Test FileNotFoundError when features file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Features file not found"):
            load_features(tmp_path)

    def test_load_features_corrupted_file(self, tmp_path):
        """Test ValueError when file is corrupted."""
        features_file = tmp_path / "stock_history_features.parquet"
        features_file.write_text("corrupted parquet data")

        with pytest.raises(ValueError, match="Error loading features file"):
            load_features(tmp_path)


class TestPrepareTrainingData:
    """Test suite for prepare_training_data function."""

    @patch("stock_market_analytics.modeling.modeling_steps.split_data")
    @patch("stock_market_analytics.modeling.modeling_steps.create_modeling_datasets")
    def test_prepare_training_data_default_params(
        self, mock_create_datasets, mock_split
    ):
        """Test prepare training data with default parameters."""
        # Setup test data
        test_data = pd.DataFrame(
            {
                "date": [date(2023, 1, 1), date(2023, 1, 2)],
                "symbol": ["AAPL", "AAPL"],
                "feature1": [0.5, 0.6],
                "y_log_returns": [0.01, -0.02],
            }
        )

        # Setup mocks
        mock_split_data = test_data.copy()
        mock_split_data["fold"] = ["train", "test"]
        mock_split.return_value = mock_split_data

        mock_datasets = {"xtrain": Mock(), "ytrain": Mock()}
        mock_create_datasets.return_value = mock_datasets

        result = prepare_training_data(test_data)

        assert "split_data" in result
        assert "modeling_datasets" in result
        assert result["modeling_datasets"] == mock_datasets

        # Verify mocks were called with config defaults
        mock_split.assert_called_once()
        mock_create_datasets.assert_called_once()

    @patch("stock_market_analytics.modeling.modeling_steps.split_data")
    @patch("stock_market_analytics.modeling.modeling_steps.create_modeling_datasets")
    def test_prepare_training_data_custom_params(
        self, mock_create_datasets, mock_split
    ):
        """Test prepare training data with custom parameters."""
        test_data = pd.DataFrame(
            {"date": [date(2023, 1, 1)], "feature1": [0.5], "y_log_returns": [0.01]}
        )

        mock_split.return_value = test_data
        mock_create_datasets.return_value = {}

        result = prepare_training_data(
            test_data, time_span=100, features=["feature1"], target="y_log_returns"
        )

        assert "split_data" in result
        mock_split.assert_called_once_with(df=test_data, time_span=100)
        mock_create_datasets.assert_called_once_with(
            split_data=test_data, features=["feature1"], target="y_log_returns"
        )


class TestCreateModelingDatasets:
    """Test suite for create_modeling_datasets function."""

    def test_create_modeling_datasets_success(self):
        """Test successful modeling datasets creation."""
        split_data = pd.DataFrame(
            {
                "date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
                "fold": ["train", "validation", "test"],
                "feature1": [0.1, 0.2, 0.3],
                "feature2": [1.1, 1.2, 1.3],
                "y_log_returns": [0.01, -0.02, 0.03],
            }
        )

        features = ["feature1", "feature2"]
        target = "y_log_returns"

        result = create_modeling_datasets(split_data, features, target)

        # Check all required datasets are created
        assert "xtrain" in result
        assert "ytrain" in result
        assert "xval" in result
        assert "yval" in result
        assert "xtest" in result
        assert "ytest" in result

        # Check data shapes and content
        assert len(result["xtrain"]) == 1  # Only train fold
        assert len(result["xval"]) == 1  # Only validation fold
        assert len(result["xtest"]) == 1  # Only test fold

        assert list(result["xtrain"].columns) == features
        assert result["ytrain"].iloc[0] == 0.01
        assert result["yval"].iloc[0] == -0.02
        assert result["ytest"].iloc[0] == 0.03

    def test_create_modeling_datasets_default_target(self):
        """Test modeling datasets creation with default target."""
        split_data = pd.DataFrame(
            {"fold": ["train"], "feature1": [0.1], "y_log_returns": [0.01]}
        )

        with patch(
            "stock_market_analytics.modeling.modeling_steps.config"
        ) as mock_config:
            mock_config.modeling.target = "y_log_returns"

            result = create_modeling_datasets(split_data, ["feature1"])

            assert "ytrain" in result
            assert result["ytrain"].iloc[0] == 0.01


class TestCreateFeatureNameMapping:
    """Test suite for _create_feature_name_mapping function."""

    def test_create_feature_name_mapping_with_transformations(self):
        """Test feature name mapping with transformations pipeline."""
        # Create mock pipeline with transformations
        mock_pipeline = MagicMock()
        mock_transformations = Mock()
        mock_transformations.get_feature_names_out.return_value = [
            "pca0",
            "pca1",
            "pca2",
        ]
        mock_pipeline.named_steps.get.return_value = mock_transformations
        mock_pipeline.__getitem__.return_value = mock_transformations

        mock_xtrain = Mock()

        result = _create_feature_name_mapping(mock_pipeline, mock_xtrain)

        expected = {"0": "pca0", "1": "pca1", "2": "pca2"}
        assert result == expected

    def test_create_feature_name_mapping_without_transformations(self):
        """Test feature name mapping without transformations pipeline."""
        # Create mock pipeline without transformations
        mock_pipeline = Mock()
        mock_pipeline.named_steps.get.return_value = None

        mock_xtrain = Mock()
        mock_xtrain.columns = ["feature1", "feature2", "feature3"]

        result = _create_feature_name_mapping(mock_pipeline, mock_xtrain)

        expected = {"0": "feature1", "1": "feature2", "2": "feature3"}
        assert result == expected


class TestFormatFeatureImportanceDataframe:
    """Test suite for _format_feature_importance_dataframe function."""

    def test_format_feature_importance_dataframe(self):
        """Test feature importance dataframe formatting."""
        # Create test dataframe
        importance_df = pd.DataFrame(
            {"Feature Id": ["0", "1", "2"], "Importances": [0.5, 0.3, 0.2]}
        )

        index_to_name = {"0": "feature1", "1": "feature2", "2": "feature3"}

        result = _format_feature_importance_dataframe(importance_df, index_to_name)

        # Check column renaming
        assert "Feature" in result.columns
        assert "Importance" in result.columns
        assert "Feature Id" not in result.columns
        assert "Importances" not in result.columns

        # Check feature name mapping
        assert result["Feature"].tolist() == ["feature1", "feature2", "feature3"]

        # Check sorting (highest importance first)
        assert result["Importance"].tolist() == [0.5, 0.3, 0.2]


class TestAnalyzeFeatureImportance:
    """Test suite for analyze_feature_importance function."""

    @patch(
        "stock_market_analytics.modeling.modeling_steps._create_feature_name_mapping"
    )
    @patch(
        "stock_market_analytics.modeling.modeling_steps._format_feature_importance_dataframe"
    )
    def test_analyze_feature_importance(self, mock_format, mock_mapping):
        """Test feature importance analysis."""
        # Create mock pipeline
        mock_pipeline = Mock()
        mock_regressor = Mock()
        mock_model = Mock()

        # Setup mock feature importance
        importance_df = pd.DataFrame(
            {"Feature Id": ["0", "1"], "Importances": [0.6, 0.4]}
        )
        mock_model.get_feature_importance.return_value = importance_df
        mock_regressor._model = mock_model
        mock_pipeline.named_steps = {"quantile_regressor": mock_regressor}

        mock_xtrain = Mock()

        # Setup mock functions
        mock_mapping.return_value = {"0": "feature1", "1": "feature2"}
        formatted_df = pd.DataFrame(
            {"Feature": ["feature1", "feature2"], "Importance": [0.6, 0.4]}
        )
        mock_format.return_value = formatted_df

        result = analyze_feature_importance(mock_pipeline, mock_xtrain)

        assert result.equals(formatted_df)
        mock_model.get_feature_importance.assert_called_once_with(prettified=True)
        mock_mapping.assert_called_once_with(mock_pipeline, mock_xtrain)
        mock_format.assert_called_once()


class TestEvaluateModel:
    """Test suite for evaluate_model function."""

    @patch("stock_market_analytics.modeling.modeling_steps.ModelEvaluator")
    def test_evaluate_model(self, mock_evaluator_class):
        """Test model evaluation."""
        # Setup mock evaluator
        mock_evaluator = Mock()
        mock_loss = 0.025
        mock_metrics = {"metric1": 0.8, "metric2": 0.9}
        mock_evaluator.evaluate_training.return_value = (mock_loss, mock_metrics)
        mock_evaluator_class.return_value = mock_evaluator

        # Setup mock data
        modeling_datasets = {"xval": Mock(), "yval": Mock()}
        mock_pipeline = Mock()

        loss, metrics = evaluate_model(mock_pipeline, modeling_datasets)

        assert loss == mock_loss
        assert metrics == mock_metrics
        mock_evaluator.evaluate_training.assert_called_once_with(
            mock_pipeline, modeling_datasets["xval"], modeling_datasets["yval"]
        )


class TestTrainBaselineModels:
    """Test suite for train_baseline_models function."""

    @patch("stock_market_analytics.modeling.modeling_steps.get_baseline_pipeline")
    @patch("stock_market_analytics.modeling.modeling_steps.ModelEvaluator")
    def test_train_baseline_models(self, mock_evaluator_class, mock_get_baseline):
        """Test baseline model training."""
        # Setup mock evaluator
        mock_evaluator = Mock()
        mock_loss = 0.045
        mock_metrics = {"metric1": 0.7}
        mock_evaluator.evaluate_training.return_value = (mock_loss, mock_metrics)
        mock_evaluator_class.return_value = mock_evaluator

        # Setup mock baseline pipeline
        mock_baseline_pipeline = Mock()
        mock_get_baseline.return_value = mock_baseline_pipeline

        # Setup mock data
        modeling_datasets = {
            "xtrain": Mock(),
            "ytrain": Mock(),
            "xval": Mock(),
            "yval": Mock(),
        }

        result = train_baseline_models(modeling_datasets)

        assert "historical" in result
        baseline_result = result["historical"]
        assert baseline_result["pipeline"] == mock_baseline_pipeline
        assert baseline_result["loss"] == mock_loss
        assert baseline_result["metrics"] == mock_metrics

        # Verify pipeline was fitted
        mock_baseline_pipeline.fit.assert_called_once_with(
            modeling_datasets["xtrain"], modeling_datasets["ytrain"]
        )


class TestCreateCalibratedPipeline:
    """Test suite for create_calibrated_pipeline function."""

    @patch("stock_market_analytics.modeling.modeling_steps.PipelineWithCalibrator")
    def test_create_calibrated_pipeline(self, mock_calibrator_class):
        """Test calibrated pipeline creation."""
        # Setup mock calibrator
        mock_calibrated_pipeline = Mock()
        mock_calibrator = Mock()
        mock_calibrator_class.create_calibrated_pipeline.return_value = (
            mock_calibrated_pipeline,
            mock_calibrator,
        )

        # Setup mock data
        mock_base_pipeline = Mock()
        modeling_datasets = {"xval": Mock(), "yval": Mock()}

        calibrated_pipeline, calibrator = create_calibrated_pipeline(
            mock_base_pipeline, modeling_datasets
        )

        assert calibrated_pipeline == mock_calibrated_pipeline
        assert calibrator == mock_calibrator

        mock_calibrator_class.create_calibrated_pipeline.assert_called_once_with(
            base_pipeline=mock_base_pipeline,
            X_cal=modeling_datasets["xval"],
            y_cal=modeling_datasets["yval"],
        )


class TestEvaluateCalibratedPredictions:
    """Test suite for evaluate_calibrated_predictions function."""

    @patch("stock_market_analytics.modeling.modeling_steps.ModelEvaluator")
    @patch("stock_market_analytics.modeling.modeling_steps.config")
    def test_evaluate_calibrated_predictions(self, mock_config, mock_evaluator_class):
        """Test calibrated predictions evaluation."""
        # Setup mock config
        mock_config.modeling.quantile_indices = {"MID": 2}

        # Setup mock evaluator
        mock_evaluator = Mock()
        mock_conformal_results = {
            "coverage": 0.82,
            "mean_width": 0.15,
            "pinball_loss": 0.021,
        }
        mock_evaluator.evaluate_calibrated_predictions.return_value = (
            mock_conformal_results
        )
        mock_evaluator_class.return_value = mock_evaluator

        # Setup mock pipelines and data
        mock_calibrated_pipeline = Mock()
        mock_calibrated_bounds = np.array([[0.1, 0.9], [0.2, 0.8]])
        mock_calibrated_pipeline.predict.return_value = mock_calibrated_bounds

        mock_base_pipeline = Mock()
        mock_raw_predictions = np.array(
            [[0.05, 0.25, 0.5, 0.75, 0.95], [0.15, 0.35, 0.6, 0.85, 0.99]]
        )
        mock_base_pipeline.predict.return_value = mock_raw_predictions

        modeling_datasets = {"xtest": Mock(), "ytest": Mock()}

        result = evaluate_calibrated_predictions(
            mock_calibrated_pipeline, mock_base_pipeline, modeling_datasets
        )

        # Check result structure
        assert "coverage" in result
        assert "mean_width" in result
        assert "pinball_loss" in result

        # Check values are wrapped in lists as expected
        assert result["coverage"] == [0.82]
        assert result["mean_width"] == [0.15]
        assert result["pinball_loss"] == [0.021]

        # Verify median predictions were extracted correctly
        expected_median = np.array([0.5, 0.6])  # Middle quantile values
        mock_evaluator.evaluate_calibrated_predictions.assert_called_once()
        args = mock_evaluator.evaluate_calibrated_predictions.call_args[0]
        np.testing.assert_array_equal(args[2], expected_median)


class TestTrainAndEvaluateMainModel:
    """Test suite for _train_and_evaluate_main_model function."""

    @patch("stock_market_analytics.modeling.modeling_steps.train_catboost_model")
    @patch("stock_market_analytics.modeling.modeling_steps.evaluate_model")
    def test_train_and_evaluate_main_model(self, mock_evaluate, mock_train):
        """Test main model training and evaluation."""
        # Setup mocks
        mock_pipeline = Mock()
        mock_final_iterations = 850
        mock_train.return_value = (mock_pipeline, mock_final_iterations)

        mock_loss = 0.023
        mock_metrics = {"metric1": 0.85}
        mock_evaluate.return_value = (mock_loss, mock_metrics)

        modeling_datasets = {"mock": "data"}

        result = _train_and_evaluate_main_model(modeling_datasets)

        # Check result structure
        assert result["pipeline"] == mock_pipeline
        assert result["final_iterations"] == mock_final_iterations
        assert result["training_loss"] == mock_loss
        assert result["training_metrics"] == mock_metrics

        # Verify functions were called
        mock_train.assert_called_once_with(modeling_datasets)
        mock_evaluate.assert_called_once_with(mock_pipeline, modeling_datasets)


class TestCreateAndEvaluateCalibratedModel:
    """Test suite for _create_and_evaluate_calibrated_model function."""

    @patch("stock_market_analytics.modeling.modeling_steps.create_calibrated_pipeline")
    @patch(
        "stock_market_analytics.modeling.modeling_steps.evaluate_calibrated_predictions"
    )
    def test_create_and_evaluate_calibrated_model(
        self, mock_evaluate_cal, mock_create_cal
    ):
        """Test calibrated model creation and evaluation."""
        # Setup mocks
        mock_calibrated_pipeline = Mock()
        mock_calibrator = Mock()
        mock_create_cal.return_value = (mock_calibrated_pipeline, mock_calibrator)

        mock_final_metrics = {"coverage": [0.81], "mean_width": [0.16]}
        mock_evaluate_cal.return_value = mock_final_metrics

        mock_base_pipeline = Mock()
        modeling_datasets = {"mock": "data"}

        result = _create_and_evaluate_calibrated_model(
            mock_base_pipeline, modeling_datasets
        )

        # Check result structure
        assert result["calibrated_pipeline"] == mock_calibrated_pipeline
        assert result["calibrator"] == mock_calibrator
        assert result["final_metrics"] == mock_final_metrics

        # Verify functions were called
        mock_create_cal.assert_called_once_with(mock_base_pipeline, modeling_datasets)
        mock_evaluate_cal.assert_called_once_with(
            mock_calibrated_pipeline, mock_base_pipeline, modeling_datasets
        )


class TestCompleteTrainingWorkflow:
    """Test suite for complete_training_workflow function."""

    @patch(
        "stock_market_analytics.modeling.modeling_steps._create_and_evaluate_calibrated_model"
    )
    @patch("stock_market_analytics.modeling.modeling_steps.train_baseline_models")
    @patch(
        "stock_market_analytics.modeling.modeling_steps._train_and_evaluate_main_model"
    )
    @patch("stock_market_analytics.modeling.modeling_steps.prepare_training_data")
    @patch("stock_market_analytics.modeling.modeling_steps.load_features")
    def test_complete_training_workflow(
        self,
        mock_load,
        mock_prepare,
        mock_train_main,
        mock_train_baselines,
        mock_calibrated,
    ):
        """Test complete training workflow."""
        # Setup mock data loading
        test_data = pd.DataFrame(
            {"feature1": [0.1, 0.2], "y_log_returns": [0.01, -0.02]}
        )
        mock_load.return_value = test_data

        # Setup mock data preparation
        mock_data_prep = {"split_data": Mock(), "modeling_datasets": Mock()}
        mock_prepare.return_value = mock_data_prep

        # Setup mock main model results
        mock_main_results = {
            "pipeline": Mock(),
            "final_iterations": 800,
            "training_loss": 0.025,
            "training_metrics": {"metric1": 0.8},
        }
        mock_train_main.return_value = mock_main_results

        # Setup mock baseline results
        mock_baseline_results = {"historical": {"loss": 0.05}}
        mock_train_baselines.return_value = mock_baseline_results

        # Setup mock calibrated results
        mock_calibrated_results = {
            "calibrated_pipeline": Mock(),
            "calibrator": Mock(),
            "final_metrics": {"coverage": [0.8]},
        }
        mock_calibrated.return_value = mock_calibrated_results

        base_path = Path("/tmp/test")
        result = complete_training_workflow(base_path)

        # Check that all results are included
        assert "pipeline" in result
        assert "final_iterations" in result
        assert "training_loss" in result
        assert "training_metrics" in result
        assert "calibrated_pipeline" in result
        assert "calibrator" in result
        assert "final_metrics" in result
        assert "baseline_results" in result
        assert "data" in result
        assert "modeling_datasets" in result

        # Verify the pipeline of function calls
        mock_load.assert_called_once_with(base_path)
        mock_prepare.assert_called_once()
        mock_train_main.assert_called_once_with(mock_data_prep["modeling_datasets"])
        mock_train_baselines.assert_called_once_with(
            mock_data_prep["modeling_datasets"]
        )
        mock_calibrated.assert_called_once_with(
            mock_main_results["pipeline"], mock_data_prep["modeling_datasets"]
        )

    @patch("stock_market_analytics.modeling.modeling_steps.load_features")
    def test_complete_training_workflow_load_error(self, mock_load):
        """Test complete training workflow when loading fails."""
        mock_load.side_effect = FileNotFoundError("Features file not found")

        base_path = Path("/tmp/test")

        with pytest.raises(FileNotFoundError):
            complete_training_workflow(base_path)
