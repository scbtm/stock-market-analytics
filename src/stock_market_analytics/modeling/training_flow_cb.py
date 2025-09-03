import os
from pathlib import Path
from typing import Any

import pandas as pd
from catboost import CatBoostRegressor, Pool
from hamilton import driver
from metaflow import FlowSpec, step

import wandb
from stock_market_analytics.modeling import processing_functions
from stock_market_analytics.modeling.modeling_config import modeling_config
from stock_market_analytics.modeling.modeling_functions import (
    eval_multiquantile,
    predict_quantiles,
    conformal_adjustment,
    apply_conformal,
    mean_width,
    pinball_loss,
    coverage
)
from stock_market_analytics.modeling.baseline_predictor import BaselinePredictor
from wandb.integration.metaflow import wandb_log

wandb.login(key=os.environ.get("WANDB_KEY"))

# Constants
FEATURES_FILE = modeling_config["FEATURES_FILE"]
QUANTILES = modeling_config["QUANTILES"]
FEATURES = modeling_config["FEATURES"]
PARAMS = modeling_config["PARAMS"]
TARGET_COVERAGE = modeling_config["TARGET_COVERAGE"]
LOW, MID, HIGH = modeling_config["LOW"], modeling_config["MID"], modeling_config["HIGH"]
TARGET = modeling_config["TARGET"]
TIME_SPAN = modeling_config["TIME_SPAN"]  # days

class TrainingFlow(FlowSpec):
    """
    A Metaflow flow to train a CatBoost model for stock market analytics.
    """

    @step
    def start(self) -> None:
        """
        This is the entry point for the Metaflow pipeline. It validates the
        environment and begins the model training process.
        """
        print("🚀 Starting model training Flow...")

        # Validate required environment variables
        if not os.environ.get("BASE_DATA_PATH"):
            raise ValueError("BASE_DATA_PATH environment variable is required")

        print(f"📁 Data directory: {os.environ['BASE_DATA_PATH']}")
        self.next(self.load_inputs)

    @step
    def load_inputs(self) -> None:
        """
        Load input data for model training.
        """
        base_data_path = Path(os.environ["BASE_DATA_PATH"])
        self.data = self._load_features(base_data_path)
        self.next(self.training_model)

    def _load_features(self, base_data_path: Path) -> pd.DataFrame:
        """Load and validate stock data from Parquet file."""
        features_path = base_data_path / FEATURES_FILE  # type: ignore

        if not features_path.exists():
            raise FileNotFoundError(
                f"Features file not found at {features_path}. "
                "Features data must be provided."
            )

        try:
            return pd.read_parquet(features_path)
        except Exception as e:
            raise ValueError(f"Error loading features file: {str(e)}") from e
        
    @step
    def training_model(self) -> None:
        data = self._prepare_data()

        pools = data["pools"]
        dataset = data["dataset"]

        params = PARAMS.copy()

        early_stopping_rounds = params["num_boost_round"] * 0.08

        # Initialize Catboost regressor with current hyperparameters
        model = CatBoostRegressor(**params)
        
        # Train the regressor
        model.fit(
            pools["train_pool"],
            eval_set=pools["validation_pool"],
            early_stopping_rounds=early_stopping_rounds,
            verbose=int(early_stopping_rounds/2),
            plot=False,
        )

        final_iterations = model.best_iteration_

        params.update({
            "num_boost_round": final_iterations,
        })

        # Evaluation metrics
        preds = predict_quantiles(model, pools["validation_pool"])
        ytrue = pools["validation_pool"].get_label()

        loss, metrics = eval_multiquantile(
                y_true=ytrue, q_pred=preds, quantiles=QUANTILES, interval=(0.1, 0.9)
            )

        # Pass information to next step
        calibration_info = {
            "loss": loss,
            "metrics": metrics,
            "params": params,
            "model": model,
            "dataset": dataset,
        }

        self.calibration_info = calibration_info
        self.next(self.evaluate_baselines)

    def _prepare_data(self) -> tuple[dict[str, Pool], dict[str, Any]]:
        """
        Clean and preprocess the loaded feature data.
        """
        print("🧹 Preparing Feature Data...")
        # We only need to drop null values, since the features are already clean
        df = self.data.dropna()

        dr = driver.Builder().with_modules(processing_functions).build()

        data = dr.execute(
            final_vars=["pools", "metadata", "dataset"],
            inputs={"df": df, "time_span": TIME_SPAN, "features": FEATURES},
        )

        return data
    
    @step
    def evaluate_baselines(self) -> None:
        """
        Evaluate baseline predictors for comparison with the trained model.
        """
        print("📊 Evaluating baseline predictors...")
        
        # Use the same _prepare_data method to get properly constructed pools
        data = self._prepare_data()
        pools = data["pools"]
        
        train_pool = pools["train_pool"]
        val_pool = pools["validation_pool"]
        
        # Define baseline strategies to evaluate
        baseline_strategies = [
            "historical_quantiles",
            "random_walk", 
            "random_noise",
            "seasonal_naive"
        ]
        
        baseline_results = {}
        
        for strategy in baseline_strategies:
            print(f"  Evaluating {strategy} baseline...")
            
            # Create and fit baseline predictor
            baseline = BaselinePredictor(strategy=strategy, random_state=42)
            baseline.fit(train_pool)
            
            # Predict on validation set
            baseline_preds = baseline.predict(val_pool)
            ytrue = val_pool.get_label()
            
            # Evaluate using same metrics as trained model
            loss, metrics = eval_multiquantile(
                y_true=ytrue, q_pred=baseline_preds, quantiles=QUANTILES, interval=(0.1, 0.9)
            )
            
            baseline_results[f"baseline_{strategy}"] = {
                "loss": loss,
                "metrics": metrics,
                "model": baseline
            }
        
        # Store baseline results for comparison
        self.baseline_results = baseline_results
        self.next(self.calibrate_model)
    
    @step
    @wandb_log(
        datasets=True,
        models=True,
        others=True,
        settings=wandb.Settings(project="stock-market-analytics"),
        )
    def calibrate_model(self) -> None:
        """
        Perform conformal adjustment on the trained model.
        """

        calibration_info = self.calibration_info

        model = calibration_info["model"]

        params = calibration_info["params"]

        dataset = calibration_info["dataset"]

        dcal = dataset[dataset['fold'] == 'validation'].copy()
        dtest = dataset[dataset['fold'] == 'test'].copy()

        xcal = dcal[FEATURES]
        xtest = dtest[FEATURES]

        ycal = dcal[TARGET].values
        ytest = dtest[TARGET].values

        # Predict quantiles
        q_cal = predict_quantiles(model, xcal)
        q_tst = predict_quantiles(model, xtest)

        # Pull specific lower/upper for CQR (here 10% / 90%)
        qlo_cal, qhi_cal = q_cal[:, LOW], q_cal[:, HIGH]
        qlo_tst, qhi_tst = q_tst[:, LOW], q_tst[:, HIGH]

        # Conformal adjustment on calibration slice
        qconf = conformal_adjustment(qlo_cal, qhi_cal, ycal, alpha=1 - TARGET_COVERAGE)

        # Apply to test
        lo_cqr, hi_cqr = apply_conformal(qlo_tst, qhi_tst, qconf)
        med_pred = q_tst[:, MID]

        # Metrics
        cov   = coverage(ytest, lo_cqr, hi_cqr)
        width = mean_width(lo_cqr, hi_cqr)
        pin50 = pinball_loss(ytest, med_pred, alpha=0.5)

        final_metrics = {
            "coverage": [cov],
            "mean_width": [width],
            "pinball_loss": [pin50],
        }

        training_metrics = calibration_info["metrics"]

        # Evaluate baselines on test set for final comparison
        print("📊 Evaluating baselines on test set...")
        baseline_test_results = {}
        
        xtest = dtest[FEATURES]
        
        for strategy_name, baseline_info in self.baseline_results.items():
            baseline_model = baseline_info["model"]
            
            # Predict quantiles on test set
            q_tst_baseline = baseline_model.predict(xtest)
            
            # Pull specific lower/upper for CQR (here 10% / 90%)
            qlo_tst_baseline, qhi_tst_baseline = q_tst_baseline[:, LOW], q_tst_baseline[:, HIGH]
            
            # Apply conformal adjustment (using same qconf from trained model)
            lo_cqr_baseline, hi_cqr_baseline = apply_conformal(qlo_tst_baseline, qhi_tst_baseline, qconf)
            med_pred_baseline = q_tst_baseline[:, MID]
            
            # Metrics
            cov_baseline = coverage(ytest, lo_cqr_baseline, hi_cqr_baseline)
            width_baseline = mean_width(lo_cqr_baseline, hi_cqr_baseline)
            pin50_baseline = pinball_loss(ytest, med_pred_baseline, alpha=0.5)
            
            baseline_test_results[strategy_name] = {
                "coverage": cov_baseline,
                "mean_width": width_baseline,
                "pinball_loss": pin50_baseline,
            }
        
        # Create comparative metrics
        comparative_metrics = {}
        trained_model_metrics = final_metrics
        
        for strategy_name, baseline_metrics in baseline_test_results.items():
            for metric_name in ["coverage", "mean_width", "pinball_loss"]:
                trained_value = trained_model_metrics[metric_name][0]
                baseline_value = baseline_metrics[metric_name]
                
                # Calculate improvement (positive means trained model is better)
                if metric_name == "coverage":
                    # For coverage, closer to target coverage is better
                    trained_diff = abs(trained_value - TARGET_COVERAGE)
                    baseline_diff = abs(baseline_value - TARGET_COVERAGE) 
                    improvement = baseline_diff - trained_diff  # positive means trained is closer
                elif metric_name == "mean_width":
                    # For width, smaller is better
                    improvement = baseline_value - trained_value  # positive means trained is narrower
                else:  # pinball_loss
                    # For pinball loss, smaller is better
                    improvement = baseline_value - trained_value  # positive means trained is lower
                
                comparative_metrics[f"{strategy_name}_{metric_name}_improvement"] = improvement
                comparative_metrics[f"{strategy_name}_{metric_name}"] = baseline_value
        
        # Log results to wandb
        self.params = params
        self.model = model
        self.training_metrics = training_metrics
        self.development_data = dataset
        self.final_metrics = final_metrics
        self.baseline_test_results = baseline_test_results
        self.comparative_metrics = comparative_metrics

        self.next(self.end)


    @step
    def end(self) -> None:
        """
        This is the final step of the Metaflow pipeline. It can be used to
        perform any final actions or cleanup.
        """
        print("✅ Training Flow completed.")
        print(f"🛠️ Training Metrics: {self.training_metrics}")
        print(f"📊 Evaluation Metrics: {self.final_metrics}")
        
        # Display baseline comparison results
        print("\n📈 Model vs Baseline Performance:")
        print("-" * 60)
        
        for strategy_name, baseline_metrics in self.baseline_test_results.items():
            print(f"\n{strategy_name.replace('baseline_', '').replace('_', ' ').title()}:")
            for metric_name in ["coverage", "mean_width", "pinball_loss"]:
                trained_value = self.final_metrics[metric_name][0]
                baseline_value = baseline_metrics[metric_name]
                improvement_key = f"{strategy_name}_{metric_name}_improvement"
                improvement = self.comparative_metrics[improvement_key]
                
                improvement_sign = "✅" if improvement > 0 else "❌" if improvement < 0 else "➡️"
                print(f"  {metric_name}: {trained_value:.4f} vs {baseline_value:.4f} {improvement_sign}")
        
        print("\n📊 Summary: Trained model improvements over baselines logged to Wandb")

if __name__ == "__main__":
    TrainingFlow()
