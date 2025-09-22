"""
Model Monitoring Flow

A Metaflow pipeline for comprehensive model monitoring and drift detection.
This flow orchestrates the complete monitoring workflow following the established
steps and flow architecture pattern.
"""

from pathlib import Path

import pandas as pd
from metaflow import FlowSpec, step

import wandb
from stock_market_analytics.config import config
from stock_market_analytics.monitoring import monitoring_steps
from wandb.integration.metaflow import wandb_log

# Initialize wandb
wandb.login(key=config.wandb_key)


class MonitoringFlow(FlowSpec):
    """
    A Metaflow pipeline for comprehensive model monitoring and drift detection.

    This flow orchestrates the complete monitoring process:
    1. Downloads model and reference data artifacts from W&B
    2. Loads current production data for monitoring
    3. Performs comprehensive monitoring analysis including:
       - Covariate drift detection
       - Target drift analysis
       - Prediction drift monitoring
       - Model performance evaluation
       - Prediction interval calibration analysis
    4. Generates visualizations and comprehensive HTML report
    5. Logs monitoring results and report as W&B artifacts

    The flow expects BASE_DATA_PATH and WANDB_API_KEY environment variables.
    """

    @step
    def start(self) -> None:
        """
        Initialize the monitoring flow.

        Sets up configuration and validates required environment variables.
        """
        print("🚀 Starting Model Monitoring Flow...")

        # Validate required environment variables
        if not config.base_data_path:
            raise ValueError("BASE_DATA_PATH configuration is required")

        print(f"📁 Data directory: {config.base_data_path}")
        print("✅ Configuration loaded successfully")

        self.config = config
        self.modeling_config = config.modeling

        self.next(self.download_artifacts)

    @step
    def download_artifacts(self) -> None:
        """
        Download model and reference dataset artifacts from Weights & Biases.

        Downloads the latest trained model and reference test set that will be used
        as the baseline for drift detection and performance comparison.
        """
        print("📦 Downloading model and reference data from W&B...")

        try:
            # Download model and reference data
            model_dir, model_name, dataset_dir, dataset_name = (
                monitoring_steps.download_artifacts()
            )

            # Load model and reference data
            self.model = monitoring_steps.load_model(model_dir, model_name)
            self.reference_data = monitoring_steps.load_reference_data(
                dataset_dir, dataset_name
            )

            print(f"✅ Model loaded: {model_name}")
            print(f"✅ Reference data loaded: {dataset_name}")
            print(f"📊 Reference data shape: {self.reference_data.shape}")

        except Exception as e:
            print(f"❌ Error downloading artifacts: {e}")
            raise

        self.next(self.load_current_data)

    @step
    def load_current_data(self) -> None:
        """
        Load current production data for monitoring analysis.

        Loads the most recent production data that will be compared against
        the reference dataset to detect drift and evaluate performance.
        """
        print("📊 Loading current production data...")

        try:
            # Load current production data
            self.current_data = monitoring_steps.load_monitoring_df(
                config.features_path
            )

            print(f"📈 Current data shape: {self.current_data.shape}")
            print(
                f"📅 Current data date range: {self.current_data['date'].min()} to {self.current_data['date'].max()}"
            )

            # Validate feature alignment
            ref_features = set(self.reference_data.columns)
            curr_features = set(self.current_data.columns)
            config_features = set(self.config.modeling.features)

            self.common_features = list(config_features & ref_features & curr_features)

            print(f"🔧 Features in reference data: {len(ref_features)}")
            print(f"🔧 Features in current data: {len(curr_features)}")
            print(f"🔧 Features from config: {len(config_features)}")
            print(f"✅ Common features for analysis: {len(self.common_features)}")

            if len(self.common_features) == 0:
                raise ValueError(
                    "No common features found between reference and current data"
                )

        except Exception as e:
            print(f"❌ Error loading current data: {e}")
            raise

        self.next(self.perform_monitoring_analysis)

    @step
    def perform_monitoring_analysis(self) -> None:
        """
        Perform comprehensive monitoring analysis.

        Conducts the complete suite of monitoring checks including drift detection,
        performance evaluation, and generates visualizations.
        """
        print("🔍 Performing comprehensive monitoring analysis...")

        try:
            target_col = self.config.modeling.target

            # 1. Covariate Drift Detection
            print("📊 Analyzing covariate drift...")
            self.covariate_drift_results = monitoring_steps.get_covariate_drift_metrics(
                reference_df=self.reference_data,
                current_df=self.current_data,
                feature_columns=self.common_features,
            )

            agg_drift = self.covariate_drift_results["aggregate"]
            print(f"   Mean PSI: {agg_drift['mean_psi']:.4f}")
            print(
                f"   Features with major drift (PSI > 0.2): {agg_drift['fraction_drifted_features_psi']:.1%}"
            )

            # 2. Target Drift Analysis
            print("🎯 Analyzing target distribution drift...")
            self.target_drift_results = monitoring_steps.get_target_drift_metrics(
                reference_targets=self.reference_data[target_col],
                current_targets=self.current_data[target_col],
            )

            target_psi = self.target_drift_results["distribution_tests"]["psi"]
            print(
                f"   Target PSI: {target_psi:.4f} ({self.target_drift_results['distribution_tests']['psi_interpretation']})"
            )

            # 3. Generate Model Predictions
            print("🤖 Generating model predictions...")

            # Reference predictions (test set)
            ref_X = self.reference_data.drop(columns=[target_col])

            ref_transformed = self.model.named_steps["transformations"].transform(ref_X)
            ref_predictions = self.model.named_steps["model"].predict(
                ref_transformed, return_full_quantiles=True
            )

            # Current predictions
            curr_X = self.current_data.drop(columns=[target_col, "date"])
            curr_y = self.current_data[target_col]

            curr_transformed = self.model.named_steps["transformations"].transform(
                curr_X
            )
            curr_predictions = self.model.named_steps["model"].predict(
                curr_transformed, return_full_quantiles=True
            )

            print(f"   Reference predictions shape: {ref_predictions.shape}")
            print(f"   Current predictions shape: {curr_predictions.shape}")

            # 4. Prediction Drift Analysis
            print("📈 Analyzing prediction drift...")
            quantile_cols = [f"q_{q:.2f}" for q in self.config.modeling.quantiles]
            ref_pred_df = pd.DataFrame(ref_predictions, columns=quantile_cols)
            curr_pred_df = pd.DataFrame(curr_predictions, columns=quantile_cols)

            self.prediction_drift_results = (
                monitoring_steps.get_prediction_drift_metrics(
                    reference_predictions=ref_pred_df,
                    current_predictions=curr_pred_df,
                    quantiles=self.config.modeling.quantiles,
                )
            )

            pred_drift = self.prediction_drift_results["aggregate"]
            print(
                f"   Mean prediction KS statistic: {pred_drift['mean_ks_statistic']:.4f}"
            )

            # 5. Model Performance Evaluation
            print("📊 Evaluating model performance...")
            self.current_performance = monitoring_steps.get_predicted_quantiles_metrics(
                y_true=curr_y,
                y_pred_quantiles=curr_pred_df,
                quantiles=self.config.modeling.quantiles,
            )

            perf = self.current_performance
            print(f"   Mean Pinball Loss: {perf['pinball_losses']['mean']:.6f}")
            print(f"   CRPS: {perf['distributional']['crps']:.6f}")
            print(f"   Coverage Bias: {perf['coverage']['bias']:.6f}")

            # 6. Prediction Interval Calibration Analysis
            print("🎯 Analyzing prediction interval calibration...")

            # Get calibrated prediction intervals from model
            curr_intervals = self.model.predict(curr_X)
            y_lower = curr_intervals[:, 0]
            y_upper = curr_intervals[:, -1]

            self.calibration_results = monitoring_steps.get_calibration_metrics(
                y_true=curr_y,
                y_lower=pd.Series(y_lower),
                y_upper=pd.Series(y_upper),
                confidence_level=self.config.modeling.target_coverage,
            )

            calib = self.calibration_results
            print(f"   Observed coverage: {calib['coverage']['observed']:.3f}")
            print(f"   Target coverage: {calib['coverage']['target']:.3f}")
            print(f"   Coverage error: {calib['coverage']['error']:.3f}")

            print("✅ Monitoring analysis completed successfully")

        except Exception as e:
            print(f"❌ Error during monitoring analysis: {e}")
            raise

        self.next(self.generate_report)

    @step
    def generate_report(self) -> None:
        """
        Generate visualizations and comprehensive monitoring report.

        Creates all monitoring visualizations and generates a self-contained HTML report
        with embedded plots and detailed analysis results.
        """
        print("📊 Generating monitoring visualizations and report...")

        try:
            # Generate visualization plots as base64 images
            print("   Creating covariate drift plot...")
            cov_b64 = monitoring_steps.plot_drift_metrics(
                self.covariate_drift_results, return_image=True
            )

            print("   Creating prediction drift plot...")
            pred_b64 = monitoring_steps.plot_drift_metrics(
                self.prediction_drift_results, return_image=True
            )

            print("   Creating target drift plot...")
            tgt_b64 = monitoring_steps.plot_drift_metrics(
                self.target_drift_results, return_image=True
            )

            print("   Creating quantile performance plot...")
            quant_b64 = monitoring_steps.plot_performance_metrics(
                self.current_performance, return_image=True
            )

            print("   Creating interval performance plot...")
            pi_b64 = monitoring_steps.plot_performance_metrics(
                self.calibration_results, return_image=True
            )

            # Compile monitoring results for report generation
            self.monitoring_results = {
                "drift_results": self.covariate_drift_results,
                "performance_results": self.current_performance,
                "image_covariate_drift": cov_b64,
                "image_prediction_drift": pred_b64,
                "image_target_drift": tgt_b64,
                "image_quantile_performance": quant_b64,
                "image_interval_performance": pi_b64,
            }

            # Generate comprehensive HTML report
            print("   Generating HTML monitoring report...")
            report_path = "monitoring_report.html"
            self.html_report = monitoring_steps.generate_monitoring_report(
                self.monitoring_results,
                output_path=report_path,
                title="Stock Market Analytics - Model Monitoring Report",
            )

            # Store report path for artifact logging
            self.report_path = report_path

            print("✅ Monitoring report generated successfully")

        except Exception as e:
            print(f"❌ Error generating monitoring report: {e}")
            raise

        self.next(self.log_artifacts)

    @wandb_log(
        datasets=False,
        models=False,
        others=True,
        settings=wandb.Settings(
            project="stock-market-analytics", run_job_type="monitoring"
        ),
    )
    @step
    def log_artifacts(self) -> None:
        """
        Log monitoring results and report as artifacts to Weights & Biases.

        The wandb_log decorator automatically logs any self attributes as W&B artifacts.
        This includes the monitoring results, metrics, and HTML report.
        """
        print("📦 Logging monitoring artifacts to Weights & Biases...")

        try:
            # Store key monitoring metrics for W&B logging
            self.covariate_drift_metrics = {
                "mean_psi": self.covariate_drift_results["aggregate"]["mean_psi"],
                "max_psi": self.covariate_drift_results["aggregate"]["max_psi"],
                "fraction_drifted_features": self.covariate_drift_results["aggregate"][
                    "fraction_drifted_features_psi"
                ],
            }

            self.target_drift_metrics = {
                "target_psi": self.target_drift_results["distribution_tests"]["psi"],
                "target_ks_statistic": self.target_drift_results["distribution_tests"][
                    "ks_statistic"
                ],
                "target_ks_p_value": self.target_drift_results["distribution_tests"][
                    "ks_p_value"
                ],
            }

            self.prediction_drift_metrics = {
                "mean_pred_ks": self.prediction_drift_results["aggregate"][
                    "mean_ks_statistic"
                ],
                "max_pred_ks": self.prediction_drift_results["aggregate"][
                    "max_ks_statistic"
                ],
                "mean_wasserstein": self.prediction_drift_results["aggregate"][
                    "mean_wasserstein"
                ],
            }

            self.performance_metrics = {
                "mean_pinball_loss": self.current_performance["pinball_losses"]["mean"],
                "crps": self.current_performance["distributional"]["crps"],
                "coverage_bias": self.current_performance["coverage"]["bias"],
                "mean_coverage_error": self.current_performance["coverage"][
                    "mean_absolute_error"
                ],
                "monotonicity_violation_rate": self.current_performance["monotonicity"][
                    "violation_rate"
                ],
            }

            self.calibration_metrics = {
                "observed_coverage": self.calibration_results["coverage"]["observed"],
                "target_coverage": self.calibration_results["coverage"]["target"],
                "coverage_error": self.calibration_results["coverage"]["error"],
                "mean_interval_width": self.calibration_results["interval_width"][
                    "mean_width"
                ],
                "interval_score": self.calibration_results["scoring"]["interval_score"],
            }

            # Summary metrics for easy overview
            self.monitoring_summary = {
                "total_features_analyzed": len(self.common_features),
                "current_data_samples": len(self.current_data),
                "reference_data_samples": len(self.reference_data),
                "monitoring_time_span_days": self.config.modeling.time_span,
            }

            # Log the HTML file
            path_to_html_file = self.report_path
            with Path(path_to_html_file).open() as f:
                wandb.log({"report": wandb.Html(f)})

            print("✅ Monitoring artifacts logged successfully")

        except Exception as e:
            print(f"❌ Error logging monitoring artifacts: {e}")
            raise

        self.next(self.end)

    @step
    def end(self) -> None:
        """
        Finalize the monitoring flow.

        Provides a comprehensive summary of the monitoring analysis results.
        """
        print("\n🏁 Model Monitoring Flow completed successfully!")
        print("\n📋 Monitoring Summary:")
        print("=" * 50)
        print("✅ Model and reference data downloaded")
        print("✅ Current production data loaded and validated")
        print("✅ Covariate drift analysis completed")
        print("✅ Target drift analysis completed")
        print("✅ Prediction drift analysis completed")
        print("✅ Model performance evaluation completed")
        print("✅ Prediction interval calibration analysis completed")
        print("✅ Comprehensive monitoring report generated")
        print("✅ All results and artifacts logged to Weights & Biases")

        # Key findings summary
        print("\n🔍 Key Monitoring Findings:")
        print(f"📊 Features analyzed: {len(self.common_features)}")
        print(
            f"📈 Mean covariate PSI: {self.covariate_drift_results['aggregate']['mean_psi']:.4f}"
        )
        print(
            f"🎯 Target PSI: {self.target_drift_results['distribution_tests']['psi']:.4f}"
        )
        print(
            f"🤖 Mean prediction drift (KS): {self.prediction_drift_results['aggregate']['mean_ks_statistic']:.4f}"
        )
        print(
            f"📊 Model CRPS: {self.current_performance['distributional']['crps']:.6f}"
        )
        print(f"🎯 Coverage error: {self.calibration_results['coverage']['error']:.3f}")
        print(
            f"🔧 Monotonicity violations: {self.current_performance['monotonicity']['violation_rate']:.2%}"
        )

        print(f"\n📄 Monitoring report saved to: {self.report_path}")
        print("=" * 50)


if __name__ == "__main__":
    MonitoringFlow()
