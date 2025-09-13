import wandb
import os
import joblib
import pandas as pd
from pathlib import Path
from stock_market_analytics.config import config

import matplotlib.pyplot as plt   
import numpy as np

from datetime import datetime

from stock_market_analytics.modeling.model_factory.data_management.preprocessing import _validate

from stock_market_analytics.monitoring.monitoring_metrics import (
    quantile_regression_performance_metrics,
    multivariate_covariate_drift_metrics,
    prediction_drift_metrics,
    quantile_performance_trends,
    prediction_interval_performance_metrics,
)


def download_artifacts() -> tuple[str, str, str, str]:
    """
    Download model artifacts from Weights & Biases.
    """
    api = wandb.Api(api_key=os.environ.get("WANDB_API_KEY"))
    model_name: str = os.environ.get("MODEL_NAME") or "pipeline:latest"
    model_version: str = model_name.split(':')[1] if ':' in model_name else 'latest'
    dataset_name: str = "test_set:" + model_version

    # Reference the artifact by entity/project/name:version or :latest
    model = api.artifact(f"san-cbtm/stock-market-analytics/{model_name}", type="model")
    dataset = api.artifact(f"san-cbtm/stock-market-analytics/{dataset_name}", type="dataset")

    # Download to a specific folder (defaults to a temp dir if omitted)
    model_dir = model.download()
    dataset_dir = dataset.download()

    return model_dir, model_name, dataset_dir, dataset_name # type: ignore

def load_model(model_dir: str, model_name: str) -> object:
    """
    Load the model from the downloaded artifact directory.
    """
    file_name = model_name.split(':')[0] + '.pkl'
    model = joblib.load(f'{model_dir}/{file_name}')
    return model

def load_reference_data(dataset_dir: str, dataset_name: str) -> pd.DataFrame:
    """
    Load the reference features data for drift detection.
    """

    file_name = dataset_name.split(':')[0] + '.parquet'
    dataset = pd.read_parquet(f'{dataset_dir}/{file_name}')
    
    return dataset

def load_monitoring_df(data_path: str) -> pd.DataFrame:
    """
    Load the features data for monitoring.
    """
    features_file = Path(data_path) / config.modeling.features_file
    
    # load features data
    df = pd.read_parquet(features_file)
    df = _validate(df, date_col='date', symbol_col='symbol')
    #get fully labeled data
    df_labeled = df.dropna(subset=[config.modeling.target])

    # Filter to the last time span days of labeled data for performance monitoring
    max_date = df_labeled['date'].max()
    six_months_ago = max_date - pd.DateOffset(days=config.modeling.time_span)
    df_labeled = df_labeled[df_labeled['date'] >= six_months_ago]

    return df_labeled


# DISTRIBUTION SHIFT DETECTION

def get_covariate_drift_metrics(reference_df: pd.DataFrame, 
                               current_df: pd.DataFrame,
                               feature_columns: list[str]) -> dict:
    """
    Calculate comprehensive covariate drift metrics for features.
    
    Args:
        reference_df: Reference period data
        current_df: Current period data  
        feature_columns: List of feature columns to analyze
        
    Returns:
        Dictionary with drift metrics per feature and aggregate
    """
    
    return multivariate_covariate_drift_metrics(reference_df, current_df, feature_columns)


def get_prediction_drift_metrics(reference_predictions: pd.DataFrame,
                                current_predictions: pd.DataFrame,
                                quantiles: list[float]) -> dict:
    """
    Calculate drift metrics for model predictions across quantiles.
    
    Args:
        reference_predictions: Historical predictions DataFrame
        current_predictions: Current predictions DataFrame
        quantiles: List of quantile levels
        
    Returns:
        Dictionary with prediction drift metrics per quantile and aggregate
    """
    
    # Extract quantile columns from DataFrames
    quantile_cols = [f'q_{q:.2f}' for q in quantiles]
    ref_pred_array = reference_predictions[quantile_cols].values
    curr_pred_array = current_predictions[quantile_cols].values
    
    return prediction_drift_metrics(ref_pred_array, curr_pred_array, quantiles)


def get_target_drift_metrics(reference_targets: pd.Series,
                           current_targets: pd.Series) -> dict:
    """
    Calculate drift metrics for target variable distribution.
    
    Args:
        reference_targets: Historical target values
        current_targets: Current target values
        
    Returns:
        Dictionary with comprehensive target drift metrics
    """
    from stock_market_analytics.monitoring.monitoring_metrics import target_drift_metrics
    
    return target_drift_metrics(reference_targets.values, current_targets.values)


# MODEL PERFORMANCE MONITORING

def get_predicted_quantiles_metrics(y_true: pd.Series,
                                  y_pred_quantiles: pd.DataFrame,
                                  quantiles: list[float]) -> dict:
    """
    Calculate comprehensive performance metrics for quantile regression models.
    
    Args:
        y_true: True target values
        y_pred_quantiles: DataFrame with predicted quantiles
        quantiles: List of quantile levels
        
    Returns:
        Dictionary with comprehensive performance metrics
    """

    # Extract quantile columns
    quantile_cols = [f'q_{q:.2f}' for q in quantiles]
    pred_array = y_pred_quantiles[quantile_cols].values
    
    return quantile_regression_performance_metrics(y_true.values, pred_array, quantiles)


def get_calibration_metrics(y_true: pd.Series,
                          y_lower: pd.Series,
                          y_upper: pd.Series,
                          confidence_level: float = 0.8) -> dict:
    """
    Calculate comprehensive metrics for prediction interval performance.
    
    Args:
        y_true: True target values
        y_lower: Lower bounds of prediction intervals
        y_upper: Upper bounds of prediction intervals
        confidence_level: Target confidence level
        
    Returns:
        Dictionary with interval performance metrics
    """

    return prediction_interval_performance_metrics(
        y_true.values, y_lower.values, y_upper.values, confidence_level
    )


def get_performance_trends(y_true: pd.Series,
                         y_pred_quantiles: pd.DataFrame,
                         dates: pd.Series,
                         quantiles: list[float],
                         window_size: int = 30) -> dict:
    """
    Track performance trends over time using rolling windows.
    
    Args:
        y_true: True target values
        y_pred_quantiles: DataFrame with predicted quantiles
        dates: Date index for time series
        quantiles: List of quantile levels
        window_size: Size of rolling window for trend analysis
        
    Returns:
        Dictionary with time-based performance trends
    """
    
    # Extract quantile columns
    quantile_cols = [f'q_{q:.2f}' for q in quantiles]
    pred_array = y_pred_quantiles[quantile_cols].values
    
    return quantile_performance_trends(y_true.values, pred_array, quantiles, dates.values, window_size)


# PLOTS

def plot_drift_metrics(drift_results: dict, 
                      save_path: str = None,
                      figsize: tuple = (15, 10)) -> None:
    """
    Create comprehensive visualizations for drift detection results.
    
    Args:
        drift_results: Dictionary with drift metrics from drift detection functions
        save_path: Optional path to save the plot
        figsize: Figure size tuple
    """    
    # Determine the type of drift results and plot accordingly
    if "per_feature" in drift_results:
        # Covariate drift results
        _plot_covariate_drift(drift_results, save_path, figsize)
    elif "per_quantile" in drift_results:
        # Prediction drift results
        _plot_prediction_drift(drift_results, save_path, figsize)
    elif "distribution_tests" in drift_results:
        # Target drift results
        _plot_target_drift(drift_results, save_path, figsize)
    else:
        raise ValueError("Unknown drift results format")


def _plot_covariate_drift(drift_results: dict, save_path: str = None, figsize: tuple = (15, 10)):
    """Plot covariate drift metrics."""

    features = list(drift_results["per_feature"].keys())
    if not features:
        print("No features to plot")
        return
    
    # Extract metrics
    psi_values = [drift_results["per_feature"][f]["psi"] for f in features]
    ks_stats = [drift_results["per_feature"][f]["ks_statistic"] for f in features]
    wd_values = [drift_results["per_feature"][f]["wasserstein_distance"] for f in features]
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Covariate Drift Detection Results', fontsize=16, fontweight='bold')
    
    # PSI plot
    colors = ['green' if x < 0.1 else 'orange' if x < 0.2 else 'red' for x in psi_values]
    axes[0, 0].barh(features, psi_values, color=colors)
    axes[0, 0].axvline(x=0.1, color='orange', linestyle='--', alpha=0.7, label='Minor drift')
    axes[0, 0].axvline(x=0.2, color='red', linestyle='--', alpha=0.7, label='Major drift')
    axes[0, 0].set_xlabel('PSI Value')
    axes[0, 0].set_title('Population Stability Index (PSI)')
    axes[0, 0].legend()
    
    # KS statistic plot
    axes[0, 1].barh(features, ks_stats, color='skyblue')
    axes[0, 1].set_xlabel('KS Statistic')
    axes[0, 1].set_title('Kolmogorov-Smirnov Statistic')
    
    # Wasserstein distance plot
    axes[1, 0].barh(features, wd_values, color='lightcoral')
    axes[1, 0].set_xlabel('Wasserstein Distance')
    axes[1, 0].set_title('Wasserstein Distance')
    
    # Summary metrics
    agg = drift_results["aggregate"]
    summary_text = f"""Aggregate Metrics:
Mean PSI: {agg.get('mean_psi', 'N/A'):.3f}
Max PSI: {agg.get('max_psi', 'N/A'):.3f}
Drifted Features: {agg.get('fraction_drifted_features_psi', 0):.1%}
Features Analyzed: {agg.get('n_features_analyzed', 0)}"""
    
    axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes, 
                    fontsize=12, verticalalignment='center', fontfamily='monospace')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def _plot_prediction_drift(drift_results: dict, save_path: str = None, figsize: tuple = (12, 8)):
    """Plot prediction drift metrics."""
    quantiles = drift_results["quantiles"]
    per_quantile = drift_results["per_quantile"]
    
    # Extract metrics per quantile
    ks_stats = [per_quantile[f"q_{q:.2f}"]["ks_statistic"] for q in quantiles]
    wd_values = [per_quantile[f"q_{q:.2f}"]["wasserstein_distance"] for q in quantiles]
    psi_values = [per_quantile[f"q_{q:.2f}"]["psi"] for q in quantiles]
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle('Prediction Drift Across Quantiles', fontsize=16, fontweight='bold')
    
    # KS statistic across quantiles
    axes[0].plot(quantiles, ks_stats, 'o-', color='blue', linewidth=2, markersize=6)
    axes[0].set_xlabel('Quantile Level')
    axes[0].set_ylabel('KS Statistic')
    axes[0].set_title('KS Statistic by Quantile')
    axes[0].grid(True, alpha=0.3)
    
    # Wasserstein distance across quantiles
    axes[1].plot(quantiles, wd_values, 'o-', color='red', linewidth=2, markersize=6)
    axes[1].set_xlabel('Quantile Level')
    axes[1].set_ylabel('Wasserstein Distance')
    axes[1].set_title('Wasserstein Distance by Quantile')
    axes[1].grid(True, alpha=0.3)
    
    # PSI across quantiles
    colors = ['green' if x < 0.1 else 'orange' if x < 0.2 else 'red' for x in psi_values]
    axes[2].bar([f'{q:.2f}' for q in quantiles], psi_values, color=colors)
    axes[2].axhline(y=0.1, color='orange', linestyle='--', alpha=0.7)
    axes[2].axhline(y=0.2, color='red', linestyle='--', alpha=0.7)
    axes[2].set_xlabel('Quantile Level')
    axes[2].set_ylabel('PSI Value')
    axes[2].set_title('PSI by Quantile')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def _plot_target_drift(drift_results: dict, save_path: str = None, figsize: tuple = (15, 10)):
    """Plot target drift metrics."""
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('Target Distribution Drift Analysis', fontsize=16, fontweight='bold')
    
    # Distribution test results
    dist_tests = drift_results["distribution_tests"]
    test_names = ['KS Statistic', 'PSI']
    test_values = [dist_tests["ks_statistic"], dist_tests["psi"]]
    
    axes[0, 0].bar(test_names, test_values, color=['skyblue', 'lightgreen'])
    axes[0, 0].set_title('Distribution Test Statistics')
    axes[0, 0].set_ylabel('Test Statistic Value')
    
    # Moments comparison
    moments = drift_results["moments_comparison"]
    metrics = ['Mean', 'Std', 'Skewness', 'Kurtosis']
    ref_values = [moments["ref_mean"], moments["ref_std"], moments["ref_skewness"], moments["ref_kurtosis"]]
    curr_values = [moments["curr_mean"], moments["curr_std"], moments["curr_skewness"], moments["curr_kurtosis"]]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[0, 1].bar(x - width/2, ref_values, width, label='Reference', color='lightblue')
    axes[0, 1].bar(x + width/2, curr_values, width, label='Current', color='lightcoral')
    axes[0, 1].set_xlabel('Statistical Moments')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].set_title('Moments Comparison')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(metrics)
    axes[0, 1].legend()
    
    # Distance metrics
    distance_metrics = drift_results["distance_metrics"]
    dist_names = ['Wasserstein', 'Jensen-Shannon']
    dist_values = [distance_metrics["wasserstein_distance"], distance_metrics["jensen_shannon_distance"]]
    
    axes[0, 2].bar(dist_names, dist_values, color=['orange', 'purple'])
    axes[0, 2].set_title('Distance Metrics')
    axes[0, 2].set_ylabel('Distance Value')
    
    # Statistical test p-values
    stat_tests = drift_results["statistical_tests"]
    test_names = ['Mean Diff\n(t-test)', 'Variance Diff\n(Levene)']
    p_values = [stat_tests["mean_diff_p_value"], stat_tests["variance_levene_p_value"]]
    
    colors = ['green' if p > 0.05 else 'red' for p in p_values]
    axes[1, 0].bar(test_names, p_values, color=colors)
    axes[1, 0].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α=0.05')
    axes[1, 0].set_title('Statistical Test P-values')
    axes[1, 0].set_ylabel('P-value')
    axes[1, 0].legend()
    
    # Sample sizes
    sample_info = drift_results["sample_sizes"]
    sample_names = ['Reference', 'Current']
    sample_sizes = [sample_info["reference_n"], sample_info["current_n"]]
    
    axes[1, 1].bar(sample_names, sample_sizes, color='lightgray')
    axes[1, 1].set_title('Sample Sizes')
    axes[1, 1].set_ylabel('Number of Samples')
    
    # Summary text
    summary_text = f"""Distribution Tests:
KS p-value: {dist_tests.get('ks_p_value', 'N/A'):.4f}
PSI: {dist_tests.get('psi', 'N/A'):.4f} ({dist_tests.get('psi_interpretation', 'N/A')})

Distance Metrics:
Wasserstein: {distance_metrics.get('wasserstein_distance', 'N/A'):.4f}
Jensen-Shannon: {distance_metrics.get('jensen_shannon_distance', 'N/A'):.4f}

Mean Shift: {moments.get('mean_shift', 'N/A'):.4f}
Std Ratio: {moments.get('std_ratio', 'N/A'):.4f}"""
    
    axes[1, 2].text(0.1, 0.5, summary_text, transform=axes[1, 2].transAxes,
                    fontsize=10, verticalalignment='center', fontfamily='monospace')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_performance_metrics(performance_results: dict,
                           save_path: str = None,
                           figsize: tuple = (15, 12)) -> None:
    """
    Create comprehensive visualizations for model performance metrics.
    
    Args:
        performance_results: Dictionary with performance metrics from performance functions
        save_path: Optional path to save the plot
        figsize: Figure size tuple
    """    
    # Determine the type of performance results and plot accordingly
    if "pinball_losses" in performance_results:
        # Quantile regression performance results
        _plot_quantile_performance(performance_results, save_path, figsize)
    elif "coverage" in performance_results and "interval_width" in performance_results:
        # Interval performance results
        _plot_interval_performance(performance_results, save_path, figsize)
    elif "dates" in performance_results and "metrics" in performance_results:
        # Performance trends results
        _plot_performance_trends(performance_results, save_path, figsize)
    else:
        raise ValueError("Unknown performance results format")


def _plot_quantile_performance(performance_results: dict, save_path: str = None, figsize: tuple = (15, 12)):
    """Plot quantile regression performance metrics."""
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('Quantile Regression Performance Metrics', fontsize=16, fontweight='bold')
    
    # Extract quantile information
    pinball_losses = performance_results["pinball_losses"]["per_quantile"]
    coverage_metrics = performance_results["coverage"]["per_quantile"]
    coverage_errors = performance_results["coverage"]["errors"]
    
    quantiles = list(pinball_losses.keys())
    quantile_values = [float(q.split('_')[1]) for q in quantiles]
    
    # Pinball losses by quantile
    losses = list(pinball_losses.values())
    axes[0, 0].bar(quantiles, losses, color='skyblue')
    axes[0, 0].set_title('Pinball Loss by Quantile')
    axes[0, 0].set_xlabel('Quantile')
    axes[0, 0].set_ylabel('Pinball Loss')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Coverage by quantile
    coverages = list(coverage_metrics.values())
    axes[0, 1].plot(quantile_values, coverages, 'o-', color='green', linewidth=2, markersize=6, label='Observed')
    axes[0, 1].plot(quantile_values, quantile_values, '--', color='red', linewidth=2, label='Target')
    axes[0, 1].set_title('Coverage by Quantile')
    axes[0, 1].set_xlabel('Quantile Level')
    axes[0, 1].set_ylabel('Coverage')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Coverage errors
    errors = list(coverage_errors.values())
    colors = ['green' if abs(e) < 0.05 else 'orange' if abs(e) < 0.1 else 'red' for e in errors]
    axes[0, 2].bar(quantiles, errors, color=colors)
    axes[0, 2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[0, 2].axhline(y=0.05, color='orange', linestyle='--', alpha=0.7)
    axes[0, 2].axhline(y=-0.05, color='orange', linestyle='--', alpha=0.7)
    axes[0, 2].set_title('Coverage Errors')
    axes[0, 2].set_xlabel('Quantile')
    axes[0, 2].set_ylabel('Coverage Error')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # PIT histogram (if available)
    if performance_results["calibration"]["pit_values"]:
        pit_values = performance_results["calibration"]["pit_values"]
        axes[1, 0].hist(pit_values, bins=20, density=True, alpha=0.7, color='lightblue', edgecolor='black')
        axes[1, 0].axhline(y=1.0, color='red', linestyle='--', label='Uniform')
        axes[1, 0].set_title('PIT Histogram')
        axes[1, 0].set_xlabel('PIT Value')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].legend()
    else:
        axes[1, 0].text(0.5, 0.5, 'PIT values\nnot available', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('PIT Histogram')
    
    # Summary metrics
    summary_text = f"""Performance Summary:
Mean Pinball Loss: {performance_results['pinball_losses']['mean']:.4f}
Mean Coverage Error: {performance_results['coverage']['mean_absolute_error']:.4f}
Coverage Bias: {performance_results['coverage']['bias']:.4f}
CRPS: {performance_results['distributional']['crps']:.4f}

Calibration:
PIT KS Statistic: {performance_results['calibration']['pit_ks_statistic']:.4f}
PIT ECE: {performance_results['calibration']['pit_ece']:.4f}

Monotonicity:
Violation Rate: {performance_results['monotonicity']['violation_rate']:.2%}
Valid Samples: {performance_results['sample_info']['n_valid_samples']}"""
    
    axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='center', fontfamily='monospace')
    axes[1, 1].axis('off')
    
    # Monotonicity violations
    mono_rate = performance_results['monotonicity']['violation_rate']
    mono_count = performance_results['monotonicity']['violation_count']
    total_samples = performance_results['monotonicity']['total_samples']
    
    mono_data = ['Valid', 'Violations']
    mono_values = [total_samples - mono_count, mono_count]
    colors = ['green', 'red']
    
    axes[1, 2].pie(mono_values, labels=mono_data, colors=colors, autopct='%1.1f%%')
    axes[1, 2].set_title(f'Monotonicity Violations\n({mono_rate:.2%} violation rate)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def _plot_interval_performance(performance_results: dict, save_path: str = None, figsize: tuple = (12, 8)):
    """Plot prediction interval performance metrics."""
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Prediction Interval Performance', fontsize=16, fontweight='bold')
    
    coverage_info = performance_results["coverage"]
    width_info = performance_results["interval_width"]
    
    # Coverage vs target
    coverage_data = ['Target', 'Observed']
    coverage_values = [coverage_info["target"], coverage_info["observed"]]
    colors = ['blue', 'green' if abs(coverage_info["error"]) < 0.05 else 'red']
    
    axes[0, 0].bar(coverage_data, coverage_values, color=colors)
    axes[0, 0].set_title(f'Coverage: {coverage_info["observed"]:.3f} vs {coverage_info["target"]:.3f}')
    axes[0, 0].set_ylabel('Coverage Probability')
    axes[0, 0].set_ylim([0, 1])
    
    # Width distribution
    width_stats = width_info["width_statistics"]
    width_metrics = ['Min', 'Q25', 'Median', 'Q75', 'Max']
    width_values = [width_stats["min"], width_stats["q25"], 
                   width_stats["median"], width_stats["q75"], width_stats["max"]]
    
    axes[0, 1].bar(width_metrics, width_values, color='lightcoral')
    axes[0, 1].set_title('Interval Width Distribution')
    axes[0, 1].set_ylabel('Width')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Coverage breakdown
    misc_data = ['Below Lower', 'Covered', 'Above Upper']
    misc_values = [coverage_info["below_lower_rate"], 
                  coverage_info["observed"],
                  coverage_info["above_upper_rate"]]
    misc_colors = ['red', 'green', 'red']
    
    axes[1, 0].bar(misc_data, misc_values, color=misc_colors)
    axes[1, 0].set_title('Coverage Breakdown')
    axes[1, 0].set_ylabel('Rate')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Summary metrics
    summary_text = f"""Interval Metrics:
Mean Width: {width_info['mean_width']:.4f}
Normalized Width: {width_info['normalized_width']:.4f}
Interval Score: {performance_results['scoring']['interval_score']:.4f}
Efficiency Ratio: {performance_results['scoring']['efficiency_ratio']:.4f}

Coverage Error: {coverage_info['error']:.4f}
Below Lower: {coverage_info['below_lower_rate']:.3f}
Above Upper: {coverage_info['above_upper_rate']:.3f}

Sample Info:
Valid Samples: {performance_results['sample_info']['n_valid_samples']}
Total Samples: {performance_results['sample_info']['n_total_samples']}"""
    
    axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='center', fontfamily='monospace')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def _plot_performance_trends(performance_results: dict, save_path: str = None, figsize: tuple = (15, 8)):
    """Plot performance trends over time."""
    
    dates = pd.to_datetime(performance_results["dates"])
    metrics = performance_results["metrics"]
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle('Performance Trends Over Time', fontsize=16, fontweight='bold')
    
    # Mean pinball loss trend
    axes[0].plot(dates, metrics["mean_pinball_loss"], 'o-', color='blue', linewidth=2, markersize=4)
    axes[0].set_title('Mean Pinball Loss Trend')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Mean Pinball Loss')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3)
    
    # Coverage error trend
    axes[1].plot(dates, metrics["mean_coverage_error"], 'o-', color='red', linewidth=2, markersize=4)
    axes[1].axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='5% threshold')
    axes[1].set_title('Mean Coverage Error Trend')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Mean Coverage Error')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # CRPS trend
    axes[2].plot(dates, metrics["crps"], 'o-', color='green', linewidth=2, markersize=4)
    axes[2].set_title('CRPS Trend')
    axes[2].set_xlabel('Date')
    axes[2].set_ylabel('CRPS')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# REPORTING

def generate_monitoring_report(monitoring_results: dict,
                             output_path: str = None,
                             title: str = "Model Monitoring Report") -> str:
    """
    Generate a comprehensive HTML monitoring report.
    
    Args:
        monitoring_results: Dictionary containing all monitoring results
        output_path: Optional path to save the HTML report
        title: Report title
        
    Returns:
        HTML string of the report
    """
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
            .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007cba; }}
            .metric {{ background-color: #f9f9f9; padding: 10px; margin: 5px 0; border-radius: 3px; }}
            .warning {{ border-left-color: #ff9800; background-color: #fff3cd; }}
            .error {{ border-left-color: #f44336; background-color: #f8d7da; }}
            .success {{ border-left-color: #4caf50; background-color: #d4edda; }}
            table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .code {{ font-family: monospace; background-color: #f5f5f5; padding: 2px 4px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{title}</h1>
            <p>Generated on: {timestamp}</p>
        </div>
    """
    
    # Add sections based on available results
    if "drift_results" in monitoring_results:
        html_template += _generate_drift_section(monitoring_results["drift_results"])
    
    if "performance_results" in monitoring_results:
        html_template += _generate_performance_section(monitoring_results["performance_results"])
    
    if "trends_results" in monitoring_results:
        html_template += _generate_trends_section(monitoring_results["trends_results"])
    
    html_template += """
    </body>
    </html>
    """
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(html_template)
        print(f"Report saved to: {output_path}")
    
    return html_template


def _generate_drift_section(drift_results: dict) -> str:
    """Generate HTML section for drift results."""
    if "per_feature" in drift_results:
        # Covariate drift
        section_type = "Covariate Drift"
        features = drift_results["per_feature"]
        
        table_rows = ""
        for feature, metrics in features.items():
            psi_class = "success" if metrics["psi"] < 0.1 else "warning" if metrics["psi"] < 0.2 else "error"
            table_rows += f"""
            <tr>
                <td>{feature}</td>
                <td class="{psi_class}">{metrics["psi"]:.4f}</td>
                <td>{metrics["ks_statistic"]:.4f}</td>
                <td>{metrics["ks_p_value"]:.4f}</td>
                <td>{metrics["wasserstein_distance"]:.4f}</td>
                <td>{metrics["psi_interpretation"]}</td>
            </tr>
            """
        
        return f"""
        <div class="section">
            <h2>{section_type} Analysis</h2>
            <table>
                <tr>
                    <th>Feature</th>
                    <th>PSI</th>
                    <th>KS Statistic</th>
                    <th>KS P-value</th>
                    <th>Wasserstein</th>
                    <th>Interpretation</th>
                </tr>
                {table_rows}
            </table>
            <div class="metric">
                <strong>Summary:</strong> Mean PSI: {drift_results["aggregate"]["mean_psi"]:.4f}, 
                Features with drift: {drift_results["aggregate"]["fraction_drifted_features_psi"]:.1%}
            </div>
        </div>
        """
    
    elif "distribution_tests" in drift_results:
        # Target drift
        dist_tests = drift_results["distribution_tests"]
        psi_class = "success" if dist_tests["psi"] < 0.1 else "warning" if dist_tests["psi"] < 0.2 else "error"
        
        return f"""
        <div class="section">
            <h2>Target Drift Analysis</h2>
            <div class="metric {psi_class}">
                <strong>PSI:</strong> {dist_tests["psi"]:.4f} ({dist_tests["psi_interpretation"]})
            </div>
            <div class="metric">
                <strong>KS Test:</strong> Statistic = {dist_tests["ks_statistic"]:.4f}, P-value = {dist_tests["ks_p_value"]:.4f}
            </div>
            <div class="metric">
                <strong>Mean Shift:</strong> {drift_results["moments_comparison"]["mean_shift"]:.4f}
            </div>
        </div>
        """
    
    return ""


def _generate_performance_section(performance_results: dict) -> str:
    """Generate HTML section for performance results."""
    if "pinball_losses" in performance_results:
        # Quantile performance
        coverage_bias_class = "success" if abs(performance_results["coverage"]["bias"]) < 0.05 else "warning"
        
        return f"""
        <div class="section">
            <h2>Model Performance Analysis</h2>
            <div class="metric">
                <strong>Mean Pinball Loss:</strong> {performance_results["pinball_losses"]["mean"]:.4f}
            </div>
            <div class="metric {coverage_bias_class}">
                <strong>Coverage Bias:</strong> {performance_results["coverage"]["bias"]:.4f}
            </div>
            <div class="metric">
                <strong>CRPS:</strong> {performance_results["distributional"]["crps"]:.4f}
            </div>
            <div class="metric">
                <strong>Monotonicity Violations:</strong> {performance_results["monotonicity"]["violation_rate"]:.2%}
            </div>
        </div>
        """
    
    elif "coverage" in performance_results:
        # Interval performance
        coverage_error_class = "success" if abs(performance_results["coverage"]["error"]) < 0.05 else "warning"
        
        return f"""
        <div class="section">
            <h2>Prediction Interval Performance</h2>
            <div class="metric {coverage_error_class}">
                <strong>Coverage:</strong> {performance_results["coverage"]["observed"]:.3f} 
                (Target: {performance_results["coverage"]["target"]:.3f})
            </div>
            <div class="metric">
                <strong>Mean Width:</strong> {performance_results["interval_width"]["mean_width"]:.4f}
            </div>
            <div class="metric">
                <strong>Interval Score:</strong> {performance_results["scoring"]["interval_score"]:.4f}
            </div>
        </div>
        """
    
    return ""


def _generate_trends_section(trends_results: dict) -> str:
    """Generate HTML section for trends results."""
    if "metrics" in trends_results:
        recent_pinball = trends_results["metrics"]["mean_pinball_loss"][-1] if trends_results["metrics"]["mean_pinball_loss"] else "N/A"
        recent_coverage_error = trends_results["metrics"]["mean_coverage_error"][-1] if trends_results["metrics"]["mean_coverage_error"] else "N/A"
        
        return f"""
        <div class="section">
            <h2>Performance Trends</h2>
            <div class="metric">
                <strong>Latest Pinball Loss:</strong> {recent_pinball}
            </div>
            <div class="metric">
                <strong>Latest Coverage Error:</strong> {recent_coverage_error}
            </div>
            <div class="metric">
                <strong>Trend Window Size:</strong> {trends_results["window_size"]} days
            </div>
        </div>
        """
    
    return ""