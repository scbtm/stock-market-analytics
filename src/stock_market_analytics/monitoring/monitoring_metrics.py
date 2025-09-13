"""
Core monitoring metrics for ML model monitoring.

This module provides scientifically robust functions for detecting distribution
drift and monitoring model performance for quantile regression models.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any
from scipy import stats

# Import existing evaluation functions
from stock_market_analytics.modeling.model_factory.evaluation.evaluation_functions import (
    pinball_loss_vectorized,
    quantile_coverage,
    crps_from_quantiles,
    interval_score,
    prediction_interval_coverage_probability,
    mean_interval_width,
    normalized_interval_width,
    pit_values,
    pit_ks_statistic,
    pit_ece,
)


# ============================================
# DISTRIBUTION SHIFT DETECTION
# ============================================

# --------------------------------------------
# Covariate Drift Metrics
# --------------------------------------------

def kolmogorov_smirnov_test(reference_data: np.ndarray, current_data: np.ndarray) -> Dict[str, float]:
    """
    Two-sample Kolmogorov-Smirnov test for distribution drift detection.
    
    Args:
        reference_data: Historical/reference data distribution
        current_data: Current data distribution to test
        
    Returns:
        Dictionary with KS statistic and p-value
    """
    reference = np.asarray(reference_data).flatten()
    current = np.asarray(current_data).flatten()
    
    if len(reference) == 0 or len(current) == 0:
        return {"ks_statistic": np.nan, "p_value": np.nan}
    
    statistic, p_value = stats.ks_2samp(reference, current)
    
    return {
        "ks_statistic": float(statistic),
        "p_value": float(p_value)
    }


def wasserstein_distance(reference_data: np.ndarray, current_data: np.ndarray) -> float:
    """
    Earth Mover's Distance (Wasserstein-1) between two distributions.
    
    Args:
        reference_data: Historical/reference data distribution
        current_data: Current data distribution to test
        
    Returns:
        Wasserstein distance
    """
    reference = np.asarray(reference_data).flatten()
    current = np.asarray(current_data).flatten()
    
    if len(reference) == 0 or len(current) == 0:
        return np.nan
    
    return float(stats.wasserstein_distance(reference, current))


def population_stability_index(reference_data: np.ndarray, current_data: np.ndarray, 
                             n_bins: int = 10) -> Dict[str, Any]:
    """
    Population Stability Index (PSI) for feature drift detection.
    
    Args:
        reference_data: Historical/reference data distribution
        current_data: Current data distribution to test
        n_bins: Number of bins for discretization
        
    Returns:
        Dictionary with PSI value, bin info, and interpretation
    """
    reference = np.asarray(reference_data).flatten()
    current = np.asarray(current_data).flatten()
    
    if len(reference) == 0 or len(current) == 0:
        return {"psi": np.nan, "interpretation": "insufficient_data"}
    
    # Create bins based on reference data quantiles
    bin_edges = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    bin_edges[0] = -np.inf  # Handle edge cases
    bin_edges[-1] = np.inf
    
    # Calculate frequencies
    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    curr_counts, _ = np.histogram(current, bins=bin_edges)
    
    # Convert to proportions (add small epsilon to avoid log(0))
    epsilon = 1e-6
    ref_props = (ref_counts + epsilon) / (len(reference) + n_bins * epsilon)
    curr_props = (curr_counts + epsilon) / (len(current) + n_bins * epsilon)
    
    # Calculate PSI
    psi = np.sum((curr_props - ref_props) * np.log(curr_props / ref_props))
    
    # Interpretation
    if psi < 0.1:
        interpretation = "no_drift"
    elif psi < 0.2:
        interpretation = "minor_drift"
    else:
        interpretation = "major_drift"
    
    return {
        "psi": float(psi),
        "interpretation": interpretation,
        "n_bins": n_bins,
        "ref_props": ref_props.tolist(),
        "curr_props": curr_props.tolist()
    }


def jensen_shannon_divergence(reference_data: np.ndarray, current_data: np.ndarray,
                            n_bins: int = 50) -> float:
    """
    Jensen-Shannon divergence between two distributions.
    
    Args:
        reference_data: Historical/reference data distribution
        current_data: Current data distribution to test
        n_bins: Number of bins for histogram estimation
        
    Returns:
        Jensen-Shannon divergence (0 = identical, 1 = completely different)
    """
    reference = np.asarray(reference_data).flatten()
    current = np.asarray(current_data).flatten()
    
    if len(reference) == 0 or len(current) == 0:
        return np.nan
    
    # Create common bins
    all_data = np.concatenate([reference, current])
    bin_edges = np.linspace(all_data.min(), all_data.max(), n_bins + 1)
    
    # Calculate normalized histograms
    ref_hist, _ = np.histogram(reference, bins=bin_edges, density=True)
    curr_hist, _ = np.histogram(current, bins=bin_edges, density=True)
    
    # Normalize to probabilities
    ref_hist = ref_hist / ref_hist.sum()
    curr_hist = curr_hist / curr_hist.sum()
    
    # Add epsilon to avoid log(0)
    epsilon = 1e-10
    ref_hist = ref_hist + epsilon
    curr_hist = curr_hist + epsilon
    
    # Calculate JS divergence
    m = 0.5 * (ref_hist + curr_hist)
    js_div = 0.5 * stats.entropy(ref_hist, m) + 0.5 * stats.entropy(curr_hist, m)
    
    return float(np.sqrt(js_div))  # Return JS distance (square root of divergence)


def multivariate_covariate_drift_metrics(reference_df: pd.DataFrame, 
                                        current_df: pd.DataFrame,
                                        feature_columns: list[str]) -> Dict[str, Any]:
    """
    Comprehensive multivariate drift detection for feature sets.
    
    Args:
        reference_df: Reference period data
        current_df: Current period data
        feature_columns: List of feature columns to analyze
        
    Returns:
        Dictionary with per-feature and aggregate drift metrics
    """
    results = {
        "per_feature": {},
        "aggregate": {},
        "feature_columns": feature_columns
    }
    
    psi_values = []
    ks_p_values = []
    wasserstein_distances = []
    
    for feature in feature_columns:
        #skip conditions:
        c1 = feature not in reference_df.columns
        c2 = feature not in current_df.columns
        c3 = current_df[feature].dtype in ['object', 'category']
        c4 = reference_df[feature].dtype in ['object', 'category']
        if c1 or c2 or c3 or c4:
            continue
            
        ref_data = reference_df[feature].dropna().values
        curr_data = current_df[feature].dropna().values
        
        # Calculate individual metrics
        ks_result = kolmogorov_smirnov_test(ref_data, curr_data)
        psi_result = population_stability_index(ref_data, curr_data)
        wd = wasserstein_distance(ref_data, curr_data)
        js_div = jensen_shannon_divergence(ref_data, curr_data)
        
        results["per_feature"][feature] = {
            "ks_statistic": ks_result["ks_statistic"],
            "ks_p_value": ks_result["p_value"],
            "psi": psi_result["psi"],
            "psi_interpretation": psi_result["interpretation"],
            "wasserstein_distance": wd,
            "jensen_shannon_distance": js_div
        }
        
        # Collect for aggregate metrics
        if not np.isnan(psi_result["psi"]):
            psi_values.append(psi_result["psi"])
        if not np.isnan(ks_result["p_value"]):
            ks_p_values.append(ks_result["p_value"])
        if not np.isnan(wd):
            wasserstein_distances.append(wd)
    
    # Aggregate metrics
    results["aggregate"] = {
        "mean_psi": float(np.mean(psi_values)) if psi_values else np.nan,
        "max_psi": float(np.max(psi_values)) if psi_values else np.nan,
        "fraction_drifted_features_psi": float(np.mean([x > 0.2 for x in psi_values])) if psi_values else np.nan,
        "mean_wasserstein": float(np.mean(wasserstein_distances)) if wasserstein_distances else np.nan,
        "fraction_significant_ks": float(np.mean([x < 0.05 for x in ks_p_values])) if ks_p_values else np.nan,
        "n_features_analyzed": len([f for f in feature_columns if f in reference_df.columns and f in current_df.columns])
    }
    
    return results


# --------------------------------------------
# Prediction Drift Metrics
# --------------------------------------------

def prediction_drift_metrics(reference_predictions: np.ndarray,
                            current_predictions: np.ndarray,
                            quantiles: np.ndarray) -> Dict[str, Any]:
    """
    Detect drift in model predictions across quantiles.
    
    Args:
        reference_predictions: Historical predictions (n_samples, n_quantiles)
        current_predictions: Current predictions (n_samples, n_quantiles)
        quantiles: Quantile levels
        
    Returns:
        Dictionary with drift metrics per quantile and aggregate
    """
    reference = np.asarray(reference_predictions)
    current = np.asarray(current_predictions)
    quantiles = np.asarray(quantiles)
    
    if reference.shape[1] != len(quantiles) or current.shape[1] != len(quantiles):
        raise ValueError("Number of quantiles must match prediction dimensions")
    
    results = {
        "per_quantile": {},
        "aggregate": {},
        "quantiles": quantiles.tolist()
    }
    
    ks_statistics = []
    wasserstein_distances = []
    
    for i, q in enumerate(quantiles):
        ref_q = reference[:, i]
        curr_q = current[:, i]
        
        # Drift metrics for this quantile
        ks_result = kolmogorov_smirnov_test(ref_q, curr_q)
        wd = wasserstein_distance(ref_q, curr_q)
        psi_result = population_stability_index(ref_q, curr_q)
        
        results["per_quantile"][f"q_{q:.2f}"] = {
            "ks_statistic": ks_result["ks_statistic"],
            "ks_p_value": ks_result["p_value"], 
            "wasserstein_distance": wd,
            "psi": psi_result["psi"],
            "psi_interpretation": psi_result["interpretation"]
        }
        
        if not np.isnan(ks_result["ks_statistic"]):
            ks_statistics.append(ks_result["ks_statistic"])
        if not np.isnan(wd):
            wasserstein_distances.append(wd)
    
    # Aggregate metrics
    results["aggregate"] = {
        "mean_ks_statistic": float(np.mean(ks_statistics)) if ks_statistics else np.nan,
        "max_ks_statistic": float(np.max(ks_statistics)) if ks_statistics else np.nan,
        "mean_wasserstein": float(np.mean(wasserstein_distances)) if wasserstein_distances else np.nan,
        "max_wasserstein": float(np.max(wasserstein_distances)) if wasserstein_distances else np.nan
    }
    
    return results


# --------------------------------------------
# Target Drift Metrics
# --------------------------------------------

def target_drift_metrics(reference_targets: np.ndarray,
                        current_targets: np.ndarray) -> Dict[str, Any]:
    """
    Detect drift in target variable distribution.
    
    Args:
        reference_targets: Historical target values
        current_targets: Current target values
        
    Returns:
        Dictionary with comprehensive drift metrics
    """
    reference = np.asarray(reference_targets).flatten()
    current = np.asarray(current_targets).flatten()
    
    # Remove NaN values
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    
    if len(reference) == 0 or len(current) == 0:
        return {"error": "Insufficient non-NaN data"}
    
    # Statistical tests
    ks_result = kolmogorov_smirnov_test(reference, current)
    psi_result = population_stability_index(reference, current)
    
    # Distance metrics
    wd = wasserstein_distance(reference, current)
    js_div = jensen_shannon_divergence(reference, current)
    
    # Statistical moments comparison
    ref_mean, curr_mean = np.mean(reference), np.mean(current)
    ref_std, curr_std = np.std(reference), np.std(current)
    ref_skew, curr_skew = stats.skew(reference), stats.skew(current)
    ref_kurt, curr_kurt = stats.kurtosis(reference), stats.kurtosis(current)
    
    # Two-sample t-test for means
    t_stat, t_p = stats.ttest_ind(reference, current, equal_var=False)
    
    # F-test approximation for variances (Levene's test)
    levene_stat, levene_p = stats.levene(reference, current)
    
    return {
        "distribution_tests": {
            "ks_statistic": ks_result["ks_statistic"],
            "ks_p_value": ks_result["p_value"],
            "psi": psi_result["psi"],
            "psi_interpretation": psi_result["interpretation"]
        },
        "distance_metrics": {
            "wasserstein_distance": wd,
            "jensen_shannon_distance": js_div
        },
        "moments_comparison": {
            "ref_mean": float(ref_mean),
            "curr_mean": float(curr_mean),
            "mean_shift": float(curr_mean - ref_mean),
            "ref_std": float(ref_std),
            "curr_std": float(curr_std),
            "std_ratio": float(curr_std / ref_std) if ref_std > 0 else np.nan,
            "ref_skewness": float(ref_skew),
            "curr_skewness": float(curr_skew),
            "skewness_shift": float(curr_skew - ref_skew),
            "ref_kurtosis": float(ref_kurt),
            "curr_kurtosis": float(curr_kurt),
            "kurtosis_shift": float(curr_kurt - ref_kurt)
        },
        "statistical_tests": {
            "mean_diff_t_stat": float(t_stat),
            "mean_diff_p_value": float(t_p),
            "variance_levene_stat": float(levene_stat),
            "variance_levene_p_value": float(levene_p)
        },
        "sample_sizes": {
            "reference_n": len(reference),
            "current_n": len(current)
        }
    }


# ============================================
# MODEL PERFORMANCE MONITORING
# ============================================

# --------------------------------------------
# Predicted Quantiles Metrics (Full set of quantiles, i.e. catboost_quantiles, output by "model" step in the pipeline)
# --------------------------------------------

def quantile_regression_performance_metrics(y_true: np.ndarray,
                                          y_pred_quantiles: np.ndarray,
                                          quantiles: np.ndarray) -> Dict[str, Any]:
    """
    Comprehensive performance metrics for quantile regression models.
    
    Args:
        y_true: True target values
        y_pred_quantiles: Predicted quantiles (n_samples, n_quantiles)
        quantiles: Quantile levels
        
    Returns:
        Dictionary with comprehensive performance metrics
    """
    y_true = np.asarray(y_true).flatten()
    y_pred_quantiles = np.asarray(y_pred_quantiles)
    quantiles = np.asarray(quantiles)
    
    # Remove rows with any NaN values
    mask = ~np.isnan(y_true)
    mask = mask & ~np.any(np.isnan(y_pred_quantiles), axis=1)
    
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred_quantiles[mask]
    
    if len(y_true_clean) == 0:
        return {"error": "No valid data after removing NaNs"}
    
    # Core metrics using existing functions
    pinball_losses = pinball_loss_vectorized(y_true_clean, y_pred_clean, quantiles)
    coverages = quantile_coverage(y_true_clean, y_pred_clean, quantiles)
    
    # Coverage errors
    coverage_errors = coverages - quantiles
    
    # CRPS (requires dense quantile grid)
    try:
        crps = crps_from_quantiles(y_true_clean, y_pred_clean, quantiles)
    except Exception:
        crps = np.nan
    
    # PIT values for calibration assessment
    try:
        pit = pit_values(y_true_clean, y_pred_clean, quantiles)
        pit_ks = pit_ks_statistic(pit)
        pit_ece_score = pit_ece(pit)
    except Exception:
        pit = np.array([])
        pit_ks = np.nan
        pit_ece_score = np.nan
    
    # Monotonicity check
    from stock_market_analytics.modeling.model_factory.evaluation.evaluation_functions import monotonicity_violation_rate
    mono_violation_rate, mono_violation_count = monotonicity_violation_rate(y_pred_clean)
    
    return {
        "pinball_losses": {
            "per_quantile": {f"q_{q:.2f}": float(loss) for q, loss in zip(quantiles, pinball_losses)},
            "mean": float(np.mean(pinball_losses)),
            "median": float(np.median(pinball_losses)),
            "max": float(np.max(pinball_losses))
        },
        "coverage": {
            "per_quantile": {f"q_{q:.2f}": float(cov) for q, cov in zip(quantiles, coverages)},
            "errors": {f"q_{q:.2f}": float(err) for q, err in zip(quantiles, coverage_errors)},
            "mean_absolute_error": float(np.mean(np.abs(coverage_errors))),
            "max_absolute_error": float(np.max(np.abs(coverage_errors))),
            "bias": float(np.mean(coverage_errors))  # negative = under-coverage
        },
        "calibration": {
            "pit_ks_statistic": float(pit_ks),
            "pit_ece": float(pit_ece_score),
            "pit_values": pit.tolist() if len(pit) <= 1000 else []  # Limit size for storage
        },
        "distributional": {
            "crps": float(crps)
        },
        "monotonicity": {
            "violation_rate": float(mono_violation_rate),
            "violation_count": int(mono_violation_count),
            "total_samples": int(len(y_pred_clean))
        },
        "sample_info": {
            "n_valid_samples": int(len(y_true_clean)),
            "n_total_samples": int(len(y_true)),
            "n_nan_removed": int(len(y_true) - len(y_true_clean))
        }
    }


def quantile_performance_trends(y_true: np.ndarray,
                               y_pred_quantiles: np.ndarray,
                               quantiles: np.ndarray,
                               dates: np.ndarray,
                               window_size: int = 30) -> Dict[str, Any]:
    """
    Track performance trends over time using rolling windows.
    
    Args:
        y_true: True target values
        y_pred_quantiles: Predicted quantiles (n_samples, n_quantiles)
        quantiles: Quantile levels
        dates: Date index for time series
        window_size: Size of rolling window for trend analysis
        
    Returns:
        Dictionary with time-based performance trends
    """
    y_true = np.asarray(y_true).flatten()
    y_pred_quantiles = np.asarray(y_pred_quantiles)
    quantiles = np.asarray(quantiles)
    dates = pd.to_datetime(dates).values
    
    # Create DataFrame for easier time-based operations
    df = pd.DataFrame({
        'date': dates,
        'y_true': y_true
    })
    
    # Add quantile predictions
    for i, q in enumerate(quantiles):
        df[f'q_{q:.2f}'] = y_pred_quantiles[:, i]
    
    # Sort by date and remove NaN rows
    df = df.sort_values('date').dropna()
    
    if len(df) < window_size:
        return {"error": f"Insufficient data for window size {window_size}"}
    
    # Calculate rolling metrics
    results = {
        "dates": [],
        "metrics": {
            "mean_pinball_loss": [],
            "mean_coverage_error": [],
            "crps": []
        },
        "window_size": window_size
    }
    
    quantile_cols = [f'q_{q:.2f}' for q in quantiles]
    
    for i in range(window_size - 1, len(df)):
        window_df = df.iloc[i - window_size + 1:i + 1]
        
        y_true_window = window_df['y_true'].values
        y_pred_window = window_df[quantile_cols].values
        
        # Calculate metrics for this window
        try:
            pinball_losses = pinball_loss_vectorized(y_true_window, y_pred_window, quantiles)
            coverages = quantile_coverage(y_true_window, y_pred_window, quantiles)
            coverage_errors = coverages - quantiles
            
            # Try to calculate CRPS
            try:
                crps = crps_from_quantiles(y_true_window, y_pred_window, quantiles)
            except Exception:
                crps = np.nan
            
            results["dates"].append(window_df.iloc[-1]['date'])
            results["metrics"]["mean_pinball_loss"].append(float(np.mean(pinball_losses)))
            results["metrics"]["mean_coverage_error"].append(float(np.mean(np.abs(coverage_errors))))
            results["metrics"]["crps"].append(float(crps))
            
        except Exception:
            continue
    
    return results


# --------------------------------------------
# Calibration Metrics (interval of lower and upper quantiles, predicted by the "calibrator" step in the pipeline)
# --------------------------------------------

def prediction_interval_performance_metrics(y_true: np.ndarray,
                                           y_lower: np.ndarray,
                                           y_upper: np.ndarray,
                                           confidence_level: float = 0.8) -> Dict[str, Any]:
    """
    Comprehensive metrics for prediction interval performance (calibrated intervals).
    
    Args:
        y_true: True target values
        y_lower: Lower bounds of prediction intervals
        y_upper: Upper bounds of prediction intervals
        confidence_level: Target confidence level (e.g., 0.8 for 80% intervals)
        
    Returns:
        Dictionary with interval performance metrics
    """
    y_true = np.asarray(y_true).flatten()
    y_lower = np.asarray(y_lower).flatten()
    y_upper = np.asarray(y_upper).flatten()
    
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_lower) | np.isnan(y_upper))
    y_true_clean = y_true[mask]
    y_lower_clean = y_lower[mask]
    y_upper_clean = y_upper[mask]
    
    if len(y_true_clean) == 0:
        return {"error": "No valid data after removing NaNs"}
    
    # Core metrics using existing functions
    coverage = prediction_interval_coverage_probability(y_true_clean, y_lower_clean, y_upper_clean)
    mean_width = mean_interval_width(y_lower_clean, y_upper_clean)
    normalized_width = normalized_interval_width(y_lower_clean, y_upper_clean, y_true_clean)
    
    # Interval score (lower is better)
    alpha = 1.0 - confidence_level
    interval_score_value = interval_score(y_true_clean, y_lower_clean, y_upper_clean, alpha)
    
    # Coverage error
    coverage_error = coverage - confidence_level
    
    # Miscoverage patterns
    below_lower = np.mean(y_true_clean < y_lower_clean)
    above_upper = np.mean(y_true_clean > y_upper_clean)
    
    # Width statistics
    widths = y_upper_clean - y_lower_clean
    width_stats = {
        "min": float(np.min(widths)),
        "max": float(np.max(widths)),
        "mean": float(np.mean(widths)),
        "median": float(np.median(widths)),
        "std": float(np.std(widths)),
        "q25": float(np.percentile(widths, 25)),
        "q75": float(np.percentile(widths, 75))
    }
    
    # Efficiency metrics
    efficiency_ratio = coverage / normalized_width if normalized_width > 0 else np.nan
    
    return {
        "coverage": {
            "observed": float(coverage),
            "target": float(confidence_level),
            "error": float(coverage_error),
            "below_lower_rate": float(below_lower),
            "above_upper_rate": float(above_upper)
        },
        "interval_width": {
            "mean_width": float(mean_width),
            "normalized_width": float(normalized_width),
            "width_statistics": width_stats
        },
        "scoring": {
            "interval_score": float(interval_score_value),
            "efficiency_ratio": float(efficiency_ratio)  # Coverage per unit normalized width
        },
        "sample_info": {
            "n_valid_samples": int(len(y_true_clean)),
            "n_total_samples": int(len(y_true)),
            "n_nan_removed": int(len(y_true) - len(y_true_clean))
        }
    }


def interval_calibration_curve(y_true: np.ndarray,
                             y_lower: np.ndarray,
                             y_upper: np.ndarray,
                             n_bins: int = 10) -> Dict[str, Any]:
    """
    Generate calibration curve for prediction intervals by binning interval widths.
    
    Args:
        y_true: True target values
        y_lower: Lower bounds of prediction intervals
        y_upper: Upper bounds of prediction intervals
        n_bins: Number of bins for calibration curve
        
    Returns:
        Dictionary with calibration curve data
    """
    y_true = np.asarray(y_true).flatten()
    y_lower = np.asarray(y_lower).flatten()
    y_upper = np.asarray(y_upper).flatten()
    
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_lower) | np.isnan(y_upper))
    y_true_clean = y_true[mask]
    y_lower_clean = y_lower[mask]
    y_upper_clean = y_upper[mask]
    
    if len(y_true_clean) == 0:
        return {"error": "No valid data after removing NaNs"}
    
    # Calculate interval widths and coverage indicators
    widths = y_upper_clean - y_lower_clean
    covered = (y_true_clean >= y_lower_clean) & (y_true_clean <= y_upper_clean)
    
    # Create bins based on width percentiles
    bin_edges = np.percentile(widths, np.linspace(0, 100, n_bins + 1))
    bin_edges[0] = bin_edges[0] - 1e-10  # Ensure all points are included
    bin_edges[-1] = bin_edges[-1] + 1e-10
    
    # Assign to bins
    bin_indices = np.digitize(widths, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Calculate metrics per bin
    calibration_data = {
        "bin_centers": [],
        "observed_coverage": [],
        "bin_counts": [],
        "mean_widths": []
    }
    
    for i in range(n_bins):
        bin_mask = bin_indices == i
        if np.sum(bin_mask) == 0:
            continue
            
        bin_widths = widths[bin_mask]
        bin_coverage = covered[bin_mask]
        
        calibration_data["bin_centers"].append(float(np.mean(bin_widths)))
        calibration_data["observed_coverage"].append(float(np.mean(bin_coverage)))
        calibration_data["bin_counts"].append(int(np.sum(bin_mask)))
        calibration_data["mean_widths"].append(float(np.mean(bin_widths)))
    
    return calibration_data


def interval_sharpness_metrics(y_lower: np.ndarray, y_upper: np.ndarray) -> Dict[str, Any]:
    """
    Metrics for interval sharpness (narrower is better, given adequate coverage).
    
    Args:
        y_lower: Lower bounds of prediction intervals
        y_upper: Upper bounds of prediction intervals
        
    Returns:
        Dictionary with sharpness metrics
    """
    y_lower = np.asarray(y_lower).flatten()
    y_upper = np.asarray(y_upper).flatten()
    
    # Remove NaN values
    mask = ~(np.isnan(y_lower) | np.isnan(y_upper))
    y_lower_clean = y_lower[mask]
    y_upper_clean = y_upper[mask]
    
    if len(y_lower_clean) == 0:
        return {"error": "No valid data after removing NaNs"}
    
    widths = y_upper_clean - y_lower_clean
    
    return {
        "mean_width": float(np.mean(widths)),
        "median_width": float(np.median(widths)),
        "width_std": float(np.std(widths)),
        "width_iqr": float(np.percentile(widths, 75) - np.percentile(widths, 25)),
        "width_90th_percentile": float(np.percentile(widths, 90)),
        "width_cv": float(np.std(widths) / np.mean(widths)) if np.mean(widths) > 0 else np.nan,
        "n_samples": int(len(y_lower_clean))
    }