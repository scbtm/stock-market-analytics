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
from pandas.api.types import is_numeric_dtype  # robust numeric check

# Import existing evaluation functions
from stock_market_analytics.modeling.model_factory.evaluation.evaluation_functions import (
    pinball_loss_vectorized,
    quantile_coverage,
    crps_from_quantiles,
    interval_score,  # Winkler score
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
    Two-sample Kolmogorov–Smirnov (KS) test for distribution drift detection.

    What: nonparametric test comparing the empirical CDFs of two samples.
    Why: sensitive to general distributional change, not just mean/variance.
    Interpret: returns {"ks_statistic", "p_value"}; larger statistic and small p-value
    (< 0.05) suggest the current distribution differs from the reference.

    Args:
        reference_data: Historical/reference data distribution.
        current_data: Current data distribution to test.
    """
    reference = np.asarray(reference_data).flatten()
    current = np.asarray(current_data).flatten()

    if len(reference) == 0 or len(current) == 0:
        return {"ks_statistic": np.nan, "p_value": np.nan}

    statistic, p_value = stats.ks_2samp(reference, current)
    return {"ks_statistic": float(statistic), "p_value": float(p_value)}


def wasserstein_distance(reference_data: np.ndarray, current_data: np.ndarray) -> float:
    """
    1-Wasserstein (Earth Mover’s) distance between two distributions.

    What: minimal “work” to move probability mass from one distribution to the other.
    Why: gives a scale-aware, robust measure of shift (units of the variable).
    Interpret: 0 means identical; larger means greater shift (no fixed upper bound).

    Args:
        reference_data: Historical/reference data distribution.
        current_data: Current data distribution to test.
    """
    reference = np.asarray(reference_data).flatten()
    current = np.asarray(current_data).flatten()

    if len(reference) == 0 or len(current) == 0:
        return np.nan

    return float(stats.wasserstein_distance(reference, current))


def population_stability_index(
    reference_data: np.ndarray, current_data: np.ndarray, n_bins: int = 10
) -> Dict[str, Any]:
    """
    Population Stability Index (PSI) for feature drift detection.

    What: compares binned proportions between reference and current distributions.
    Why: widely used in risk/credit monitoring; easy-to-interpret thresholds.
    Interpret: PSI < 0.1 ~ no drift; 0.1–0.2 ~ minor; > 0.2 ~ major drift.

    Args:
        reference_data: Historical/reference data distribution.
        current_data: Current data distribution to test.
        n_bins: Number of bins for discretization.
    """
    reference = np.asarray(reference_data).flatten()
    current = np.asarray(current_data).flatten()

    if len(reference) == 0 or len(current) == 0:
        return {"psi": np.nan, "interpretation": "insufficient_data"}

    # Create bins based on reference data quantiles
    bin_edges = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    bin_edges[0] = -np.inf
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
        "curr_props": curr_props.tolist(),
    }


def jensen_shannon_divergence(
    reference_data: np.ndarray, current_data: np.ndarray, n_bins: int = 50
) -> float:
    """
    Jensen–Shannon **distance** between two distributions (bounded in [0, 1]).

    What: symmetrized and smoothed KL divergence; we return its square root (“distance”).
    Why: stable even with zero-probability bins; interpretable 0–1 scale.
    Interpret: 0 = identical, 1 = maximally different (given equal supports).

    Args:
        reference_data: Historical/reference data distribution.
        current_data: Current data distribution to test.
        n_bins: Number of bins for histogram estimation.
    """
    reference = np.asarray(reference_data).flatten()
    current = np.asarray(current_data).flatten()

    if len(reference) == 0 or len(current) == 0:
        return np.nan

    # Common bins
    all_data = np.concatenate([reference, current])
    bin_edges = np.linspace(all_data.min(), all_data.max(), n_bins + 1)

    # Normalized histograms -> probabilities
    ref_hist, _ = np.histogram(reference, bins=bin_edges, density=True)
    curr_hist, _ = np.histogram(current, bins=bin_edges, density=True)
    ref_hist = ref_hist / np.sum(ref_hist) if ref_hist.sum() > 0 else ref_hist
    curr_hist = curr_hist / np.sum(curr_hist) if curr_hist.sum() > 0 else curr_hist

    epsilon = 1e-10
    ref_hist = ref_hist + epsilon
    curr_hist = curr_hist + epsilon

    m = 0.5 * (ref_hist + curr_hist)
    # Use base=2 so JS divergence ∈ [0,1]; return JS distance = sqrt(JS divergence)
    js_div = 0.5 * stats.entropy(ref_hist, m, base=2) + 0.5 * stats.entropy(curr_hist, m, base=2)
    return float(np.sqrt(js_div))


def multivariate_covariate_drift_metrics(
    reference_df: pd.DataFrame, current_df: pd.DataFrame, feature_columns: list[str]
) -> Dict[str, Any]:
    """
    Comprehensive multivariate drift detection over a set of **numeric** features.

    What: applies KS/Wasserstein/PSI/JS per feature and aggregates.
    Why: monitors broad and per-feature changes affecting model validity.
    Interpret: look at per-feature stats; aggregates (mean/max PSI, fraction KS-significant)
    summarize the overall degree of drift across the feature set.

    Args:
        reference_df: Reference period data.
        current_df: Current period data.
        feature_columns: List of feature columns to analyze.
    """
    results: Dict[str, Any] = {"per_feature": {}, "aggregate": {}, "feature_columns": feature_columns}

    psi_values: list[float] = []
    ks_p_values: list[float] = []
    wasserstein_distances: list[float] = []

    analyzed_features = []

    for feature in feature_columns:
        if feature not in reference_df.columns or feature not in current_df.columns:
            continue
        # Only analyze numeric dtypes
        if not (is_numeric_dtype(reference_df[feature]) and is_numeric_dtype(current_df[feature])):
            continue

        ref_data = reference_df[feature].dropna().values
        curr_data = current_df[feature].dropna().values
        if ref_data.size == 0 or curr_data.size == 0:
            continue

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
            "jensen_shannon_distance": js_div,
        }

        analyzed_features.append(feature)
        if not np.isnan(psi_result["psi"]):
            psi_values.append(psi_result["psi"])
        if not np.isnan(ks_result["p_value"]):
            ks_p_values.append(ks_result["p_value"])
        if not np.isnan(wd):
            wasserstein_distances.append(wd)

    results["aggregate"] = {
        "mean_psi": float(np.mean(psi_values)) if psi_values else np.nan,
        "max_psi": float(np.max(psi_values)) if psi_values else np.nan,
        "fraction_drifted_features_psi": float(np.mean([x > 0.2 for x in psi_values])) if psi_values else np.nan,
        "mean_wasserstein": float(np.mean(wasserstein_distances)) if wasserstein_distances else np.nan,
        "fraction_significant_ks": float(np.mean([x < 0.05 for x in ks_p_values])) if ks_p_values else np.nan,
        "n_features_analyzed": len(analyzed_features),
    }

    return results


# --------------------------------------------
# Prediction Drift Metrics
# --------------------------------------------

def prediction_drift_metrics(
    reference_predictions: np.ndarray, current_predictions: np.ndarray, quantiles: np.ndarray
) -> Dict[str, Any]:
    """
    Drift metrics on **predicted quantiles**.

    What: compares the distributions of predicted q-levels (per-quantile) over time windows.
    Why: detects model-behavior changes even if inputs or targets aren’t available.
    Interpret: per-quantile KS/psi/Wasserstein; aggregates summarize typical magnitude.

    Args:
        reference_predictions: Historical predictions (n_samples, n_quantiles).
        current_predictions: Current predictions (n_samples, n_quantiles).
        quantiles: Quantile levels.
    """
    reference = np.asarray(reference_predictions)
    current = np.asarray(current_predictions)
    quantiles = np.asarray(quantiles)

    if reference.shape[1] != len(quantiles) or current.shape[1] != len(quantiles):
        raise ValueError("Number of quantiles must match prediction dimensions")

    results: Dict[str, Any] = {"per_quantile": {}, "aggregate": {}, "quantiles": quantiles.tolist()}

    ks_statistics: list[float] = []
    wasserstein_distances: list[float] = []

    for i, q in enumerate(quantiles):
        ref_q = reference[:, i]
        curr_q = current[:, i]

        ks_result = kolmogorov_smirnov_test(ref_q, curr_q)
        wd = wasserstein_distance(ref_q, curr_q)
        psi_result = population_stability_index(ref_q, curr_q)

        results["per_quantile"][f"q_{q:.2f}"] = {
            "ks_statistic": ks_result["ks_statistic"],
            "ks_p_value": ks_result["p_value"],
            "wasserstein_distance": wd,
            "psi": psi_result["psi"],
            "psi_interpretation": psi_result["interpretation"],
        }

        if not np.isnan(ks_result["ks_statistic"]):
            ks_statistics.append(ks_result["ks_statistic"])
        if not np.isnan(wd):
            wasserstein_distances.append(wd)

    results["aggregate"] = {
        "mean_ks_statistic": float(np.mean(ks_statistics)) if ks_statistics else np.nan,
        "max_ks_statistic": float(np.max(ks_statistics)) if ks_statistics else np.nan,
        "mean_wasserstein": float(np.mean(wasserstein_distances)) if wasserstein_distances else np.nan,
        "max_wasserstein": float(np.max(wasserstein_distances)) if wasserstein_distances else np.nan,
    }
    return results


# --------------------------------------------
# Target Drift Metrics
# --------------------------------------------

def target_drift_metrics(reference_targets: np.ndarray, current_targets: np.ndarray) -> Dict[str, Any]:
    """
    Detect drift in the **target** distribution.

    What: compares distributions and moments; tests mean and variance differences.
    Why: target drift affects the validity of learned quantiles/intervals.
    Interpret: KS/PSI/JS/Wasserstein for shape; mean/std/skew/kurt shifts;
    t-test (means) and Levene’s (variances) p-values for significance.

    Args:
        reference_targets: Historical target values.
        current_targets: Current target values.
    """
    reference = np.asarray(reference_targets).flatten()
    current = np.asarray(current_targets).flatten()

    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]

    if len(reference) == 0 or len(current) == 0:
        return {"error": "Insufficient non-NaN data"}

    ks_result = kolmogorov_smirnov_test(reference, current)
    psi_result = population_stability_index(reference, current)

    wd = wasserstein_distance(reference, current)
    js_div = jensen_shannon_divergence(reference, current)

    ref_mean, curr_mean = np.mean(reference), np.mean(current)
    ref_std, curr_std = np.std(reference), np.std(current)
    ref_skew, curr_skew = stats.skew(reference), stats.skew(current)
    ref_kurt, curr_kurt = stats.kurtosis(reference), stats.kurtosis(current)  # excess kurtosis

    # Two-sample Welch t-test for means
    t_stat, t_p = stats.ttest_ind(reference, current, equal_var=False)

    # Levene's test for variance equality (robust to non-normality)
    levene_stat, levene_p = stats.levene(reference, current)

    return {
        "distribution_tests": {
            "ks_statistic": ks_result["ks_statistic"],
            "ks_p_value": ks_result["p_value"],
            "psi": psi_result["psi"],
            "psi_interpretation": psi_result["interpretation"],
        },
        "distance_metrics": {"wasserstein_distance": wd, "jensen_shannon_distance": js_div},
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
            "kurtosis_shift": float(curr_kurt - ref_kurt),
        },
        "statistical_tests": {
            "mean_diff_t_stat": float(t_stat),
            "mean_diff_p_value": float(t_p),
            "variance_levene_stat": float(levene_stat),
            "variance_levene_p_value": float(levene_p),
        },
        "sample_sizes": {"reference_n": len(reference), "current_n": len(current)},
    }


# ============================================
# MODEL PERFORMANCE MONITORING
# ============================================

# --------------------------------------------
# Predicted Quantiles Metrics
# --------------------------------------------

def quantile_regression_performance_metrics(
    y_true: np.ndarray, y_pred_quantiles: np.ndarray, quantiles: np.ndarray
) -> Dict[str, Any]:
    """
    Performance metrics for **quantile regression** predictions.

    What:
      - Pinball loss per q (lower is better).
      - Coverage per q vs nominal q (calibration).
      - CRPS from quantiles (lower is better; requires reasonably dense grid).
      - PIT-based calibration (uniformity via KS / ECE).
      - Monotonicity violations (non-crossing).

    Why: captures accuracy, calibration, and distributional fidelity of quantile forecasts.
    Interpret: near-zero pinball/CRPS and small |coverage − q| are good; PIT KS/ECE near 0
    indicates good calibration; violation rate should be ~0 after enforcing monotonicity.

    Args:
        y_true: True target values (n,).
        y_pred_quantiles: Predicted quantiles (n, n_quantiles).
        quantiles: Quantile levels (n_quantiles,).
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

    pinball_losses = pinball_loss_vectorized(y_true_clean, y_pred_clean, quantiles)
    coverages = quantile_coverage(y_true_clean, y_pred_clean, quantiles)
    coverage_errors = coverages - quantiles

    # CRPS (approx. from quantiles; requires a reasonably dense grid)
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

    from stock_market_analytics.modeling.model_factory.evaluation.evaluation_functions import (
        monotonicity_violation_rate,
    )
    mono_violation_rate, mono_violation_count = monotonicity_violation_rate(y_pred_clean)

    return {
        "pinball_losses": {
            "per_quantile": {f"q_{q:.2f}": float(loss) for q, loss in zip(quantiles, pinball_losses)},
            "mean": float(np.mean(pinball_losses)),
            "median": float(np.median(pinball_losses)),
            "max": float(np.max(pinball_losses)),
        },
        "coverage": {
            "per_quantile": {f"q_{q:.2f}": float(cov) for q, cov in zip(quantiles, coverages)},
            "errors": {f"q_{q:.2f}": float(err) for q, err in zip(quantiles, coverage_errors)},
            "mean_absolute_error": float(np.mean(np.abs(coverage_errors))),
            "max_absolute_error": float(np.max(np.abs(coverage_errors))),
            "bias": float(np.mean(coverage_errors)),  # negative = under-coverage
        },
        "calibration": {
            "pit_ks_statistic": float(pit_ks),
            "pit_ece": float(pit_ece_score),
            "pit_values": pit.tolist() if len(pit) <= 1000 else [],
        },
        "distributional": {"crps": float(crps)},
        "monotonicity": {
            "violation_rate": float(mono_violation_rate),
            "violation_count": int(mono_violation_count),
            "total_samples": int(len(y_pred_clean)),
        },
        "sample_info": {
            "n_valid_samples": int(len(y_true_clean)),
            "n_total_samples": int(len(y_true)),
            "n_nan_removed": int(len(y_true) - len(y_true_clean)),
        },
    }


def quantile_performance_trends(
    y_true: np.ndarray,
    y_pred_quantiles: np.ndarray,
    quantiles: np.ndarray,
    dates: np.ndarray,
    window_size: int = 30,
) -> Dict[str, Any]:
    """
    Rolling performance trends for quantile regression.

    What: computes rolling mean pinball loss, |coverage error|, and CRPS.
    Why: monitors degradation/regime changes over time.
    Interpret: sustained upward trends signal accuracy/calibration deterioration.

    Args:
        y_true: True target values.
        y_pred_quantiles: Predicted quantiles (n_samples, n_quantiles).
        quantiles: Quantile levels.
        dates: Date index for time series (any datetime-like array).
        window_size: Rolling window length (number of samples).
    """
    y_true = np.asarray(y_true).flatten()
    y_pred_quantiles = np.asarray(y_pred_quantiles)
    quantiles = np.asarray(quantiles)
    dates = pd.to_datetime(dates).values

    df = pd.DataFrame({"date": dates, "y_true": y_true})
    for i, q in enumerate(quantiles):
        df[f"q_{q:.2f}"] = y_pred_quantiles[:, i]

    df = df.sort_values("date").dropna()

    if len(df) < window_size:
        return {"error": f"Insufficient data for window size {window_size}"}

    results: Dict[str, Any] = {
        "dates": [],
        "metrics": {"mean_pinball_loss": [], "mean_coverage_error": [], "crps": []},
        "window_size": window_size,
    }

    quantile_cols = [f"q_{q:.2f}" for q in quantiles]

    for i in range(window_size - 1, len(df)):
        window_df = df.iloc[i - window_size + 1 : i + 1]
        y_true_window = window_df["y_true"].values
        y_pred_window = window_df[quantile_cols].values

        try:
            pinball_losses = pinball_loss_vectorized(y_true_window, y_pred_window, quantiles)
            coverages = quantile_coverage(y_true_window, y_pred_window, quantiles)
            coverage_errors = coverages - quantiles

            try:
                crps = crps_from_quantiles(y_true_window, y_pred_window, quantiles)
            except Exception:
                crps = np.nan

            results["dates"].append(window_df.iloc[-1]["date"])
            results["metrics"]["mean_pinball_loss"].append(float(np.mean(pinball_losses)))
            results["metrics"]["mean_coverage_error"].append(float(np.mean(np.abs(coverage_errors))))
            results["metrics"]["crps"].append(float(crps))
        except Exception:
            continue

    return results


# --------------------------------------------
# Calibration Metrics (interval of lower and upper quantiles)
# --------------------------------------------

def prediction_interval_performance_metrics(
    y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray, confidence_level: float = 0.8
) -> Dict[str, Any]:
    """
    Performance metrics for **prediction intervals** (PI).

    What:
      - Empirical coverage vs target (miscoverage split below/above).
      - Width (mean and normalized), plus Winkler/interval score (lower is better).
      - Efficiency ratio = coverage / normalized width.

    Why: balances **calibration** (hit rate) and **sharpness** (narrowness).
    Interpret: observed coverage ≈ target with minimal width and low interval score is ideal;
    below/above split indicates under- vs over-shooting behavior.

    Args:
        y_true: True target values.
        y_lower: Lower bounds of prediction intervals.
        y_upper: Upper bounds of prediction intervals.
        confidence_level: Target confidence level (e.g., 0.8 for 80% intervals).
    """
    y_true = np.asarray(y_true).flatten()
    y_lower = np.asarray(y_lower).flatten()
    y_upper = np.asarray(y_upper).flatten()

    mask = ~(np.isnan(y_true) | np.isnan(y_lower) | np.isnan(y_upper))
    y_true_clean = y_true[mask]
    y_lower_clean = y_lower[mask]
    y_upper_clean = y_upper[mask]

    if len(y_true_clean) == 0:
        return {"error": "No valid data after removing NaNs"}

    coverage = prediction_interval_coverage_probability(y_true_clean, y_lower_clean, y_upper_clean)
    mean_width = mean_interval_width(y_lower_clean, y_upper_clean)
    normalized_width = normalized_interval_width(y_lower_clean, y_upper_clean, y_true_clean)

    alpha = 1.0 - confidence_level
    interval_score_value = interval_score(y_true_clean, y_lower_clean, y_upper_clean, alpha)

    coverage_error = coverage - confidence_level
    below_lower = np.mean(y_true_clean < y_lower_clean)
    above_upper = np.mean(y_true_clean > y_upper_clean)

    widths = y_upper_clean - y_lower_clean
    width_stats = {
        "min": float(np.min(widths)),
        "max": float(np.max(widths)),
        "mean": float(np.mean(widths)),
        "median": float(np.median(widths)),
        "std": float(np.std(widths)),
        "q25": float(np.percentile(widths, 25)),
        "q75": float(np.percentile(widths, 75)),
    }

    efficiency_ratio = coverage / normalized_width if normalized_width > 0 else np.nan

    return {
        "coverage": {
            "observed": float(coverage),
            "target": float(confidence_level),
            "error": float(coverage_error),
            "below_lower_rate": float(below_lower),
            "above_upper_rate": float(above_upper),
        },
        "interval_width": {
            "mean_width": float(mean_width),
            "normalized_width": float(normalized_width),
            "width_statistics": width_stats,
        },
        "scoring": {"interval_score": float(interval_score_value), "efficiency_ratio": float(efficiency_ratio)},
        "sample_info": {
            "n_valid_samples": int(len(y_true_clean)),
            "n_total_samples": int(len(y_true)),
            "n_nan_removed": int(len(y_true) - len(y_true_clean)),
        },
    }


def interval_calibration_curve(
    y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray, n_bins: int = 10
) -> Dict[str, Any]:
    """
    Calibration-by-width curve for prediction intervals.

    What: bins intervals by their **width** and reports observed coverage per bin.
    Why: reveals **conditional coverage** (e.g., narrow intervals under-cover).
    Interpret: observed coverage should be ~constant near the target across widths;
    strong slope indicates miscalibration tied to interval width.

    Args:
        y_true: True target values.
        y_lower: Lower bounds of prediction intervals.
        y_upper: Upper bounds of prediction intervals.
        n_bins: Number of bins for the calibration curve.
    """
    y_true = np.asarray(y_true).flatten()
    y_lower = np.asarray(y_lower).flatten()
    y_upper = np.asarray(y_upper).flatten()

    mask = ~(np.isnan(y_true) | np.isnan(y_lower) | np.isnan(y_upper))
    y_true_clean = y_true[mask]
    y_lower_clean = y_lower[mask]
    y_upper_clean = y_upper[mask]

    if len(y_true_clean) == 0:
        return {"error": "No valid data after removing NaNs"}

    widths = y_upper_clean - y_lower_clean
    covered = (y_true_clean >= y_lower_clean) & (y_true_clean <= y_upper_clean)

    bin_edges = np.percentile(widths, np.linspace(0, 100, n_bins + 1))
    bin_edges[0] = bin_edges[0] - 1e-10
    bin_edges[-1] = bin_edges[-1] + 1e-10

    bin_indices = np.digitize(widths, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    calibration_data: Dict[str, Any] = {"bin_centers": [], "observed_coverage": [], "bin_counts": [], "mean_widths": []}

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
    Sharpness metrics for prediction intervals (narrower is better, if coverage holds).

    What: summarizes the distribution of interval widths.
    Why: complements coverage — calibrated but overly wide intervals are unhelpful.
    Interpret: lower mean/median/IQR and 90th percentile are sharper; monitor CV for stability.

    Args:
        y_lower: Lower bounds of prediction intervals.
        y_upper: Upper bounds of prediction intervals.
    """
    y_lower = np.asarray(y_lower).flatten()
    y_upper = np.asarray(y_upper).flatten()

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
        "n_samples": int(len(y_lower_clean)),
    }


# --------------------------------------------
# Optional: simple point-median metrics (recommended addition)
# --------------------------------------------

def median_point_metrics(y_true: np.ndarray, y_pred_quantiles: np.ndarray, quantiles: np.ndarray) -> Dict[str, float]:
    """
    Point-error sanity checks using the median (q=0.5) prediction.

    What: MAE, RMSE, and bias of q=0.5 vs y_true.
    Why: quick central-tendency check alongside full quantile metrics.
    Interpret: lower is better; bias near 0 indicates no systematic over/underestimation.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred_quantiles = np.asarray(y_pred_quantiles)
    quantiles = np.asarray(quantiles)

    if y_pred_quantiles.ndim != 2 or quantiles.ndim != 1:
        return {"mae": np.nan, "rmse": np.nan, "bias": np.nan}

    # Find closest quantile to 0.5
    idx = int(np.argmin(np.abs(quantiles - 0.5)))
    y_pred_med = y_pred_quantiles[:, idx].flatten()

    mask = ~(np.isnan(y_true) | np.isnan(y_pred_med))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred_med[mask]
    if y_true_clean.size == 0:
        return {"mae": np.nan, "rmse": np.nan, "bias": np.nan}

    err = y_pred_clean - y_true_clean
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    bias = float(np.mean(err))
    return {"mae": mae, "rmse": rmse, "bias": bias}
