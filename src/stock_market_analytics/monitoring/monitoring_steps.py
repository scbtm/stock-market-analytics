import base64
import io
import os
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import wandb
from stock_market_analytics.config import config
from stock_market_analytics.modeling.model_factory.data_management.preprocessing import (
    _validate,
)
from stock_market_analytics.monitoring.monitoring_metrics import (
    multivariate_covariate_drift_metrics,
    prediction_drift_metrics,
    prediction_interval_performance_metrics,
    quantile_performance_trends,
    quantile_regression_performance_metrics,
)

# Centralized color palette configuration (ggplot-friendly, color-blind safe)
MONITORING_COLORS = {
    # Core hues
    "primary": "#0072B2",  # deep blue
    "secondary": "#56B4E9",  # sky blue
    "success": "#009E73",  # bluish green (fixed: no RGBA)
    "warning": "#E69F00",  # amber
    "neutral": "#777777",  # ggplot-ish gray
    # Soft tints (fills/bands)
    "light_blue": "#A6D8F0",
    "light_green": "#8CD7C7",
    "light_orange": "#F6D18A",
    "light_red": "#F4A79E",
    "light_gray": "#B0B0B0",
    # Thresholds for drift detection (traffic-light)
    "no_drift": "#009E73",  # Green
    "minor_drift": "#F0E442",  # Yellow
    "major_drift": "#D55E00",  # Vermillion/Red
    # Performance colors
    "good_performance": "#009E73",
    "medium_performance": "#E69F00",
    "poor_performance": "#D55E00",
    # Helpers
    "baseline": "#777777",  # reference lines / targets
    "accent": "#CC79A7",  # optional highlight
}


def download_artifacts() -> tuple[str, str, str, str]:
    """
    Download model artifacts from Weights & Biases.
    """
    api = wandb.Api(api_key=os.environ.get("WANDB_API_KEY"))
    model_name: str = os.environ.get("MODEL_NAME") or "pipeline:latest"
    model_version: str = model_name.split(":")[1] if ":" in model_name else "latest"
    dataset_name: str = "test_set:" + model_version

    # Reference the artifact by entity/project/name:version or :latest
    model = api.artifact(f"san-cbtm/stock-market-analytics/{model_name}", type="model")
    dataset = api.artifact(
        f"san-cbtm/stock-market-analytics/{dataset_name}", type="dataset"
    )

    # Download to a specific folder (defaults to a temp dir if omitted)
    model_dir = model.download()
    dataset_dir = dataset.download()

    return model_dir, model_name, dataset_dir, dataset_name  # type: ignore


def load_model(model_dir: str, model_name: str) -> object:
    """
    Load the model from the downloaded artifact directory.
    """
    file_name = model_name.split(":")[0] + ".pkl"
    model = joblib.load(f"{model_dir}/{file_name}")
    return model


def load_reference_data(dataset_dir: str, dataset_name: str) -> pd.DataFrame:
    """
    Load the reference features data for drift detection.
    """
    file_name = dataset_name.split(":")[0] + ".parquet"
    dataset = pd.read_parquet(f"{dataset_dir}/{file_name}")
    return dataset


def load_monitoring_df(data_path: str) -> pd.DataFrame:
    """
    Load the features data for monitoring.
    """
    features_file = Path(data_path) / config.modeling.features_file

    # load features data
    df = pd.read_parquet(features_file)
    df = _validate(df, date_col="date", symbol_col="symbol")
    # get fully labeled data
    df_labeled = df.dropna(subset=[config.modeling.target])

    # Filter to the last time span days of labeled data for performance monitoring
    max_date = df_labeled["date"].max()
    six_months_ago = max_date - pd.DateOffset(days=config.modeling.time_span)
    df_labeled = df_labeled[df_labeled["date"] >= six_months_ago]

    return df_labeled  # type: ignore


# DISTRIBUTION SHIFT DETECTION


def get_covariate_drift_metrics(
    reference_df: pd.DataFrame, current_df: pd.DataFrame, feature_columns: list[str]
) -> dict:
    """
    Calculate comprehensive covariate drift metrics for features.
    """
    return multivariate_covariate_drift_metrics(
        reference_df, current_df, feature_columns
    )


def get_prediction_drift_metrics(
    reference_predictions: pd.DataFrame,
    current_predictions: pd.DataFrame,
    quantiles: list[float],
) -> dict:
    """
    Calculate drift metrics for model predictions across quantiles.
    """
    # Extract quantile columns from DataFrames
    quantile_cols = [f"q_{q:.2f}" for q in quantiles]
    ref_pred_array = reference_predictions[quantile_cols].to_numpy()
    curr_pred_array = current_predictions[quantile_cols].to_numpy()

    return prediction_drift_metrics(ref_pred_array, curr_pred_array, quantiles)


def get_target_drift_metrics(
    reference_targets: pd.Series, current_targets: pd.Series
) -> dict:
    """
    Calculate drift metrics for target variable distribution.
    """
    from stock_market_analytics.monitoring.monitoring_metrics import (
        target_drift_metrics,
    )

    return target_drift_metrics(
        reference_targets.to_numpy(), current_targets.to_numpy()
    )


# MODEL PERFORMANCE MONITORING


def get_predicted_quantiles_metrics(
    y_true: pd.Series, y_pred_quantiles: pd.DataFrame, quantiles: list[float]
) -> dict:
    """
    Calculate comprehensive performance metrics for quantile regression models.
    """
    quantile_cols = [f"q_{q:.2f}" for q in quantiles]
    pred_array = y_pred_quantiles[quantile_cols].to_numpy()
    return quantile_regression_performance_metrics(
        y_true.to_numpy(), pred_array, quantiles
    )


def get_calibration_metrics(
    y_true: pd.Series,
    y_lower: pd.Series,
    y_upper: pd.Series,
    confidence_level: float = 0.8,
) -> dict:
    """
    Calculate comprehensive metrics for prediction interval performance.
    """
    return prediction_interval_performance_metrics(
        y_true.to_numpy(), y_lower.to_numpy(), y_upper.to_numpy(), confidence_level
    )


def get_performance_trends(
    y_true: pd.Series,
    y_pred_quantiles: pd.DataFrame,
    dates: pd.Series,
    quantiles: list[float],
    window_size: int = 30,
) -> dict:
    """
    Track performance trends over time using rolling windows.
    """
    quantile_cols = [f"q_{q:.2f}" for q in quantiles]
    pred_array = y_pred_quantiles[quantile_cols].to_numpy()
    return quantile_performance_trends(
        y_true.to_numpy(), pred_array, quantiles, dates.to_numpy(), window_size
    )


# =========================
# PLOTS & REPORTING HELPERS
# =========================


def _fig_to_base64(fig) -> str:  # type: ignore
    """Encode a Matplotlib figure as base64 PNG (for self-contained HTML)."""
    buff = io.BytesIO()
    fig.savefig(buff, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    buff.seek(0)
    return base64.b64encode(buff.read()).decode("ascii")


def _render_figure(fig, save_path: str | None, return_image: bool) -> str | None:  # type: ignore
    """Save/show/return-image without changing prior behavior."""
    img_b64 = None
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if return_image:
        img_b64 = _fig_to_base64(fig)
    else:
        plt.show()
        plt.close(fig)
    return img_b64


# =========================
# PLOTS
# =========================


def plot_drift_metrics(
    drift_results: dict,
    save_path: str | None = None,
    figsize: tuple = (15, 10),
    return_image: bool = False,
) -> str | None:
    """
    Create visualizations for drift detection results.

    If return_image=True, returns a base64-encoded PNG string; otherwise shows/saves the figure.
    """
    if "per_feature" in drift_results:
        return _plot_covariate_drift(drift_results, save_path, figsize, return_image)
    elif "per_quantile" in drift_results:
        return _plot_prediction_drift(drift_results, save_path, figsize, return_image)
    elif "distribution_tests" in drift_results:
        return _plot_target_drift(drift_results, save_path, figsize, return_image)
    else:
        raise ValueError("Unknown drift results format")


def _plot_covariate_drift(
    drift_results: dict,
    save_path: str | None = None,
    figsize: tuple = (15, 10),
    return_image: bool = False,
) -> str | None:
    """Plot covariate drift metrics."""
    features = list(drift_results["per_feature"].keys())
    if not features:
        print("No features to plot")
        return None

    n_features = len(features)
    min_height_per_feature = 0.4
    dynamic_height = max(figsize[1], n_features * min_height_per_feature + 4)
    adjusted_figsize = (figsize[0], dynamic_height)

    if n_features > 20:
        feature_psi_pairs = [
            (f, drift_results["per_feature"][f]["psi"]) for f in features
        ]
        feature_psi_pairs.sort(
            key=lambda x: (x[1] if x[1] is not None else -np.inf), reverse=True
        )
        top_features = [f for f, _ in feature_psi_pairs[:20]]
        print(
            f"Showing top 20 features out of {n_features} total features (sorted by PSI)"
        )
        features = top_features

    psi_values = [drift_results["per_feature"][f]["psi"] for f in features]
    ks_stats = [drift_results["per_feature"][f]["ks_statistic"] for f in features]
    wd_values = [
        drift_results["per_feature"][f]["wasserstein_distance"] for f in features
    ]
    js_values = [
        drift_results["per_feature"][f].get("jensen_shannon_distance", np.nan)
        for f in features
    ]

    fig, axes = plt.subplots(2, 2, figsize=adjusted_figsize, constrained_layout=True)
    fig.suptitle("Covariate Drift Detection Results", fontsize=16, fontweight="bold")

    label_fontsize = max(8, min(12, 200 // max(1, len(features))))

    # PSI plot with traffic-light colors
    colors = [
        MONITORING_COLORS["no_drift"]
        if (x is not None and x < 0.1)
        else MONITORING_COLORS["minor_drift"]
        if (x is not None and x < 0.2)
        else MONITORING_COLORS["major_drift"]
        for x in psi_values
    ]
    axes[0, 0].barh(features, psi_values, color=colors, height=0.8)
    axes[0, 0].axvline(
        x=0.1,
        color=MONITORING_COLORS["minor_drift"],
        linestyle="--",
        alpha=0.7,
        label="Minor drift",
    )
    axes[0, 0].axvline(
        x=0.2,
        color=MONITORING_COLORS["major_drift"],
        linestyle="--",
        alpha=0.7,
        label="Major drift",
    )
    axes[0, 0].set_xlabel("PSI")
    axes[0, 0].set_title("Population Stability Index (PSI)")
    axes[0, 0].tick_params(axis="y", labelsize=label_fontsize)
    axes[0, 0].grid(axis="x", alpha=0.2)
    axes[0, 0].legend()

    # KS statistic
    axes[0, 1].barh(
        features, ks_stats, color=MONITORING_COLORS["secondary"], height=0.8
    )
    axes[0, 1].set_xlabel("KS Statistic")
    axes[0, 1].set_title("KS Statistic by Feature")
    axes[0, 1].tick_params(axis="y", labelsize=label_fontsize)
    axes[0, 1].grid(axis="x", alpha=0.2)

    # Wasserstein distance
    axes[1, 0].barh(features, wd_values, color=MONITORING_COLORS["primary"], height=0.8)
    axes[1, 0].set_xlabel("Wasserstein Distance")
    axes[1, 0].set_title("Wasserstein Distance by Feature")
    axes[1, 0].tick_params(axis="y", labelsize=label_fontsize)
    axes[1, 0].grid(axis="x", alpha=0.2)

    # Summary text (also includes JS distance summary)
    agg = drift_results["aggregate"]
    try:
        mean_js = float(np.nanmean(js_values))
        max_js = float(np.nanmax(js_values))
    except Exception:
        mean_js, max_js = np.nan, np.nan

    summary_text = (
        f"Aggregate Metrics:\n"
        f"Mean PSI: {agg.get('mean_psi', np.nan):.3f}\n"
        f"Max PSI: {agg.get('max_psi', np.nan):.3f}\n"
        f"Drifted Features (PSI>0.2): {agg.get('fraction_drifted_features_psi', 0):.1%}\n"
        f"Mean Wasserstein: {agg.get('mean_wasserstein', np.nan):.3f}\n"
        f"Mean JS distance: {mean_js:.3f} | Max JS: {max_js:.3f}\n"
        f"Features Analyzed: {agg.get('n_features_analyzed', 0)}"
    )
    axes[1, 1].text(
        0.05,
        0.5,
        summary_text,
        transform=axes[1, 1].transAxes,
        fontsize=11,
        va="center",
        family="monospace",
    )
    axes[1, 1].axis("off")

    return _render_figure(fig, save_path, return_image)


def _plot_prediction_drift(
    drift_results: dict,
    save_path: str | None = None,
    figsize: tuple = (12, 8),
    return_image: bool = False,
) -> str | None:
    """Plot prediction drift metrics."""
    quantiles = drift_results["quantiles"]
    per_quantile = drift_results["per_quantile"]

    ks_stats = [per_quantile[f"q_{q:.2f}"]["ks_statistic"] for q in quantiles]
    wd_values = [per_quantile[f"q_{q:.2f}"]["wasserstein_distance"] for q in quantiles]
    psi_values = [per_quantile[f"q_{q:.2f}"]["psi"] for q in quantiles]

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    fig.suptitle("Prediction Drift Across Quantiles", fontsize=16, fontweight="bold")

    axes[0].plot(
        quantiles,
        ks_stats,
        "o-",
        color=MONITORING_COLORS["primary"],
        linewidth=2,
        markersize=6,
    )
    axes[0].set_xlabel("Quantile Level")
    axes[0].set_ylabel("KS Statistic")
    axes[0].set_title("KS by Quantile")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        quantiles,
        wd_values,
        "o-",
        color=MONITORING_COLORS["secondary"],
        linewidth=2,
        markersize=6,
    )
    axes[1].set_xlabel("Quantile Level")
    axes[1].set_ylabel("Wasserstein Distance")
    axes[1].set_title("Wasserstein by Quantile")
    axes[1].grid(True, alpha=0.3)

    colors = [
        MONITORING_COLORS["no_drift"]
        if x < 0.1
        else MONITORING_COLORS["minor_drift"]
        if x < 0.2
        else MONITORING_COLORS["major_drift"]
        for x in psi_values
    ]
    axes[2].bar([f"{q:.2f}" for q in quantiles], psi_values, color=colors)
    axes[2].axhline(
        y=0.1,
        color=MONITORING_COLORS["minor_drift"],
        linestyle="--",
        alpha=0.7,
        label="0.10",
    )
    axes[2].axhline(
        y=0.2,
        color=MONITORING_COLORS["major_drift"],
        linestyle="--",
        alpha=0.7,
        label="0.20",
    )
    axes[2].set_xlabel("Quantile Level")
    axes[2].set_ylabel("PSI")
    axes[2].set_title("PSI by Quantile")
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].legend()

    return _render_figure(fig, save_path, return_image)


def _plot_target_drift(
    drift_results: dict,
    save_path: str | None = None,
    figsize: tuple = (15, 10),
    return_image: bool = False,
) -> str | None:
    """Plot target drift metrics."""
    fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)
    fig.suptitle("Target Distribution Drift Analysis", fontsize=16, fontweight="bold")

    dist_tests = drift_results["distribution_tests"]
    moments = drift_results["moments_comparison"]
    distance_metrics = drift_results["distance_metrics"]
    stat_tests = drift_results["statistical_tests"]
    sample_info = drift_results["sample_sizes"]

    # Distribution test statistics
    axes[0, 0].bar(
        ["KS Stat", "PSI"],
        [dist_tests["ks_statistic"], dist_tests["psi"]],
        color=[MONITORING_COLORS["primary"], MONITORING_COLORS["warning"]],
    )
    axes[0, 0].set_title("Distribution Tests")
    axes[0, 0].grid(axis="y", alpha=0.2)

    # Moments comparison
    metrics = ["Mean", "Std", "Skew", "Kurtosis"]
    ref_vals = [
        moments["ref_mean"],
        moments["ref_std"],
        moments["ref_skewness"],
        moments["ref_kurtosis"],
    ]
    cur_vals = [
        moments["curr_mean"],
        moments["curr_std"],
        moments["curr_skewness"],
        moments["curr_kurtosis"],
    ]
    x = np.arange(len(metrics))
    width = 0.35
    axes[0, 1].bar(
        x - width / 2,
        ref_vals,
        width,
        label="Ref",
        color=MONITORING_COLORS["light_blue"],
    )
    axes[0, 1].bar(
        x + width / 2,
        cur_vals,
        width,
        label="Cur",
        color=MONITORING_COLORS["light_green"],
    )
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(metrics)
    axes[0, 1].set_title("Moments Comparison")
    axes[0, 1].legend()
    axes[0, 1].grid(axis="y", alpha=0.2)

    # Distances
    axes[0, 2].bar(
        ["Wasserstein", "JS dist"],
        [
            distance_metrics["wasserstein_distance"],
            distance_metrics["jensen_shannon_distance"],
        ],
        color=[MONITORING_COLORS["primary"], MONITORING_COLORS["secondary"]],
    )
    axes[0, 2].set_title("Distance Metrics")
    axes[0, 2].grid(axis="y", alpha=0.2)

    # P-values
    p_values = [stat_tests["mean_diff_p_value"], stat_tests["variance_levene_p_value"]]
    colors = [
        MONITORING_COLORS["no_drift"] if p > 0.05 else MONITORING_COLORS["major_drift"]
        for p in p_values
    ]
    axes[1, 0].bar(["t-test p", "Levene p"], p_values, color=colors)
    axes[1, 0].axhline(
        y=0.05,
        color=MONITORING_COLORS["major_drift"],
        linestyle="--",
        alpha=0.7,
        label="α=0.05",
    )
    axes[1, 0].set_title("Hypothesis Tests")
    axes[1, 0].legend()
    axes[1, 0].grid(axis="y", alpha=0.2)

    # Sample sizes
    axes[1, 1].bar(
        ["Reference", "Current"],
        [sample_info["reference_n"], sample_info["current_n"]],
        color=MONITORING_COLORS["light_gray"],
    )
    axes[1, 1].set_title("Sample Sizes")
    axes[1, 1].grid(axis="y", alpha=0.2)

    # Summary text
    summary_text = (
        f"KS p-value: {dist_tests.get('ks_p_value', np.nan):.4f}\n"
        f"PSI: {dist_tests.get('psi', np.nan):.4f} ({dist_tests.get('psi_interpretation', 'N/A')})\n\n"
        f"Mean Shift: {moments.get('mean_shift', np.nan):.4f}\n"
        f"Std Ratio: {moments.get('std_ratio', np.nan):.4f}"
    )
    axes[1, 2].text(
        0.05,
        0.5,
        summary_text,
        transform=axes[1, 2].transAxes,
        fontsize=10,
        va="center",
        family="monospace",
    )
    axes[1, 2].axis("off")

    return _render_figure(fig, save_path, return_image)


def plot_performance_metrics(
    performance_results: dict,
    save_path: str | None = None,
    figsize: tuple = (15, 12),
    return_image: bool = False,
) -> str | None:
    """
    Visualizations for model performance metrics.

    If return_image=True, returns base64-encoded PNG; else shows/saves figure.
    """
    if "pinball_losses" in performance_results:
        return _plot_quantile_performance(
            performance_results, save_path, figsize, return_image
        )
    elif "coverage" in performance_results and "interval_width" in performance_results:
        return _plot_interval_performance(
            performance_results, save_path, figsize, return_image
        )
    elif "dates" in performance_results and "metrics" in performance_results:
        return _plot_performance_trends(
            performance_results, save_path, figsize, return_image
        )
    else:
        raise ValueError("Unknown performance results format")


def _plot_quantile_performance(
    performance_results: dict,
    save_path: str | None = None,
    figsize: tuple = (15, 12),
    return_image: bool = False,
) -> str | None:
    """Plot quantile regression performance metrics with reliability band."""
    fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)
    fig.suptitle(
        "Quantile Regression Performance Metrics", fontsize=16, fontweight="bold"
    )

    pinball_losses = performance_results["pinball_losses"]["per_quantile"]
    coverage_metrics = performance_results["coverage"]["per_quantile"]
    coverage_errors = performance_results["coverage"]["errors"]
    n_valid = performance_results.get("sample_info", {}).get("n_valid_samples", None)

    q_labels = list(pinball_losses.keys())
    q_vals = [float(q.split("_")[1]) for q in q_labels]

    # Pinball losses
    losses = list(pinball_losses.values())
    axes[0, 0].bar(q_labels, losses, color=MONITORING_COLORS["primary"])
    axes[0, 0].set_title("Pinball Loss by Quantile")
    axes[0, 0].set_xlabel("Quantile")
    axes[0, 0].set_ylabel("Pinball Loss")
    axes[0, 0].tick_params(axis="x", rotation=45)
    axes[0, 0].grid(axis="y", alpha=0.2)

    # Coverage vs nominal with 95% binomial band (normal approx)
    covs = np.array(list(coverage_metrics.values()))
    axes[0, 1].plot(
        q_vals,
        covs,
        "o-",
        color=MONITORING_COLORS["light_green"],
        linewidth=2,
        markersize=6,
        label="Observed",
    )
    axes[0, 1].plot(
        q_vals,
        q_vals,
        "--",
        color=MONITORING_COLORS["neutral"],
        linewidth=2,
        label="Target y=x",
    )
    if (n_valid is not None) and (n_valid > 0):
        se = np.sqrt(
            np.clip(np.array(q_vals) * (1 - np.array(q_vals)) / n_valid, 0, None)
        )
        upper = np.clip(np.array(q_vals) + 1.96 * se, 0, 1)
        lower = np.clip(np.array(q_vals) - 1.96 * se, 0, 1)
        axes[0, 1].fill_between(
            q_vals, lower, upper, alpha=0.15, label="95% sampling band"
        )
    axes[0, 1].set_title("Coverage by Quantile")
    axes[0, 1].set_xlabel("Quantile Level")
    axes[0, 1].set_ylabel("Coverage")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Coverage errors with thresholds
    errs = list(coverage_errors.values())
    colors = [
        MONITORING_COLORS["no_drift"]
        if abs(e) < 0.05
        else MONITORING_COLORS["minor_drift"]
        if abs(e) < 0.1
        else MONITORING_COLORS["major_drift"]
        for e in errs
    ]
    axes[0, 2].bar(q_labels, errs, color=colors)
    axes[0, 2].axhline(
        y=0, color=MONITORING_COLORS["baseline"], linestyle="-", alpha=0.6
    )
    axes[0, 2].axhline(
        y=0.05, color=MONITORING_COLORS["minor_drift"], linestyle="--", alpha=0.7
    )
    axes[0, 2].axhline(
        y=-0.05, color=MONITORING_COLORS["minor_drift"], linestyle="--", alpha=0.7
    )
    axes[0, 2].set_title("Coverage Errors")
    axes[0, 2].set_xlabel("Quantile")
    axes[0, 2].set_ylabel("Error")
    axes[0, 2].tick_params(axis="x", rotation=45)
    axes[0, 2].grid(axis="y", alpha=0.2)

    # PIT histogram
    pit_vals = performance_results["calibration"]["pit_values"]
    if pit_vals:
        axes[1, 0].hist(
            pit_vals,
            bins=20,
            density=True,
            alpha=0.75,
            color=MONITORING_COLORS["light_gray"],
            edgecolor="black",
        )
        axes[1, 0].axhline(
            y=1.0, color=MONITORING_COLORS["baseline"], linestyle="--", label="Uniform"
        )
        axes[1, 0].set_title("PIT Histogram")
        axes[1, 0].set_xlabel("PIT")
        axes[1, 0].set_ylabel("Density")
        axes[1, 0].legend()
    else:
        axes[1, 0].text(
            0.5,
            0.5,
            "PIT values\nnot available",
            ha="center",
            va="center",
            transform=axes[1, 0].transAxes,
        )
        axes[1, 0].set_title("PIT Histogram")

    # Summary text
    summary_text = (
        f"Mean Pinball: {performance_results['pinball_losses']['mean']:.4f}\n"
        f"Mean |Coverage Error|: {performance_results['coverage']['mean_absolute_error']:.4f}\n"
        f"Coverage Bias: {performance_results['coverage']['bias']:.4f}\n"
        f"CRPS: {performance_results['distributional']['crps']:.4f}\n\n"
        f"PIT KS: {performance_results['calibration']['pit_ks_statistic']:.4f}\n"
        f"PIT ECE: {performance_results['calibration']['pit_ece']:.4f}\n\n"
        f"Monotonicity Violation Rate: {performance_results['monotonicity']['violation_rate']:.2%}\n"
        f"Valid Samples: {performance_results['sample_info']['n_valid_samples']}"
    )
    axes[1, 1].text(
        0.05,
        0.5,
        summary_text,
        transform=axes[1, 1].transAxes,
        fontsize=10,
        va="center",
        family="monospace",
    )
    axes[1, 1].axis("off")

    # Monotonicity: bar instead of pie
    mono_count = performance_results["monotonicity"]["violation_count"]
    total_samples = performance_results["monotonicity"]["total_samples"]
    valid_count = max(total_samples - mono_count, 0)
    axes[1, 2].bar(
        ["Valid", "Violations"],
        [valid_count, mono_count],
        color=[
            MONITORING_COLORS["good_performance"],
            MONITORING_COLORS["poor_performance"],
        ],
    )
    axes[1, 2].set_title("Non-crossing Check")
    axes[1, 2].set_ylabel("Count")
    axes[1, 2].grid(axis="y", alpha=0.2)

    return _render_figure(fig, save_path, return_image)


def _plot_interval_performance(
    performance_results: dict,
    save_path: str | None = None,
    figsize: tuple = (12, 8),
    return_image: bool = False,
) -> str | None:
    """Plot prediction interval performance metrics."""
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    fig.suptitle("Prediction Interval Performance", fontsize=16, fontweight="bold")

    coverage_info = performance_results["coverage"]
    width_info = performance_results["interval_width"]

    # Coverage vs target
    coverage_values = [coverage_info["target"], coverage_info["observed"]]
    colors = [
        MONITORING_COLORS["light_blue"],
        MONITORING_COLORS["light_green"]
        if abs(coverage_info["error"]) < 0.05
        else MONITORING_COLORS["light_red"],
    ]
    axes[0, 0].bar(["Target", "Observed"], coverage_values, color=colors)
    axes[0, 0].set_title(
        f"Coverage: {coverage_info['observed']:.3f} vs {coverage_info['target']:.3f}"
    )
    axes[0, 0].set_ylabel("Coverage")
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].grid(axis="y", alpha=0.2)

    # Width distribution
    ws = width_info["width_statistics"]
    axes[0, 1].bar(
        ["Min", "Q25", "Median", "Q75", "Max"],
        [ws["min"], ws["q25"], ws["median"], ws["q75"], ws["max"]],
        color=MONITORING_COLORS["light_orange"],
    )
    axes[0, 1].set_title("Interval Width Summary")
    axes[0, 1].set_ylabel("Width")
    axes[0, 1].grid(axis="y", alpha=0.2)

    # Coverage breakdown
    axes[1, 0].bar(
        ["Below Lower", "Covered", "Above Upper"],
        [
            coverage_info["below_lower_rate"],
            coverage_info["observed"],
            coverage_info["above_upper_rate"],
        ],
        color=[
            MONITORING_COLORS["light_red"],
            MONITORING_COLORS["light_green"],
            MONITORING_COLORS["light_orange"],
        ],
    )
    axes[1, 0].set_title("Coverage Breakdown")
    axes[1, 0].set_ylabel("Rate")
    axes[1, 0].grid(axis="y", alpha=0.2)

    # Summary text
    summary_text = (
        f"Mean Width: {width_info['mean_width']:.4f}\n"
        f"Normalized Width: {width_info['normalized_width']:.4f}\n"
        f"Interval Score: {performance_results['scoring']['interval_score']:.4f}\n"
        f"Efficiency Ratio: {performance_results['scoring']['efficiency_ratio']:.4f}\n\n"
        f"Coverage Error: {coverage_info['error']:.4f}\n"
        f"Below Lower: {coverage_info['below_lower_rate']:.3f} | Above Upper: {coverage_info['above_upper_rate']:.3f}\n\n"
        f"Valid Samples: {performance_results['sample_info']['n_valid_samples']}/"
        f"{performance_results['sample_info']['n_total_samples']}"
    )
    axes[1, 1].text(
        0.05,
        0.5,
        summary_text,
        transform=axes[1, 1].transAxes,
        fontsize=10,
        va="center",
        family="monospace",
    )
    axes[1, 1].axis("off")

    return _render_figure(fig, save_path, return_image)


def _plot_performance_trends(
    performance_results: dict,
    save_path: str | None = None,
    figsize: tuple = (15, 8),
    return_image: bool = False,
) -> str | None:
    """Plot performance trends over time."""
    dates = pd.to_datetime(performance_results["dates"])
    metrics = performance_results["metrics"]

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    fig.suptitle("Performance Trends Over Time", fontsize=16, fontweight="bold")

    axes[0].plot(
        dates,
        metrics["mean_pinball_loss"],
        "o-",
        color=MONITORING_COLORS["primary"],
        linewidth=2,
        markersize=4,
    )
    axes[0].set_title("Mean Pinball Loss")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Loss")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        dates,
        metrics["mean_coverage_error"],
        "o-",
        color=MONITORING_COLORS["secondary"],
        linewidth=2,
        markersize=4,
        label="|Err|",
    )
    axes[1].axhline(
        y=0.05,
        color=MONITORING_COLORS["minor_drift"],
        linestyle="--",
        alpha=0.7,
        label="5% threshold",
    )
    axes[1].axhline(
        y=-0.05, color=MONITORING_COLORS["minor_drift"], linestyle="--", alpha=0.7
    )
    axes[1].set_title("Mean Coverage Error")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Abs Error")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(
        dates,
        metrics["crps"],
        "o-",
        color=MONITORING_COLORS["good_performance"],
        linewidth=2,
        markersize=4,
    )
    axes[2].set_title("CRPS")
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("CRPS")
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].grid(True, alpha=0.3)

    return _render_figure(fig, save_path, return_image)


# =========================
# REPORTING
# =========================


def generate_monitoring_report(
    monitoring_results: dict,
    output_path: str | None = None,
    title: str = "Model Monitoring Report",
) -> str:
    """
    Generate a comprehensive **self-contained** HTML monitoring report.

    To embed plots inline, pass base64 images under any of these optional keys in `monitoring_results`:
      - "image_covariate_drift", "image_prediction_drift", "image_target_drift",
        "image_quantile_performance", "image_interval_performance", "image_trends"

    Example:
        img = _plot_quantile_performance(perf, return_image=True)
        monitoring_results["image_quantile_performance"] = img
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _img_html(key: str) -> str:
        b64 = monitoring_results.get(key)
        if not b64:
            return ""
        return f'<div><img alt="{key}" style="max-width:100%;height:auto" src="data:image/png;base64,{b64}"/></div>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; color: #222; }}
.header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
.section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #0072B2; background:#fafafa; border-radius:4px; }}
.metric {{ background-color: #fff; padding: 10px; margin: 5px 0; border-radius: 3px; border:1px solid #eee; }}
.warning {{ border-left-color: #E69F00; background-color: #fff9ef; }}
.error {{ border-left-color: #D55E00; background-color: #fff1ec; }}
.success {{ border-left-color: #009E73; background-color: #eefaf6; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 14px; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background-color: #f2f2f2; }}
.code {{ font-family: monospace; background-color: #f5f5f5; padding: 2px 4px; }}
h2 {{ margin-top: 0; }}
img {{ display:block; margin: 10px 0; }}
</style>
</head>
<body>
<div class="header">
  <h1>{title}</h1>
  <p>Generated on: {timestamp}</p>
</div>
"""

    if "drift_results" in monitoring_results:
        html += _generate_drift_section(monitoring_results["drift_results"])
        # inline images if provided
        html += _img_html("image_covariate_drift")
        html += _img_html("image_prediction_drift")
        html += _img_html("image_target_drift")

    if "performance_results" in monitoring_results:
        html += _generate_performance_section(monitoring_results["performance_results"])
        html += _img_html("image_quantile_performance")
        html += _img_html("image_interval_performance")

    if "trends_results" in monitoring_results:
        html += _generate_trends_section(monitoring_results["trends_results"])
        html += _img_html("image_trends")

    html += "\n</body>\n</html>\n"

    if output_path:
        with Path(output_path).open("w", encoding="utf-8") as f:
            f.write(html)
        print(f"Report saved to: {output_path}")
    return html


def _generate_drift_section(drift_results: dict) -> str:
    """Generate HTML section for drift results."""
    if "per_feature" in drift_results:
        # Covariate drift
        section_type = "Covariate Drift"
        features = drift_results["per_feature"]

        table_rows = ""
        for feature, metrics in features.items():
            psi_class = (
                "success"
                if metrics["psi"] < 0.1
                else "warning"
                if metrics["psi"] < 0.2
                else "error"
            )
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
        psi_class = (
            "success"
            if dist_tests["psi"] < 0.1
            else "warning"
            if dist_tests["psi"] < 0.2
            else "error"
        )

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
        coverage_bias_class = (
            "success"
            if abs(performance_results["coverage"]["bias"]) < 0.05
            else "warning"
        )
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
        coverage_error_class = (
            "success"
            if abs(performance_results["coverage"]["error"]) < 0.05
            else "warning"
        )
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
        recent_pinball = (
            trends_results["metrics"]["mean_pinball_loss"][-1]
            if trends_results["metrics"]["mean_pinball_loss"]
            else "N/A"
        )
        recent_coverage_error = (
            trends_results["metrics"]["mean_coverage_error"][-1]
            if trends_results["metrics"]["mean_coverage_error"]
            else "N/A"
        )
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
