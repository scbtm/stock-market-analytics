"""
Metrics and evaluation utilities for multi-quantile modeling.

This module contains all evaluation metrics, conformal prediction utilities,
and plotting functions extracted from the original modeling_functions.py.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from optuna import Study
from optuna.trial import TrialState
import optuna

# Re-export all the existing functions from modeling_functions
# This maintains backward compatibility while organizing the code better

def predict_quantiles(model, X: Union[pd.DataFrame, "Pool"]) -> np.ndarray:
    """
    Predict quantiles using a trained model.
    
    Args:
        model: Trained model with predict method
        X: Input features
        
    Returns:
        Quantile predictions array of shape (n_samples, n_quantiles)
    """
    from catboost import Pool
    
    is_df = isinstance(X, pd.DataFrame)

    if is_df:
        cat_idx = np.where((X.dtypes == "category") | (X.dtypes == "object"))[0]
        pool = Pool(X, cat_features=cat_idx)
    else:
        pool = X

    qhat = model.predict(pool)
    qhat = np.asarray(qhat)
    # Enforce non-crossing (cheap rearrangement)
    qhat.sort(axis=1)
    return qhat


def _weighted_mean(x: np.ndarray, w: Optional[np.ndarray]) -> float:
    """Compute weighted mean."""
    if w is None:
        return float(np.mean(x))
    
    w = np.asarray(w, dtype=float).ravel()
    w = w / np.sum(w)
    return float(np.sum(w * x))


def _pinball(
    y: np.ndarray, 
    q: np.ndarray, 
    alpha: float, 
    w: Optional[np.ndarray] = None
) -> float:
    """Compute pinball loss for a single quantile."""
    e = y - q
    return _weighted_mean(np.maximum(alpha * e, (alpha - 1.0) * e), w)


def _interp(alpha: float, Q: List[float], qhat: np.ndarray) -> np.ndarray:
    """Interpolate quantile predictions for a specific alpha level."""
    idx = np.where(np.isclose(Q, alpha))[0]
    if idx.size:
        return qhat[:, idx[0]]
    # linear interpolate between nearest quantiles
    if alpha <= Q[0] or alpha >= Q[-1]:
        raise ValueError(
            "interval alpha outside provided quantiles; add tails or change interval."
        )
    j_hi = np.searchsorted(Q, alpha)
    j_lo = j_hi - 1
    w_hi = (alpha - Q[j_lo]) / (Q[j_hi] - Q[j_lo])
    return (1.0 - w_hi) * qhat[:, j_lo] + w_hi * qhat[:, j_hi]


def eval_multiquantile(
    y_true: np.ndarray,
    q_pred: np.ndarray,
    quantiles: List[float],
    interval: Tuple[float, float] = (0.10, 0.90),
    sample_weight: Optional[np.ndarray] = None,
    lambda_cross: float = 0.0,
    return_per_quantile: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluate multi-quantile predictions.
    
    Args:
        y_true: True values
        q_pred: Quantile predictions of shape (n_samples, n_quantiles)
        quantiles: List of quantiles corresponding to q_pred columns
        interval: Interval for coverage calculation
        sample_weight: Optional sample weights
        lambda_cross: Penalty for quantile crossing
        return_per_quantile: Whether to include per-quantile metrics
        
    Returns:
        Tuple of (loss, metrics_dict)
    """
    y = np.asarray(y_true).ravel()
    qhat = np.asarray(q_pred)
    Q = np.asarray(quantiles, dtype=float)

    assert qhat.ndim == 2 and qhat.shape[0] == y.shape[0], "Shape mismatch."
    assert len(Q) == qhat.shape[1], "quantiles must align with q_pred columns."
    assert 0.0 < interval[0] < interval[1] < 1.0, "interval must be within (0,1)."

    # Pinball loss (objective base)
    pinballs = [
        _pinball(y, qhat[:, j], Q[j], sample_weight) for j in range(qhat.shape[1])
    ]
    pinball_mean = float(np.mean(pinballs))

    # Optional crossing penalty
    if lambda_cross > 0.0:
        diffs = qhat[:, :-1] - qhat[:, 1:]  # >0 means crossing
        cross_pen = _weighted_mean(np.clip(diffs, 0.0, None).sum(axis=1), sample_weight)
    else:
        cross_pen = 0.0

    loss = pinball_mean + lambda_cross * cross_pen

    q_lo = _interp(alpha=interval[0], Q=Q, qhat=qhat)
    q_hi = _interp(alpha=interval[1], Q=Q, qhat=qhat)
    covered = (y >= q_lo) & (y <= q_hi)

    coverage = _weighted_mean(covered.astype(float), sample_weight)
    mean_width = _weighted_mean(q_hi - q_lo, sample_weight)

    # Quantile calibration mismatch
    cal_errs = [
        abs(_weighted_mean((y <= qhat[:, j]).astype(float), sample_weight) - Q[j])
        for j in range(len(Q))
    ]
    cal_err_mean = float(np.mean(cal_errs))

    metrics = {
        "loss": loss,
        "pinball_mean": pinball_mean,
        f"coverage_{int(interval[0] * 100)}_{int(interval[1] * 100)}": coverage,
        "mean_width": mean_width,
        "crossing_penalty": cross_pen,
        "calibration_error_mean": cal_err_mean,
    }
    
    if return_per_quantile:
        for j, a in enumerate(Q):
            metrics[f"pinball@{a:.2f}"] = float(pinballs[j])

    return loss, metrics


def conformal_adjustment(
    q_lo_cal: np.ndarray, 
    q_hi_cal: np.ndarray, 
    y_cal: np.ndarray, 
    alpha: float
) -> float:
    """
    Compute conformal adjustment for prediction intervals.
    
    Args:
        q_lo_cal: Lower quantile predictions on calibration set
        q_hi_cal: Upper quantile predictions on calibration set
        y_cal: True values on calibration set
        alpha: Miscoverage level (1 - target_coverage)
        
    Returns:
        Conformal quantile adjustment
    """
    s = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)
    s_sorted = np.sort(s)
    n = len(s_sorted)
    # Finite-sample index per CQR (Romano et al.): ceil((n+1)*(1-alpha)) - 1
    k = int(np.ceil((n + 1) * (1 - alpha))) - 1
    k = np.clip(k, 0, n - 1)
    return float(s_sorted[k])


def apply_conformal(
    q_lo: np.ndarray, 
    q_hi: np.ndarray, 
    q_conformal: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply conformal adjustment to prediction intervals.
    
    Args:
        q_lo: Lower quantile predictions
        q_hi: Upper quantile predictions  
        q_conformal: Conformal adjustment value
        
    Returns:
        Tuple of (adjusted_lower, adjusted_upper) predictions
    """
    lo = q_lo - q_conformal
    hi = q_hi + q_conformal
    return lo, hi


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Compute coverage (fraction of true values within prediction intervals)."""
    return float(np.mean((y >= lo) & (y <= hi)))


def mean_width(lo: np.ndarray, hi: np.ndarray) -> float:
    """Compute mean width of prediction intervals."""
    return float(np.mean(hi - lo))


def pinball_loss(y: np.ndarray, q_pred: np.ndarray, alpha: float) -> float:
    """Compute pinball loss for a single quantile."""
    e = y - q_pred
    return float(np.mean(np.maximum(alpha*e, (alpha-1)*e)))


# Visualization utilities
DEFAULT_PLOT_PALETTE = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e", 
    "success": "#2ca02c",
    "danger": "#d62728",
    "warning": "#ff7f0e",
    "info": "#17a2b8",
    "light": "#f8f9fa",
    "dark": "#343a40",
    "gradient": [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ],
}


def plot_optuna_parallel_coordinates(
    study: Study,
    palette: Optional[Dict[str, Any]] = None,
    title: str = "Optuna Study - Parallel Coordinates Plot",
    show_best_n: Optional[int] = None,
    width: int = 1000,
    height: int = 600,
) -> go.Figure:
    """Create a parallel coordinates plot to visualize optuna study metrics."""
    if palette is None:
        palette = DEFAULT_PLOT_PALETTE

    # Extract data from trials
    trials_data = []
    for trial in study.trials:
        if trial.state != TrialState.COMPLETE:
            continue

        row = {"trial_number": trial.number, "objective_value": trial.value}

        # Add metrics from user_attrs if available
        if hasattr(trial, "user_attrs") and "metrics" in trial.user_attrs:
            metrics = trial.user_attrs["metrics"]
            row.update(metrics)

        trials_data.append(row)

    if not trials_data:
        raise ValueError("No completed trials found in the study")

    df = pd.DataFrame(trials_data)

    # Filter to best N trials if specified
    if show_best_n is not None:
        df = df.nsmallest(show_best_n, "objective_value")

    # Separate numeric columns for parallel coordinates
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    plot_dimensions = [
        col for col in numeric_cols if col not in ["trial_number", "objective_value"]
    ]

    # Create parallel coordinates plot
    dimensions = [
        dict(
            range=[df["trial_number"].min(), df["trial_number"].max()],
            label="Trial",
            values=df["trial_number"],
        )
    ]

    dimensions.extend([
        dict(
            range=[df[col].min(), df[col].max()],
            label=col.replace("_", " ").title(),
            values=df[col],
        )
        for col in plot_dimensions
    ])

    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=df["objective_value"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Objective Value"),
            ),
            dimensions=dimensions,
            labelfont=dict(size=12, color=palette["dark"]),
            tickfont=dict(size=10, color=palette["dark"]),
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=palette["dark"])),
        width=width,
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color=palette["dark"]),
        hovermode="closest",
    )

    return fig


def plot_optuna_metrics_distribution(
    study: Study,
    metric_name: str = "loss",
    palette: Optional[Dict[str, Any]] = None,
    title: Optional[str] = None,
    width: int = 800,
    height: int = 400,
) -> go.Figure:
    """Create a histogram showing the distribution of a specific metric across trials."""
    if palette is None:
        palette = DEFAULT_PLOT_PALETTE

    if title is None:
        title = f"Distribution of {metric_name.replace('_', ' ').title()}"

    # Extract metric values
    metric_values = []
    for trial in study.trials:
        if (
            trial.state == TrialState.COMPLETE
            and hasattr(trial, "user_attrs")
            and "metrics" in trial.user_attrs
            and metric_name in trial.user_attrs["metrics"]
        ):
            metric_values.append(trial.user_attrs["metrics"][metric_name])

    if not metric_values:
        raise ValueError(
            f"No values found for metric '{metric_name}' in completed trials"
        )

    fig = go.Figure(
        data=[
            go.Histogram(
                x=metric_values, 
                nbinsx=20, 
                marker_color=palette["primary"], 
                opacity=0.7
            )
        ]
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=palette["dark"])),
        xaxis_title=metric_name.replace("_", " ").title(),
        yaxis_title="Count",
        width=width,
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color=palette["dark"]),
        showlegend=False,
    )

    return fig