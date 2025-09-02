import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Any, Dict, List
import optuna
from optuna.study import Study

def eval_multiquantile(
    y_true: np.ndarray,
    q_pred: np.ndarray,              # shape (n_samples, n_quantiles), raw model output
    quantiles: list[float],          # e.g., [0.10, 0.25, 0.50, 0.75, 0.90] in the same column order as q_pred
    interval: tuple[float, float] = (0.10, 0.90),   # for coverage/width tracking
    sample_weight: Optional[np.ndarray] = None,     # optional time-weights
    lambda_cross: float = 0.0,        # >0 to discourage quantile crossing (penalty added to objective)
    return_per_quantile: bool = False # include per-quantile pinballs in metrics
) -> tuple[float, dict[str, Any]]:
    """
    Returns:
      loss: float  (scalar to minimize in Optuna)
      metrics: dict (coverage, mean_width, pinball_mean, etc.)

    Notes:
      - Uses raw q_pred (no sorting), so the crossing penalty measures real violations.
      - Pinball loss is a proper score for quantiles; averaging across quantiles is a solid single-number objective.
    """
    y = np.asarray(y_true).ravel()
    qhat = np.asarray(q_pred)
    Q = np.asarray(quantiles, dtype=float)

    assert qhat.ndim == 2 and qhat.shape[0] == y.shape[0], "Shape mismatch."
    assert len(Q) == qhat.shape[1], "quantiles must align with q_pred columns."
    assert 0.0 < interval[0] < interval[1] < 1.0, "interval must be within (0,1)."

    # --- helpers ---
    def _weighted_mean(x: np.ndarray, w: Optional[np.ndarray]) -> float:
        if w is None:
            return float(np.mean(x))
        w = np.asarray(w, dtype=float).ravel()
        w = w / np.sum(w)
        return float(np.sum(w * x))

    def _pinball(y: np.ndarray, q: np.ndarray, alpha: float, w: Optional[np.ndarray] = None) -> float:
        e = y - q
        return _weighted_mean(np.maximum(alpha * e, (alpha - 1.0) * e), w)

    # --- pinball loss (objective base) ---
    pinballs = [_pinball(y, qhat[:, j], Q[j], sample_weight) for j in range(qhat.shape[1])]
    pinball_mean = float(np.mean(pinballs))

    # --- optional crossing penalty (keeps it simple but nudges toward monotone) ---
    if lambda_cross > 0.0:
        diffs = qhat[:, :-1] - qhat[:, 1:]                 # >0 means crossing
        cross_pen = _weighted_mean(np.clip(diffs, 0.0, None).sum(axis=1), sample_weight)
    else:
        cross_pen = 0.0

    loss = pinball_mean + lambda_cross * cross_pen

    # --- coverage & width for a chosen interval (interpolate if interval alphas not present) ---
    def _interp(alpha:float):
        idx = np.where(np.isclose(Q, alpha))[0]
        if idx.size:
            return qhat[:, idx[0]]
        # linear interpolate between nearest quantiles
        if alpha <= Q[0] or alpha >= Q[-1]:
            raise ValueError("interval alpha outside provided quantiles; add tails or change interval.")
        j_hi = np.searchsorted(Q, alpha)
        j_lo = j_hi - 1
        w_hi = (alpha - Q[j_lo]) / (Q[j_hi] - Q[j_lo])
        return (1.0 - w_hi) * qhat[:, j_lo] + w_hi * qhat[:, j_hi]

    q_lo = _interp(interval[0])
    q_hi = _interp(interval[1])
    covered = (y >= q_lo) & (y <= q_hi)

    coverage = _weighted_mean(covered.astype(float), sample_weight)
    mean_width = _weighted_mean(q_hi - q_lo, sample_weight)

    # --- quantile calibration mismatch (track-only) ---
    cal_errs = [abs(_weighted_mean((y <= qhat[:, j]).astype(float), sample_weight) - Q[j]) for j in range(len(Q))]
    cal_err_mean = float(np.mean(cal_errs))

    metrics = {
        "loss": loss,
        "pinball_mean": pinball_mean,
        f"coverage_{int(interval[0]*100)}_{int(interval[1]*100)}": coverage,
        "mean_width": mean_width,
        "crossing_penalty": cross_pen,
        "calibration_error_mean": cal_err_mean,
    }
    if return_per_quantile:
        for j, a in enumerate(Q):
            metrics[f"pinball@{a:.2f}"] = float(pinballs[j])

    return loss, metrics

# Configurable color palette for visualizations
DEFAULT_PLOT_PALETTE = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e', 
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff7f0e',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#343a40',
    'gradient': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
}

def plot_optuna_parallel_coordinates(
    study: Study,
    palette: Dict[str, Any] = None,
    title: str = "Optuna Study - Parallel Coordinates Plot",
    show_best_n: Optional[int] = None,
    width: int = 1000,
    height: int = 600
) -> go.Figure:
    """
    Creates a parallel coordinates plot to visualize optuna study metrics.
    
    Args:
        study: Optuna study object with trials containing user_attrs['metrics']
        palette: Color palette dictionary. If None, uses DEFAULT_PLOT_PALETTE
        title: Title for the plot
        show_best_n: If specified, only show the best N trials (by objective value)
        width: Plot width in pixels
        height: Plot height in pixels
    
    Returns:
        Plotly figure object
    
    Example:
        >>> fig = plot_optuna_parallel_coordinates(study)
        >>> fig.show()
    """
    if palette is None:
        palette = DEFAULT_PLOT_PALETTE
    
    # Extract data from trials
    trials_data = []
    
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
            
        row = {
            'trial_number': trial.number,
            'objective_value': trial.value
        }
        
        # Add hyperparameters
        # row.update(trial.params)
        
        # Add metrics from user_attrs if available
        if hasattr(trial, 'user_attrs') and 'metrics' in trial.user_attrs:
            metrics = trial.user_attrs['metrics']
            row.update(metrics)
        
        trials_data.append(row)
    
    if not trials_data:
        raise ValueError("No completed trials found in the study")
    
    df = pd.DataFrame(trials_data)
    
    # Filter to best N trials if specified
    if show_best_n is not None:
        df = df.nsmallest(show_best_n, 'objective_value')
    
    # Separate numeric columns for parallel coordinates
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove trial_number from dimensions (keep for coloring)
    plot_dimensions = [col for col in numeric_cols if col != 'trial_number']
    
    # Create parallel coordinates plot
    fig = go.Figure(data=
        go.Parcoords(
            line=dict(
                color=df['objective_value'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Objective Value")
            ),
            dimensions=[
                dict(
                    range=[df[col].min(), df[col].max()],
                    label=col.replace('_', ' ').title(),
                    values=df[col]
                ) for col in plot_dimensions
            ]
        )
    )
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, color=palette['dark'])
        ),
        width=width,
        height=height,
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color=palette['dark'])
    )
    
    return fig


def plot_optuna_metrics_distribution(
    study: Study,
    metric_name: str = 'loss',
    palette: Dict[str, Any] = None,
    title: str = None,
    width: int = 800,
    height: int = 400
) -> go.Figure:
    """
    Creates a histogram showing the distribution of a specific metric across trials.
    
    Args:
        study: Optuna study object
        metric_name: Name of the metric to visualize from user_attrs['metrics']
        palette: Color palette dictionary. If None, uses DEFAULT_PLOT_PALETTE
        title: Title for the plot. If None, auto-generated
        width: Plot width in pixels
        height: Plot height in pixels
    
    Returns:
        Plotly figure object
    """
    if palette is None:
        palette = DEFAULT_PLOT_PALETTE
    
    if title is None:
        title = f"Distribution of {metric_name.replace('_', ' ').title()}"
    
    # Extract metric values
    metric_values = []
    for trial in study.trials:
        if (trial.state == optuna.trial.TrialState.COMPLETE and 
            hasattr(trial, 'user_attrs') and 
            'metrics' in trial.user_attrs and
            metric_name in trial.user_attrs['metrics']):
            metric_values.append(trial.user_attrs['metrics'][metric_name])
    
    if not metric_values:
        raise ValueError(f"No values found for metric '{metric_name}' in completed trials")
    
    fig = go.Figure(data=[
        go.Histogram(
            x=metric_values,
            nbinsx=20,
            marker_color=palette['primary'],
            opacity=0.7
        )
    ])
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, color=palette['dark'])
        ),
        xaxis_title=metric_name.replace('_', ' ').title(),
        yaxis_title='Count',
        width=width,
        height=height,
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color=palette['dark']),
        showlegend=False
    )
    
    return fig


