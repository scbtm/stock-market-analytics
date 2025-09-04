from typing import Any

import numpy as np
import optuna
import pandas as pd
import plotly.graph_objects as go
from catboost import CatBoostRegressor, Pool
from optuna.study import Study


def predict_quantiles(model: CatBoostRegressor, X: pd.DataFrame | Pool) -> np.ndarray:
    # MultiQuantile returns (n_samples, n_quantiles)

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

# --- helpers ---
def _weighted_mean(x: np.ndarray, w: np.ndarray | None) -> float:
    if w is None:
        return float(np.mean(x))

    w = np.asarray(w, dtype=float).ravel()
    w = w / np.sum(w)
    return float(np.sum(w * x))

def _pinball(
        y: np.ndarray, q: np.ndarray, alpha: float, w: np.ndarray | None = None
        ) -> float:

    e = y - q
    return _weighted_mean(np.maximum(alpha * e, (alpha - 1.0) * e), w)

# --- coverage & width for a chosen interval (interpolate if interval alphas not present) ---
def _interp(alpha: float, Q: list[float], qhat: np.ndarray) -> np.ndarray:
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
    q_pred: np.ndarray,  # shape (n_samples, n_quantiles), raw model output
    quantiles: list[float],  # e.g., [0.10, 0.25, 0.50, 0.75, 0.90] in the same column order as q_pred
    interval: tuple[float, float] = (0.10, 0.90),  # for coverage/width tracking
    sample_weight: np.ndarray | None = None,  # optional time-weights
    lambda_cross: float = 0.0,  # >0 to discourage quantile crossing (penalty added to objective)
    return_per_quantile: bool = False,  # include per-quantile pinballs in metrics
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

    # --- pinball loss (objective base) ---
    pinballs = [
        _pinball(y, qhat[:, j], Q[j], sample_weight) for j in range(qhat.shape[1])
    ]
    pinball_mean = float(np.mean(pinballs))

    # --- optional crossing penalty (keeps it simple but nudges toward monotone) ---
    if lambda_cross > 0.0:
        diffs = qhat[:, :-1] - qhat[:, 1:]  # >0 means crossing
        cross_pen = _weighted_mean(np.clip(diffs, 0.0, None).sum(axis=1), sample_weight)
    else:
        cross_pen = 0.0

    loss = pinball_mean + lambda_cross * cross_pen

    q_lo = _interp(alpha = interval[0], Q=Q, qhat=qhat)
    q_hi = _interp(alpha = interval[1], Q=Q, qhat=qhat)
    covered = (y >= q_lo) & (y <= q_hi)

    coverage = _weighted_mean(covered.astype(float), sample_weight)
    mean_width = _weighted_mean(q_hi - q_lo, sample_weight)

    # --- quantile calibration mismatch (track-only) ---
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


# --------------------------
# Conformalized Quantile Regression (CQR) for a lower/upper pair
# --------------------------
def conformal_adjustment(q_lo_cal: np.ndarray, q_hi_cal: np.ndarray, y_cal: np.ndarray, alpha: float) -> float:
    """
    Nonconformity scores s_i = max(q_lo - y, y - q_hi).
    Return the (1-alpha) empirical quantile with finite-sample correction.
    """
    s = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)
    s_sorted = np.sort(s)
    n = len(s_sorted)
    # Finite-sample index per CQR (Romano et al.): ceil((n+1)*(1-alpha)) - 1
    k = int(np.ceil((n + 1) * (1 - alpha))) - 1
    k = np.clip(k, 0, n - 1)

    return float(s_sorted[k])

def apply_conformal(q_lo: np.ndarray, q_hi: np.ndarray, q_conformal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = q_lo - q_conformal
    hi = q_hi + q_conformal
    return lo, hi

# --------------------------
# Metrics
# --------------------------
def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean((y >= lo) & (y <= hi)))

def mean_width(lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean(hi - lo))

def pinball_loss(y: np.ndarray, q_pred: np.ndarray, alpha: float) -> float:
    e = y - q_pred
    return float(np.mean(np.maximum(alpha*e, (alpha-1)*e)))


# Configurable color palette for visualizations
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
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ],
}


def plot_optuna_parallel_coordinates(
    study: Study,
    palette: dict[str, Any] = None,
    title: str = "Optuna Study - Parallel Coordinates Plot",
    show_best_n: int | None = None,
    width: int = 1000,
    height: int = 600,
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

        row = {"trial_number": trial.number, "objective_value": trial.value}

        # Add hyperparameters
        # row.update(trial.params)

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

    # Remove trial_number from dimensions (keep for coloring)
    plot_dimensions = [
        col for col in numeric_cols if col not in ["trial_number", "objective_value"]
    ]

    # Create parallel coordinates plot
    # Add trial_number as the first dimension for easy identification
    dimensions = [
        dict(
            range=[df["trial_number"].min(), df["trial_number"].max()],
            label="Trial",
            values=df["trial_number"],
        )
    ]

    # Add other dimensions
    dimensions.extend(
        [
            dict(
                range=[df[col].min(), df[col].max()],
                label=col.replace("_", " ").title(),
                values=df[col],
            )
            for col in plot_dimensions
        ]
    )

    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=df["objective_value"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Objective Value"),
            ),
            dimensions=dimensions,
            # Enable interactive features
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
        # Enable better interactivity
        hovermode="closest",
    )

    return fig


def plot_optuna_metrics_distribution(
    study: Study,
    metric_name: str = "loss",
    palette: dict[str, Any] = None,
    title: str = None,
    width: int = 800,
    height: int = 400,
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
        if (
            trial.state == optuna.trial.TrialState.COMPLETE
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
                x=metric_values, nbinsx=20, marker_color=palette["primary"], opacity=0.7
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
