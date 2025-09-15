from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

import numpy as np
import pandas as pd

from stock_market_analytics.modeling.model_factory.protocols import DataSplitter


def _as_dt(s: pd.Series) -> pd.Series:
    if not np.issubdtype(s.dtype, np.datetime64):
        s = pd.to_datetime(s, utc=False)
    return s


def _unique_sorted_dates(d: pd.Series) -> np.ndarray:  # type: ignore
    return np.asarray(pd.Index(d).unique().sort_values())


def _cut_by_fractions(  # type: ignore
    unique_dates: np.ndarray, fracs: tuple[float, float, float, float]
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """
    Return (train_end, val_end, cal_end) date cutoffs using cumulative fractions.
    """
    if not np.isclose(sum(fracs), 1.0):
        raise ValueError("fractions must sum to 1.")
    n = len(unique_dates)
    if n < 10:
        raise ValueError("Too few unique dates for fractional splits.")
    f_train, f_val, f_cal, _ = fracs
    i_train_end = max(int(np.floor(n * f_train)) - 1, 0)
    i_val_end = max(int(np.floor(n * (f_train + f_val))) - 1, i_train_end + 1)
    i_cal_end = max(int(np.floor(n * (f_train + f_val + f_cal))) - 1, i_val_end + 1)
    return unique_dates[i_train_end], unique_dates[i_val_end], unique_dates[i_cal_end]


def _validate(df: pd.DataFrame, date_col: str, symbol_col: str) -> pd.DataFrame:  # type: ignore
    if date_col not in df.columns:
        raise ValueError(f"'{date_col}' column missing.")

    if symbol_col not in df.columns:
        raise ValueError(f"'{symbol_col}' column missing.")

    df[date_col] = _as_dt(df[date_col])
    return df


def _purge_around(
    window_mask: np.ndarray, dates: np.ndarray, d: pd.Series, embargo: int, h: int
) -> tuple[np.datetime64, np.datetime64]:
    """
    Given a boolean mask defining a time window, return the union of [t - embargo, t + h + embargo) for t in the window.
    """
    if not window_mask.any():
        return np.datetime64(d.min()), np.datetime64(d.min())

    w_start = dates[window_mask].min()
    w_end = dates[window_mask].max()
    union_start = w_start - np.timedelta64(embargo, "D")
    union_end = w_end + np.timedelta64(h + embargo, "D")
    return union_start, union_end


def _no_overlap_with(
    union0: np.datetime64, union1: np.datetime64, dates: np.ndarray, end_i: np.ndarray
) -> np.ndarray:
    # keep samples whose [t, t+h) lies completely outside [union0, union1]

    start_i = dates

    return ~((start_i < union1) & (end_i > union0))


def _apply_segment_masks(  # type: ignore
    df: pd.DataFrame,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
    cal_end: pd.Timestamp,
    date_col: str = "date",
    horizon_days: int = 5,
    embargo_days: int | None = None,
) -> dict[str, np.ndarray]:
    d = df[date_col]
    # Embargo defaults to horizon (safe for 5-day-ahead labels)
    embargo = int(embargo_days if embargo_days is not None else horizon_days)
    h = int(horizon_days)

    # Base boundaries
    train_mask_base = d <= train_end
    val_mask_base = (d > train_end) & (d <= val_end)
    cal_mask_base = (d > val_end) & (d <= cal_end)
    test_mask_base = d > cal_end

    # Purge/embargo around boundaries by label overlap logic:
    dates = df[date_col].to_numpy()  # numpy datetime64[D]
    end_i = np.array([np.datetime64(ts + pd.Timedelta(days=h)) for ts in df[date_col]])

    # Purge train against val union and test union (calibration sits between, so train shouldn't see it either)
    val_u0, val_u1 = _purge_around(val_mask_base, dates, d, embargo, h)
    cal_u0, cal_u1 = _purge_around(cal_mask_base, dates, d, embargo, h)
    test_u0, test_u1 = _purge_around(test_mask_base, dates, d, embargo, h)

    train_mask = (
        train_mask_base
        & _no_overlap_with(val_u0, val_u1, dates, end_i)
        & _no_overlap_with(cal_u0, cal_u1, dates, end_i)
        & _no_overlap_with(test_u0, test_u1, dates, end_i)
    )
    val_mask = val_mask_base & _no_overlap_with(
        test_u0, test_u1, dates, end_i
    )  # prevent val bleeding into test
    cal_mask = cal_mask_base & _no_overlap_with(
        test_u0, test_u1, dates, end_i
    )  # prevent cal bleeding into test
    test_mask = test_mask_base  # test is last block

    return {
        "train_idx": np.flatnonzero(train_mask),
        "val_idx": np.flatnonzero(val_mask),
        "cal_idx": np.flatnonzero(cal_mask),
        "test_idx": np.flatnonzero(test_mask),
    }


def _xy(
    frame: pd.DataFrame, feature_cols: list[str], target_col: str
) -> tuple[pd.DataFrame, pd.Series]:  # type: ignore
    X = frame[feature_cols].copy()
    y = frame[target_col].copy()
    return X, y  # type: ignore


def _build_test_windows(
    u: np.ndarray, n_splits: int, test_span_days: int | None = None
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if test_span_days is None:
        # equal partitions of the unique-date grid
        idx = np.linspace(0, len(u), n_splits + 1, dtype=int)
        wins = [(u[idx[i]], u[idx[i + 1] - 1]) for i in range(n_splits)]
    else:
        span = pd.Timedelta(days=test_span_days)
        anchors = np.linspace(0, len(u) - 1, n_splits, dtype=int)
        wins = []
        for a in anchors:
            start = u[a]
            end = min(u[-1], start + span)
            wins.append((start, end))
    # ensure chronological order
    wins.sort(key=lambda t: t[0])
    return wins  # type: ignore


# --------------------------------------- Public API --------------------------------------- #
def make_holdout_splits(
    df: pd.DataFrame,
    date_col: str,
    symbol_col: str,
    *,
    fractions: tuple[float, float, float, float] = (0.7, 0.10, 0.10, 0.10),
    cut_dates: tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp] | None = None,
    return_frames: bool = False,
) -> Mapping[str, pd.DataFrame] | Mapping[str, np.ndarray]:
    """
    Build leakage-safe Train/Val/Cal/Test.

    Either pass `cut_dates=(train_end, val_end, cal_end)` or `fractions` (sum=1).
    Embargo around boundaries defaults to `horizon_days`.
    """
    df = _validate(df, date_col, symbol_col)
    u = _unique_sorted_dates(df[date_col])

    if cut_dates is None:
        train_end, val_end, cal_end = _cut_by_fractions(u, fractions)

    else:
        train_end, val_end, cal_end = cut_dates

    idx = _apply_segment_masks(df, train_end, val_end, cal_end)

    if not return_frames:
        return idx

    return {
        "train": df.iloc[idx["train_idx"]],
        "val": df.iloc[idx["val_idx"]],
        "cal": df.iloc[idx["cal_idx"]],
        "test": df.iloc[idx["test_idx"]],
    }


def get_modeling_sets(
    df: pd.DataFrame,
    date_col: str = "date",
    symbol_col: str = "symbol",
    *,
    feature_cols: list[str],
    target_col: str,
    fractions: tuple[float, float, float, float] = (0.7, 0.10, 0.10, 0.10),
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """
    Convenience for CatBoost:
      - Train set for fitting
      - Validation set for early stopping
      - Calibration set reserved for conformal
      - Test set final evaluation
    """
    frames = make_holdout_splits(
        df,
        date_col=date_col,
        symbol_col=symbol_col,
        fractions=fractions,
        return_frames=True,
    )

    X_train, y_train = _xy(frames["train"], feature_cols, target_col)
    X_val, y_val = _xy(frames["val"], feature_cols, target_col)
    X_cal, y_cal = _xy(frames["cal"], feature_cols, target_col)
    X_test, y_test = _xy(frames["test"], feature_cols, target_col)
    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "cal": (X_cal, y_cal),
        "test": (X_test, y_test),
    }


class PurgedTimeSeriesSplit(DataSplitter):
    """
    Purged & embargoed time-series CV for panel data with fixed forecast horizon.
    Past-only by default: training samples must lie strictly BEFORE the test window
    (with an embargo buffer and no label-window overlap).

    train_side: 'past' (default) or 'past_and_future' (old behavior).
    """

    def __init__(
        self,
        n_splits: int = 5,
        date: pd.Series | None = None,
        horizon_days: int = 5,
        embargo_days: int | None = None,
        test_span_days: int | None = None,
        min_train_fraction: float = 0.05,  # optional safety
    ):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2.")

        self.n_splits = n_splits
        self.h = int(horizon_days)
        self.embargo = int(embargo_days if embargo_days is not None else self.h)
        self.test_span_days = test_span_days
        self.min_train_fraction = float(min_train_fraction)
        self._date = _as_dt(date) if date is not None else None

    def split(
        self,
        X: pd.DataFrame,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        # Resolve date series
        if self._date is None:
            if "date" not in X.columns:
                raise ValueError(
                    "X must contain a 'date' column if 'date' not provided at init."
                )
            d = _as_dt(X["date"])
        else:
            d = self._date

        u = _unique_sorted_dates(d)
        test_windows = _build_test_windows(u, self.n_splits, self.test_span_days)

        dates: np.ndarray = np.asarray(d.values)  # numpy datetime64[D]
        n = len(dates)

        # Strict causal cutoff per fold
        gap = np.timedelta64(self.embargo + self.h, "D")  # (embargo + horizon)

        for t_start, t_end in test_windows:
            test_mask = (dates >= np.datetime64(t_start)) & (
                dates <= np.datetime64(t_end)
            )

            # Strictly causal train: date <= t_start - (embargo + horizon)
            cutoff = np.datetime64(t_start) - gap
            train_mask = (dates <= cutoff) & (~test_mask)

            # Safety: drop fold if too small train
            if train_mask.sum() < self.min_train_fraction * n:
                continue

            yield np.nonzero(train_mask)[0], np.nonzero(test_mask)[0]
