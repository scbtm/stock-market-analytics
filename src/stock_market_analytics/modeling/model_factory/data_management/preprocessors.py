from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Optional
import numpy as np
import pandas as pd
from collections.abc import Mapping

from stock_market_analytics.modeling.model_factory.data_management.splitting_functions import (
    _as_dt,
    _cut_by_fractions,
    _unique_sorted_dates,
    _validate,
    _apply_segment_masks,
    _xy,
    _build_test_windows
)

class PurgedTimeSeriesSplit:
    """
    Purged & embargoed time-series CV for panel data with fixed forecast horizon.
    Past-only by default: training samples must lie strictly BEFORE the test window
    (with an embargo buffer and no label-window overlap).

    train_side: 'past' (default) or 'past_and_future' (old behavior).
    """

    def __init__(
        self,
        n_splits: int = 5,
        date: Optional[pd.Series] = None,
        horizon_days: int = 5,
        embargo_days: Optional[int] = None,
        test_span_days: Optional[int] = None,
        min_train_fraction: float = 0.05,      # optional safety
    ):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2.")

        self.n_splits = n_splits
        self.h = int(horizon_days)
        self.embargo = int(embargo_days if embargo_days is not None else self.h)
        self.test_span_days = test_span_days
        self.min_train_fraction = float(min_train_fraction)
        self._date = _as_dt(date) if date is not None else None

    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None, groups: Optional[pd.Series] = None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        # Resolve date series
        if self._date is None:
            if "date" not in X.columns:
                raise ValueError("X must contain a 'date' column if 'date' not provided at init.")
            d = _as_dt(X["date"])
        else:
            d = self._date

        u = _unique_sorted_dates(d)
        test_windows = _build_test_windows(u, self.n_splits, self.test_span_days)

        dates: np.ndarray = np.asarray(d.values)  # numpy datetime64[D]
        n = len(dates)

        # Strict causal cutoff per fold
        gap = np.timedelta64(self.embargo + self.h, 'D')  # (embargo + horizon)

        for (t_start, t_end) in test_windows:
            test_mask = (dates >= np.datetime64(t_start)) & (dates <= np.datetime64(t_end))

            # Strictly causal train: date <= t_start - (embargo + horizon)
            cutoff = np.datetime64(t_start) - gap
            train_mask = (dates <= cutoff) & (~test_mask)

            # Safety: drop fold if too small train
            if train_mask.sum() < self.min_train_fraction * n:
                continue

            yield np.nonzero(train_mask)[0], np.nonzero(test_mask)[0]

    # def split(
    #     self, X: pd.DataFrame, y: Optional[pd.Series] = None, groups: Optional[pd.Series] = None
    # ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    #     if self._date is None:
    #         if "date" not in X.columns:
    #             raise ValueError("X must contain a 'date' column when 'date' not provided at init.")
    #         d = _as_dt(X["date"])
    #     else:
    #         d = _as_dt(self._date)

    #     u = _unique_sorted_dates(d)
    #     if self.test_span_days is None:
    #         idx = np.linspace(0, len(u), self.n_splits + 1, dtype=int)
    #         test_windows = [(u[idx[i]], u[idx[i+1]-1]) for i in range(self.n_splits)]
    #     else:
    #         span = pd.Timedelta(days=int(self.test_span_days))
    #         anchors = np.linspace(0, len(u) - 1, self.n_splits, dtype=int)
    #         test_windows = []
    #         for a in anchors:
    #             start = u[a]
    #             end = min(u[-1], start + span)
    #             test_windows.append((start, end))

    #     dates: np.ndarray = np.asanyarray(d.values)
    #     n = len(d)
    #     end_i = np.array([np.datetime64(pd.Timestamp(t) + pd.Timedelta(days=self.h)) for t in d])

    #     for (t_start, t_end) in test_windows:
    #         # test_mask = (dates >= np.datetime64(t_start)) & (dates <= np.datetime64(t_end))

    #         # # Embargoed union of test label windows
    #         # union_start = np.datetime64(t_start) - np.timedelta64(self.embargo, 'D')
    #         # union_end   = np.datetime64(t_end)   + np.timedelta64(self.h + self.embargo, 'D')

    #         # # Purge label-window overlap with the union
    #         # start_i = dates
    #         # overlap = (start_i < union_end) & (end_i > union_start)

    #         # # Base: not test, not overlapping
    #         # base_train = (~test_mask) & (~overlap)

    #         # side_mask = end_i <= union_start

    #         # train_mask = base_train & side_mask

    #         # # Optional: drop folds with too-small train set
    #         # if train_mask.sum() < self.min_train_fraction * n:
    #         #     continue

    #         test_mask = (dates >= np.datetime64(t_start)) & (dates <= np.datetime64(t_end))

    #         # Strictly causal train: date <= t_start - (embargo + horizon)
    #         cutoff = np.datetime64(t_start) - gap
    #         train_mask = (dates <= cutoff) & (~test_mask)

    #         # Safety: drop fold if too small train
    #         if train_mask.sum() < self.min_train_fraction * n:
    #             continue

    #         yield np.nonzero(train_mask)[0], np.nonzero(test_mask)[0]

# =========================
# Purged, Embargoed time CV
# =========================
# class PurgedTimeSeriesSplit:
#     """
#     Purged & embargoed time-series CV for panel data with fixed forecast horizon (in calendar days).
#     Test folds are contiguous in time (by date). Training folds exclude any sample whose
#     label window [t, t+horizon) overlaps the test union, then an additional symmetric embargo.

#     Compatible with sklearn API: split(X, y=None, groups=None), but X must carry a date Series.
#     """

#     def __init__(
#         self,
#         n_splits: int = 5,
#         date: Optional[pd.Series] = None,
#         horizon_days: int = 5,
#         embargo_days: Optional[int] = None,
#         test_span_days: Optional[int] = None,   # if None, splits the unique date grid into n_splits
#     ):
#         if n_splits < 2:
#             raise ValueError("n_splits must be >= 2.")
#         self.n_splits = n_splits
#         self.h = int(horizon_days)
#         self.embargo = int(embargo_days if embargo_days is not None else self.h)
#         self.test_span_days = test_span_days
#         self._date = _as_dt(date) if date is not None else None

#     def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None, groups: Optional[pd.Series] = None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
#         if self._date is None:
#             if "date" not in X.columns:
#                 raise ValueError("X must contain a 'date' column when 'date' not provided at init.")
#             d = _as_dt(X["date"])
#         else:
#             d = _as_dt(self._date)

#         # Unique date grid and fold allocations
#         u = _unique_sorted_dates(d)
#         if self.test_span_days is None:
#             # Split date grid into n_splits contiguous blocks
#             idx = np.linspace(0, len(u), self.n_splits + 1, dtype=int)
#             test_windows = [(u[idx[i]], u[idx[i+1]-1]) for i in range(self.n_splits)]
#         else:
#             # rolling fixed-span windows
#             span = pd.Timedelta(days=int(self.test_span_days))
#             # choose evenly spaced starting anchors
#             anchors = np.linspace(0, len(u) - 1, self.n_splits, dtype=int)
#             test_windows = []
#             for a in anchors:
#                 start = u[a]
#                 end = min(u[-1], start + span)
#                 test_windows.append((start, end))

#         # Precompute arrays
#         # n = len(d)
#         dates: np.ndarray = np.asanyarray(d.values)
#         horizon: np.ndarray = np.array([np.datetime64(pd.Timestamp(t) + pd.Timedelta(days=self.h)) for t in d])

#         for (t_start, t_end) in test_windows:
#             # Test membership: t in [t_start, t_end]
#             test_mask = (dates >= np.datetime64(t_start)) & (dates <= np.datetime64(t_end))

#             # Union of test label windows is [t_start, t_end + h)
#             union_start = np.datetime64(t_start) - np.timedelta64(self.embargo, 'D')
#             union_end   = np.datetime64(t_end) + np.timedelta64(self.h + self.embargo, 'D')

#             # Purge criterion for a train sample with interval [t_i, t_i+h):
#             # overlap iff (t_i < union_end) & (t_i+h > union_start)
#             start_i = dates
#             end_i   = horizon
#             overlap = (start_i < union_end) & (end_i > union_start)

#             train_mask = (~test_mask) & (~overlap)

#             yield np.nonzero(train_mask)[0], np.nonzero(test_mask)[0]


# =========================
# Main splitters
# =========================
@dataclass
class PanelHorizonSplitter:
    """
    Data splitter for panel stock data targeting a fixed natural-day horizon.
    Produces leakage-safe Train/Val/Cal/Test slices and a PurgedTimeSeriesSplit for HPO.
    """
    date_col: str = "date"
    symbol_col: str = "symbol"
    horizon_days: int = 5
    embargo_days: Optional[int] = None  # defaults to horizon_days

    # ---------- Public API ----------

    def make_holdout_splits(
        self,
        df: pd.DataFrame,
        *,
        fractions: tuple[float, float, float, float] = (0.7, 0.10, 0.10, 0.10),
        cut_dates: Optional[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = None,
        return_frames: bool = False,
    ) -> Mapping[str, pd.DataFrame] | Mapping[str, np.ndarray]:
        """
        Build leakage-safe Train/Val/Cal/Test.

        Either pass `cut_dates=(train_end, val_end, cal_end)` or `fractions` (sum=1).
        Embargo around boundaries defaults to `horizon_days`.
        """
        df = _validate(df, self.date_col, self.symbol_col)
        u = _unique_sorted_dates(df[self.date_col])

        if cut_dates is None:
            train_end, val_end, cal_end = _cut_by_fractions(u, fractions)
        else:
            train_end, val_end, cal_end = cut_dates

        idx = _apply_segment_masks(df, train_end, val_end, cal_end)

        if not return_frames:
            return idx

        return {
            "train": df.iloc[idx["train_idx"]],
            "val"  : df.iloc[idx["val_idx"]],
            "cal"  : df.iloc[idx["cal_idx"]],
            "test" : df.iloc[idx["test_idx"]],
        }

    def catboost_early_stopping_sets(
        self,
        df: pd.DataFrame,
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
        frames = self.make_holdout_splits(df, fractions=fractions, return_frames=True)

        X_train, y_train = _xy(frames["train"], feature_cols, target_col)
        X_val, y_val     = _xy(frames["val"], feature_cols, target_col)
        X_cal, y_cal     = _xy(frames["cal"], feature_cols, target_col)
        X_test, y_test   = _xy(frames["test"], feature_cols, target_col)
        return {
            "train": (X_train, y_train),
            "val"  : (X_val  , y_val),
            "cal"  : (X_cal  , y_cal),
            "test" : (X_test , y_test),
        }

    def cv_splitter(
        self,
        df: pd.DataFrame,
        *,
        n_splits: int = 5,
        test_span_days: Optional[int] = None,
    ) -> PurgedTimeSeriesSplit:
        """
        Return a PurgedTimeSeriesSplit object bound to df's date column.
        Use in sklearn GridSearchCV / randomized HPO for leakage-safe scoring.
        """
        df = _validate(df, self.date_col, self.symbol_col)
        return PurgedTimeSeriesSplit(
            n_splits=n_splits,
            date=df[self.date_col],
            horizon_days=self.horizon_days,
            embargo_days=self.embargo_days if self.embargo_days is not None else self.horizon_days,
            test_span_days=test_span_days,
        )