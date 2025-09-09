from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd

def _as_dt(s: pd.Series) -> pd.Series:
    if not np.issubdtype(s.dtype, np.datetime64):
        s = pd.to_datetime(s, utc=False)
    return s

def _unique_sorted_dates(d: pd.Series) -> np.ndarray:
    return np.asarray(pd.Index(d).unique().sort_values())

def _cut_by_fractions(unique_dates: np.ndarray, fracs: tuple[float, float, float, float]) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Return (train_end, val_end, cal_end) date cutoffs using cumulative fractions."""
    if not np.isclose(sum(fracs), 1.0):
        raise ValueError("fractions must sum to 1.")
    n = len(unique_dates)
    if n < 10:
        raise ValueError("Too few unique dates for fractional splits.")
    f_train, f_val, f_cal, _ = fracs
    i_train_end = max(int(np.floor(n * f_train)) - 1, 0)
    i_val_end   = max(int(np.floor(n * (f_train + f_val))) - 1, i_train_end + 1)
    i_cal_end   = max(int(np.floor(n * (f_train + f_val + f_cal))) - 1, i_val_end + 1)
    return unique_dates[i_train_end], unique_dates[i_val_end], unique_dates[i_cal_end]

def _validate(df: pd.DataFrame, date_col: str, symbol_col: str) -> pd.DataFrame:
    if date_col not in df.columns:
        raise ValueError(f"'{date_col}' column missing.")
    
    if symbol_col not in df.columns:
        raise ValueError(f"'{symbol_col}' column missing.")
    
    df = df.copy()
    df[date_col] = _as_dt(df[date_col])
    return df.sort_values([date_col, symbol_col]).reset_index(drop=True)


def _purge_around(
        window_mask: np.ndarray,
        dates: np.ndarray,
        d: pd.Series,
        embargo: int,
        h: int
) -> tuple[np.datetime64, np.datetime64]:
    """
    Given a boolean mask defining a time window, return the union of [t - embargo, t + h + embargo) for t in the window.
    """
    if not window_mask.any():
        return np.datetime64(d.min()), np.datetime64(d.min())
    
    w_start = dates[window_mask].min()
    w_end   = dates[window_mask].max()
    union_start = w_start - np.timedelta64(embargo, 'D')
    union_end   = w_end + np.timedelta64(h + embargo, 'D')
    return union_start, union_end

def _no_overlap_with(union0, union1, dates: np.ndarray, end_i: np.ndarray) -> np.ndarray:
    # keep samples whose [t, t+h) lies completely outside [union0, union1]
    
    start_i = dates
    
    return ~((start_i < union1) & (end_i > union0))

def _apply_segment_masks(
    df: pd.DataFrame,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
    cal_end: pd.Timestamp,
    date_col: str = "date",
    horizon_days: int = 5,
    embargo_days: Optional[int] = None,
    ) -> dict[str, np.ndarray]:
        
    d = df[date_col]
    # Embargo defaults to horizon (safe for 5-day-ahead labels)
    embargo = int(embargo_days if embargo_days is not None else horizon_days)
    h = int(horizon_days)

    # Base boundaries
    train_mask_base = d <= train_end
    val_mask_base   = (d > train_end) & (d <= val_end)
    cal_mask_base   = (d > val_end)   & (d <= cal_end)
    test_mask_base  = d > cal_end

    # Purge/embargo around boundaries by label overlap logic:
    dates = df[date_col].values
    end_i = np.array([np.datetime64(ts + pd.Timedelta(days=h)) for ts in df[date_col]])

    # Purge train against val union and test union (calibration sits between, so train shouldn't see it either)
    val_u0, val_u1 = _purge_around(val_mask_base, dates, d, embargo, h)
    cal_u0, cal_u1 = _purge_around(cal_mask_base, dates, d, embargo, h)
    test_u0, test_u1 = _purge_around(test_mask_base, dates, d, embargo, h)

    train_mask = train_mask_base & _no_overlap_with(val_u0, val_u1, dates, end_i) & _no_overlap_with(cal_u0, cal_u1, dates, end_i) & _no_overlap_with(test_u0, test_u1, dates, end_i)
    val_mask   = val_mask_base   & _no_overlap_with(test_u0, test_u1, dates, end_i)  # prevent val bleeding into test
    cal_mask   = cal_mask_base   & _no_overlap_with(test_u0, test_u1, dates, end_i)  # prevent cal bleeding into test
    test_mask  = test_mask_base  # test is last block

    return {
        "train_idx": np.flatnonzero(train_mask),
        "val_idx"  : np.flatnonzero(val_mask),
        "cal_idx"  : np.flatnonzero(cal_mask),
        "test_idx" : np.flatnonzero(test_mask),
        }

def _xy(frame: pd.DataFrame, feature_cols: list[str], target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    X = frame[feature_cols].copy()
    y = frame[target_col].copy()
    return X, y