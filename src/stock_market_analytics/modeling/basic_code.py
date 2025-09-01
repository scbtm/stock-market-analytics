# pip install catboost pandas numpy scikit-learn
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Iterator, Tuple, List, Optional
from catboost import CatBoostRegressor, Pool

# --------------------------
# Utils: splitter (purge + embargo)
# --------------------------
@dataclass
class PurgedEmbargoedSplitter:
    n_splits: int
    horizon_days: int          # label horizon H in trading days (e.g., 63 ~ 3 months)
    embargo_days: int = 30     # add gap after each test block
    time_col: str = "timestamp"

    def split(self, df: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, test_idx) for each fold with purging + embargo."""
        # Ensure chronological order
        df_sorted = df.sort_values(self.time_col).reset_index(drop=False).rename(columns={"index": "_orig_idx"})
        times = df_sorted[self.time_col].values.astype("datetime64[D]")

        unique_days = np.unique(times)
        # Split timeline into n_splits contiguous test blocks on unique days
        cutpoints = np.linspace(0, len(unique_days), self.n_splits + 1, dtype=int)

        for k in range(self.n_splits):
            test_start_day = unique_days[cutpoints[k]]
            test_end_day = unique_days[cutpoints[k+1]-1]

            # Boolean masks for test block
            is_test = (times >= test_start_day) & (times <= test_end_day)

            # Purge: drop any train sample whose [t, t+H] overlaps test window
            # We operate in days; label_end = t + horizon_days
            t = times
            label_end = t + np.timedelta64(self.horizon_days, 'D')
            overlaps = (label_end >= test_start_day) & (t <= test_end_day)

            # Embargo: drop any train sample in (test_end, test_end + E]
            embargo_end = test_end_day + np.timedelta64(self.embargo_days, 'D')
            in_embargo = (t > test_end_day) & (t <= embargo_end)

            is_train = ~(is_test | overlaps | in_embargo)

            train_idx = df_sorted.loc[is_train, "_orig_idx"].values
            test_idx  = df_sorted.loc[is_test,  "_orig_idx"].values
            yield train_idx, test_idx

# --------------------------
# Utils: temporal calibration slice from the end of train
# --------------------------
def split_train_calib(df: pd.DataFrame, train_idx: np.ndarray, time_col="timestamp", calib_frac=0.2):
    train = df.loc[train_idx].sort_values(time_col)
    unique_days = train[time_col].values.astype("datetime64[D]")
    u = np.unique(unique_days)
    split_point = u[int((1 - calib_frac) * len(u))]
    is_calib = train[time_col].values >= split_point
    calib_idx = train.index.values[is_calib]
    pure_train_idx = train.index.values[~is_calib]
    return pure_train_idx, calib_idx

# --------------------------
# Model: CatBoost MultiQuantile
# --------------------------
def make_catboost_mq(quantiles: List[float], **kwargs) -> CatBoostRegressor:
    # CatBoost takes MultiQuantile with alpha list encoded in the loss string
    alpha_str = ",".join([str(q) for q in quantiles])
    params = dict(
        loss_function=f"MultiQuantile:alpha={alpha_str}",
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=42,
        # task_type="CPU"  # uncomment explicitly if you ever run on mixed hardware
    )
    params.update(kwargs)
    return CatBoostRegressor(**params)

def fit_catboost_mq(model, X_train, y_train, X_valid=None, y_valid=None, cat_idx: Optional[List[int]] = None):
    train_pool = Pool(X_train, y_train, cat_features=cat_idx)
    eval_set = None
    if X_valid is not None and y_valid is not None:
        eval_set = Pool(X_valid, y_valid, cat_features=cat_idx)
    model.fit(train_pool, eval_set=eval_set, verbose=200, use_best_model=False)
    return model

def predict_quantiles(model, X, cat_idx: Optional[List[int]] = None) -> np.ndarray:
    # MultiQuantile returns (n_samples, n_quantiles)
    pool = Pool(X, cat_features=cat_idx)
    qhat = model.predict(pool)
    qhat = np.asarray(qhat)
    # Enforce non-crossing (cheap rearrangement)
    qhat.sort(axis=1)
    return qhat

# --------------------------
# Conformalized Quantile Regression (CQR) for a lower/upper pair
# --------------------------
def conformal_adjustment(q_lo_cal, q_hi_cal, y_cal, alpha: float) -> float:
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
    return s_sorted[k]

def apply_conformal(q_lo, q_hi, q_conformal) -> Tuple[np.ndarray, np.ndarray]:
    lo = q_lo - q_conformal
    hi = q_hi + q_conformal
    return lo, hi

# --------------------------
# Metrics
# --------------------------
def coverage(y, lo, hi):
    return np.mean((y >= lo) & (y <= hi))

def mean_width(lo, hi):
    return np.mean(hi - lo)

def pinball_loss(y, q_pred, alpha):
    e = y - q_pred
    return np.mean(np.maximum(alpha*e, (alpha-1)*e))

# --------------------------
# End-to-end example
# --------------------------
# Assumptions:
#  - df has columns ['timestamp', 'y', <features...>]
#  - df is already free of look-ahead
Q = [0.10, 0.25, 0.50, 0.75, 0.90]
LOW, MID, HIGH = 0, 2, 4  # indices in Q for 0.10, 0.50, 0.90
TARGET_COVERAGE = 0.80     # matches 10%/90% pair
H = 63                     # ~3 months
E = 30

# Pick features
exclude = {"timestamp", "y", "symbol"}  # keep symbol if you need it (prefer engineered dummies/ranks instead)
feature_cols = [c for c in df.columns if c not in exclude]
cat_idx = None  # OR: indices of categorical features in feature_cols

# Ensure timestamp is datetime and sorted
df = df.dropna(subset=["timestamp", "y"]).copy()
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

splitter = PurgedEmbargoedSplitter(n_splits=5, horizon_days=H, embargo_days=E)
fold_results = []

for fold, (train_idx, test_idx) in enumerate(splitter.split(df), 1):
    # Create calibration slice from the end of train
    pure_train_idx, calib_idx = split_train_calib(df, train_idx, time_col="timestamp", calib_frac=0.2)

    X_train = df.loc[pure_train_idx, feature_cols]; y_train = df.loc[pure_train_idx, "y"].values
    X_cal   = df.loc[calib_idx,     feature_cols]; y_cal   = df.loc[calib_idx,     "y"].values
    X_test  = df.loc[test_idx,      feature_cols]; y_test  = df.loc[test_idx,      "y"].values

    # Train MQ model
    model = make_catboost_mq(Q)
    fit_catboost_mq(model, X_train, y_train)  # you can pass eval_set=X_cal,y_cal if you want early diagnostics

    # Predict quantiles
    q_cal = predict_quantiles(model, X_cal,  cat_idx)
    q_tst = predict_quantiles(model, X_test, cat_idx)

    # Pull specific lower/upper for CQR (here 10% / 90%)
    qlo_cal, qhi_cal = q_cal[:, LOW], q_cal[:, HIGH]
    qlo_tst, qhi_tst = q_tst[:, LOW], q_tst[:, HIGH]

    # Conformal adjustment on calibration slice
    qconf = conformal_adjustment(qlo_cal, qhi_cal, y_cal, alpha=1 - TARGET_COVERAGE)

    # Apply to test
    lo_cqr, hi_cqr = apply_conformal(qlo_tst, qhi_tst, qconf)
    med_pred = q_tst[:, MID]

    # Metrics
    cov   = coverage(y_test, lo_cqr, hi_cqr)
    width = mean_width(lo_cqr, hi_cqr)
    pin50 = pinball_loss(y_test, med_pred, alpha=0.5)

    fold_results.append({"fold": fold, "coverage": cov, "mean_width": width, "pinball@0.5": pin50})

# Inspect results
pd.DataFrame(fold_results)
