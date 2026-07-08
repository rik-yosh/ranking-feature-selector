"""Utility functions used by ranking_feature_selector."""

from __future__ import annotations

from typing import Optional, Sequence, Union
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)

VALID_TASKS = {"classification", "regression", "survival"}


def as_dataframe(
    X: Union[pd.DataFrame, np.ndarray], columns: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """Return ``X`` as a pandas DataFrame with stable column names."""
    if isinstance(X, pd.DataFrame):
        return X.copy()
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError("X must be a 2D array or a pandas DataFrame.")
    if columns is None:
        columns = [f"x{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=list(columns))


def as_series(y: Union[pd.Series, np.ndarray, list]) -> pd.Series:
    """Return ``y`` as a reset-index pandas Series."""
    if isinstance(y, pd.Series):
        return y.reset_index(drop=True)
    return pd.Series(y).reset_index(drop=True)


def make_survival_y(
    event: Sequence,
    time: Sequence,
    event_name: str = "event",
    time_name: str = "time",
) -> np.ndarray:
    """Create a scikit-survival compatible structured survival target.

    The returned array has two fields: a boolean event indicator and a floating
    observed time. It intentionally does not import scikit-survival so that the
    package can be imported without the optional survival dependency installed.
    """
    event_arr = np.asarray(event).astype(bool)
    time_arr = np.asarray(time, dtype=float)
    if event_arr.ndim != 1 or time_arr.ndim != 1:
        raise ValueError("event and time must be one-dimensional arrays.")
    if len(event_arr) != len(time_arr):
        raise ValueError("event and time must have the same length.")
    out = np.empty(len(event_arr), dtype=[(event_name, "?"), (time_name, "<f8")])
    out[event_name] = event_arr
    out[time_name] = time_arr
    return out


def is_survival_y(y) -> bool:
    """Return True if ``y`` looks like a structured survival target."""
    return isinstance(y, np.ndarray) and y.dtype.names is not None and len(y.dtype.names) >= 2


def survival_field_names(y: np.ndarray) -> tuple[str, str]:
    """Return event and time field names from a structured survival array."""
    if not is_survival_y(y):
        raise ValueError("y is not a structured survival array.")
    names = y.dtype.names
    return names[0], names[1]


def get_event_time(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``event`` and ``time`` arrays from a survival target."""
    event_name, time_name = survival_field_names(y)
    return np.asarray(y[event_name], dtype=bool), np.asarray(y[time_name], dtype=float)


def _infer_survival_columns(data: pd.DataFrame) -> tuple[str, str]:
    event_candidates = ["event", "status", "death", "outcome", "event_indicator"]
    time_candidates = ["time", "duration", "followup", "followup_time", "survival_time"]

    lower_to_original = {str(c).lower(): c for c in data.columns}
    event_col = next((lower_to_original[c] for c in event_candidates if c in lower_to_original), None)
    time_col = next((lower_to_original[c] for c in time_candidates if c in lower_to_original), None)

    if event_col is None or time_col is None:
        if data.shape[1] == 2:
            event_col, time_col = data.columns[:2]
        else:
            raise ValueError(
                "For survival task, pass y as a structured array, a tuple (event, time), "
                "or a DataFrame with event_col and time_col specified."
            )
    return str(event_col), str(time_col)


def as_survival_array(
    y,
    event_col: Optional[str] = None,
    time_col: Optional[str] = None,
) -> np.ndarray:
    """Return ``y`` as a structured array compatible with scikit-survival."""
    if is_survival_y(y):
        arr = np.asarray(y).copy()
        validate_survival_y(arr)
        return arr

    if isinstance(y, pd.DataFrame):
        event_col, time_col = (event_col, time_col) if event_col and time_col else _infer_survival_columns(y)
        arr = make_survival_y(y[event_col].to_numpy(), y[time_col].to_numpy(), event_col, time_col)
        validate_survival_y(arr)
        return arr

    if isinstance(y, dict):
        if event_col is None:
            event_col = "event" if "event" in y else "status" if "status" in y else None
        if time_col is None:
            time_col = "time" if "time" in y else "duration" if "duration" in y else None
        if event_col is None or time_col is None:
            raise ValueError("For dict y, specify event_col and time_col or use keys event/time.")
        arr = make_survival_y(y[event_col], y[time_col], str(event_col), str(time_col))
        validate_survival_y(arr)
        return arr

    if isinstance(y, tuple) and len(y) == 2:
        arr = make_survival_y(y[0], y[1])
        validate_survival_y(arr)
        return arr

    arr = np.asarray(y)
    if arr.ndim == 2 and arr.shape[1] == 2:
        out = make_survival_y(arr[:, 0], arr[:, 1])
        validate_survival_y(out)
        return out

    raise ValueError(
        "For survival task, y must be a structured array, a tuple (event, time), "
        "a two-column array, or a DataFrame/dict with event and time columns."
    )


def as_target(
    y,
    task: str,
    event_col: Optional[str] = None,
    time_col: Optional[str] = None,
):
    """Convert ``y`` to the target representation used internally."""
    if task not in VALID_TASKS:
        raise ValueError("task must be one of: classification, regression, survival.")
    if task == "survival":
        return as_survival_array(y, event_col=event_col, time_col=time_col)
    return as_series(y)


def target_len(y) -> int:
    """Return number of samples in a task target."""
    return len(y)


def target_take(y, indices: Sequence[int], task: str):
    """Take rows from a target and reset index where applicable."""
    if task == "survival":
        return np.asarray(y)[np.asarray(indices)]
    return y.iloc[indices].reset_index(drop=True)


def target_reset(y, task: str):
    """Reset target index if it has one."""
    if task == "survival":
        return np.asarray(y).copy()
    return as_series(y)


def make_k_grid(n_features: int, max_k: Optional[int] = None) -> list[int]:
    """Build a practical candidate grid for the number of selected features.

    For small feature counts, all values are evaluated. For larger problems, the
    grid is dense for small ``k`` and coarser afterward.
    """
    if max_k is None:
        max_k = n_features
    max_k = int(min(max_k, n_features))
    if max_k < 1:
        raise ValueError("max_k must be at least 1.")

    if max_k <= 60:
        return list(range(1, max_k + 1))

    grid = set(range(1, 21))
    grid.update(range(25, min(max_k, 100) + 1, 5))
    grid.update(range(110, max_k + 1, 10))
    grid.add(max_k)
    return sorted(k for k in grid if 1 <= k <= max_k)


def _survival_strata(y, mode: str = "event", time_bins: int = 4) -> np.ndarray:
    """Build discrete strata for survival CV splitting."""
    if mode in {None, "none", "false", "off"}:
        raise ValueError("_survival_strata called with stratification disabled.")
    event, time = get_event_time(y)
    mode = str(mode).lower()
    if mode == "event":
        return event.astype(int)
    if mode in {"event_time", "event+time", "event_time_bin"}:
        if len(np.unique(time)) <= 1:
            bins = np.zeros_like(time, dtype=int)
        else:
            q = max(2, int(time_bins))
            try:
                bins = pd.qcut(time, q=min(q, len(np.unique(time))), labels=False, duplicates="drop")
                bins = np.asarray(bins, dtype=int)
            except Exception:
                bins = np.zeros_like(time, dtype=int)
        return event.astype(int) * (int(np.nanmax(bins)) + 1) + bins
    raise ValueError("survival_stratify must be 'event', 'event_time', or 'none'.")


class _SurvivalCV:
    """Small wrapper that stratifies survival splits on event status or event/time bins."""

    def __init__(
        self,
        n_splits: int,
        random_state: int,
        stratify: str = "event",
        time_bins: int = 4,
        group_aware: bool = False,
    ):
        self.n_splits = int(n_splits)
        self.random_state = random_state
        self.stratify = stratify
        self.time_bins = int(time_bins)
        self.group_aware = group_aware

    def split(self, X, y, groups=None):
        try:
            strata = _survival_strata(y, mode=self.stratify, time_bins=self.time_bins)
            counts = pd.Series(strata).value_counts()
            if len(counts) < 2 or int(counts.min()) < self.n_splits:
                raise ValueError(
                    "Not enough samples in each survival stratum for the requested number of splits."
                )
            if self.group_aware:
                splitter = StratifiedGroupKFold(
                    n_splits=self.n_splits, shuffle=True, random_state=self.random_state
                )
                yield from splitter.split(X, strata, groups)
            else:
                splitter = StratifiedKFold(
                    n_splits=self.n_splits, shuffle=True, random_state=self.random_state
                )
                yield from splitter.split(X, strata)
        except Exception as exc:
            warnings.warn(
                "Survival event-stratified CV failed; falling back to "
                f"{'GroupKFold' if self.group_aware else 'KFold'}. Reason: {exc}",
                RuntimeWarning,
            )
            if self.group_aware:
                yield from GroupKFold(n_splits=self.n_splits).split(X, y, groups)
            else:
                yield from KFold(
                    n_splits=self.n_splits, shuffle=True, random_state=self.random_state
                ).split(X, y)


def make_cv(
    task: str,
    n_splits: int,
    random_state: int,
    groups: Optional[Sequence] = None,
    survival_stratify: str = "event",
    survival_time_bins: int = 4,
):
    """Create a CV splitter for classification, regression, or survival.

    Classification uses class stratification. Survival uses event-stratified
    splitting by default, because small survival datasets can otherwise produce
    folds with too few observed events. If groups are supplied, group-aware CV is
    used so the same group cannot appear in both train and validation/test.
    """
    if task == "classification":
        if groups is None:
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    if task == "regression":
        if groups is None:
            return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return GroupKFold(n_splits=n_splits)
    if task == "survival":
        if survival_stratify not in {None, "none", "false", "off"}:
            return _SurvivalCV(
                n_splits=n_splits,
                random_state=random_state,
                stratify=survival_stratify,
                time_bins=survival_time_bins,
                group_aware=groups is not None,
            )
        if groups is None:
            return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return GroupKFold(n_splits=n_splits)
    raise ValueError("task must be one of: classification, regression, survival.")


def iter_split(cv, X: pd.DataFrame, y, groups: Optional[Sequence] = None):
    """Yield train/test indices from a possibly group-aware CV splitter."""
    if groups is None:
        return cv.split(X, y)
    return cv.split(X, y, groups)


def validate_survival_y(y: np.ndarray) -> None:
    """Validate a structured survival target."""
    if not is_survival_y(y):
        raise ValueError("survival target y must be a structured array with event and time fields.")
    event, time = get_event_time(y)
    if event.ndim != 1 or time.ndim != 1:
        raise ValueError("survival event/time fields must be one-dimensional.")
    if len(event) != len(time):
        raise ValueError("survival event/time fields must have the same length.")
    if len(event) == 0:
        raise ValueError("survival target y must not be empty.")
    if np.isnan(time).any():
        raise ValueError("survival time field must not contain NaN.")
    if np.any(time < 0):
        raise ValueError("survival time field must be non-negative.")
    if event.sum() == 0:
        raise ValueError("survival target must contain at least one observed event.")


def validate_task_y(task: str, y) -> None:
    """Validate target according to task."""
    if task not in VALID_TASKS:
        raise ValueError("task must be one of: classification, regression, survival.")
    if task == "classification":
        unique = sorted(pd.unique(as_series(y).dropna()))
        if unique != [0, 1]:
            raise ValueError(
                "classification currently expects a binary target encoded as 0/1. "
                f"Got classes={unique}."
            )
    elif task == "regression":
        y_series = as_series(y)
        if y_series.isna().any():
            raise ValueError("regression target y must not contain missing values.")
    else:
        validate_survival_y(y)
