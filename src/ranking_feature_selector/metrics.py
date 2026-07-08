"""Metric functions for classification, regression, and survival."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from .utils import get_event_time


def classification_metrics(
    y_true: Sequence[int],
    y_proba: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute binary classification metrics using a fixed probability threshold."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)
    y_proba = np.clip(y_proba, 1e-15, 1 - 1e-15)
    y_pred = (y_proba >= threshold).astype(int)

    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    out = {
        "log_loss": log_loss(y_true, y_proba, labels=[0, 1]),
        "brier": brier_score_loss(y_true, y_proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    try:
        out["auroc"] = roc_auc_score(y_true, y_proba)
    except ValueError:
        out["auroc"] = np.nan
    try:
        out["auprc"] = average_precision_score(y_true, y_proba)
    except ValueError:
        out["auprc"] = np.nan
    return out


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    """Compute common regression metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "median_ae": median_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def _harrell_c_index_fallback(event: np.ndarray, time: np.ndarray, risk: np.ndarray) -> tuple[float, int]:
    """Small, dependency-free Harrell C-index fallback.

    This fallback is O(n^2) and intended only for environments where
    scikit-survival is unavailable. It follows the common comparable-pair rule:
    a pair is comparable if the shorter observed time is an event.
    """
    concordant = 0.0
    comparable = 0
    n = len(time)
    for i in range(n):
        for j in range(i + 1, n):
            if time[i] == time[j]:
                continue
            if time[i] < time[j] and event[i]:
                comparable += 1
                if risk[i] > risk[j]:
                    concordant += 1.0
                elif risk[i] == risk[j]:
                    concordant += 0.5
            elif time[j] < time[i] and event[j]:
                comparable += 1
                if risk[j] > risk[i]:
                    concordant += 1.0
                elif risk[i] == risk[j]:
                    concordant += 0.5
    if comparable == 0:
        return np.nan, comparable
    return float(concordant / comparable), comparable


def survival_c_index_value(y_true, risk_score: Sequence[float]) -> float:
    """Return Harrell's concordance index for survival risk scores."""
    event, time = get_event_time(y_true)
    risk = np.asarray(risk_score, dtype=float)
    if len(risk) != len(time):
        raise ValueError("risk_score and y_true must have the same length.")

    try:
        from sksurv.metrics import concordance_index_censored

        return float(concordance_index_censored(event, time, risk)[0])
    except Exception:
        c_index, _ = _harrell_c_index_fallback(event, time, risk)
        return c_index


def survival_metrics(y_true, risk_score: Sequence[float]) -> Dict[str, float]:
    """Compute survival metrics from a predicted risk score.

    The default metric is Harrell's C-index. Higher risk scores are assumed to
    correspond to a shorter time-to-event.
    """
    event, time = get_event_time(y_true)
    risk = np.asarray(risk_score, dtype=float)
    c_index = survival_c_index_value(y_true, risk)
    _, n_comparable = _harrell_c_index_fallback(event, time, risk)
    return {
        "c_index": c_index,
        "n_events": int(event.sum()),
        "n_censored": int((~event).sum()),
        "n_comparable": int(n_comparable),
    }


def metric_direction(task: str, optimization_metric: str | None = None) -> tuple[str, bool]:
    """Return default optimization metric and whether lower values are better."""
    if task == "classification":
        metric = optimization_metric or "log_loss"
        lower_is_better = metric in {"log_loss", "brier"}
        if metric not in {"log_loss", "brier", "auroc", "auprc", "accuracy", "f1"}:
            raise ValueError(
                "Unsupported classification optimization_metric. Use one of: "
                "log_loss, brier, auroc, auprc, accuracy, f1."
            )
        return metric, lower_is_better
    if task == "regression":
        metric = optimization_metric or "rmse"
        lower_is_better = metric in {"rmse", "mse", "mae", "median_ae"}
        if metric not in {"rmse", "mse", "mae", "median_ae", "r2"}:
            raise ValueError(
                "Unsupported regression optimization_metric. Use one of: "
                "rmse, mse, mae, median_ae, r2."
            )
        return metric, lower_is_better
    if task == "survival":
        metric = optimization_metric or "c_index"
        lower_is_better = False
        if metric not in {"c_index"}:
            raise ValueError("Unsupported survival optimization_metric. Use: c_index.")
        return metric, lower_is_better
    raise ValueError("task must be one of: classification, regression, survival.")
