"""Survival-model risk-score utilities.

The selector uses survival risk scores only for within-fold rank-based
metrics such as Harrell's C-index. Raw scores are not assumed to be calibrated
or comparable across CV folds.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

RiskScoreSpec = Union[str, Callable[[object, pd.DataFrame], object]]


def _as_1d_float(values, name: str = "risk score") -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        arr = np.ravel(arr)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return arr


def _clip_time_to_step_domain(fn, prediction_time: float) -> float:
    """Clip a time point to the domain of a sksurv StepFunction if available."""
    t = float(prediction_time)
    x = getattr(fn, "x", None)
    if x is None:
        return t
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return t
    return float(np.clip(t, np.nanmin(x), np.nanmax(x)))


def _evaluate_step_functions(functions, prediction_time: float) -> np.ndarray:
    values = []
    for fn in functions:
        t = _clip_time_to_step_domain(fn, prediction_time)
        values.append(float(fn(t)))
    return np.asarray(values, dtype=float)


def _survival_probability_at_time(model, X: pd.DataFrame, prediction_time: float) -> np.ndarray:
    if not hasattr(model, "predict_survival_function"):
        raise AttributeError(
            "risk_score='event_probability' requires a model with "
            "predict_survival_function(X)."
        )
    try:
        surv_fns = model.predict_survival_function(X, return_array=False)
    except TypeError:
        surv_fns = model.predict_survival_function(X)
    return _evaluate_step_functions(surv_fns, prediction_time)


def _cumulative_hazard_at_time(model, X: pd.DataFrame, prediction_time: float) -> np.ndarray:
    if not hasattr(model, "predict_cumulative_hazard_function"):
        raise AttributeError(
            "risk_score='cumulative_hazard' requires a model with "
            "predict_cumulative_hazard_function(X)."
        )
    try:
        chf_fns = model.predict_cumulative_hazard_function(X, return_array=False)
    except TypeError:
        chf_fns = model.predict_cumulative_hazard_function(X)
    return _evaluate_step_functions(chf_fns, prediction_time)


def survival_risk_score(
    model,
    X: pd.DataFrame,
    risk_score: RiskScoreSpec = "predict",
    prediction_time: Optional[float] = None,
    risk_score_direction: str = "higher",
) -> np.ndarray:
    """Return a one-dimensional survival risk score.

    Parameters
    ----------
    model:
        Fitted survival estimator.
    X:
        Feature matrix.
    risk_score:
        How to obtain the raw risk score.

        - ``"predict"`` or ``"auto"``: use ``model.predict(X)``.
        - ``"event_probability"``: use ``1 - S(t | X)`` at ``prediction_time``.
        - ``"cumulative_hazard"``: use ``H(t | X)`` at ``prediction_time``.
        - callable: custom ``callable(model, X) -> array-like``.

    prediction_time:
        Fixed time horizon for ``"event_probability"`` or ``"cumulative_hazard"``.
    risk_score_direction:
        ``"higher"`` means larger returned values indicate higher event risk.
        ``"lower"`` means smaller raw values indicate higher event risk; the raw
        score is multiplied by ``-1`` before being returned.

    Notes
    -----
    The returned scores are intended for rank-based metrics within the same CV
    fold. They are not treated as calibrated risks or as comparable across folds.
    """
    if risk_score_direction not in {"higher", "lower"}:
        raise ValueError("risk_score_direction must be 'higher' or 'lower'.")

    if callable(risk_score):
        raw = risk_score(model, X)
    else:
        method = str(risk_score).lower().strip()
        if method in {"auto", "predict", "risk", "risk_score"}:
            if not hasattr(model, "predict"):
                raise AttributeError("risk_score='predict' requires a model with predict(X).")
            raw = model.predict(X)
        elif method in {"event_probability", "event", "risk_probability", "1-survival"}:
            if prediction_time is None:
                raise ValueError(
                    "prediction_time must be specified when risk_score='event_probability'."
                )
            raw = 1.0 - _survival_probability_at_time(model, X, float(prediction_time))
        elif method in {"cumulative_hazard", "cumhaz", "chf"}:
            if prediction_time is None:
                raise ValueError(
                    "prediction_time must be specified when risk_score='cumulative_hazard'."
                )
            raw = _cumulative_hazard_at_time(model, X, float(prediction_time))
        else:
            raise ValueError(
                "risk_score must be 'predict', 'event_probability', 'cumulative_hazard', "
                "or a callable(model, X)."
            )

    score = _as_1d_float(raw, name="survival risk score")
    if risk_score_direction == "lower":
        score = -score
    return score
