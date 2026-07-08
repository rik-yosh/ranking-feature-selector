"""Feature ranking with optional shadow features.

For classification and regression, the default importance method is Tree SHAP
when the optional ``shap`` dependency is installed. For survival models such as
scikit-survival's RandomSurvivalForest, the default importance method is
permutation importance with C-index scoring, because RSF models usually do not
expose impurity-based ``feature_importances_``.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance

from .metrics import (
    classification_metrics,
    metric_direction,
    regression_metrics,
    survival_c_index_value,
)
from .survival import survival_risk_score
from .utils import target_reset, target_take


def _load_shap():
    """Import SHAP only when a SHAP-based backend is actually requested."""
    try:  # pragma: no cover - import itself is environment dependent
        import shap  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "SHAP-based importance requires the optional dependency 'shap'. "
            "Install it with: pip install ranking-feature-selector[shap] or pip install shap."
        ) from exc
    return shap


def transform_same_columns(transformer, X: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Apply a fitted transformer that must preserve feature count and order."""
    arr = transformer.transform(X.loc[:, columns])
    if hasattr(arr, "toarray"):
        arr = arr.toarray()
    arr = np.asarray(arr)
    if arr.ndim != 2 or arr.shape[1] != len(columns):
        raise ValueError(
            "The imputer/preprocessor must preserve the same number of selected columns. "
            "For one-hot encoding or other feature-expanding preprocessing, perform it before "
            "calling ranking_feature_selector or implement group-level selection separately."
        )
    return pd.DataFrame(arr, columns=list(columns), index=X.index)


def fit_transform_same_columns(transformer, X: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Fit-transform a same-column transformer and return a DataFrame."""
    arr = transformer.fit_transform(X.loc[:, columns])
    if hasattr(arr, "toarray"):
        arr = arr.toarray()
    arr = np.asarray(arr)
    if arr.ndim != 2 or arr.shape[1] != len(columns):
        raise ValueError(
            "The imputer/preprocessor must preserve the same number of selected columns. "
            "For one-hot encoding or other feature-expanding preprocessing, perform it before "
            "calling ranking_feature_selector or implement group-level selection separately."
        )
    return pd.DataFrame(arr, columns=list(columns), index=X.index)


def extract_shap_array(shap_values, task: str, positive_class_index: int = 1) -> np.ndarray:
    """Normalize SHAP values to shape ``(n_samples, n_features)``.

    Binary classification uses the positive class. Regression uses the single
    output, or averages absolute contributions across outputs for multi-output
    regressors.
    """
    if isinstance(shap_values, list):
        if task == "classification" and len(shap_values) > positive_class_index:
            values = shap_values[positive_class_index]
        else:
            values = shap_values[0]
    else:
        values = shap_values

    values = np.asarray(values)
    if values.ndim == 2:
        return values

    if values.ndim == 3:
        # SHAP often returns (n_samples, n_features, n_outputs/classes).
        if values.shape[0] != values.shape[2] and values.shape[1] >= 1:
            if task == "classification" and values.shape[2] > positive_class_index:
                return values[:, :, positive_class_index]
            return np.mean(np.abs(values), axis=2)

        # Some outputs are (n_outputs/classes, n_samples, n_features).
        if task == "classification" and values.shape[0] > positive_class_index:
            return values[positive_class_index, :, :]
        return np.mean(np.abs(values), axis=0)

    raise ValueError(f"Unsupported SHAP value shape: {values.shape}")


def _coef_importance(model, task: str, positive_class_index: int = 1) -> Optional[np.ndarray]:
    if not hasattr(model, "coef_"):
        return None
    coef = np.asarray(model.coef_)
    if coef.ndim == 1:
        return np.abs(coef)
    if coef.ndim == 2:
        if task == "classification" and coef.shape[0] > positive_class_index:
            return np.abs(coef[positive_class_index, :])
        return np.mean(np.abs(coef), axis=0)
    return None


def _predict_risk_or_score(model, X: pd.DataFrame, task: str, positive_class_index: int = 1):
    if task == "classification":
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            return np.asarray(proba[:, positive_class_index], dtype=float)
        return np.asarray(model.predict(X), dtype=float)
    return np.asarray(model.predict(X), dtype=float)


def _permutation_scorer(
    task: str,
    positive_class_index: int = 1,
    survival_risk_score_method: Any = "predict",
    prediction_time: Optional[float] = None,
    risk_score_direction: str = "higher",
    optimization_metric: Optional[str] = None,
    permutation_scoring: str = "auto",
):
    """Return a scorer suitable for ``sklearn.inspection.permutation_importance``.

    ``permutation_scoring='auto'`` aligns the permutation ranking with the same
    optimization metric used to choose the feature count, e.g. log loss for
    classification, RMSE for regression, and C-index for survival. This avoids
    ranking features by accuracy/R2 while choosing ``k`` by log loss/RMSE.

    Set ``permutation_scoring='native'`` to use ``estimator.score`` when present.
    """

    def scorer(estimator, X, y):
        if task == "survival":
            risk = survival_risk_score(
                estimator,
                X,
                risk_score=survival_risk_score_method,
                prediction_time=prediction_time,
                risk_score_direction=risk_score_direction,
            )
            return survival_c_index_value(y, risk)

        if permutation_scoring == "native":
            if hasattr(estimator, "score"):
                return estimator.score(X, y)
            warnings.warn(
                "permutation_scoring='native' was requested but the estimator does not expose "
                "score(). Falling back to permutation_scoring='auto'.",
                RuntimeWarning,
            )

        metric = optimization_metric if permutation_scoring in {None, "auto"} else permutation_scoring
        metric, lower_is_better = metric_direction(task, metric)
        pred = _predict_risk_or_score(estimator, X, task, positive_class_index)

        if task == "classification":
            metrics = classification_metrics(y, pred)
        elif task == "regression":
            metrics = regression_metrics(y, pred)
        else:  # pragma: no cover - survival handled above
            raise ValueError("Unsupported task for permutation scorer.")

        value = float(metrics[metric])
        if np.isnan(value):
            return value
        return -value if lower_is_better else value

    return scorer


def model_importance(
    model,
    X_explain: pd.DataFrame,
    task: str,
    y_explain=None,
    positive_class_index: int = 1,
    importance_method: str = "auto",
    permutation_n_repeats: int = 10,
    permutation_random_state: int = 0,
    permutation_n_jobs: Optional[int] = None,
    survival_risk_score_method: Any = "predict",
    prediction_time: Optional[float] = None,
    risk_score_direction: str = "higher",
    optimization_metric: Optional[str] = None,
    permutation_scoring: str = "auto",
) -> np.ndarray:
    """Return feature importances.

    Parameters
    ----------
    importance_method:
        ``"auto"`` uses SHAP for classification/regression and permutation
        importance for survival. ``"shap"`` forces Tree SHAP. ``"permutation"``
        uses ``sklearn.inspection.permutation_importance``.
    permutation_scoring:
        ``"auto"`` aligns permutation importance with ``optimization_metric``.
        ``"native"`` uses the estimator's own ``score`` method.
    """
    if importance_method not in {"auto", "shap", "permutation"}:
        raise ValueError("importance_method must be 'auto', 'shap', or 'permutation'.")
    method = "permutation" if importance_method == "auto" and task == "survival" else importance_method
    method = "shap" if method == "auto" else method

    if method == "permutation":
        if y_explain is None:
            raise ValueError("y_explain is required for permutation importance.")
        scoring = _permutation_scorer(
            task,
            positive_class_index=positive_class_index,
            survival_risk_score_method=survival_risk_score_method,
            prediction_time=prediction_time,
            risk_score_direction=risk_score_direction,
            optimization_metric=optimization_metric,
            permutation_scoring=permutation_scoring,
        )
        result = permutation_importance(
            model,
            X_explain,
            y_explain,
            scoring=scoring,
            n_repeats=permutation_n_repeats,
            random_state=permutation_random_state,
            n_jobs=permutation_n_jobs,
        )
        return np.asarray(result.importances_mean, dtype=float)

    try:
        shap = _load_shap()
        explainer = shap.TreeExplainer(model)
        raw_values = explainer.shap_values(X_explain)
        values = extract_shap_array(raw_values, task=task, positive_class_index=positive_class_index)
        if values.shape[1] != X_explain.shape[1]:
            raise ValueError(
                f"SHAP value feature dimension mismatch: got {values.shape[1]}, "
                f"expected {X_explain.shape[1]}."
            )
        return np.abs(values).mean(axis=0)
    except Exception as shap_error:
        if hasattr(model, "feature_importances_"):
            try:
                return np.asarray(model.feature_importances_, dtype=float)
            except Exception:
                # Some estimators expose an attribute/property that raises
                # NotImplementedError, e.g. scikit-survival RSF.
                pass
            warnings.warn(
                "Tree SHAP failed and model.feature_importances_ could not be used. "
                f"Original error: {repr(shap_error)}",
                RuntimeWarning,
            )
        coef_imp = _coef_importance(model, task=task, positive_class_index=positive_class_index)
        if coef_imp is not None:
            warnings.warn(
                "Tree SHAP failed; falling back to absolute model coefficients. "
                f"Original error: {repr(shap_error)}",
                RuntimeWarning,
            )
            return coef_imp
        raise


def add_shadow_features(X: pd.DataFrame, rng: np.random.Generator) -> Tuple[pd.DataFrame, List[str]]:
    """Append independently permuted copies of all features."""
    shadow = {}
    shadow_names = []
    for col in X.columns:
        name = f"__shadow__{col}"
        shadow[name] = rng.permutation(X[col].to_numpy())
        shadow_names.append(name)
    X_shadow = pd.DataFrame(shadow, index=X.index)
    return pd.concat([X, X_shadow], axis=1), shadow_names


def shap_feature_ranking(
    X_train: pd.DataFrame,
    y_train,
    model,
    task: str,
    imputer=None,
    sampler=None,
    random_state: int = 0,
    use_shadow: bool = True,
    shadow_quantile: float = 1.0,
    shap_sample_size: Optional[int] = 300,
    positive_class_index: int = 1,
    importance_method: str = "auto",
    permutation_n_repeats: int = 10,
    permutation_n_jobs: Optional[int] = None,
    survival_risk_score_method: Any = "predict",
    prediction_time: Optional[float] = None,
    risk_score_direction: str = "higher",
    optimization_metric: Optional[str] = None,
    permutation_scoring: str = "auto",
) -> pd.DataFrame:
    """Fit preprocessing, optional sampler, and model, then rank features.

    Parameters
    ----------
    task:
        ``"classification"``, ``"regression"``, or ``"survival"``.
    sampler:
        Optional imbalanced-learn sampler. It is valid only for classification.
    importance_method:
        ``"auto"`` uses Tree SHAP for classification/regression and permutation
        C-index importance for survival models.
    use_shadow:
        If true, permuted shadow features are added and each real feature is
        compared with the shadow-importance threshold.
    shadow_quantile:
        Quantile of shadow importances used as the null threshold. ``1.0`` means
        the maximum shadow importance and is deliberately conservative.
    shap_sample_size:
        Number of original training rows used to compute importances. ``None``
        uses all rows.
    """
    if task != "classification" and sampler is not None:
        raise ValueError("sampler is classification-only; pass sampler=None for regression/survival.")
    if not 0 < shadow_quantile <= 1:
        raise ValueError("shadow_quantile must be in (0, 1].")

    if imputer is None:
        imputer = SimpleImputer(strategy="median")

    X_train = X_train.reset_index(drop=True)
    y_train = target_reset(y_train, task)
    real_features = list(X_train.columns)

    imp = clone(imputer)
    X_imp = fit_transform_same_columns(imp, X_train, real_features)

    rng = np.random.default_rng(random_state)
    if use_shadow:
        X_model, shadow_features = add_shadow_features(X_imp, rng)
    else:
        X_model, shadow_features = X_imp, []

    if sampler is not None:
        X_fit, y_fit = clone(sampler).fit_resample(X_model, y_train)
        X_fit = pd.DataFrame(X_fit, columns=X_model.columns)
        y_fit = pd.Series(y_fit)
    else:
        X_fit, y_fit = X_model, y_train

    fitted_model = clone(model)
    fitted_model.fit(X_fit, y_fit)

    if shap_sample_size is not None and len(X_model) > shap_sample_size:
        explain_idx = rng.choice(len(X_model), size=shap_sample_size, replace=False)
        X_explain = X_model.iloc[explain_idx].copy()
        y_explain = target_take(y_train, explain_idx, task)
    else:
        X_explain = X_model.copy()
        y_explain = y_train

    importances = model_importance(
        fitted_model,
        X_explain,
        task=task,
        y_explain=y_explain,
        positive_class_index=positive_class_index,
        importance_method=importance_method,
        permutation_n_repeats=permutation_n_repeats,
        permutation_random_state=random_state,
        permutation_n_jobs=permutation_n_jobs,
        survival_risk_score_method=survival_risk_score_method,
        prediction_time=prediction_time,
        risk_score_direction=risk_score_direction,
        optimization_metric=optimization_metric,
        permutation_scoring=permutation_scoring,
    )
    if len(importances) != X_model.shape[1]:
        raise ValueError(
            f"Importance length mismatch: got {len(importances)}, expected {X_model.shape[1]}."
        )

    all_imp = pd.DataFrame({"feature": list(X_model.columns), "importance": importances})
    real = all_imp[all_imp["feature"].isin(real_features)].copy()

    if use_shadow:
        shadow_imp = all_imp[all_imp["feature"].isin(shadow_features)]["importance"].to_numpy()
        shadow_threshold = float(np.quantile(shadow_imp, shadow_quantile))
        real["shadow_threshold"] = shadow_threshold
        real["passes_shadow"] = real["importance"] > shadow_threshold
        real["importance_minus_shadow"] = real["importance"] - shadow_threshold
    else:
        real["shadow_threshold"] = np.nan
        real["passes_shadow"] = True
        real["importance_minus_shadow"] = real["importance"]

    real = real.sort_values(
        ["importance", "importance_minus_shadow"], ascending=False
    ).reset_index(drop=True)
    real["rank"] = np.arange(1, len(real) + 1)
    return real
