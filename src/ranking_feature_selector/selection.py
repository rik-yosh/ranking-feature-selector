"""Leakage-safe nested-CV feature selection."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
import platform
import warnings

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer

from .importance import fit_transform_same_columns, shap_feature_ranking, transform_same_columns
from .metrics import classification_metrics, metric_direction, regression_metrics, survival_metrics
from .results import FittedModelBundle, NestedShapFSResult
from .preprocessing import make_preprocessor
from .survival import survival_risk_score
from .utils import (
    as_dataframe,
    as_series,
    as_target,
    iter_split,
    make_cv,
    make_k_grid,
    target_len,
    target_take,
    validate_task_y,
)


def _default_survival_model(random_state: int):
    """Create a default scikit-survival RandomSurvivalForest."""
    try:
        from sksurv.ensemble import RandomSurvivalForest
    except ImportError as exc:
        raise ImportError(
            "Survival task requires scikit-survival. Install with: "
            "pip install -e '.[survival]' or pip install scikit-survival."
        ) from exc

    return RandomSurvivalForest(
        n_estimators=500,
        min_samples_split=6,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )


def _default_model_for_task(task: str, random_state: int):
    """Create a reasonable default estimator for a task."""
    if task == "classification":
        return RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )
    if task == "regression":
        return RandomForestRegressor(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )
    if task == "survival":
        return _default_survival_model(random_state)
    raise ValueError("task must be one of: classification, regression, survival.")


def _predict_vector(
    model,
    X: pd.DataFrame,
    task: str,
    positive_class_index: int = 1,
    survival_risk_score_method: Any = "predict",
    prediction_time: Optional[float] = None,
    risk_score_direction: str = "higher",
) -> np.ndarray:
    """Return positive-class probabilities, regression predictions, or survival risk scores."""
    if task == "classification":
        if not hasattr(model, "predict_proba"):
            raise AttributeError("Classification model must implement predict_proba.")
        proba = model.predict_proba(X)
        if proba.ndim != 2 or proba.shape[1] <= positive_class_index:
            raise ValueError(
                "predict_proba must return shape (n_samples, n_classes) with the requested "
                f"positive_class_index={positive_class_index}."
            )
        return np.asarray(proba[:, positive_class_index], dtype=float)
    if task == "survival":
        return survival_risk_score(
            model,
            X,
            risk_score=survival_risk_score_method,
            prediction_time=prediction_time,
            risk_score_direction=risk_score_direction,
        )
    return np.asarray(model.predict(X), dtype=float)


def compute_metrics(
    task: str,
    y_true,
    y_pred_or_proba: Sequence,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute task-specific metrics."""
    if task == "classification":
        return classification_metrics(y_true, y_pred_or_proba, threshold=threshold)
    if task == "regression":
        return regression_metrics(y_true, y_pred_or_proba)
    if task == "survival":
        return survival_metrics(y_true, y_pred_or_proba)
    raise ValueError("task must be one of: classification, regression, survival.")


def fit_predict_on_features(
    X_train: pd.DataFrame,
    y_train,
    X_valid: pd.DataFrame,
    features: Sequence[str],
    model,
    task: str,
    imputer=None,
    preprocessor=None,
    standardize: bool = False,
    sampler=None,
    positive_class_index: int = 1,
    survival_risk_score_method: Any = "predict",
    prediction_time: Optional[float] = None,
    risk_score_direction: str = "higher",
) -> np.ndarray:
    """Fit preprocessing + optional sampler + model on selected features and predict validation.

    ``imputer`` is retained as a backward-compatible alias. New code should use
    ``preprocessor``. The preprocessor must preserve the selected columns.
    """
    if len(features) == 0:
        raise ValueError("features must contain at least one feature.")
    if task != "classification" and sampler is not None:
        raise ValueError("sampler is classification-only; pass sampler=None for regression/survival.")
    if preprocessor is not None and imputer is not None:
        raise ValueError("Pass either preprocessor or imputer, not both.")
    if preprocessor is None:
        preprocessor = imputer
    preprocessor = make_preprocessor(preprocessor, standardize=standardize)

    features = list(features)
    prep = clone(preprocessor)
    X_train_prep = fit_transform_same_columns(prep, X_train, features)
    X_valid_prep = transform_same_columns(prep, X_valid, features)

    if sampler is not None:
        X_fit, y_fit = clone(sampler).fit_resample(X_train_prep, y_train)
        X_fit = pd.DataFrame(X_fit, columns=features)
        y_fit = pd.Series(y_fit)
    else:
        X_fit, y_fit = X_train_prep, y_train

    fitted_model = clone(model)
    fitted_model.fit(X_fit, y_fit)
    return _predict_vector(
        fitted_model,
        X_valid_prep,
        task=task,
        positive_class_index=positive_class_index,
        survival_risk_score_method=survival_risk_score_method,
        prediction_time=prediction_time,
        risk_score_direction=risk_score_direction,
    )


def fit_final_model_on_features(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train,
    features: Sequence[str],
    model,
    task: str = "classification",
    imputer=None,
    preprocessor=None,
    standardize: bool = False,
    sampler=None,
    positive_class_index: int = 1,
    event_col: Optional[str] = None,
    time_col: Optional[str] = None,
    survival_risk_score_method: Any = "predict",
    prediction_time: Optional[float] = None,
    risk_score_direction: str = "higher",
) -> FittedModelBundle:
    """Fit final preprocessing and model on a fixed feature set.

    This should be called only after the feature-selection protocol and all tuning
    decisions have been fixed. For classification, optional samplers are fitted
    only on the provided training data. Survival targets can be supplied as a
    structured array, ``(event, time)`` tuple, two-column array, or DataFrame.
    """
    X_train = as_dataframe(X_train)
    y_train = as_target(y_train, task=task, event_col=event_col, time_col=time_col)
    validate_task_y(task, y_train)

    if target_len(y_train) != len(X_train):
        raise ValueError("X_train and y_train must have the same number of rows.")
    if len(features) == 0:
        raise ValueError("features must contain at least one feature.")
    if task != "classification" and sampler is not None:
        raise ValueError("sampler is classification-only; pass sampler=None for regression/survival.")
    if preprocessor is not None and imputer is not None:
        raise ValueError("Pass either preprocessor or imputer, not both.")
    if preprocessor is None:
        preprocessor = imputer
    preprocessor = make_preprocessor(preprocessor, standardize=standardize)

    features = list(features)
    feature_names_in = list(X_train.columns)
    missing = sorted(set(features) - set(feature_names_in))
    if missing:
        raise ValueError(f"The following features are missing from X_train: {missing}")
    name_to_idx = {name: i for i, name in enumerate(feature_names_in)}
    selected_indices = [name_to_idx[f] for f in features]

    prep = clone(preprocessor)
    X_prep = fit_transform_same_columns(prep, X_train, features)

    sampler_fit = None
    if sampler is not None:
        sampler_fit = clone(sampler)
        X_fit, y_fit = sampler_fit.fit_resample(X_prep, y_train)
        X_fit = pd.DataFrame(X_fit, columns=features)
        y_fit = pd.Series(y_fit)
    else:
        X_fit, y_fit = X_prep, y_train

    fitted_model = clone(model)
    fitted_model.fit(X_fit, y_fit)
    return FittedModelBundle(
        task=task,
        features=features,
        preprocessor=prep,
        model=fitted_model,
        sampler=sampler_fit,
        positive_class_index=positive_class_index,
        survival_risk_score_method=survival_risk_score_method,
        prediction_time=prediction_time,
        risk_score_direction=risk_score_direction,
        feature_names_in_=feature_names_in,
        selected_indices_=selected_indices,
    )


def predict_with_fitted(fitted: FittedModelBundle, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """Return the positive-class probability, regression prediction, or survival risk score."""
    return fitted.predict_score(X)


def _choose_k(
    inner_summary: pd.DataFrame,
    optimization_metric: str,
    lower_is_better: bool,
    rule: str = "one_se",
) -> int:
    """Choose a feature count from an inner-CV summary."""
    if inner_summary.empty:
        raise ValueError("inner_summary is empty; cannot choose k.")
    if optimization_metric not in inner_summary.columns:
        raise ValueError(f"optimization_metric={optimization_metric!r} is not in inner_summary.")
    if rule not in {"min", "one_se"}:
        raise ValueError("choose_k_rule must be 'min' or 'one_se'.")

    summary = inner_summary.sort_values("k").copy()
    se_col = f"se_{optimization_metric}"
    if se_col not in summary.columns:
        summary[se_col] = 0.0

    if lower_is_better:
        best_idx = summary[optimization_metric].idxmin()
        best = summary.loc[best_idx]
        if rule == "min":
            return int(best["k"])
        limit = float(best[optimization_metric] + best[se_col])
        eligible = summary[summary[optimization_metric] <= limit]
        return int(eligible["k"].min())

    best_idx = summary[optimization_metric].idxmax()
    best = summary.loc[best_idx]
    if rule == "min":
        return int(best["k"])
    limit = float(best[optimization_metric] - best[se_col])
    eligible = summary[summary[optimization_metric] >= limit]
    return int(eligible["k"].min())


def _metric_columns(task: str) -> list[str]:
    if task == "classification":
        return [
            "log_loss",
            "brier",
            "accuracy",
            "sensitivity",
            "specificity",
            "precision",
            "f1",
            "auroc",
            "auprc",
        ]
    if task == "regression":
        return ["mae", "mse", "rmse", "median_ae", "r2"]
    if task == "survival":
        return ["c_index", "n_events", "n_censored", "n_comparable"]
    raise ValueError("task must be one of: classification, regression, survival.")


def _summarize_inner_details(
    detail: pd.DataFrame,
    task: str,
    n_inner: int,
) -> pd.DataFrame:
    metric_cols = _metric_columns(task)
    agg_spec = {f"mean_{m}": (m, "mean") for m in metric_cols if m in detail.columns}
    agg_spec.update({f"std_{m}": (m, "std") for m in metric_cols if m in detail.columns})
    first_metric = metric_cols[0]
    agg_spec["n_folds"] = (first_metric, "count")

    summary = detail.groupby("k", as_index=False).agg(**agg_spec).sort_values("k")
    for m in metric_cols:
        mean_col = f"mean_{m}"
        std_col = f"std_{m}"
        se_col = f"se_{m}"
        if mean_col in summary.columns:
            summary[m] = summary[mean_col]
        if std_col in summary.columns:
            summary[se_col] = summary[std_col].fillna(0.0) / np.sqrt(summary["n_folds"])

    complete = summary[summary["n_folds"] == n_inner].copy()
    if complete.empty:
        warnings.warn(
            "No k was evaluated in all inner folds. Falling back to incomplete summary. "
            "Consider using shadow_filter=False or a smaller k_grid.",
            RuntimeWarning,
        )
        complete = summary.copy()
    return complete


def evaluate_k_grid_strict_inner_cv(
    X: pd.DataFrame,
    y,
    model,
    task: str,
    k_grid: Sequence[int],
    inner_cv,
    imputer=None,
    sampler=None,
    groups: Optional[Sequence] = None,
    random_state: int = 0,
    use_shadow: bool = True,
    shadow_quantile: float = 1.0,
    shadow_filter: bool = False,
    shap_sample_size: Optional[int] = 300,
    threshold: float = 0.5,
    positive_class_index: int = 1,
    importance_method: str = "auto",
    permutation_n_repeats: int = 10,
    permutation_n_jobs: Optional[int] = None,
    optimization_metric: Optional[str] = None,
    permutation_scoring: str = "auto",
    survival_risk_score_method: Any = "predict",
    prediction_time: Optional[float] = None,
    risk_score_direction: str = "higher",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate each candidate ``k`` using strict inner CV.

    The feature ranking is recomputed inside every inner-training fold. This is
    the key leakage-control step: an inner-validation fold is never used to rank
    the features that are evaluated on that same fold.
    """
    if task != "classification" and sampler is not None:
        raise ValueError("sampler is classification-only; pass sampler=None for regression/survival.")

    rows = []
    rankings = []
    n_inner = 0

    groups_arr = None if groups is None else np.asarray(groups)
    for inner_i, (tr_idx, va_idx) in enumerate(iter_split(inner_cv, X, y, groups_arr), start=1):
        n_inner += 1
        X_tr = X.iloc[tr_idx].reset_index(drop=True)
        y_tr = target_take(y, tr_idx, task)
        X_va = X.iloc[va_idx].reset_index(drop=True)
        y_va = target_take(y, va_idx, task)

        ranking = shap_feature_ranking(
            X_tr,
            y_tr,
            model=model,
            task=task,
            imputer=imputer,
            sampler=sampler,
            random_state=random_state + inner_i,
            use_shadow=use_shadow,
            shadow_quantile=shadow_quantile,
            shap_sample_size=shap_sample_size,
            positive_class_index=positive_class_index,
            importance_method=importance_method,
            permutation_n_repeats=permutation_n_repeats,
            permutation_n_jobs=permutation_n_jobs,
            optimization_metric=optimization_metric,
            permutation_scoring=permutation_scoring,
            survival_risk_score_method=survival_risk_score_method,
            prediction_time=prediction_time,
            risk_score_direction=risk_score_direction,
        )
        ranking["inner_fold"] = inner_i
        rankings.append(ranking)

        if shadow_filter:
            ranked_features = ranking.loc[ranking["passes_shadow"], "feature"].tolist()
            if len(ranked_features) == 0:
                ranked_features = ranking["feature"].head(1).tolist()
        else:
            ranked_features = ranking["feature"].tolist()

        for k in k_grid:
            k = int(k)
            if k > len(ranked_features):
                continue
            features_k = ranked_features[:k]
            pred = fit_predict_on_features(
                X_tr,
                y_tr,
                X_va,
                features=features_k,
                model=model,
                task=task,
                imputer=imputer,
                sampler=sampler,
                positive_class_index=positive_class_index,
                survival_risk_score_method=survival_risk_score_method,
                prediction_time=prediction_time,
                risk_score_direction=risk_score_direction,
            )
            metrics = compute_metrics(task, y_va, pred, threshold=threshold)
            rows.append(
                {
                    "inner_fold": inner_i,
                    "k": k,
                    "n_features_evaluated": len(features_k),
                    **metrics,
                }
            )

    detail = pd.DataFrame(rows)
    if detail.empty:
        raise RuntimeError("No inner-CV evaluations were completed. Check k_grid/shadow_filter.")

    summary = _summarize_inner_details(detail, task=task, n_inner=n_inner)
    rankings_df = pd.concat(rankings, ignore_index=True)
    return summary, detail, rankings_df


def nested_shap_feature_selection_cv(
    X: Union[pd.DataFrame, np.ndarray],
    y,
    model=None,
    task: str = "classification",
    imputer=None,
    sampler=None,
    outer_splits: int = 5,
    inner_splits: int = 4,
    random_state: int = 42,
    k_grid: Optional[Sequence[int]] = None,
    max_k: Optional[int] = None,
    choose_k_rule: str = "one_se",
    optimization_metric: Optional[str] = None,
    use_shadow: bool = True,
    shadow_quantile: float = 1.0,
    shadow_filter: bool = False,
    shap_sample_size: Optional[int] = 300,
    groups: Optional[Sequence] = None,
    final_min_frequency: float = 0.5,
    max_final_features: Optional[int] = None,
    threshold: float = 0.5,
    positive_class_index: int = 1,
    event_col: Optional[str] = None,
    time_col: Optional[str] = None,
    importance_method: str = "auto",
    permutation_n_repeats: int = 10,
    permutation_n_jobs: Optional[int] = None,
    permutation_scoring: str = "auto",
    survival_stratify: str = "event",
    survival_time_bins: int = 4,
    survival_risk_score_method: Any = "predict",
    prediction_time: Optional[float] = None,
    risk_score_direction: str = "higher",
    verbose: bool = True,
) -> NestedShapFSResult:
    """Run leakage-safe nested CV for feature selection.

    Parameters
    ----------
    task:
        ``"classification"`` for binary 0/1 outcomes, ``"regression"`` for
        continuous outcomes, or ``"survival"`` for right-censored time-to-event
        outcomes. For survival, ``y`` can be a scikit-survival structured array,
        ``(event, time)`` tuple, two-column array, or DataFrame.
    optimization_metric:
        Inner-loop metric used to choose ``k``. Defaults are ``log_loss`` for
        classification, ``rmse`` for regression, and ``c_index`` for survival.
    importance_method:
        ``"auto"`` uses Tree SHAP for classification/regression and permutation
        C-index importance for survival. You can force ``"shap"`` or
        ``"permutation"``.
    choose_k_rule:
        ``"min"`` chooses the empirically best ``k``. ``"one_se"`` chooses the
        smallest ``k`` within one standard error of the best inner-CV score.
    sampler:
        Optional imbalanced-learn sampler for classification only. It is applied
        only inside training folds.
    groups:
        Optional group labels for group-aware outer and inner CV.
    final_min_frequency:
        Outer-fold selection frequency threshold for constructing ``final_features``.
    max_final_features:
        Optional hard cap on the number of features returned in ``final_features``.
        This is separate from ``max_k``: ``max_k`` limits each fold-specific
        selected feature set and the inner-CV search range, while
        ``max_final_features`` limits the stability-aggregated final feature list.
    """
    X = as_dataframe(X)
    y = as_target(y, task=task, event_col=event_col, time_col=time_col)
    if len(X) != target_len(y):
        raise ValueError("X and y must have the same number of rows.")
    validate_task_y(task, y)

    if task != "classification" and sampler is not None:
        raise ValueError("sampler is classification-only; pass sampler=None for regression/survival.")
    if task == "survival" and risk_score_direction not in {"higher", "lower"}:
        raise ValueError("risk_score_direction must be 'higher' or 'lower'.")

    if model is None:
        model = _default_model_for_task(task, random_state)
    if imputer is None:
        imputer = SimpleImputer(strategy="median")
    if not 0 < final_min_frequency <= 1:
        raise ValueError("final_min_frequency must be in (0, 1].")
    if max_final_features is not None:
        max_final_features = int(max_final_features)
        if max_final_features < 1:
            raise ValueError("max_final_features must be at least 1 when specified.")

    optimization_metric, lower_is_better = metric_direction(task, optimization_metric)
    if max_k is not None:
        max_k = int(max_k)
        if max_k < 1:
            raise ValueError("max_k must be at least 1 when specified.")

    if k_grid is None:
        k_grid = make_k_grid(X.shape[1], max_k=max_k)
    else:
        max_allowed_k = X.shape[1] if max_k is None else min(max_k, X.shape[1])
        k_grid = sorted({int(k) for k in k_grid if 1 <= int(k) <= max_allowed_k})
    if len(k_grid) == 0:
        raise ValueError("k_grid has no valid values after applying max_k and feature-count bounds.")

    outer_cv = make_cv(
        task,
        outer_splits,
        random_state,
        groups=groups,
        survival_stratify=survival_stratify,
        survival_time_bins=survival_time_bins,
    )
    groups_arr = None if groups is None else np.asarray(groups)

    outer_rows = []
    inner_summary_rows = []
    inner_detail_rows = []
    selected_by_fold: Dict[str, list[str]] = {}
    rankings_by_fold: Dict[str, pd.DataFrame] = {}
    chosen_k_by_fold: Dict[str, int] = {}

    for outer_i, (tr_idx, te_idx) in enumerate(iter_split(outer_cv, X, y, groups_arr), start=1):
        fold_name = f"outer_{outer_i}"
        if verbose:
            print(f"\n[{fold_name}] training={len(tr_idx)}, test={len(te_idx)}")

        X_tr = X.iloc[tr_idx].reset_index(drop=True)
        y_tr = target_take(y, tr_idx, task)
        X_te = X.iloc[te_idx].reset_index(drop=True)
        y_te = target_take(y, te_idx, task)
        groups_tr = None if groups_arr is None else groups_arr[tr_idx]

        inner_cv = make_cv(
            task,
            inner_splits,
            random_state + 1000 + outer_i,
            groups=groups_tr,
            survival_stratify=survival_stratify,
            survival_time_bins=survival_time_bins,
        )
        inner_summary, inner_detail, _inner_rankings = evaluate_k_grid_strict_inner_cv(
            X=X_tr,
            y=y_tr,
            model=model,
            task=task,
            k_grid=k_grid,
            inner_cv=inner_cv,
            imputer=imputer,
            sampler=sampler,
            groups=groups_tr,
            random_state=random_state + 10000 + outer_i * 100,
            use_shadow=use_shadow,
            shadow_quantile=shadow_quantile,
            shadow_filter=shadow_filter,
            shap_sample_size=shap_sample_size,
            threshold=threshold,
            positive_class_index=positive_class_index,
            importance_method=importance_method,
            permutation_n_repeats=permutation_n_repeats,
            permutation_n_jobs=permutation_n_jobs,
            optimization_metric=optimization_metric,
            permutation_scoring=permutation_scoring,
            survival_risk_score_method=survival_risk_score_method,
            prediction_time=prediction_time,
            risk_score_direction=risk_score_direction,
        )
        inner_summary["outer_fold"] = fold_name
        inner_detail["outer_fold"] = fold_name
        inner_summary_rows.append(inner_summary)
        inner_detail_rows.append(inner_detail)

        chosen_k = _choose_k(
            inner_summary,
            optimization_metric=optimization_metric,
            lower_is_better=lower_is_better,
            rule=choose_k_rule,
        )
        chosen_k_by_fold[fold_name] = int(chosen_k)

        outer_ranking = shap_feature_ranking(
            X_tr,
            y_tr,
            model=model,
            task=task,
            imputer=imputer,
            sampler=sampler,
            random_state=random_state + 20000 + outer_i,
            use_shadow=use_shadow,
            shadow_quantile=shadow_quantile,
            shap_sample_size=shap_sample_size,
            positive_class_index=positive_class_index,
            importance_method=importance_method,
            permutation_n_repeats=permutation_n_repeats,
            permutation_n_jobs=permutation_n_jobs,
            optimization_metric=optimization_metric,
            permutation_scoring=permutation_scoring,
            survival_risk_score_method=survival_risk_score_method,
            prediction_time=prediction_time,
            risk_score_direction=risk_score_direction,
        )
        if shadow_filter:
            candidate_features = outer_ranking.loc[outer_ranking["passes_shadow"], "feature"].tolist()
            if len(candidate_features) == 0:
                candidate_features = outer_ranking["feature"].head(1).tolist()
        else:
            candidate_features = outer_ranking["feature"].tolist()

        selected_features = candidate_features[: min(chosen_k, len(candidate_features))]
        selected_by_fold[fold_name] = selected_features
        rankings_by_fold[fold_name] = outer_ranking

        fitted = fit_final_model_on_features(
            X_tr,
            y_tr,
            features=selected_features,
            model=model,
            task=task,
            imputer=imputer,
            sampler=sampler,
            positive_class_index=positive_class_index,
            survival_risk_score_method=survival_risk_score_method,
            prediction_time=prediction_time,
            risk_score_direction=risk_score_direction,
        )
        pred = predict_with_fitted(fitted, X_te)
        metrics = compute_metrics(task, y_te, pred, threshold=threshold)

        outer_row = {
            "outer_fold": fold_name,
            "chosen_k": int(chosen_k),
            "n_selected": len(selected_features),
            **metrics,
        }
        outer_rows.append(outer_row)

        if verbose:
            metric_value = outer_row.get(optimization_metric, np.nan)
            print(
                f"[{fold_name}] chosen_k={chosen_k}, n_selected={len(selected_features)}, "
                f"{optimization_metric}={metric_value:.4f}"
            )

    outer_results = pd.DataFrame(outer_rows)
    inner_results = pd.concat(inner_summary_rows, ignore_index=True)
    inner_details = pd.concat(inner_detail_rows, ignore_index=True)

    all_selected = []
    for fold_name, feats in selected_by_fold.items():
        for rank, feat in enumerate(feats, start=1):
            all_selected.append({"outer_fold": fold_name, "feature": feat, "selected_rank": rank})
    selected_long = pd.DataFrame(all_selected)

    if selected_long.empty:
        selection_frequency = pd.DataFrame(columns=["feature", "count", "frequency"])
        final_features = []
    else:
        selection_frequency = (
            selected_long.groupby("feature", as_index=False)
            .agg(count=("outer_fold", "nunique"), mean_selected_rank=("selected_rank", "mean"))
        )
        selection_frequency["frequency"] = selection_frequency["count"] / outer_splits

        rank_rows = []
        for fold_name, ranking in rankings_by_fold.items():
            r = ranking.copy()
            r["outer_fold"] = fold_name
            rank_rows.append(r)
        rank_long = pd.concat(rank_rows, ignore_index=True)
        rank_summary = (
            rank_long.groupby("feature", as_index=False)
            .agg(
                mean_importance=("importance", "mean"),
                mean_rank=("rank", "mean"),
                shadow_pass_rate=("passes_shadow", "mean"),
            )
        )
        selection_frequency = selection_frequency.merge(rank_summary, on="feature", how="left")
        selection_frequency = selection_frequency.sort_values(
            ["frequency", "mean_selected_rank", "mean_importance"],
            ascending=[False, True, False],
        ).reset_index(drop=True)

        final_features = selection_frequency.loc[
            selection_frequency["frequency"] >= final_min_frequency,
            "feature",
        ].tolist()
        if len(final_features) == 0:
            fallback_k = int(round(np.median(outer_results["n_selected"])))
            final_features = selection_frequency.head(max(fallback_k, 1))["feature"].tolist()

        if max_final_features is not None and len(final_features) > max_final_features:
            final_features = final_features[:max_final_features]

    config = {
        "api": "nested_shap_feature_selection_cv",
        "task": task,
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "outer_splits": int(outer_splits),
        "inner_splits": int(inner_splits),
        "random_state": int(random_state),
        "k_grid": list(map(int, k_grid)),
        "max_k": None if max_k is None else int(max_k),
        "max_final_features": None if max_final_features is None else int(max_final_features),
        "choose_k_rule": choose_k_rule,
        "optimization_metric": optimization_metric,
        "use_shadow": bool(use_shadow),
        "shadow_quantile": float(shadow_quantile),
        "shadow_filter": bool(shadow_filter),
        "final_min_frequency": float(final_min_frequency),
        "importance_method": importance_method,
        "permutation_n_repeats": int(permutation_n_repeats),
        "permutation_n_jobs": permutation_n_jobs,
        "permutation_scoring": permutation_scoring,
        "survival_stratify": survival_stratify,
        "survival_time_bins": int(survival_time_bins),
        "risk_score": repr(survival_risk_score_method),
        "prediction_time": prediction_time,
        "risk_score_direction": risk_score_direction,
        "groups_provided": groups is not None,
        "model": repr(model),
        "preprocessor": repr(imputer),
        "sampler": repr(sampler),
    }
    metadata = _runtime_metadata()

    return NestedShapFSResult(
        task=task,
        optimization_metric=optimization_metric,
        lower_is_better=lower_is_better,
        outer_results=outer_results,
        inner_results=inner_results,
        inner_details=inner_details,
        selection_frequency=selection_frequency,
        final_features=final_features,
        outer_selected_features=selected_by_fold,
        outer_rankings=rankings_by_fold,
        chosen_k_by_outer_fold=chosen_k_by_fold,
        config=config,
        metadata=metadata,
    )


def summarize_outer_performance(outer_results: pd.DataFrame) -> pd.DataFrame:
    """Return mean and standard deviation of available outer-CV metrics."""
    metric_cols = [
        "log_loss",
        "brier",
        "accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "auroc",
        "auprc",
        "mae",
        "mse",
        "rmse",
        "median_ae",
        "r2",
        "c_index",
        "n_events",
        "n_censored",
        "n_comparable",
        "n_selected",
    ]
    rows = []
    for col in metric_cols:
        if col in outer_results.columns:
            rows.append(
                {
                    "metric": col,
                    "mean": outer_results[col].mean(),
                    "std": outer_results[col].std(ddof=1),
                }
            )
    return pd.DataFrame(rows)



def _runtime_metadata() -> Dict[str, Any]:
    """Return lightweight runtime metadata for result persistence."""
    try:
        package_version = importlib_metadata.version("ranking-feature-selector")
    except Exception:
        package_version = "unknown"
    versions = {"ranking_feature_selector": package_version}
    for dist_name, key in [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scikit-learn", "scikit_learn"),
        ("scikit-survival", "scikit_survival"),
        ("shap", "shap"),
    ]:
        try:
            versions[key] = importlib_metadata.version(dist_name)
        except Exception:
            versions[key] = None
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": versions,
    }


def _result_config(
    *,
    task: str,
    max_features: Optional[int],
    preset: str,
    preprocessor,
    standardize: bool,
    random_state: int,
    groups_provided: bool,
    cv_config: Dict[str, Any],
    selection_config: Dict[str, Any],
    importance_config: Dict[str, Any],
    model,
) -> Dict[str, Any]:
    """Return a JSON-friendly configuration summary."""
    return {
        "task": task,
        "max_features": max_features,
        "preset": preset,
        "preprocessor": repr(preprocessor),
        "standardize": bool(standardize),
        "random_state": random_state,
        "groups_provided": bool(groups_provided),
        "cv_config": dict(cv_config),
        "selection_config": dict(selection_config),
        "importance_config": dict(importance_config),
        "model": repr(model),
    }


_PRESET_CONFIGS = {
    "fast": {
        "cv_config": {"outer_splits": 3, "inner_splits": 3},
        "selection_config": {
            "selection_rule": "one_se",
            "use_shadow": False,
            "shadow_filter": False,
            "shadow_quantile": 1.0,
            "min_selection_rate": 0.5,
            "threshold": 0.5,
            "positive_class_index": 1,
        },
        "importance_config": {
            "method": "auto",
            "shap_sample_size": 150,
            "n_repeats": 5,
            "n_jobs": None,
            "scoring": "auto",
        },
    },
    "safe": {
        "cv_config": {"outer_splits": 5, "inner_splits": 4},
        "selection_config": {
            "selection_rule": "one_se",
            "use_shadow": True,
            "shadow_filter": False,
            "shadow_quantile": 1.0,
            "min_selection_rate": 0.5,
            "threshold": 0.5,
            "positive_class_index": 1,
        },
        "importance_config": {
            "method": "auto",
            "shap_sample_size": 300,
            "n_repeats": 10,
            "n_jobs": None,
            "scoring": "auto",
        },
    },
    "publication": {
        "cv_config": {"outer_splits": 5, "inner_splits": 5},
        "selection_config": {
            "selection_rule": "one_se",
            "use_shadow": True,
            "shadow_filter": False,
            "shadow_quantile": 1.0,
            "min_selection_rate": 0.6,
            "threshold": 0.5,
            "positive_class_index": 1,
        },
        "importance_config": {
            "method": "auto",
            "shap_sample_size": None,
            "n_repeats": 30,
            "n_jobs": None,
            "scoring": "auto",
        },
    },
    "custom": {
        "cv_config": {"outer_splits": 5, "inner_splits": 4},
        "selection_config": {
            "selection_rule": "one_se",
            "use_shadow": True,
            "shadow_filter": False,
            "shadow_quantile": 1.0,
            "min_selection_rate": 0.5,
            "threshold": 0.5,
            "positive_class_index": 1,
        },
        "importance_config": {
            "method": "auto",
            "shap_sample_size": 300,
            "n_repeats": 10,
            "n_jobs": None,
            "scoring": "auto",
        },
    },
}


def _merge_config(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(base)
    if override:
        out.update(override)
    return out


def _resolve_public_configs(
    preset: str,
    cv_config: Optional[Dict[str, Any]],
    selection_config: Optional[Dict[str, Any]],
    importance_config: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    if preset not in _PRESET_CONFIGS:
        raise ValueError("preset must be one of: 'fast', 'safe', 'publication', 'custom'.")
    base = _PRESET_CONFIGS[preset]
    resolved_cv = _merge_config(base["cv_config"], cv_config)
    resolved_selection = _merge_config(base["selection_config"], selection_config)
    resolved_importance = _merge_config(base["importance_config"], importance_config)
    return resolved_cv, resolved_selection, resolved_importance


def nested_feature_selection_cv(
    X: Union[pd.DataFrame, np.ndarray],
    y,
    model=None,
    task: str = "classification",
    max_features: Optional[int] = None,
    preset: str = "safe",
    preprocessor="auto",
    standardize: bool = False,
    random_state: int = 42,
    groups: Optional[Sequence] = None,
    cv_config: Optional[Dict[str, Any]] = None,
    selection_config: Optional[Dict[str, Any]] = None,
    importance_config: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> NestedShapFSResult:
    """Run nested-CV feature selection using the simplified public API.

    The usual user-facing controls are ``task``, ``model``, ``max_features``,
    ``preset``, ``preprocessor`` and ``standardize``. Advanced behavior is
    configured through ``cv_config``, ``selection_config`` and
    ``importance_config``.

    Notes
    -----
    The preprocessor must keep the same number and order of columns. Apply
    one-hot encoding or other feature-expanding transformations before calling
    this selector.
    """
    if task not in {"classification", "regression", "survival"}:
        raise ValueError("task must be one of: classification, regression, survival.")
    if max_features is not None:
        max_features = int(max_features)
        if max_features < 1:
            raise ValueError("max_features must be at least 1 when specified.")

    cv_cfg, sel_cfg, imp_cfg = _resolve_public_configs(
        preset=preset,
        cv_config=cv_config,
        selection_config=selection_config,
        importance_config=importance_config,
    )

    sampler = sel_cfg.pop("sampler", None)
    if task == "survival" and sampler is not None:
        raise ValueError("sampler is not exposed for survival; use sampler=None.")
    if task == "regression" and sampler is not None:
        raise ValueError("sampler is classification-only; use sampler=None for regression.")

    prepared_preprocessor = make_preprocessor(preprocessor, standardize=standardize)

    max_features_per_fold = sel_cfg.pop("max_features_per_fold", max_features)
    max_final_features = sel_cfg.pop("max_final_features", max_features)
    min_selection_rate = sel_cfg.pop("min_selection_rate", 0.5)
    selection_rule = sel_cfg.pop("selection_rule", sel_cfg.pop("choose_k_rule", "one_se"))
    optimization_metric = sel_cfg.pop("optimization_metric", None)
    k_grid = sel_cfg.pop("k_grid", None)
    event_col = sel_cfg.pop("event_col", None)
    time_col = sel_cfg.pop("time_col", None)
    threshold = sel_cfg.pop("threshold", 0.5)
    positive_class_index = sel_cfg.pop("positive_class_index", 1)
    survival_risk_score_method = sel_cfg.pop("risk_score", sel_cfg.pop("survival_risk_score_method", "predict"))
    prediction_time = sel_cfg.pop("prediction_time", None)
    risk_score_direction = sel_cfg.pop("risk_score_direction", "higher")
    use_shadow = sel_cfg.pop("use_shadow", True)
    shadow_quantile = sel_cfg.pop("shadow_quantile", 1.0)
    shadow_filter = sel_cfg.pop("shadow_filter", False)

    importance_method = imp_cfg.pop("method", imp_cfg.pop("importance_method", "auto"))
    shap_sample_size = imp_cfg.pop("shap_sample_size", 300)
    permutation_n_repeats = imp_cfg.pop("n_repeats", imp_cfg.pop("permutation_n_repeats", 10))
    permutation_n_jobs = imp_cfg.pop("n_jobs", imp_cfg.pop("permutation_n_jobs", None))
    permutation_scoring = imp_cfg.pop("scoring", imp_cfg.pop("permutation_scoring", "auto"))

    outer_splits = cv_cfg.pop("outer_splits", 5)
    inner_splits = cv_cfg.pop("inner_splits", 4)
    survival_stratify = cv_cfg.pop("survival_stratify", "event")
    survival_time_bins = cv_cfg.pop("survival_time_bins", 4)

    if sel_cfg:
        raise ValueError(f"Unknown selection_config keys: {sorted(sel_cfg)}")
    if imp_cfg:
        raise ValueError(f"Unknown importance_config keys: {sorted(imp_cfg)}")
    if cv_cfg:
        raise ValueError(f"Unknown cv_config keys: {sorted(cv_cfg)}")

    used_cv_config = {
        "outer_splits": outer_splits,
        "inner_splits": inner_splits,
        "survival_stratify": survival_stratify,
        "survival_time_bins": survival_time_bins,
    }
    used_selection_config = {
        "max_features_per_fold": max_features_per_fold,
        "max_final_features": max_final_features,
        "min_selection_rate": min_selection_rate,
        "selection_rule": selection_rule,
        "optimization_metric": optimization_metric,
        "k_grid": k_grid,
        "threshold": threshold,
        "positive_class_index": positive_class_index,
        "event_col": event_col,
        "time_col": time_col,
        "risk_score": survival_risk_score_method,
        "prediction_time": prediction_time,
        "risk_score_direction": risk_score_direction,
        "use_shadow": use_shadow,
        "shadow_quantile": shadow_quantile,
        "shadow_filter": shadow_filter,
        "sampler": repr(sampler),
    }
    used_importance_config = {
        "method": importance_method,
        "shap_sample_size": shap_sample_size,
        "n_repeats": permutation_n_repeats,
        "n_jobs": permutation_n_jobs,
        "scoring": permutation_scoring,
    }

    result = nested_shap_feature_selection_cv(
        X=X,
        y=y,
        model=model,
        task=task,
        imputer=prepared_preprocessor,
        sampler=sampler,
        outer_splits=outer_splits,
        inner_splits=inner_splits,
        random_state=random_state,
        k_grid=k_grid,
        max_k=max_features_per_fold,
        max_final_features=max_final_features,
        choose_k_rule=selection_rule,
        optimization_metric=optimization_metric,
        use_shadow=use_shadow,
        shadow_quantile=shadow_quantile,
        shadow_filter=shadow_filter,
        shap_sample_size=shap_sample_size,
        groups=groups,
        final_min_frequency=min_selection_rate,
        threshold=threshold,
        positive_class_index=positive_class_index,
        event_col=event_col,
        time_col=time_col,
        importance_method=importance_method,
        permutation_n_repeats=permutation_n_repeats,
        permutation_n_jobs=permutation_n_jobs,
        permutation_scoring=permutation_scoring,
        survival_stratify=survival_stratify,
        survival_time_bins=survival_time_bins,
        survival_risk_score_method=survival_risk_score_method,
        prediction_time=prediction_time,
        risk_score_direction=risk_score_direction,
        verbose=verbose,
    )
    result.config = _result_config(
        task=task,
        max_features=max_features,
        preset=preset,
        preprocessor=preprocessor,
        standardize=standardize,
        random_state=random_state,
        groups_provided=groups is not None,
        cv_config=used_cv_config,
        selection_config=used_selection_config,
        importance_config=used_importance_config,
        model=model,
    )
    result.metadata = _runtime_metadata()
    return result


class RobustFeatureSelectorCV(BaseEstimator, TransformerMixin):
    """User-friendly nested-CV feature selector.

    Parameters
    ----------
    model:
        Base estimator. If omitted, a random forest model suitable for the task
        is used.
    task:
        ``"classification"``, ``"regression"`` or ``"survival"``.
    max_features:
        Maximum number of features returned in ``selected_features_``. It also
        limits the fold-wise feature-count search unless overridden by
        ``selection_config["max_features_per_fold"]``.
    preset:
        ``"fast"``, ``"safe"``, ``"publication"`` or ``"custom"``.
    preprocessor:
        ``"auto"``/``"median_impute"`` for median imputation,
        ``"standardize"`` for median imputation + standardization, ``"none"``
        for no preprocessing, or a same-column sklearn transformer.
    standardize:
        Convenience flag that adds ``StandardScaler`` after median imputation
        when ``preprocessor`` is ``"auto"`` or ``None``.
    """

    def __init__(
        self,
        model=None,
        task: str = "classification",
        max_features: Optional[int] = None,
        preset: str = "safe",
        preprocessor="auto",
        standardize: bool = False,
        random_state: int = 42,
        cv_config: Optional[Dict[str, Any]] = None,
        selection_config: Optional[Dict[str, Any]] = None,
        importance_config: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ):
        self.model = model
        self.task = task
        self.max_features = max_features
        self.preset = preset
        self.preprocessor = preprocessor
        self.standardize = standardize
        self.random_state = random_state
        self.cv_config = cv_config
        self.selection_config = selection_config
        self.importance_config = importance_config
        self.verbose = verbose

    def fit(self, X, y, groups: Optional[Sequence] = None):
        self.result_ = nested_feature_selection_cv(
            X=X,
            y=y,
            model=self.model,
            task=self.task,
            max_features=self.max_features,
            preset=self.preset,
            preprocessor=self.preprocessor,
            standardize=self.standardize,
            random_state=self.random_state,
            groups=groups,
            cv_config=self.cv_config,
            selection_config=self.selection_config,
            importance_config=self.importance_config,
            verbose=self.verbose,
        )
        self.selected_features_ = self.result_.final_features
        self.n_features_in_ = as_dataframe(X).shape[1]
        return self

    def transform(self, X):
        if not hasattr(self, "selected_features_"):
            raise AttributeError("Call fit before transform.")
        X_df = as_dataframe(X)
        return X_df.loc[:, self.selected_features_].copy()

    def fit_final_model(self, X, y) -> FittedModelBundle:
        """Fit the configured base model on ``selected_features_``."""
        if not hasattr(self, "selected_features_"):
            raise AttributeError("Call fit before fit_final_model.")
        _, sel_cfg, _ = _resolve_public_configs(
            preset=self.preset,
            cv_config=self.cv_config,
            selection_config=self.selection_config,
            importance_config=self.importance_config,
        )
        sampler = sel_cfg.get("sampler", None)
        if self.task != "classification" and sampler is not None:
            raise ValueError("sampler is classification-only; use sampler=None for regression/survival.")
        event_col = sel_cfg.get("event_col", None)
        time_col = sel_cfg.get("time_col", None)
        survival_risk_score_method = sel_cfg.get("risk_score", sel_cfg.get("survival_risk_score_method", "predict"))
        prediction_time = sel_cfg.get("prediction_time", None)
        risk_score_direction = sel_cfg.get("risk_score_direction", "higher")
        prepared_preprocessor = make_preprocessor(self.preprocessor, standardize=self.standardize)
        final_model = self.model
        if final_model is None:
            final_model = _default_model_for_task(self.task, self.random_state)
        return fit_final_model_on_features(
            X_train=X,
            y_train=y,
            features=self.selected_features_,
            model=final_model,
            task=self.task,
            preprocessor=prepared_preprocessor,
            sampler=sampler,
            positive_class_index=sel_cfg.get("positive_class_index", 1),
            event_col=event_col,
            time_col=time_col,
            survival_risk_score_method=survival_risk_score_method,
            prediction_time=prediction_time,
            risk_score_direction=risk_score_direction,
        )

    def summary(self) -> pd.DataFrame:
        """Return mean/std outer-CV performance after ``fit``."""
        if not hasattr(self, "result_"):
            raise AttributeError("Call fit before summary.")
        return summarize_outer_performance(self.result_.outer_results)


class RobustClassificationFeatureSelectorCV(RobustFeatureSelectorCV):
    """Nested-CV feature selector for binary classification.

    The base model must implement ``fit(X, y)`` and ``predict_proba(X)``.
    Optional imbalanced-learn samplers can be supplied via ``sampler``; they are
    applied only within training folds.
    """

    def __init__(
        self,
        model=None,
        max_features: Optional[int] = None,
        preset: str = "safe",
        preprocessor="auto",
        standardize: bool = False,
        sampler=None,
        random_state: int = 42,
        cv_config: Optional[Dict[str, Any]] = None,
        selection_config: Optional[Dict[str, Any]] = None,
        importance_config: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ):
        super().__init__(
            model=model,
            task="classification",
            max_features=max_features,
            preset=preset,
            preprocessor=preprocessor,
            standardize=standardize,
            random_state=random_state,
            cv_config=cv_config,
            selection_config=selection_config,
            importance_config=importance_config,
            verbose=verbose,
        )
        self.sampler = sampler

    def _selection_config_with_sampler(self) -> Dict[str, Any]:
        cfg = dict(self.selection_config or {})
        if self.sampler is not None:
            if "sampler" in cfg and cfg["sampler"] is not self.sampler:
                raise ValueError("Pass sampler either as sampler=... or selection_config['sampler'], not both.")
            cfg["sampler"] = self.sampler
        return cfg

    def fit(self, X, y, groups: Optional[Sequence] = None):
        self.result_ = nested_feature_selection_cv(
            X=X,
            y=y,
            model=self.model,
            task="classification",
            max_features=self.max_features,
            preset=self.preset,
            preprocessor=self.preprocessor,
            standardize=self.standardize,
            random_state=self.random_state,
            groups=groups,
            cv_config=self.cv_config,
            selection_config=self._selection_config_with_sampler(),
            importance_config=self.importance_config,
            verbose=self.verbose,
        )
        self.selected_features_ = self.result_.final_features
        self.n_features_in_ = as_dataframe(X).shape[1]
        return self

    def fit_final_model(self, X, y) -> FittedModelBundle:
        if not hasattr(self, "selected_features_"):
            raise AttributeError("Call fit before fit_final_model.")
        _, sel_cfg, _ = _resolve_public_configs(
            preset=self.preset,
            cv_config=self.cv_config,
            selection_config=self._selection_config_with_sampler(),
            importance_config=self.importance_config,
        )
        event_col = sel_cfg.get("event_col", None)
        time_col = sel_cfg.get("time_col", None)
        prepared_preprocessor = make_preprocessor(self.preprocessor, standardize=self.standardize)
        final_model = self.model
        if final_model is None:
            final_model = _default_model_for_task("classification", self.random_state)
        return fit_final_model_on_features(
            X_train=X,
            y_train=y,
            features=self.selected_features_,
            model=final_model,
            task="classification",
            preprocessor=prepared_preprocessor,
            sampler=sel_cfg.get("sampler", None),
            positive_class_index=sel_cfg.get("positive_class_index", 1),
            event_col=event_col,
            time_col=time_col,
        )


class RobustRegressionFeatureSelectorCV(RobustFeatureSelectorCV):
    """Nested-CV feature selector for single-output regression.

    The base model must implement ``fit(X, y)`` and ``predict(X)``.
    Samplers are not supported for regression.
    """

    def __init__(
        self,
        model=None,
        max_features: Optional[int] = None,
        preset: str = "safe",
        preprocessor="auto",
        standardize: bool = False,
        random_state: int = 42,
        cv_config: Optional[Dict[str, Any]] = None,
        selection_config: Optional[Dict[str, Any]] = None,
        importance_config: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ):
        super().__init__(
            model=model,
            task="regression",
            max_features=max_features,
            preset=preset,
            preprocessor=preprocessor,
            standardize=standardize,
            random_state=random_state,
            cv_config=cv_config,
            selection_config=selection_config,
            importance_config=importance_config,
            verbose=verbose,
        )


class RobustSurvivalFeatureSelectorCV(RobustFeatureSelectorCV):
    """Nested-CV feature selector for right-censored survival analysis.

    The base model must implement ``fit(X, y)`` and a risk-score interface.
    By default, ``model.predict(X)`` is used and larger values are treated as
    higher event risk. Alternatively, use ``risk_score='event_probability'``
    with ``prediction_time`` to evaluate ``1 - S(t | X)`` at a fixed horizon,
    or ``risk_score='cumulative_hazard'`` for ``H(t | X)``. Samplers/SMOTE are
    intentionally not exposed for survival analysis.
    """

    def __init__(
        self,
        model=None,
        max_features: Optional[int] = None,
        preset: str = "safe",
        preprocessor="auto",
        standardize: bool = False,
        random_state: int = 42,
        event_col: Optional[str] = None,
        time_col: Optional[str] = None,
        risk_score: Any = "predict",
        prediction_time: Optional[float] = None,
        risk_score_direction: str = "higher",
        cv_config: Optional[Dict[str, Any]] = None,
        selection_config: Optional[Dict[str, Any]] = None,
        importance_config: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ):
        super().__init__(
            model=model,
            task="survival",
            max_features=max_features,
            preset=preset,
            preprocessor=preprocessor,
            standardize=standardize,
            random_state=random_state,
            cv_config=cv_config,
            selection_config=selection_config,
            importance_config=importance_config,
            verbose=verbose,
        )
        self.event_col = event_col
        self.time_col = time_col
        self.risk_score = risk_score
        self.prediction_time = prediction_time
        self.risk_score_direction = risk_score_direction

    def _selection_config_with_survival_columns(self) -> Dict[str, Any]:
        cfg = dict(self.selection_config or {})
        if self.event_col is not None:
            if "event_col" in cfg and cfg["event_col"] != self.event_col:
                raise ValueError("Pass event_col either as event_col=... or selection_config['event_col'], not both.")
            cfg["event_col"] = self.event_col
        if self.time_col is not None:
            if "time_col" in cfg and cfg["time_col"] != self.time_col:
                raise ValueError("Pass time_col either as time_col=... or selection_config['time_col'], not both.")
            cfg["time_col"] = self.time_col
        if self.risk_score is not None:
            if "risk_score" in cfg and cfg["risk_score"] != self.risk_score:
                raise ValueError("Pass risk_score either as risk_score=... or selection_config['risk_score'], not both.")
            cfg["risk_score"] = self.risk_score
        if self.prediction_time is not None:
            if "prediction_time" in cfg and cfg["prediction_time"] != self.prediction_time:
                raise ValueError("Pass prediction_time either as prediction_time=... or selection_config['prediction_time'], not both.")
            cfg["prediction_time"] = self.prediction_time
        if self.risk_score_direction is not None:
            if "risk_score_direction" in cfg and cfg["risk_score_direction"] != self.risk_score_direction:
                raise ValueError("Pass risk_score_direction either as risk_score_direction=... or selection_config['risk_score_direction'], not both.")
            cfg["risk_score_direction"] = self.risk_score_direction
        if "sampler" in cfg and cfg["sampler"] is not None:
            raise ValueError("sampler/SMOTE is not supported by RobustSurvivalFeatureSelectorCV.")
        return cfg

    def fit(self, X, y, groups: Optional[Sequence] = None):
        self.result_ = nested_feature_selection_cv(
            X=X,
            y=y,
            model=self.model,
            task="survival",
            max_features=self.max_features,
            preset=self.preset,
            preprocessor=self.preprocessor,
            standardize=self.standardize,
            random_state=self.random_state,
            groups=groups,
            cv_config=self.cv_config,
            selection_config=self._selection_config_with_survival_columns(),
            importance_config=self.importance_config,
            verbose=self.verbose,
        )
        self.selected_features_ = self.result_.final_features
        self.n_features_in_ = as_dataframe(X).shape[1]
        return self

    def fit_final_model(self, X, y) -> FittedModelBundle:
        if not hasattr(self, "selected_features_"):
            raise AttributeError("Call fit before fit_final_model.")
        _, sel_cfg, _ = _resolve_public_configs(
            preset=self.preset,
            cv_config=self.cv_config,
            selection_config=self._selection_config_with_survival_columns(),
            importance_config=self.importance_config,
        )
        prepared_preprocessor = make_preprocessor(self.preprocessor, standardize=self.standardize)
        final_model = self.model
        if final_model is None:
            final_model = _default_model_for_task("survival", self.random_state)
        return fit_final_model_on_features(
            X_train=X,
            y_train=y,
            features=self.selected_features_,
            model=final_model,
            task="survival",
            preprocessor=prepared_preprocessor,
            sampler=None,
            positive_class_index=1,
            event_col=sel_cfg.get("event_col", None),
            time_col=sel_cfg.get("time_col", None),
            survival_risk_score_method=sel_cfg.get("risk_score", sel_cfg.get("survival_risk_score_method", "predict")),
            prediction_time=sel_cfg.get("prediction_time", None),
            risk_score_direction=sel_cfg.get("risk_score_direction", "higher"),
        )


# Backward-compatible class-name alias. New code should use task-specific classes
# or RobustFeatureSelectorCV.
RobustShapFeatureSelectorCV = RobustFeatureSelectorCV
