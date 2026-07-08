"""Documented model compatibility for ranking_feature_selector."""

from __future__ import annotations

import pandas as pd

SUPPORTED_MODELS = {
    "classification": {
        "required_api": "fit(X, y) and predict_proba(X)",
        "default_importance": "Tree SHAP; falls back to feature_importances_ or coef_ when available",
        "recommended": [
            "sklearn.ensemble.RandomForestClassifier",
            "sklearn.ensemble.ExtraTreesClassifier",
            "sklearn.linear_model.LogisticRegression",
            "lightgbm.LGBMClassifier",
            "xgboost.XGBClassifier",
            "catboost.CatBoostClassifier",
            "sklearn.svm.SVC(probability=True)",
        ],
        "notes": "Binary classification is the primary target. LinearSVC is not supported unless wrapped to provide predict_proba.",
    },
    "regression": {
        "required_api": "fit(X, y) and predict(X)",
        "default_importance": "Tree SHAP; falls back to feature_importances_ or coef_ when available",
        "recommended": [
            "sklearn.ensemble.RandomForestRegressor",
            "sklearn.ensemble.ExtraTreesRegressor",
            "sklearn.linear_model.ElasticNet / Ridge / Lasso",
            "lightgbm.LGBMRegressor",
            "xgboost.XGBRegressor",
            "catboost.CatBoostRegressor",
            "sklearn.svm.SVR with importance_config={'method': 'permutation'}",
        ],
        "notes": "Single-output regression is the primary target. Non-tree/non-linear models should usually use permutation importance.",
    },
    "survival": {
        "required_api": "fit(X, y) and predict(X), where predict returns a risk score; larger means higher event risk",
        "default_importance": "Permutation importance using Harrell C-index",
        "recommended": [
            "sksurv.ensemble.RandomSurvivalForest",
            "sksurv.linear_model.CoxPHSurvivalAnalysis",
            "sksurv.ensemble.GradientBoostingSurvivalAnalysis",
            "sksurv.ensemble.ComponentwiseGradientBoostingSurvivalAnalysis",
        ],
        "notes": "y should be a scikit-survival structured array, a (event, time) tuple, a 2-column array, or a DataFrame with event/time columns.",
    },
}


def supported_models() -> pd.DataFrame:
    """Return a compact table describing supported model APIs by task."""
    rows = []
    for task, spec in SUPPORTED_MODELS.items():
        rows.append(
            {
                "task": task,
                "required_api": spec["required_api"],
                "default_importance": spec["default_importance"],
                "recommended_models": "; ".join(spec["recommended"]),
                "notes": spec["notes"],
            }
        )
    return pd.DataFrame(rows)
