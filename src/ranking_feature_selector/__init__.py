"""Robust feature selection for small tabular data.

The package provides nested-CV feature selection for binary classification,
regression, and survival analysis. Feature rankings are recomputed inside
inner-CV training folds, so validation/test outcomes are not used to rank
features. Classification/regression use SHAP when available and otherwise fall back to
model-native importances; survival uses permutation importance by default.
"""

from .importance import shap_feature_ranking
from .preprocessing import make_preprocessor
from .results import FittedModelBundle, NestedShapFSResult
from .survival import survival_risk_score
from .selection import (
    RobustClassificationFeatureSelectorCV,
    RobustFeatureSelectorCV,
    RobustRegressionFeatureSelectorCV,
    RobustShapFeatureSelectorCV,
    RobustSurvivalFeatureSelectorCV,
    compute_metrics,
    evaluate_k_grid_strict_inner_cv,
    fit_final_model_on_features,
    fit_predict_on_features,
    nested_feature_selection_cv,
    nested_shap_feature_selection_cv,
    predict_with_fitted,
    summarize_outer_performance,
)
from .utils import make_k_grid, make_survival_y
from .models import SUPPORTED_MODELS, supported_models

__all__ = [
    "FittedModelBundle",
    "NestedShapFSResult",
    "RobustClassificationFeatureSelectorCV",
    "RobustFeatureSelectorCV",
    "RobustRegressionFeatureSelectorCV",
    "RobustShapFeatureSelectorCV",
    "RobustSurvivalFeatureSelectorCV",
    "compute_metrics",
    "evaluate_k_grid_strict_inner_cv",
    "fit_final_model_on_features",
    "fit_predict_on_features",
    "make_k_grid",
    "make_preprocessor",
    "make_survival_y",
    "nested_feature_selection_cv",
    "nested_shap_feature_selection_cv",
    "predict_with_fitted",
    "shap_feature_ranking",
    "summarize_outer_performance",
    "survival_risk_score",
    "SUPPORTED_MODELS",
    "supported_models",
]

__version__ = "0.4.3"
