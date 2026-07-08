"""Result containers for ranking_feature_selector."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

import numpy as np
import pandas as pd

from .survival import survival_risk_score


def _json_default(obj):
    """Best-effort JSON serializer for configs and metadata."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Index,)):
        return list(obj)
    try:
        return repr(obj)
    except Exception:  # pragma: no cover
        return f"<unserializable {type(obj).__name__}>"


@dataclass
class NestedShapFSResult:
    """Container for nested-CV feature-selection results.

    Attributes
    ----------
    task:
        ``"classification"``, ``"regression"``, or ``"survival"``.
    optimization_metric:
        Metric used to choose the feature count in the inner loop.
    lower_is_better:
        Whether smaller values of ``optimization_metric`` are better.
    outer_results:
        One row per outer fold, including the chosen number of features and test metrics.
    inner_results:
        Inner-CV summary by outer fold and candidate feature count ``k``.
    inner_details:
        Raw inner-CV fold-level evaluations. Useful for diagnostics.
    selection_frequency:
        Frequency with which each feature was selected across outer folds.
    final_features:
        Stability-selected final feature list from ``selection_frequency``.
    outer_selected_features:
        Fold-specific selected feature lists. These are the actual features used for each
        outer-fold performance estimate.
    outer_rankings:
        Feature ranking produced after refitting the selector on each full outer-training fold.
    chosen_k_by_outer_fold:
        Feature count selected in each outer fold.
    config:
        User-facing configuration used to produce this result.
    metadata:
        Runtime/package metadata for reproducibility.
    """

    task: str
    optimization_metric: str
    lower_is_better: bool
    outer_results: pd.DataFrame
    inner_results: pd.DataFrame
    inner_details: pd.DataFrame
    selection_frequency: pd.DataFrame
    final_features: List[str]
    outer_selected_features: Dict[str, List[str]]
    outer_rankings: Dict[str, pd.DataFrame]
    chosen_k_by_outer_fold: Dict[str, int]
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save(self, directory: str | Path) -> None:
        """Save result components as CSV plus JSON config/metadata files.

        Parameters
        ----------
        directory:
            Output directory. It will be created if it does not exist.
        """
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.outer_results.to_csv(out_dir / "outer_results.csv", index=False)
        self.inner_results.to_csv(out_dir / "inner_results.csv", index=False)
        self.inner_details.to_csv(out_dir / "inner_details.csv", index=False)
        self.selection_frequency.to_csv(out_dir / "selection_frequency.csv", index=False)
        pd.DataFrame({"feature": self.final_features}).to_csv(
            out_dir / "final_features.csv", index=False
        )
        for fold_name, ranking in self.outer_rankings.items():
            ranking.to_csv(out_dir / f"ranking_{fold_name}.csv", index=False)
        with open(out_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False, default=_json_default)
        with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False, default=_json_default)


@dataclass
class FittedModelBundle:
    """Preprocessor + model bundle fitted on selected features."""

    task: str
    features: List[str]
    preprocessor: object
    model: object
    sampler: Optional[object] = None
    positive_class_index: int = 1
    survival_risk_score_method: Any = "predict"
    prediction_time: Optional[float] = None
    risk_score_direction: str = "higher"
    feature_names_in_: Optional[List[str]] = None
    selected_indices_: Optional[List[int]] = None

    @property
    def imputer(self):
        """Backward-compatible alias for ``preprocessor``."""
        return self.preprocessor

    def _select_input_columns(self, X) -> pd.DataFrame:
        """Return selected-feature DataFrame from a DataFrame or full/selected ndarray."""
        if isinstance(X, pd.DataFrame):
            missing = sorted(set(self.features) - set(X.columns))
            if missing:
                raise ValueError(f"Input is missing selected features: {missing}")
            return X.loc[:, self.features].copy()

        arr = np.asarray(X)
        if arr.ndim != 2:
            raise ValueError("X must be a 2D array or DataFrame.")

        # Full matrix with original feature order.
        if self.feature_names_in_ is not None and arr.shape[1] == len(self.feature_names_in_):
            if self.selected_indices_ is None:
                name_to_idx = {name: i for i, name in enumerate(self.feature_names_in_)}
                selected_indices = [name_to_idx[f] for f in self.features]
            else:
                selected_indices = self.selected_indices_
            return pd.DataFrame(arr[:, selected_indices], columns=self.features)

        # Already-selected matrix.
        if arr.shape[1] == len(self.features):
            return pd.DataFrame(arr, columns=self.features)

        expected_full = len(self.feature_names_in_) if self.feature_names_in_ is not None else "unknown"
        raise ValueError(
            "For ndarray input, X must contain either all original columns "
            f"(n_features={expected_full}) or exactly the selected columns "
            f"(n_features={len(self.features)}). Got n_features={arr.shape[1]}."
        )

    def transform(self, X):
        """Apply the fitted preprocessor to selected columns and return a DataFrame."""
        X_sel = self._select_input_columns(X)
        arr = self.preprocessor.transform(X_sel)
        if hasattr(arr, "toarray"):
            arr = arr.toarray()
        arr = np.asarray(arr)
        if arr.ndim != 2 or arr.shape[1] != len(self.features):
            raise ValueError(
                "The fitted preprocessor did not preserve the selected feature count."
            )
        return pd.DataFrame(arr, columns=self.features, index=X_sel.index)

    def predict(self, X):
        """Predict labels, continuous values, or survival risk scores."""
        X_imp = self.transform(X)
        return self.model.predict(X_imp)

    def predict_proba(self, X):
        """Predict probabilities for classification models."""
        if self.task != "classification":
            raise AttributeError("predict_proba is available only for classification bundles.")
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("The fitted model does not implement predict_proba.")
        X_imp = self.transform(X)
        return self.model.predict_proba(X_imp)

    def predict_survival_function(self, X, return_array: bool = False):
        """Predict survival functions for survival models."""
        if self.task != "survival":
            raise AttributeError("predict_survival_function is available only for survival bundles.")
        if not hasattr(self.model, "predict_survival_function"):
            raise AttributeError("The fitted model does not implement predict_survival_function.")
        X_imp = self.transform(X)
        return self.model.predict_survival_function(X_imp, return_array=return_array)

    def predict_cumulative_hazard_function(self, X, return_array: bool = False):
        """Predict cumulative hazard functions for survival models."""
        if self.task != "survival":
            raise AttributeError(
                "predict_cumulative_hazard_function is available only for survival bundles."
            )
        if not hasattr(self.model, "predict_cumulative_hazard_function"):
            raise AttributeError(
                "The fitted model does not implement predict_cumulative_hazard_function."
            )
        X_imp = self.transform(X)
        return self.model.predict_cumulative_hazard_function(X_imp, return_array=return_array)

    def predict_score(self, X):
        """Return the optimization-style prediction vector.

        Classification returns the positive-class probability. Regression returns
        continuous predictions. Survival returns a risk score according to
        ``survival_risk_score_method``. Survival risk scores are intended for
        rank-based metrics within the same fold, not as calibrated probabilities
        unless ``survival_risk_score_method='event_probability'`` is used with a
        fixed ``prediction_time``.
        """
        X_imp = self.transform(X)
        if self.task == "classification":
            proba = self.model.predict_proba(X_imp)
            return proba[:, self.positive_class_index]
        if self.task == "survival":
            return survival_risk_score(
                self.model,
                X_imp,
                risk_score=self.survival_risk_score_method,
                prediction_time=self.prediction_time,
                risk_score_direction=self.risk_score_direction,
            )
        return self.model.predict(X_imp)
