"""Preprocessing helpers for ranking_feature_selector.

The selector expects preprocessing steps to preserve the same number and order of
columns. This keeps selected feature names interpretable. Feature-expanding steps
such as one-hot encoding should be applied before calling the selector.
"""

from __future__ import annotations

from typing import Any

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def make_preprocessor(preprocessor: Any = "auto", standardize: bool = False):
    """Return a same-column preprocessor.

    Parameters
    ----------
    preprocessor:
        One of ``"auto"``, ``"median_impute"``, ``"standardize"``, ``"none"``,
        ``None``, or a fitted/unfitted sklearn-compatible transformer.
        ``"auto"`` and ``None`` mean median imputation, plus standardization when
        ``standardize=True``.
    standardize:
        If true, append ``StandardScaler`` after median imputation when the
        preprocessor is ``"auto"``/``None``/``"median_impute"``. For a custom
        preprocessor, include scaling in the custom pipeline instead.
    """
    if isinstance(preprocessor, str):
        key = preprocessor.lower().strip()
    else:
        key = None

    if preprocessor is None or key in {"auto", "median_impute", "impute", "median"}:
        if standardize:
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
        return SimpleImputer(strategy="median")

    if key in {"standardize", "impute_and_standardize", "scale", "scaled"}:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

    if key in {"none", "passthrough", "identity"}:
        if standardize:
            return StandardScaler()
        return FunctionTransformer(validate=False)

    if key is not None:
        raise ValueError(
            "preprocessor must be 'auto', 'median_impute', 'standardize', 'none', "
            "None, or a sklearn-compatible transformer."
        )

    if standardize:
        raise ValueError(
            "standardize=True is only supported with preprocessor='auto', "
            "preprocessor='median_impute', preprocessor='none', or None. "
            "For a custom transformer, add StandardScaler to your custom Pipeline."
        )
    return clone(preprocessor)
