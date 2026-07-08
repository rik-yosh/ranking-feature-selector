import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ranking_feature_selector import (
    RobustClassificationFeatureSelectorCV,
    RobustFeatureSelectorCV,
    RobustRegressionFeatureSelectorCV,
    RobustShapFeatureSelectorCV,
    RobustSurvivalFeatureSelectorCV,
    fit_final_model_on_features,
    make_survival_y,
    nested_feature_selection_cv,
    nested_shap_feature_selection_cv,
    predict_with_fitted,
)


class ToySurvivalEstimator(BaseEstimator):
    """Minimal sklearn-compatible survival-risk estimator for smoke tests."""

    def __init__(self, scale=1.0):
        self.scale = scale

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.coef_ = self.scale * np.linspace(1.0, 0.1, X.shape[1])
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_


def test_classification_public_api_smoke():
    X_np, y_np = make_classification(
        n_samples=50,
        n_features=8,
        n_informative=3,
        n_redundant=1,
        random_state=1,
    )
    X = pd.DataFrame(X_np, columns=[f"x{i}" for i in range(X_np.shape[1])])
    y = pd.Series(y_np)
    model = RandomForestClassifier(n_estimators=10, random_state=1, min_samples_leaf=2)

    selector = RobustClassificationFeatureSelectorCV(
        model=model,
        max_features=2,
        preset="fast",
        preprocessor="standardize",
        selection_config={"k_grid": [1, 2]},
        importance_config={"method": "permutation", "n_repeats": 1},
        random_state=1,
        verbose=False,
    )
    selector.fit(X, y)
    assert 1 <= len(selector.selected_features_) <= 2
    assert selector.transform(X).shape[1] == len(selector.selected_features_)
    assert selector.summary().shape[0] > 0

    fitted = selector.fit_final_model(X, y)
    pred = predict_with_fitted(fitted, X)
    assert pred.shape[0] == X.shape[0]


def test_regression_public_api_and_alias_smoke():
    X_np, y_np = make_regression(
        n_samples=50,
        n_features=8,
        n_informative=3,
        noise=5.0,
        random_state=1,
    )
    X = pd.DataFrame(X_np, columns=[f"x{i}" for i in range(X_np.shape[1])])
    y = pd.Series(y_np)
    model = RandomForestRegressor(n_estimators=10, random_state=1, min_samples_leaf=2)

    selector = RobustRegressionFeatureSelectorCV(
        model=model,
        max_features=2,
        preset="fast",
        preprocessor="standardize",
        selection_config={"k_grid": [1, 2]},
        importance_config={"method": "permutation", "n_repeats": 1},
        random_state=2,
        verbose=False,
    )
    selector.fit(X, y)
    assert 1 <= len(selector.selected_features_) <= 2
    assert selector.transform(X).shape[1] == len(selector.selected_features_)

    alias = RobustShapFeatureSelectorCV(task="regression", model=model, max_features=2, preset="fast", verbose=False)
    assert alias.task == "regression"


def test_legacy_function_still_works():
    X_np, y_np = make_classification(
        n_samples=40,
        n_features=6,
        n_informative=3,
        random_state=4,
    )
    X = pd.DataFrame(X_np, columns=[f"x{i}" for i in range(X_np.shape[1])])
    y = pd.Series(y_np)
    model = RandomForestClassifier(n_estimators=8, random_state=4, min_samples_leaf=2)

    result = nested_shap_feature_selection_cv(
        X,
        y,
        model=model,
        task="classification",
        outer_splits=2,
        inner_splits=2,
        max_k=2,
        max_final_features=2,
        use_shadow=False,
        importance_method="permutation",
        permutation_n_repeats=1,
        random_state=4,
        verbose=False,
    )
    assert len(result.outer_results) == 2
    assert 1 <= len(result.final_features) <= 2


def test_survival_smoke_with_permutation_importance():
    rng = np.random.default_rng(3)
    n = 50
    X_np = rng.normal(size=(n, 6))
    linear_risk = X_np[:, 0] + 0.5 * X_np[:, 1]
    time = np.exp(3.0 - linear_risk + rng.normal(scale=0.1, size=n))
    event = np.ones(n, dtype=bool)
    event[::4] = False
    y = make_survival_y(event, time)
    X = pd.DataFrame(X_np, columns=[f"x{i}" for i in range(X_np.shape[1])])

    selector = RobustSurvivalFeatureSelectorCV(
        model=ToySurvivalEstimator(),
        max_features=2,
        preset="fast",
        selection_config={"k_grid": [1, 2]},
        importance_config={"method": "permutation", "n_repeats": 1},
        random_state=3,
        verbose=False,
    )
    selector.fit(X, y)
    result = selector.result_
    assert len(result.outer_results) == 3
    assert 1 <= len(result.final_features) <= 2
    assert "c_index" in result.outer_results.columns

    fitted = fit_final_model_on_features(
        X,
        y,
        features=result.final_features,
        model=ToySurvivalEstimator(),
        task="survival",
    )
    risk = predict_with_fitted(fitted, X)
    assert risk.shape[0] == X.shape[0]


def test_survival_requires_dependency_for_default_model():
    try:
        import sksurv  # noqa: F401
    except ImportError:
        X = pd.DataFrame(np.random.default_rng(1).normal(size=(20, 4)))
        y = make_survival_y(np.ones(20, dtype=bool), np.arange(1, 21))
        with pytest.raises(ImportError):
            nested_feature_selection_cv(
                X,
                y,
                model=None,
                task="survival",
                max_features=2,
                preset="fast",
                verbose=False,
            )

class ToyStepFunction:
    def __init__(self, value):
        self.value = float(value)
        self.x = np.array([0.0, 10.0])

    def __call__(self, t):
        return self.value


class ToySurvivalCurveEstimator(BaseEstimator):
    """Minimal estimator exposing predict_survival_function for fixed-horizon risks."""

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.coef_ = np.linspace(1.0, 0.1, X.shape[1])
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_

    def predict_survival_function(self, X, return_array=False):
        risk = self.predict(X)
        surv = 1.0 / (1.0 + np.exp(risk))
        if return_array:
            return surv.reshape(-1, 1)
        return [ToyStepFunction(v) for v in surv]


def test_survival_fixed_time_event_probability_risk_score():
    from ranking_feature_selector import survival_risk_score

    rng = np.random.default_rng(7)
    X_np = rng.normal(size=(48, 5))
    linear_risk = X_np[:, 0] + 0.5 * X_np[:, 1]
    time = np.exp(3.0 - linear_risk + rng.normal(scale=0.1, size=len(linear_risk)))
    event = np.ones(len(time), dtype=bool)
    event[::5] = False
    y = make_survival_y(event, time)
    X = pd.DataFrame(X_np, columns=[f"x{i}" for i in range(X_np.shape[1])])

    selector = RobustSurvivalFeatureSelectorCV(
        model=ToySurvivalCurveEstimator(),
        max_features=2,
        preset="fast",
        risk_score="event_probability",
        prediction_time=5.0,
        selection_config={"k_grid": [1, 2]},
        importance_config={"method": "permutation", "n_repeats": 1},
        random_state=7,
        verbose=False,
    )
    selector.fit(X, y)
    assert 1 <= len(selector.selected_features_) <= 2
    fitted = selector.fit_final_model(X, y)
    fixed_time_risk = predict_with_fitted(fitted, X)
    assert fixed_time_risk.shape[0] == X.shape[0]
    assert np.all((0.0 <= fixed_time_risk) & (fixed_time_risk <= 1.0))

    model = ToySurvivalEstimator().fit(X, y)
    risk_high = survival_risk_score(model, X, risk_score="predict", risk_score_direction="higher")
    risk_low = survival_risk_score(model, X, risk_score="predict", risk_score_direction="lower")
    np.testing.assert_allclose(risk_low, -risk_high)


def test_fitted_bundle_accepts_full_ndarray_and_save_outputs(tmp_path):
    X_np, y_np = make_regression(
        n_samples=42,
        n_features=7,
        n_informative=3,
        noise=3.0,
        random_state=11,
    )
    X = pd.DataFrame(X_np, columns=[f"x{i}" for i in range(X_np.shape[1])])
    y = pd.Series(y_np)
    model = RandomForestRegressor(n_estimators=8, random_state=11, min_samples_leaf=2)

    selector = RobustRegressionFeatureSelectorCV(
        model=model,
        max_features=2,
        preset="fast",
        selection_config={"k_grid": [1, 2]},
        importance_config={"method": "permutation", "n_repeats": 1, "scoring": "auto"},
        random_state=11,
        verbose=False,
    )
    selector.fit(X, y)
    fitted = selector.fit_final_model(X, y)

    # Full ndarray input should be subset internally by selected_indices_.
    pred_full = fitted.predict_score(X_np)
    pred_df = fitted.predict_score(X)
    assert pred_full.shape == pred_df.shape
    np.testing.assert_allclose(pred_full, pred_df)

    selector.result_.save(tmp_path)
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "metadata.json").exists()


def test_survival_cv_config_accepts_event_stratification():
    rng = np.random.default_rng(12)
    X_np = rng.normal(size=(48, 5))
    risk = X_np[:, 0] - 0.5 * X_np[:, 1]
    time = np.exp(2.5 - risk + rng.normal(scale=0.2, size=48))
    event = np.zeros(48, dtype=bool)
    event[::3] = True
    y = make_survival_y(event, time)
    X = pd.DataFrame(X_np, columns=[f"x{i}" for i in range(X_np.shape[1])])

    selector = RobustSurvivalFeatureSelectorCV(
        model=ToySurvivalEstimator(),
        max_features=2,
        preset="fast",
        cv_config={"survival_stratify": "event", "survival_time_bins": 3},
        selection_config={"k_grid": [1, 2]},
        importance_config={"method": "permutation", "n_repeats": 1},
        random_state=12,
        verbose=False,
    )
    selector.fit(X, y)
    assert len(selector.result_.outer_results) == 3
    assert selector.result_.config["cv_config"]["survival_stratify"] == "event"


def test_backward_compatible_import_aliases():
    import robust_feature_selector
    import robust_shap_selector

    assert robust_feature_selector.RobustFeatureSelectorCV is RobustFeatureSelectorCV
    assert robust_shap_selector.RobustFeatureSelectorCV is RobustFeatureSelectorCV
