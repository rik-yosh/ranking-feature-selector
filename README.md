# ranking-feature-selector

`ranking-feature-selector` is a leakage-safe, nested-CV feature selection package for small tabular data. It supports binary classification, single-output regression, and right-censored survival analysis.

It is designed for research workflows where feature ranking, feature-count selection, preprocessing, and performance estimation should be performed inside cross-validation folds to reduce data leakage.

## Installation

```bash
pip install ranking-feature-selector
```

For survival analysis / Random Survival Forests:

```bash
pip install "ranking-feature-selector[survival]"
```

For SHAP-based ranking backends:

```bash
pip install "ranking-feature-selector[shap]"
```

For imbalanced-learn samplers in binary classification:

```bash
pip install "ranking-feature-selector[imbalance]"
```

For all optional backends:

```bash
pip install "ranking-feature-selector[all]"
```

## Quick start

```python
from ranking_feature_selector import RobustSurvivalFeatureSelectorCV

selector = RobustSurvivalFeatureSelectorCV(
    model=rsf,
    max_features=20,
    preset="safe",
    preprocessor="standardize",
    random_state=42,
)
selector.fit(X, y, groups=center_id)
print(selector.selected_features_)
print(selector.summary())
```

For most use cases, start with a task-specific class, `model`, `max_features`, `preset`, `preprocessor`, and `random_state`. Advanced controls are grouped under `cv_config`, `selection_config`, and `importance_config`.

## Main classes

```python
from ranking_feature_selector import (
    RobustClassificationFeatureSelectorCV,
    RobustRegressionFeatureSelectorCV,
    RobustSurvivalFeatureSelectorCV,
)
```

| Class | Task | Required model API |
|---|---|---|
| `RobustClassificationFeatureSelectorCV` | Binary classification | `fit(X, y)` and `predict_proba(X)` |
| `RobustRegressionFeatureSelectorCV` | Regression | `fit(X, y)` and `predict(X)` |
| `RobustSurvivalFeatureSelectorCV` | Right-censored survival analysis | `fit(X, y)` and a risk-score interface, usually `predict(X)` |

`RobustFeatureSelectorCV(task=...)` is also available as a generic interface. `RobustShapFeatureSelectorCV` is kept as a backward-compatible class alias.

Backward-compatible import aliases are also provided:

```python
from robust_feature_selector import RobustSurvivalFeatureSelectorCV
from robust_shap_selector import RobustSurvivalFeatureSelectorCV
```

New code should prefer:

```python
from ranking_feature_selector import RobustSurvivalFeatureSelectorCV
```

## What the package does

For each outer CV fold, the selector performs feature ranking and feature-count selection using only the training portion of that fold:

1. Fit preprocessing on the training fold only.
2. Rank features inside inner-CV training folds.
3. Select the number of features using inner-CV performance.
4. Refit on the outer-training fold using the selected features.
5. Evaluate on the untouched outer-test fold.
6. Aggregate fold-wise selections into `selected_features_` using selection frequency.

This means that the reported outer-CV performance reflects fold-specific selected feature sets. The final `selected_features_` list is a stability-based summary across outer folds.

## Presets

| Preset | Intended use | Behavior |
|---|---|---|
| `"fast"` | Initial checks | Fewer repeats and smaller CV |
| `"safe"` | Default research use | Nested CV, one-standard-error rule, shadow features enabled |
| `"publication"` | More conservative analysis | More repeats and stricter stability threshold |
| `"custom"` | Full control | Override settings via config dictionaries |

## Preprocessing and standardization

`preprocessor` must preserve the same number and order of columns. One-hot encoding, target encoding, PCA, or other feature-expanding transformations should be performed before calling the selector.

| Setting | Behavior |
|---|---|
| `preprocessor="auto"` | Median imputation |
| `preprocessor="median_impute"` | Median imputation |
| `preprocessor="standardize"` | Median imputation + `StandardScaler` |
| `preprocessor="none"` | No preprocessing |
| `standardize=True` | Adds `StandardScaler` after median imputation when using `"auto"` |
| custom transformer | Any same-column sklearn-compatible transformer |

Example:

```python
selector = RobustRegressionFeatureSelectorCV(
    model=model,
    max_features=15,
    preprocessor="standardize",
)
```

Custom preprocessing:

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler()),
])

selector = RobustRegressionFeatureSelectorCV(
    model=model,
    max_features=15,
    preprocessor=preprocessor,
)
```

## Supported models

`model` should be a scikit-learn compatible estimator that can be cloned with `sklearn.base.clone()`.

### Binary classification

Required API:

```python
model.fit(X_train, y_train)
model.predict_proba(X_valid)
```

Recommended models include:

- `sklearn.ensemble.RandomForestClassifier`
- `sklearn.ensemble.ExtraTreesClassifier`
- `sklearn.linear_model.LogisticRegression`
- `lightgbm.LGBMClassifier`
- `xgboost.XGBClassifier`
- `catboost.CatBoostClassifier`
- `sklearn.svm.SVC(probability=True)`

Classification samplers such as SMOTE can be passed only to `RobustClassificationFeatureSelectorCV`:

```python
from imblearn.over_sampling import SMOTE
from ranking_feature_selector import RobustClassificationFeatureSelectorCV

selector = RobustClassificationFeatureSelectorCV(
    model=model,
    max_features=20,
    sampler=SMOTE(random_state=42),
    preset="safe",
)
```

### Regression

Required API:

```python
model.fit(X_train, y_train)
model.predict(X_valid)
```

Recommended models include:

- `sklearn.ensemble.RandomForestRegressor`
- `sklearn.ensemble.ExtraTreesRegressor`
- `sklearn.linear_model.ElasticNet`, `Ridge`, `Lasso`
- `lightgbm.LGBMRegressor`
- `xgboost.XGBRegressor`
- `catboost.CatBoostRegressor`
- `SVR` or `KNeighborsRegressor` with `importance_config={"method": "permutation"}`

### Survival analysis

Required API:

```python
model.fit(X_train, y_train)
risk_score = model.predict(X_valid)
```

Recommended models include:

- `sksurv.ensemble.RandomSurvivalForest`
- `sksurv.linear_model.CoxPHSurvivalAnalysis`
- `sksurv.ensemble.GradientBoostingSurvivalAnalysis`
- `sksurv.ensemble.ComponentwiseGradientBoostingSurvivalAnalysis`

`y` can be a `sksurv.util.Surv.from_arrays(event, time)` structured array, an `(event, time)` tuple, a two-column array, or a DataFrame with event/time columns.

```python
from sksurv.util import Surv

y = Surv.from_arrays(event=event, time=time)
```

## Survival risk-score configuration

By default, `RobustSurvivalFeatureSelectorCV` uses `model.predict(X)` as the risk score for C-index and permutation importance. Larger values are assumed to mean higher event risk.

| Setting | Meaning | Required model API |
|---|---|---|
| `risk_score="predict"` | Uses `model.predict(X)` | `predict(X)` |
| `risk_score="event_probability"` | Uses `1 - S(t | X)` at `prediction_time` | `predict_survival_function(X)` |
| `risk_score="cumulative_hazard"` | Uses `H(t | X)` at `prediction_time` | `predict_cumulative_hazard_function(X)` |
| callable | Uses `callable(model, X)` | User-defined |

Example using fixed-time event probability:

```python
selector = RobustSurvivalFeatureSelectorCV(
    model=rsf,
    max_features=20,
    risk_score="event_probability",
    prediction_time=365,
    preset="safe",
)
```

Raw RSF risk scores may have different scales across CV folds. This package does not assume that raw survival risk scores are calibrated or directly comparable across folds. They are used for within-fold rank-based evaluation such as C-index.

If your risk-score function returns lower values for higher risk, use:

```python
selector = RobustSurvivalFeatureSelectorCV(
    model=model,
    max_features=20,
    risk_score_direction="lower",
)
```

## Advanced usage

Advanced settings are grouped into dictionaries.

```python
selector = RobustSurvivalFeatureSelectorCV(
    model=rsf,
    max_features=20,
    preset="custom",
    cv_config={
        "outer_splits": 5,
        "inner_splits": 4,
        "survival_stratify": "event",
    },
    selection_config={
        "selection_rule": "one_se",
        "min_selection_rate": 0.5,
        "use_shadow": True,
        "k_grid": [5, 10, 15, 20],
    },
    importance_config={
        "method": "permutation",
        "scoring": "auto",
        "n_repeats": 10,
        "n_jobs": -1,
    },
)
```

### Importance backends

| Method | Classification | Regression | Survival |
|---|---:|---:|---:|
| `"auto"` | SHAP when available, fallback otherwise | SHAP when available, fallback otherwise | permutation C-index |
| `"shap"` | SHAP-based ranking | SHAP-based ranking | not the default |
| `"permutation"` | metric-aligned permutation | metric-aligned permutation | metric-aligned permutation |

When `importance_config={"method": "permutation", "scoring": "auto"}`, permutation importance is aligned with the optimization metric: log loss for classification, RMSE for regression, and C-index for survival.

## Results

After fitting:

```python
selector.selected_features_
selector.result_.outer_results
selector.result_.inner_results
selector.result_.selection_frequency
selector.summary()
```

Save all result tables and metadata:

```python
selector.result_.save("fs_results")
```

This writes:

```text
outer_results.csv
inner_results.csv
inner_details.csv
selection_frequency.csv
final_features.csv
ranking_outer_*.csv
config.json
metadata.json
```

## Requirements files

The `requirements/` directory provides install shortcuts. The authoritative dependency definitions remain in `pyproject.toml`.

```bash
pip install -r requirements/base.txt
pip install -r requirements/survival.txt
pip install -r requirements/shap.txt
pip install -r requirements/imbalance.txt
pip install -r requirements/all.txt
```

## Development

```bash
git clone https://github.com/rik-yosh/ranking-feature-selector.git
cd ranking-feature-selector
pip install -e ".[dev]"
pytest -q
python -m build
python -m twine check dist/*
```

## GitHub Actions publishing

This repository includes GitHub Actions workflows for tests and PyPI publishing.

- `.github/workflows/test.yml` runs tests and package checks on pushes and pull requests.
- `.github/workflows/publish.yml` builds and publishes on tags matching `v*`.

For PyPI Trusted Publishing, configure the PyPI project with your GitHub owner, repository, workflow filename, and environment before pushing the release tag.

## Important limitations

- The package targets research workflows and should not be used as the sole basis for clinical decision-making.
- Outer-CV performance reflects fold-specific selected feature sets.
- `selected_features_` is a stability-based aggregate across outer folds.
- Column-expanding preprocessing should be performed before using the selector.
- Survival raw risk scores are not assumed to be comparable across CV folds.

## License

MIT License. See [LICENSE](LICENSE).
