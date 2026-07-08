# Changelog

All notable changes to this project will be documented in this file.

The project follows semantic versioning once the public API is stable. Releases below `1.0.0` may still introduce API refinements.

## [0.4.3] - 2026-07-08

### Changed

- Renamed the distribution package to `ranking-feature-selector`.
- Added the primary import package `ranking_feature_selector`.
- Kept backward-compatible import aliases for `robust_feature_selector` and `robust_shap_selector`.
- Updated package metadata for PyPI publication.
- Updated README with `pip install` examples, model compatibility, survival risk-score guidance, advanced usage, result saving, and development instructions.
- Added GitHub Actions workflows for tests and PyPI publishing.
- Added `requirements/` install shortcut files.

### Added

- `CHANGELOG.md`.
- Explicit MIT `LICENSE` file.
- PyPI-oriented `project.urls` placeholders in `pyproject.toml`.

## [0.4.2] - 2026-07-08

### Added

- Lazy / optional SHAP import.
- Metric-aligned permutation importance scoring.
- Event-stratified survival cross-validation by default.
- Safer ndarray handling in fitted model bundles.
- Config and metadata export in `result.save()`.

## [0.4.1] - 2026-07-08

### Added

- Survival risk-score configuration via `risk_score`, `prediction_time`, and `risk_score_direction`.
- Fixed-time event probability and cumulative hazard risk-score options.

## [0.4.0] - 2026-07-08

### Added

- Task-specific selector classes:
  - `RobustClassificationFeatureSelectorCV`
  - `RobustRegressionFeatureSelectorCV`
  - `RobustSurvivalFeatureSelectorCV`
- Simplified public API with `max_features`, `preset`, `preprocessor`, and grouped config dictionaries.

## [0.3.0] - 2026-07-08

### Added

- `RobustFeatureSelectorCV` as the primary class.
- Presets: `fast`, `safe`, `publication`, and `custom`.
- `preprocessor` and standardization support.

## [0.2.0] - 2026-07-08

### Added

- Initial survival-analysis support.
- Permutation C-index importance for survival models.

## [0.1.0] - 2026-07-08

### Added

- Initial nested-CV feature selection prototype for classification and regression.
