import pandas as pd
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

from ranking_feature_selector import RobustRegressionFeatureSelectorCV

X_np, y_np = make_regression(
    n_samples=160,
    n_features=30,
    n_informative=7,
    noise=10.0,
    random_state=42,
)
X = pd.DataFrame(X_np, columns=[f"x{i}" for i in range(X_np.shape[1])])
y = pd.Series(y_np)

model = RandomForestRegressor(
    n_estimators=300,
    max_features="sqrt",
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

selector = RobustRegressionFeatureSelectorCV(
    model=model,
    max_features=15,
    preset="safe",
    preprocessor="standardize",
    random_state=42,
    importance_config={"shap_sample_size": 120},
)

selector.fit(X, y)
print(selector.summary())
print(selector.selected_features_)

final_model = selector.fit_final_model(X, y)
