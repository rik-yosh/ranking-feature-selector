import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

from ranking_feature_selector import RobustClassificationFeatureSelectorCV

X_np, y_np = make_classification(
    n_samples=180,
    n_features=25,
    n_informative=6,
    n_redundant=4,
    weights=[0.75, 0.25],
    random_state=42,
)
X = pd.DataFrame(X_np, columns=[f"x{i}" for i in range(X_np.shape[1])])
y = pd.Series(y_np)

model = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

selector = RobustClassificationFeatureSelectorCV(
    model=model,
    max_features=15,
    preset="safe",
    preprocessor="auto",
    random_state=42,
    sampler=SMOTE(k_neighbors=3, random_state=42),
    importance_config={"shap_sample_size": 120},
)

selector.fit(X, y)
print(selector.summary())
print(selector.selected_features_)

final_model = selector.fit_final_model(X, y)
