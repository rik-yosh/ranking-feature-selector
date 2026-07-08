import pandas as pd
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

from ranking_feature_selector import RobustSurvivalFeatureSelectorCV

# X: pandas DataFrame
# event: True = observed event, False = right-censored
# time: event or censoring time
# y = Surv.from_arrays(event=event, time=time)

rsf = RandomSurvivalForest(
    n_estimators=500,
    min_samples_split=6,
    min_samples_leaf=3,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)

selector = RobustSurvivalFeatureSelectorCV(
    model=rsf,
    max_features=20,
    preset="safe",
    preprocessor="auto",
    random_state=42,
    importance_config={"n_repeats": 10, "n_jobs": -1},
)

selector.fit(X, y, groups=center_id)
print(selector.summary())
print(selector.selected_features_)

final_model = selector.fit_final_model(X, y)
risk_score = final_model.predict_score(X)
survival_curves = final_model.predict_survival_function(X, return_array=True)
