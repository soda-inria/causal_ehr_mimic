# pipelines
import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge


ESTIMATOR_RIDGE = {
    "name": "Regularized LR",
    "treatment_estimator": LogisticRegression(),
    "treatment_param_distributions": {
        "estimator__C": np.logspace(-3, 2, 10),
    },
    "outcome_estimator": Ridge(),
    "outcome_param_distributions": {
        "estimator__alpha": np.logspace(-3, 2, 10),
    },
}
ESTIMATOR_LR = {
    "name": "LR",
    "treatment_estimator": LogisticRegression(n_jobs=-1),
}
ESTIMATOR_RF = {
    "name": "Forests",
    "treatment_estimator": RandomForestClassifier(n_jobs=-1),
    "treatment_param_distributions": {
        "estimator__n_estimators": [10, 100, 200],
        "estimator__max_depth": [3, 10, 50],
    },
    "outcome_estimator": RandomForestRegressor(n_jobs=-1),
    "outcome_param_distributions": {
        "estimator__n_estimators": [10, 100, 200],
        "estimator__max_depth": [3, 10, 50],
    },
}

ESTIMATOR_HGB = {
    "name": "HGB",
    "treatment_estimator": HistGradientBoostingClassifier(early_stopping=True),
    "treatment_param_distributions": {
        "estimator__learning_rate": [0.001, 0.01, 0.1, 1],
        "estimator__max_iter": [10, 50, 100],
    },
    "outcome_estimator": HistGradientBoostingRegressor(early_stopping=True),
    "outcome_param_distributions": {
        "estimator__n_estimators": [10, 100, 200],
        "estimator__max_iter": [10, 50, 100],
    },
}

ESTIMATOR_DUMMY = {
    "name": "Uniform random classifier",
    "treatment_estimator": DummyClassifier(strategy="uniform", random_state=0),
    "treatment_param_distributions": None,
    "outcome_estimator": DummyRegressor(strategy="mean"),
    "outcome_param_distributions": None,
}
