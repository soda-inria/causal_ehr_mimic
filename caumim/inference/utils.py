import numpy as np
import pandas as pd
from sklearn.calibration import column_or_1d
from typing import Dict, Iterable, List
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import is_classifier


def make_random_search_pipeline(
    estimator: BaseEstimator,
    param_distributions: Dict,
    column_transformer: ColumnTransformer = None,
    # estimation_methods: str ,
    n_iter: int = 10,
    random_state: int = 42,
):
    pipeline_steps = []
    if column_transformer is not None:
        pipeline_steps.append(("preprocessor", column_transformer))
    pipeline_steps.append(("estimator", estimator))
    # better to optimize for brier score if we target a binary treatment
    if is_classifier(estimator):
        scoring_ = "neg_brier_score"
    else:
        scoring_ = None
    pipeline = RandomizedSearchCV(
        Pipeline(pipeline_steps),
        param_distributions=param_distributions,
        random_state=random_state,
        n_iter=n_iter,
        n_jobs=-1,
        scoring=scoring_,
        # pre_dispatch=2,
    )

    return pipeline


class dummy1Fold:
    def __init__(self) -> None:
        pass

    def split(self, X, y=None, groups=None):
        indices = np.arange(X.shape[0])
        yield indices, indices


def cross_val_predict_from_fitted(
    estimators, X: np.array, A=None, cv=None, method: str = "predict"
):
    """Compute cross-validation predictions from fitted estimators.

    Args:
        estimators (List): List of fitted estimators.
        X (np.array): Predictors
        splitter ([type], optional): splitter. Defaults to None.
        method (str, optional): estimator method for prediction : ["predict", "predict_proba"]. Defaults to "predict".

    Returns:
        hat_Y (np.array): predictions

    """
    hat_Y = []
    indices = []
    if A is None:
        iterator = dummy1Fold().split(X)
    elif hasattr(cv, "split"):
        iterator = cv.split(X, A)
    for i, (train_ix, test_ix) in enumerate(iterator):
        estimator = estimators[i]
        func = getattr(estimator, method)
        hat_Y.append(func(X[test_ix]))
        indices.append(test_ix)

    if A is None:
        # Average in case of no CV (ie. leftout)
        hat_Y = np.mean(hat_Y, axis=0)
    else:
        indices = np.argsort(np.concatenate(indices, axis=0))
        hat_Y = np.concatenate(hat_Y, axis=0)[indices]
    return hat_Y


def get_treatment_and_covariates(X, treament_col=None):
    """Split treatment and covariates from full covariate matrix $X=[a, X_cov]$.

    Require that the first column of $X$ is the treatment indicator.

    Args:
        X (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    if isinstance(X, pd.DataFrame):
        if treament_col is not None:
            a = X[treament_col]
            X_cov = X[[c for c in X.columns if c != treament_col]]
        else:
            raise ValueError(
                "If X is a pandas DataFrame, treatment_col should be specified."
            )
    elif isinstance(X, np.ndarray):
        a = X[:, 0]
        X_cov = X[:, 1:]
        a = column_or_1d(a, warn=True)
    else:
        raise ValueError(
            "X should be either a pandas DataFrame or a numpy array."
        )
    if not np.array_equal(a, a.astype(bool)):
        raise ValueError(
            "First column of covariates should contains the treatment indicator as binary values."
        )
    return a, X_cov


def cast_to_dataframe(X, treatment_column: str) -> pd.DataFrame:
    if isinstance(X, np.ndarray):
        X_ = pd.DataFrame(
            X,
            columns=[
                treatment_column,
                *[f"X_{i}" for i in range(X.shape[1] - 1)],
            ],
        )
    else:
        X_ = X.copy()
    return X_


# ### Causal Estimators for ATE ### #
# TODO: add  closed form CI
def tau_diff_means(y, a):
    """Simple difference in means estimator

    Args:
        y ([type]): [description]
        a ([type]): [description]

    Returns:
        [type]: [description]
    """
    y1 = y[a == 1]  # Outcome in treatment grp
    y0 = y[a == 0]  # Outcome in control group
    n1 = a.sum()  # Number of obs in treatment
    n0 = len(a) - n1  # Number of obs in control
    # Difference in means is ATE
    ate = np.mean(y1) - np.mean(y0)
    # 95% Confidence intervals
    se_hat = np.sqrt(np.var(y0) / (n0 - 1) + np.var(y1) / (n1 - 1))
    lower_ci = ate - 1.96 * se_hat
    upper_ci = ate + 1.96 * se_hat
    return {
        "hat_ate": ate,
        "hat_cate": None,
        "lower_ci": lower_ci,
        "upper_ci": upper_ci,
    }


def tau_ipw(y, a, a_hat_proba):
    cate = a * y / a_hat_proba - (1 - a) * y / (1 - a_hat_proba)
    return {
        "hat_ate": cate.mean(axis=0),
        "hat_cate": None,
        "lower_ci": None,
        "upper_ci": None,
    }


def tau_g_formula(mu_1, mu_0):
    cate = mu_1 - mu_0
    return {
        "hat_ate": cate.mean(axis=0),
        "hat_cate": cate,
        "lower_ci": None,
        "upper_ci": None,
    }


def aipw_formula(y, mu_1, mu_0, a, a_hat_proba):
    return (
        mu_1
        - mu_0
        + a * (y - mu_1) / a_hat_proba
        - (1 - a) * (y - mu_0) / (1 - a_hat_proba)
    )


def tau_aipw(y, mu_1, mu_0, a, a_hat_proba):
    # NOTE: AIPW does not estimate CATE, HTE should be done with Residual learners (cf. Wager 361)
    cate = aipw_formula(y, mu_1, mu_0, a, a_hat_proba)
    return {
        "hat_ate": cate.mean(axis=0),
        "hat_cate": cate,
        "lower_ci": None,
        "upper_ci": None,
    }


def safe_hstack(it: List):
    """Safe hstack for numpy and pandas"""
    first_elem = it[0]
    if isinstance(first_elem, pd.DataFrame):
        return pd.concat(it, axis=1)
    elif isinstance(first_elem, np.ndarray):
        return np.hstack(it)
    else:
        raise ValueError(
            "X should be either a pandas DataFrame or a numpy array."
        )


def safe_vstack(it: List):
    """Safe vstack for numpy and pandas"""
    first_elem = it[0]
    if isinstance(first_elem, pd.DataFrame):
        return pd.concat(it, axis=0)
    elif isinstance(first_elem, np.ndarray):
        return np.vstack(it)
    else:
        raise ValueError(
            "X should be either a pandas DataFrame or a numpy array."
        )
