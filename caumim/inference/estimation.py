import logging
from typing import Dict, Tuple
from matplotlib.transforms import IdentityTransform
import numpy as np
import pandas as pd
import polars as pl
from sklearn import clone
from sklearn.dummy import check_random_state
from sklearn.utils import _safe_indexing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble._stacking import _BaseStacking
from caumim.inference.scores import (
    print_metrics_binary,
    print_metrics_regression,
)

from caumim.inference.utils import (
    cast_to_dataframe,
    dummy1Fold,
    cross_val_predict_from_fitted,
    dummy1Fold,
    get_treatment_and_covariates,
    tau_aipw,
    tau_diff_means,
    tau_ipw,
    tau_g_formula,
)

# ### MetaLearners ### #
SLEARNER_LABEL = "SLearner"
TLEARNER_LABEL = "TLearner"
RLEARNER_LABEL = "RLearner"
AVAILABLE_LEARNERS = [
    SLEARNER_LABEL,
    TLEARNER_LABEL,
    RLEARNER_LABEL,
]
# ### Inference methods ### #
AIPW_LABEL = "AIPW"
IPW_LABEL = "IPW"
DM_LABEL = "DM"
G_FORMULA_LABEL = "G-formula"

## Own implementation of the (A)IPW estimator:
# Rational: The IPW does not exists in causalml, or econml (which wrongly deals
# with imputation by [encouraging missing data information
# leakage](https://github.com/py-why/EconML/issues/664)). The implementation of
# dowhy : does not do "honest" splitting, in the sense that it uses the same
# samples to learn and predict the propensity score. For flexible models, this
# leads to overfitting, so we cannot use RF for propensity score weighting.

# Note that the CI for simple nuisance inference with ML estimators are not
# asymptotically normals so cannot be given by closed form. It is not the case
# for AIPW which are asymptotically normal at the condition of sufficient
# (hard to define) smoothness and well-specification of the nuisance.
# For simplicity, we will estimate the CI by bootstrap, which should be valid for every method.


# ### ATE and CATE estimator ### #
class AteEstimator(object):
    def __init__(
        self,
        outcome_model: BaseEstimator = None,
        propensity_model: BaseEstimator = None,
        treatment_column: str = "a",
        tau: str = AIPW_LABEL,
        n_splits=5,
        random_state_cv=42,
        clip=1e-4,
        meta_learner: str = SLEARNER_LABEL,
    ) -> None:
        """ATE estimator supporting AIPW, IPW and G-computation with any sklearn
        pipeline for outcome and treatment models.

        Args:
            outcome_model (BaseEstimator, optional): [description]. Defaults to
            None. propensity_model (BaseEstimator, optional): [description].
            Defaults to None. tau (str, optional): [description]. Defaults to
            "AIPW". nb_splits (int, optional): [description]. Defaults to 5.
            random_state_cv (int, optional): [description]. Defaults to 42.
            min_propensity ([type], optional): Used to trim the high and low
            propensity scores that violate the overlap assumption. Defaults to
            1e-4.
        """
        super().__init__()
        self.tau = tau
        if self.tau in [DM_LABEL, IPW_LABEL]:
            outcome_model = None
            logging.info("TAU set to IPW, forcing outcome_model to be None")
        if self.tau in [DM_LABEL, G_FORMULA_LABEL]:
            propensity_model = None
            logging.info("TAU set to REG, forcing propensity_model to be None")
        self.outcome_model = outcome_model
        self.propensity_model = propensity_model
        self.outcome_models = []
        self.propensity_models = []
        self.random_state_cv = random_state_cv
        self.n_splits = n_splits
        self.clip = clip
        self.treatment_column = treatment_column
        if self.tau in [G_FORMULA_LABEL, AIPW_LABEL]:
            # meta_learner is necessary for G-formula and AIPW only
            self.meta_learner_type = meta_learner
            if self.meta_learner_type not in [TLEARNER_LABEL, SLEARNER_LABEL]:
                raise ValueError(
                    f"Meta-learner {self.meta_learner_type} not supported"
                )
            self.meta_learner = set_meta_learner(
                self.meta_learner_type,
                self.outcome_model,
                treatment_column=self.treatment_column,
            )
        else:
            self.meta_learner = None
            self.meta_learner_type = None
        # TODO: should have different kfold for propensity and outcome
        if self.n_splits == 1:
            self.kfold = dummy1Fold()
        else:
            if self.outcome_model is not None:
                if self.outcome_model._estimator_type == "classifier":
                    self.kfold = StratifiedKFold(
                        n_splits=self.n_splits,
                        random_state=self.random_state_cv,
                        shuffle=True,
                    )
                else:
                    self.kfold = KFold(
                        n_splits=self.n_splits,
                        random_state=self.random_state_cv,
                        shuffle=True,
                    )
            else:
                self.kfold = StratifiedKFold(
                    n_splits=self.n_splits,
                    random_state=self.random_state_cv,
                    shuffle=True,
                )

        self.predictions = {
            "hat_mu_1": [],
            "hat_mu_0": [],
            "hat_a_proba": [],
            "y": [],
            "a": [],
        }
        self.metrics = {}
        self.in_sample_cate = None
        self.in_sample_ate = None

    def fit(self, X, y):
        # force to process pandas DataFrame instead of numpy array
        X_ = cast_to_dataframe(X, self.treatment_column)
        a, X_cov = get_treatment_and_covariates(X_, self.treatment_column)
        if self.outcome_model is not None:
            # a is passed for stratified split
            splitter_ix = self.kfold.split(np.arange(len(X_cov)), a)
            for train_index, _ in splitter_ix:
                outcome_estimator_fold = clone(self.meta_learner)
                outcome_estimator_fold.fit(
                    _safe_indexing(X, train_index, axis=0),
                    _safe_indexing(y, train_index, axis=0),
                )
                self.outcome_models.append(outcome_estimator_fold)

        if self.propensity_model is not None:
            self._fit_propensity(X_cov, a)

    def _fit_propensity(self, X, a):
        # TODO: add recalibration here with a train/test split
        splitter_ix = self.kfold.split(np.arange(len(X)), a)
        # a = a.reshape(-1, 1)
        for train_index, _ in splitter_ix:
            if self.propensity_model is not None:
                propensity_estimator_fold = clone(
                    self.propensity_model, safe=True
                )
                propensity_estimator_fold.fit(
                    _safe_indexing(X, train_index, axis=0),
                    _safe_indexing(a, train_index, axis=0),
                )
                self.propensity_models.append(propensity_estimator_fold)

    # TODO: scoring is extern to a model in scikitlearn api and there is no reason to act differently for causal inference.
    def predict(
        self,
        X,
        y,
        leftout: bool = False,
    ) -> Tuple[Dict, Dict]:
        """Run the predictions for the nuisance models.
        If one or both of the nuisance models are set to none, do nothing but return dummy predictions proba = (-1, -1) and empty metrics.

        Args:
            X ([type]): Covariates
            y ([type]): Binary Outcome
            a ([type]): Binary Intervention
            leftout (binary, optional): If True, average the results over the folds and models. Defaults to False.
        Returns:
            predictions [Dict]: [description]
            metrics [Dict]: [description]
        """
        # Forcing one dimension for y and a
        y = np.array(y).ravel()
        X = cast_to_dataframe(X, self.treatment_column)
        a, X_cov = get_treatment_and_covariates(X, self.treatment_column)
        # a = a.ravel()
        # checking wich nuisance models are used
        n_outcome_models = len(self.outcome_models)
        n_ps_models = len(self.propensity_models)
        n_models = max(n_outcome_models, n_ps_models)

        if not leftout:
            chunks_indices = [
                np.arange(len(X))[test_ix]
                for _, test_ix in self.kfold.split(np.arange(len(X)), a)
            ]
            chunks_X_cov = [
                _safe_indexing(X_cov, test_ix, axis=0)
                for _, test_ix in self.kfold.split(np.arange(len(X)), a)
            ]
            chunks_y = [
                np.array(y)[test_ix]
                for _, test_ix in self.kfold.split(np.arange(len(X)), a)
            ]
            chunks_a = [
                np.array(a)[test_ix]
                for _, test_ix in self.kfold.split(np.arange(len(X)), a)
            ]
            # if no folds is provided, this is test data, we average over the cv models
        else:
            chunks_indices = [np.arange(len(X)) for i in range(n_models)]
            chunks_X_cov = [X_cov for i in range(n_models)]
            chunks_y = [np.array(y) for i in range(n_models)]
            chunks_a = [np.array(a) for i in range(n_models)]
        mu_1 = []
        mu_0 = []
        a_hat_proba = []
        for i in range(n_models):
            # reshaped_a = np.expand_dims(, axis=1)
            if n_outcome_models > 0:
                zeros_a = pd.DataFrame(
                    np.zeros_like(chunks_a[i]), columns=[self.treatment_column]
                )
                ones_a = pd.DataFrame(
                    np.ones_like(chunks_a[i]), columns=[self.treatment_column]
                )
                if self.outcome_model._estimator_type == "classifier":
                    chunk_y_hat_0 = self.outcome_models[i].predict_proba(
                        pd.DataFrame([zeros_a, chunks_X_cov[i]])
                    )[:, 1]
                    chunk_y_hat_1 = self.outcome_models[i].predict_proba(
                        pd.DataFrame([ones_a, chunks_X_cov[i]])
                    )[:, 1]
                else:
                    chunk_y_hat_0 = self.outcome_models[i].predict(
                        pd.DataFrame([zeros_a, chunks_X_cov[i]])
                    )
                    chunk_y_hat_1 = self.outcome_models[i].predict(
                        pd.DataFrame([ones_a, chunks_X_cov[i]])
                    )
                mu_1.append(chunk_y_hat_1)
                mu_0.append(chunk_y_hat_0)
            if n_ps_models > 0:
                a_hat_proba.append(
                    self.propensity_models[i].predict_proba(chunks_X_cov[i])[
                        :, 1
                    ]
                )
        # concatenate the results
        if not leftout:
            if n_outcome_models > 0:
                mu_1 = np.concatenate(mu_1, axis=0)
                mu_0 = np.concatenate(mu_0, axis=0)
            else:
                mu_1, mu_0 = (
                    np.repeat([np.nan], len(y), axis=0),
                    np.repeat([np.nan], len(y), axis=0),
                )
            if n_ps_models > 0:
                a_hat_proba = np.concatenate(a_hat_proba, axis=0)
            else:
                a_hat_proba = np.repeat([np.nan], len(a), axis=0)
            y = np.concatenate(chunks_y, axis=0)
            a = np.concatenate(chunks_a, axis=0)
            indices = np.concatenate(chunks_indices, axis=0)
        else:
            if n_outcome_models > 0:
                mu_1 = np.array(mu_1).mean(axis=0)
                mu_0 = np.array(mu_0).mean(axis=0)
            else:
                mu_1, mu_0 = (
                    np.repeat([np.nan], len(y), axis=0),
                    np.repeat([np.nan], len(y), axis=0),
                )
            if n_ps_models > 0:
                a_hat_proba = np.array(a_hat_proba).mean(axis=0)
            else:
                a_hat_proba = np.repeat([np.nan], len(a), axis=0)
            y = chunks_y[0]
            a = chunks_a[0]
            indices = chunks_indices[0]

        predictions = (
            pd.DataFrame(
                {
                    "idx": indices,
                    "hat_mu_1": mu_1,
                    "hat_mu_0": mu_0,
                    "hat_a_proba": a_hat_proba,
                    "y": y,
                    "a": a,
                }
            )
            .sort_values("idx")
            .reset_index(drop=True)
        )

        # metrics
        metrics = {}
        if self.outcome_model is not None:
            y_hat = a * mu_1 + (1 - a) * mu_0 * (1 - a)
            if self.outcome_model._estimator_type == "classifier":
                outcome_metrics = print_metrics_binary(y, y_hat, verbose=0)
            else:
                outcome_metrics = print_metrics_regression(y, y_hat, verbose=0)
            for metric_name, metric in outcome_metrics.items():
                metrics[f"outcome_{metric_name}"] = metric
        if self.propensity_model is not None:
            if np.all(a == 0) or np.all(a == 1):
                Warning(
                    "All propensity scores are 0 or 1, cannot compute scores for propensity models, returning dummy scores"
                )
                a = np.random.randint(0, 2, size=len(a))
            propensity_metrics = print_metrics_binary(a, a_hat_proba, verbose=0)
            for metric_name, metric in propensity_metrics.items():
                metrics[f"propensity_{metric_name}"] = metric

        # update if we are predicting the insample data
        if not leftout:
            self.predictions = predictions.copy()
            self.metrics = metrics.copy()
        return predictions, metrics

    def estimate(self, predictions, clip=None) -> float:
        if clip is None:
            clip = self.clip
        y = predictions["y"]
        mu_1 = predictions["hat_mu_1"]
        mu_0 = predictions["hat_mu_0"]
        a = predictions["a"]
        # clipping at inference time
        a_hat_proba = np.clip(predictions["hat_a_proba"], clip, 1 - clip)
        if self.tau == AIPW_LABEL:
            estimate = tau_aipw(
                y=y,
                mu_1=mu_1,
                mu_0=mu_0,
                a=a,
                a_hat_proba=a_hat_proba,
            )
        elif self.tau == DM_LABEL:
            estimate = tau_diff_means(y, a)
        elif self.tau == IPW_LABEL:
            estimate = tau_ipw(y=y, a=a, a_hat_proba=a_hat_proba)
        elif self.tau == G_FORMULA_LABEL:
            estimate = tau_g_formula(mu_1=mu_1, mu_0=mu_0)
        else:
            raise ValueError(f"{self.tau} is not a valid ATE estimator")
        return estimate

    def predict_estimate(self, X, y, leftout=False):
        predictions, metrics = self.predict(X, y, leftout=leftout)
        estimate = self.estimate(predictions)
        return predictions, estimate, metrics

    def fit_predict(self, X, y):
        self.fit(X, y)
        predictions, _, metrics = self.predict_estimate(X, y)
        return predictions, metrics

    def fit_predict_estimate(self, X, y) -> Tuple[Dict, Dict, Dict]:
        """
        - Fit the nuisance models
        - Predict the nuisance parameters
        - Estimate the causal target (ie. estimand)

        Args:
            X ([type]): Covariates
            y ([type]): Target (binary or continuous)
            a ([type]): Intervention (ie. Treatment)

        Returns:
            Tuple[Dict, Dict, Dict]: [predictions, estimate, metrics]
        """
        self.fit(X, y)
        predictions, estimate, metrics = self.predict_estimate(X, y)
        estimate = self.estimate(predictions)
        self.in_sample_cate = estimate["hat_cate"]
        self.in_sample_ate = estimate["hat_ate"]
        return predictions, estimate, metrics

    def get_confidence_intervals(
        self, X, y, n_reps: int = 5
    ) -> Tuple[float, float, float]:
        """
        Confidence intervals for the ATE using classical bootstrap.
        TODO: Do I have to refit the models ? I think so, but it's not clear to me, since
        dowhy is not doing this.

        Args:
            X (_type_): _description_ y (_type_): _description_ n_reps (int,
            optional): _description_. Defaults to 5.
        Returns:
            Tuple[float, float, float]: Lower bound, mean effect and upper
            bounds of the confidence interval at 95%.
        """
        self.bs_estimates = []
        self.bs_prediction = []
        self.bs_metrics = []
        for i in range(n_reps):
            rg = check_random_state(self.random_state_cv + i)
            index_bs = rg.choice(np.arange(len(X)), size=len(X), replace=True)
            X_bs = _safe_indexing(X, index_bs, axis=0).reset_index(drop=True)
            y_bs = _safe_indexing(y, index_bs, axis=0).reset_index(drop=True)
            # TODO: list index out of range, since we are fitting two time the
            # models
            predictions, estimate, metrics = self.fit_predict_estimate(
                X_bs, y_bs
            )

            self.bs_estimates.append(estimate)
            self.bs_prediction.append(predictions)
            self.bs_metrics.append(metrics)

        mean_effect = np.mean([e["hat_ate"] for e in self.bs_estimates])
        std_effect = np.std([e["hat_ate"] for e in self.bs_estimates])
        self.lb = mean_effect - 1.96 * std_effect
        self.mean_effect = mean_effect
        self.ub = mean_effect + 1.96 * std_effect
        return self.lb, self.mean_effect, self.ub


# TODO: adapt to Pandas Frame
class CateEstimator(object):
    """Estimator class for CATE model. Supports SLearner, TLearner and RLearner.
    Args:
        object ([type]): [description]
    """

    def __init__(
        self,
        meta_learner: BaseEstimator,
        a_estimator: BaseEstimator = None,
        y_estimator: BaseEstimator = None,
        a_hyperparameters: Dict = None,
        y_hyperparameters: Dict = None,
        a_scoring: str = "roc_auc",
        y_scoring: str = "r2",
        n_iter: int = 10,
        cv: int = None,
        random_state_hp_search=0,
        n_jobs=-1,
        strict_overlap=1e-10,
    ) -> None:
        self.meta_learner = meta_learner
        self.a_estimator = a_estimator
        self.y_estimator = y_estimator
        self.logger = logging.getLogger(__name__)
        if self.y_estimator is not None:
            self.y_hyperparameters = y_hyperparameters
            assert (
                self.y_estimator._estimator_type == "regressor"
            ), "Mean outcome estimator must be a regressor"
        else:
            self.y_hyperparameters = None
            if self.y_hyperparameters is not None:
                self.logger.warning(
                    "No mean outcome estimator provided, forcing y_hyperparameters to None"
                )
        if self.a_estimator is not None:
            self.a_hyperparameters = a_hyperparameters
            assert (
                self.a_estimator._estimator_type == "classifier"
            ), "Treatment estimator must be a classifier"
        else:
            self.a_hyperparameters = None
            if self.a_hyperparameters is not None:
                self.logger.warning(
                    "No treatment estimator provided, forcing a_hyperparameters to None"
                )

        if cv is None:
            self.cv = StratifiedKFold(n_splits=5)
        elif cv == 1:
            self.cv = dummy1Fold()
        else:
            self.cv = cv
        self.random_state_hp_search = random_state_hp_search
        self.n_jobs = n_jobs
        self.strict_overlap = strict_overlap
        self.y_scoring = y_scoring
        self.a_scoring = a_scoring
        self.n_iter = n_iter
        assert (
            self.meta_learner.final_estimator._estimator_type == "regressor"
        ), "CATE estimator must be a regressor"

    def fit_nuisances(self, X, y):
        """Learn unknown nuisance ($\\check e$, $\\check m$) necessary for:
        - estimation (R meta-learning)
        - model selection ($\widehat{\mu\mathrm{-risk}}_{IPTW}(\frac{1}{\check e_a}, f)$)
        Args:
            X ([type]): [description]
            Y ([type]): [description]
        Returns:
            [type]: [description]
        """
        if (self.a_estimator is None) or (self.y_estimator is None):
            raise ValueError(
                "No nuisance estimators provided for CATE, cannot estimate nuisances."
            )
        a, X_cov = get_treatment_and_covariates(X)

        # Find appropriate parameters for nuisance models
        self.y_model_rs_ = RandomizedSearchCV(
            estimator=self.y_estimator,
            param_distributions=self.y_hyperparameters,
            scoring=self.y_scoring,
            n_iter=self.n_iter,
            random_state=self.random_state_hp_search,
            cv=None,
        )
        self.a_model_rs_ = RandomizedSearchCV(
            estimator=self.a_estimator,
            param_distributions=self.a_hyperparameters,
            scoring=self.a_scoring,
            n_iter=self.n_iter,
            n_jobs=self.n_jobs,
            random_state=self.random_state_hp_search,
            cv=None,
        )
        self.y_model_rs_results_ = self.y_model_rs_.fit(X_cov, y)
        self.a_model_rs_results_ = self.a_model_rs_.fit(X_cov, a)
        # Refit best model with CV
        splitter_y = self.cv.split(X_cov, a)
        self.y_nuisance_estimators_cv_ = cross_validate(
            clone(self.y_model_rs_results_.best_estimator_),
            X_cov,
            y,
            cv=splitter_y,
            return_estimator=True,
            scoring="neg_mean_squared_error",
        )

        splitter_a = self.cv.split(X_cov, a)
        self.a_nuisance_estimators_cv_ = cross_validate(
            clone(self.a_model_rs_results_.best_estimator_),
            X_cov,
            a,
            cv=splitter_a,
            return_estimator=True,
            scoring="neg_brier_score",
        )
        self.a_nuisance_estimators_ = self.a_nuisance_estimators_cv_[
            "estimator"
        ]
        self.y_nuisance_estimators_ = self.y_nuisance_estimators_cv_[
            "estimator"
        ]
        return self

    def fit(self, X, y):
        # Rq: In case of R-learner, fit should always be done on leftout data (or we should include a nested CV procedure)
        self.meta_learner_ = clone(self.meta_learner)
        self.meta_learner_.fit(X, y)
        return self

    def predict(self, X):
        predictions = {}
        a, X_cov = get_treatment_and_covariates(X)
        if hasattr(self, "a_nuisance_estimators_"):
            hat_e = cross_val_predict_from_fitted(
                estimators=self.a_nuisance_estimators_,
                X=X_cov,
                A=a,
                cv=self.cv,
                method="predict_proba",
            )[:, 1]
            if self.strict_overlap is not None:
                hat_e[hat_e <= 0.5] = hat_e[hat_e <= 0.5] + self.strict_overlap
                hat_e[hat_e > 0.5] = hat_e[hat_e > 0.5] - self.strict_overlap
            predictions["check_e"] = hat_e
        if hasattr(self, "y_nuisance_estimators_"):
            hat_m = cross_val_predict_from_fitted(
                estimators=self.y_nuisance_estimators_,
                X=X_cov,
                A=None,
                method="predict",
                cv=self.cv,
            )
            predictions["check_m"] = hat_m
        if self.meta_learner.__class__.__name__ == RLEARNER_LABEL:
            predictions["hat_tau"] = self.meta_learner_.predict(X)
            predictions["hat_mu_0"] = hat_m - hat_e * predictions["hat_tau"]
            predictions["hat_mu_1"] = (
                predictions["hat_mu_0"] + predictions["hat_tau"]
            )
        else:
            predictions["hat_mu_0"] = self.meta_learner_.predict(
                np.column_stack((np.zeros(X.shape[0]) * 1.0, X_cov))
            )
            predictions["hat_mu_1"] = self.meta_learner_.predict(
                np.column_stack((np.ones(X.shape[0]) * 1.0, X_cov))
            )
            predictions["hat_tau"] = (
                predictions["hat_mu_1"] - predictions["hat_mu_0"]
            )

        return pd.DataFrame(predictions)

    def describe(self):
        model_names = ["Outcome, m", "Treatment, e"]
        model_descriptions = []
        for model_name, model, model_hps in zip(
            model_names,
            [self.y_estimator, self.a_estimator],
            [self.y_hyperparameters, self.a_hyperparameters],
        ):
            model_desc = {"Model": model_name}
            if isinstance(model, _BaseStacking):
                y_stacked_models = []
                for est in model.estimators:
                    if hasattr(est[1], "steps"):
                        y_stacked_models.append(est[1].steps[-1][0])
                    else:
                        y_stacked_models.append(est[0])
                model_desc["Estimator"] = (
                    "StackedRegressor(" + ", ".join(y_stacked_models) + ")"
                )
            else:
                model_desc["Estimator"] = type(model).__name__
            for i, (hp_name, hp_values) in enumerate(model_hps.items()):
                hp_str = (
                    hp_name.replace("_", " ")
                    + ": "
                    + str(hp_values).replace("_", " ")
                )
                if i == 0:
                    model_desc["Hyper-parameters grid"] = hp_str
                else:
                    model_desc = {
                        "Model": "",
                        "Estimator": "",
                        "Hyper-parameters grid": hp_str,
                    }
                model_descriptions.append(model_desc)
        return pd.DataFrame(model_descriptions).set_index(
            ["Model", "Estimator"]
        )


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, random_state=None):
        if random_state is None:
            self.random_state = 0
        else:
            self.random_state = random_state

    def fit(self, X, y=None):
        self.components_ = None
        return self

    def transform(self, X):
        X = self._validate_data(X, accept_sparse="csr", reset=False)
        return X * 1


class MetaLearner:
    _required_parameters = ["estimator"]
    """Mixin class for all meta learners."""

    def predict_cate(self, X):
        a, X_cov = get_treatment_and_covariates(X)
        hat_y_1 = self.predict(np.column_stack([np.ones_like(a), X_cov]))
        hat_y_0 = self.predict(np.column_stack([np.zeros_like(a), X_cov]))
        return hat_y_1 - hat_y_0

    def predict(self, X):
        raise NotImplementedError


def set_meta_learner(
    meta_learner_name: str,
    final_estimator: BaseEstimator,
    featurizer: TransformerMixin = None,
    treatment_column: str = "a",
):
    if meta_learner_name == TLEARNER_LABEL:
        meta_learner = TLearner(final_estimator, featurizer, treatment_column)
    elif meta_learner_name == SLEARNER_LABEL:
        meta_learner = SLearner(final_estimator, featurizer, treatment_column)
    elif meta_learner_name == RLEARNER_LABEL:
        meta_learner = make_pipeline(featurizer, final_estimator)
    else:
        raise ValueError(
            "Got {} meta_learner, but supports only following : \n {}".format(
                meta_learner_name, AVAILABLE_LEARNERS
            )
        )
    return meta_learner


class SLearner(MetaLearner, BaseEstimator):
    """Meta-learner with shared featurization and regressors between both treated and control populations.

    Fit/predict a transformation, g of the covariates, then apply a sklearn estimator, f on the transformed covariates augmtented with the treatment variable
    $m(x,a) = f \big([a, g(x)] \big)$
    Args:
        final_estimator (BaseEstimator): _description_
        featurizer (TransformerMixin): _description_
    """

    def __init__(
        self,
        final_estimator: BaseEstimator,
        featurizer: TransformerMixin = None,
        treatment_column: str = "a",
    ) -> None:
        self.final_estimator = final_estimator
        self.treatment_column = treatment_column
        if featurizer is None:
            self.featurizer = IdentityTransformer()
        else:
            self.featurizer = featurizer

    def fit(self, X, y):
        a, X_cov = get_treatment_and_covariates(X, self.treatment_column)
        self.featurizer_ = clone(self.featurizer)
        self.final_estimator_ = clone(self.final_estimator)
        X_transformed = self.featurizer_.fit_transform(X_cov)
        self.final_estimator_.fit(np.column_stack((a, X_transformed)), y)
        return self

    def predict(self, X):
        a, X_cov = get_treatment_and_covariates(X, self.treatment_column)
        X_transformed = self.featurizer_.transform(X_cov)
        return self.final_estimator_.predict(
            np.column_stack((a, X_transformed))
        )

    def predict_proba(self, X):
        a, X_cov = get_treatment_and_covariates(X, self.treatment_column)
        X_transformed = self.featurizer_.transform(X_cov)
        return self.final_estimator_.predict_proba(
            np.column_stack((a, X_transformed))
        )

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)


class TLearner(MetaLearner, BaseEstimator):
    """Meta-learner with separate featurization and regressors between two populations.

    For each population, fit/predict a transformation, g_a of the covariates, then apply a sklearn estimator, f_a on the transformed covariates:

    $m(x,a) = f_a \big([g_a(x)] \big)$
    Args:
        final_estimator (BaseEstimator): _description_
        featurizer (TransformerMixin, optional): _description_. Defaults to None.
    """

    def __init__(
        self,
        final_estimator: BaseEstimator,
        featurizer: TransformerMixin = None,
        treatment_column: str = "a",
    ) -> None:
        self.final_estimator = final_estimator
        self.treatment_column = treatment_column
        if featurizer is None:
            self.featurizer = IdentityTransform()
        else:
            self.featurizer = featurizer

    def fit(self, X, y):
        a, X_cov = get_treatment_and_covariates(X, self.treatment_column)

        self.featurizer_control_ = clone(self.featurizer)
        self.final_estimator_control_ = clone(self.final_estimator)
        self.featurizer_treated_ = clone(self.featurizer)
        self.final_estimator_treated_ = clone(self.final_estimator)

        mask_control = a == 0
        mask_treated = a == 1
        if (mask_control.sum() == 0) or (mask_treated.sum() == 0):
            raise AttributeError(
                "Provided crossfit folds contain training splits that don't contain at least one example of each treatment"
            )
        X_control_transformed = self.featurizer_control_.fit_transform(
            X_cov[mask_control]
        )
        X_treated_transformed = self.featurizer_treated_.fit_transform(
            X_cov[mask_treated]
        )

        self.final_estimator_control_.fit(
            X_control_transformed, y[mask_control]
        )
        self.final_estimator_treated_.fit(
            X_treated_transformed, y[mask_treated]
        )

        return self

    def predict(self, X):
        a, X_cov = get_treatment_and_covariates(X, self.treatment_column)

        mask_control = a == 0
        mask_treated = a == 1
        y = np.empty_like(a) * 0.0

        if sum(mask_control) != 0:
            X_control_transformed = self.featurizer_control_.transform(
                X_cov[mask_control]
            )
            y[mask_control] = self.final_estimator_control_.predict(
                X_control_transformed
            )
        if sum(mask_treated) != 0:
            X_treated_transformed = self.featurizer_treated_.transform(
                X_cov[mask_treated]
            )
            y[mask_treated] = self.final_estimator_treated_.predict(
                X_treated_transformed
            )
        return y

    def predict_proba(self, X):
        a, X_cov = get_treatment_and_covariates(X, self.treatment_column)

        mask_control = a == 0
        mask_treated = a == 1
        y = np.empty((a.shape[0], 2)) * 0.0

        if sum(mask_control) != 0:
            X_control_transformed = self.featurizer_control_.transform(
                X_cov[mask_control]
            )
            y[mask_control] = self.final_estimator_control_.predict_proba(
                X_control_transformed
            )
        if sum(mask_treated) != 0:
            X_treated_transformed = self.featurizer_treated_.transform(
                X_cov[mask_treated]
            )
            y[mask_treated] = self.final_estimator_treated_.predict_proba(
                X_treated_transformed
            )
        return y

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)


class RLearner(MetaLearner, BaseEstimator):
    def __init__(
        self,
        final_estimator: BaseEstimator,
        y_estimator: BaseEstimator,
        a_estimator: BaseEstimator,
        featurizer: TransformerMixin = None,
        treatment_column: str = "a",
    ) -> None:
        if featurizer is None:
            self.featurizer = IdentityTransformer()
        else:
            self.featurizer = featurizer
        self.treatment_column = treatment_column
        self.final_estimator = make_pipeline(self.featurizer, final_estimator)
        self.y_estimator = make_pipeline(self.featurizer, y_estimator)
        self.a_estimator = make_pipeline(self.featurizer, a_estimator)

    def fit(self, X, y):
        a, X_cov = get_treatment_and_covariates(X, self.treatment_column)
        splitter_a = self.cv.split(X_cov, a)
        self.a_estimator_cv_ = cross_validate(
            clone(self.y_estimator),
            X_cov,
            a,
            cv=splitter_a,
            return_estimator=True,
            scoring="neg_brier_score",
        )
        splitter_y = self.cv.split(X_cov, y)
        self.y_estimator_cv_ = cross_validate(
            clone(self.y_estimator),
            X_cov,
            y,
            cv=splitter_y,
            return_estimator=True,
            scoring="neg_brier_score",
        )

        self.hat_e_ = cross_val_predict_from_fitted(
            estimators=self.a_estimator_cv_["estimator"],
            X=X_cov,
            A=a,
            cv=self.cv,
            method="predict_proba",
        )[:, 1]
        self.hat_m_ = cross_val_predict_from_fitted(
            estimators=self.y_estimator_cv_["estimator"],
            X=X,
            A=None,
            cv=self.cv,
            method="predict",
        )
        weights = (a - self.hat_e_) ** 2
        y_tilde = (y - self.hat_m_) / (a - self.hat_e_)
        self.final_estimator_ = clone(self.final_estimator)
        self.final_estimator_.fit(
            X_cov, y_tilde, regression__sample_weight=weights
        )

    def predict(self, X):
        _, X_cov = get_treatment_and_covariates(X, self.treatment_column)
        return self.final_estimator.predict(X_cov)
