from datetime import datetime
from pathlib import Path
from typing import Dict
from attr import dataclass
from dowhy import CausalModel
from dowhy.causal_estimator import CausalEstimate

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder
from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression

from caumim.constants import (
    RANDOM_STATE,
    RESULT_ATE,
    RESULT_ATE_LB,
    RESULT_ATE_UB,
    MIN_PS_SCORE,
)

from econml.metalearners import TLearner
from econml.grf import CausalForest
from econml.dml import DML  # ortho-learning ie. R-like learner.
from econml.dr import DRLearner
from econml.inference import BootstrapInference
from joblib import Memory

location = "./cachedir"
memory = Memory(location, verbose=0)


@dataclass
class CateConfig:
    expe_name: None
    cohort_folder: Path
    outcome_name: str
    experience_grid_dict: Dict
    fraction: float = 1.0
    random_state: int = 0
    bootstrap_num_samples: int = 50
    train_test_random_state: int = 0
    test_size: float = 0.2


def log_estimate(estimate: Dict, estimate_folder: str):
    estimate_folder_path = Path(estimate_folder)
    estimate_folder_path.mkdir(parents=True, exist_ok=True)

    estimate_ = {k: [v] for k, v in estimate.items()}
    current_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    estimate_["time_stamp"] = [current_time]
    pd.DataFrame(estimate_).to_parquet(
        str(estimate_folder_path / f"{current_time}.parquet")
    )


@memory.cache()
def fit_randomized_search(
    X, a, y, treatment_pipeline: Pipeline, outcome_pipeline: Pipeline
):
    "Wrapper to save the model with joblib for different iterations."
    treatment_pipeline.fit(X, a)
    treatment_estimator_w_best_HP = treatment_pipeline.best_estimator_
    outcome_pipeline.fit(X, y)
    outcome_estimator_w_best_HP = outcome_pipeline.best_estimator_
    return treatment_estimator_w_best_HP, outcome_estimator_w_best_HP


def make_column_transformer(
    numerical_features: list, categorical_features: list
) -> Pipeline:
    """
    Create a simple feature preprocessing pipeline.
    """
    categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
    numerical_preprocessor = make_pipeline(
        *[
            StandardScaler(),
            # TODO: what is the effect of adding the mask on the results ?
            SimpleImputer(strategy="median"),
        ]
    )
    column_transformer = ColumnTransformer(
        [
            (
                "one-hot-encoder",
                categorical_preprocessor,
                categorical_features,
            ),
            (
                "standard_scaler",
                numerical_preprocessor,
                numerical_features,
            ),
        ],
        remainder="passthrough",
        # The passthrough is necessary for all the event features.
    )
    return column_transformer


ECONML_CATE_LEARNERS = ["DRLearner", "DML"]
ECONML_META_LEARNERS = ["TLearner"]

ECONML_LEARNERS = [*ECONML_CATE_LEARNERS, *ECONML_META_LEARNERS, "CausalForest"]

DEFAULT_BS_NUM_SAMPLES = 100


class InferenceWrapper(BaseEstimator):

    """
    Wrapper for all estimation methods (from dowhy or econml).

    treatment_pipeline: Model for the treatment allocation
    outcome_pipeline: Model for the outcome
    estimation_method: Causal estimator (IPW, matching, Doubly Machine Learning, etc...)
    outcome_name: column name of the outcome
    treatment_name: column name of the treatment
    bootstrap_num_samples: Number of bootstrap samples for the confidence interval
    model_final: Final estimator used for CATE estimator
    """

    def __init__(
        self,
        treatment_pipeline: Pipeline,
        outcome_pipeline: Pipeline,
        estimation_method: str,
        outcome_name: str,
        treatment_name: str,
        bootstrap_num_samples: int = DEFAULT_BS_NUM_SAMPLES,
        model_final: BaseEstimator = None,
    ) -> None:
        super().__init__()
        self.treatment_pipeline = treatment_pipeline
        self.outcome_pipeline = outcome_pipeline
        self.estimation_method = estimation_method
        self.outcome_name = outcome_name
        self.treatment_name = treatment_name
        self._not_supported_estimation_method = ValueError(
            f"{self.estimation_method} not supported."
        )
        self.bootstrap_num_samples = bootstrap_num_samples
        self.model_final = model_final

    def fit(self, X, y, X_cate=None):
        """Fit the inference estimator independently from the underlying package.

        Args:
            X (_type_): Confounders and treatment
            y (_type_): Outcome of interest
            X_cate (_type_, optional): Feature on which . Defaults to None.

        Raises:
            self._not_supported_estimation_method: _description_
            self._not_supported_estimation_method: _description_

        Returns:
            _type_: _description_
        """
        if self.estimation_method.startswith("backdoor."):
            self.inference_estimator_ = _fit_dowhy(
                X=X,
                y=y,
                outcome_name=self.outcome_name,
                treatment_name=self.treatment_name,
                treatment_pipeline=self.treatment_pipeline,
                estimation_method=self.estimation_method,
            )
        elif self.estimation_method in ECONML_LEARNERS:
            X_, a = _get_X_a(X, self.treatment_name)
            if self.estimation_method == "DRLearner":
                if self.model_final is None:
                    self.model_final = StatsModelsLinearRegression(
                        fit_intercept=True
                    )
                dr_learner = DRLearner(
                    model_propensity=self.treatment_pipeline["estimator"],
                    model_regression=self.outcome_pipeline["estimator"],
                    model_final=self.model_final,
                    min_propensity=MIN_PS_SCORE,
                    cv=5,
                    random_state=RANDOM_STATE,
                )
                dr_learner.fit(
                    y,
                    a,
                    X=X_cate,
                    W=X_,
                    inference=BootstrapInference(
                        n_bootstrap_samples=self.bootstrap_num_samples
                    ),
                )
                self.inference_estimator_ = dr_learner
            elif self.estimation_method == "TLearner":
                # TODO: might add a cv to learn/predict here.
                t_learner = TLearner(
                    models=[
                        self.outcome_pipeline["estimator"],
                        self.outcome_pipeline["estimator"],
                    ]
                )
                t_learner.fit(
                    y,
                    a,
                    X=X_,
                    inference=BootstrapInference(
                        n_bootstrap_samples=self.bootstrap_num_samples
                    ),
                )
                self.inference_estimator_ = t_learner
            elif self.estimation_method == "DML":
                # default to linear dml
                if self.model_final is None:
                    self.model_final = StatsModelsLinearRegression(
                        fit_intercept=False
                    )
                dml_learner = DML(
                    model_t=self.treatment_pipeline["estimator"],
                    model_y=self.outcome_pipeline["estimator"],
                    model_final=self.model_final,
                    discrete_treatment=True,
                    cv=5,
                    random_state=RANDOM_STATE,
                )
                dml_learner.fit(
                    y,
                    a,
                    X=X_cate,
                    W=X_,
                    inference=BootstrapInference(
                        n_bootstrap_samples=self.bootstrap_num_samples
                    ),
                )
                self.inference_estimator_ = dml_learner
            elif self.estimation_method == "CausalForest":
                forest_learner = CausalForest(random_state=RANDOM_STATE)
                forest_learner.fit(
                    X=X_,
                    T=a,
                    y=y,
                )
                self.inference_estimator_ = forest_learner
            else:
                raise self._not_supported_estimation_method
        else:
            raise self._not_supported_estimation_method
        return self

    def predict(self, X):
        if self.estimation_method.startswith("backdoor."):
            results = _predict_dowhy(
                X,
                estimator=self.inference_estimator_,
                num_boostrap_samples=self.bootstrap_num_samples,
            )
        elif self.estimation_method in ECONML_LEARNERS:
            X_, _ = _get_X_a(X, self.treatment_name)
            if self.estimation_method in ECONML_CATE_LEARNERS:
                ate_inference = self.inference_estimator_.ate_inference(X=None)
                results = {}
                results[RESULT_ATE] = ate_inference.mean_point
                (
                    results[RESULT_ATE_LB],
                    results[RESULT_ATE_UB],
                ) = ate_inference.conf_int_mean()
            elif self.estimation_method in ECONML_META_LEARNERS:
                ate_inference = self.inference_estimator_.ate_inference(X=X_)
                results = {}
                results[RESULT_ATE] = ate_inference.mean_point
                (
                    results[RESULT_ATE_LB],
                    results[RESULT_ATE_UB],
                ) = ate_inference.conf_int_mean()

            elif self.estimation_method == "CausalForest":
                (
                    ate_point_estimates,
                    lb_point_estimates,
                    ub_point_estimates,
                ) = self.inference_estimator_.predict(X=X_, interval=True)
                results = {}
                results[RESULT_ATE] = ate_point_estimates.mean()
                results[RESULT_ATE_LB] = lb_point_estimates.mean()
                results[RESULT_ATE_UB] = ub_point_estimates.mean()

        return results

    def predict_cate(self, X_cate, alpha=0.05):
        results = {}
        for col in X_cate.columns:
            results[f"X_cate__{col}"] = X_cate[col].values
        results["cate_predictions"] = self.inference_estimator_.effect(X_cate)
        lb, ub = self.inference_estimator_.effect_interval(X_cate, alpha=alpha)
        results["cate_lb"] = lb
        results["cate_ub"] = ub
        return results


def _fit_dowhy(
    X,
    y,
    outcome_name: str,
    treatment_name: str,
    treatment_pipeline: Pipeline,
    estimation_method: str,
    **kwargs,
):
    X_, a = _get_X_a(X, treatment_name=treatment_name)
    dowhy_X = pd.concat([X_, a, y], axis=1)
    common_causes = list(dowhy_X.columns.drop([outcome_name, treatment_name]))
    model = CausalModel(
        data=dowhy_X,
        treatment=treatment_name,
        outcome=outcome_name,
        common_causes=common_causes,
    )
    identified_estimand = model.identify_effect(
        optimize_backdoor=True, proceed_when_unidentifiable=True
    )
    estimate = model.estimate_effect(
        identified_estimand,
        method_name=estimation_method,
        method_params={
            "propensity_score_model": treatment_pipeline[-1],
            "min_ps_score": MIN_PS_SCORE,
            "max_ps_score": 1 - MIN_PS_SCORE,
        },
        confidence_intervals="bootstrap",
    )
    return estimate


def _predict_dowhy(
    X, estimator: CausalEstimate, num_boostrap_samples: int = 100
):
    lower_bound, upper_bound = estimator.get_confidence_intervals(
        num_simulations=num_boostrap_samples,
    )
    results = {}
    results[RESULT_ATE] = estimator.value
    results[RESULT_ATE_LB] = lower_bound
    results[RESULT_ATE_UB] = upper_bound
    return results


def _get_X_a(X_a, treatment_name):
    return X_a.drop(columns=treatment_name), X_a[treatment_name]


def score_binary_classification(y_true, hat_y_pred):
    scores = {}
    hat_y = hat_y_pred >= 0.5
    scores["n_samples"] = y_true.shape[0]
    scores["prevalence"] = y_true.mean()
    scores["accuracy"] = accuracy_score(y_true, hat_y)
    scores["precision"] = precision_score(y_true, hat_y)
    scores["recall"] = recall_score(y_true, hat_y)
    scores["roc_auc"] = roc_auc_score(y_true, hat_y_pred)
    scores["pr_auc"] = average_precision_score(y_true, hat_y_pred)
    scores["confusion_matrix"] = confusion_matrix(y_true, hat_y)
    return scores
