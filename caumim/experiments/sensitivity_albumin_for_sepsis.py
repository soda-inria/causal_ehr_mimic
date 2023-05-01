from copy import deepcopy
from loguru import logger
import polars as pl
import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV
from sklearn.utils import Bunch
from caumim.constants import *
from caumim.framing.albumin_for_sepsis import COHORT_CONFIG_ALBUMIN_FOR_SEPSIS
from caumim.framing.utils import create_cohort_folder
from caumim.inference.estimation import ESTIMATOR_LR, make_column_tranformer

from caumim.variables.selection import get_albumin_events_zhou_baseline
from caumim.variables.utils import (
    feature_emergency_at_admission,
    feature_insurance_medicare,
)
from caumim.experiments.utils import log_estimate

from dowhy import CausalModel
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from zepid import RiskDifference


config = Bunch(
    **{
        "outcome_name": COLNAME_MORTALITY_28D,
        "experience_grid_dict": {
            "event_aggregations": [
                # [pl.min(), pl.max()],
                # [pl.first()],
                # pl.element().first(),
                pl.element().last(),
                pl.element().median(),
            ],
            "estimation_methods": ["backdoor.propensity_score_matching"],
            "estimators": [
                ESTIMATOR_LR,
            ],
        },
    }
)


def run_experiment(config):
    # 1 - Framing
    cohort_folder = create_cohort_folder(COHORT_CONFIG_ALBUMIN_FOR_SEPSIS)
    target_trial_population = pl.read_parquet(
        cohort_folder / FILENAME_TARGET_POPULATION
    )
    # 2 - Variable selection
    # Static features
    # demographics
    target_trial_population = feature_emergency_at_admission(
        target_trial_population
    )
    target_trial_population = feature_insurance_medicare(
        target_trial_population
    ).with_columns(
        [
            pl.when(pl.col("gender") == "F")
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("Female"),
            pl.when(pl.col("race").str.to_lowercase().str.contains("white"))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("White"),
        ]
    )
    static_features = [
        "admission_age",
        "Female",
        COLNAME_EMERGENCY_ADMISSION,
        COLNAME_INSURANCE_MEDICARE,
    ]
    outcome_name = config.outcome_name

    # event features
    event_features, feature_types = get_albumin_events_zhou_baseline(
        target_trial_population
    )
    experience_grid_dict = {
        "event_aggregations": config.experience_grid_dict["event_aggregations"],
        "estimation_methods": config.experience_grid_dict["estimation_methods"],
        "estimators": config.experience_grid_dict["estimators"],
    }
    runs_to_be_launch = list(ParameterGrid(experience_grid_dict))

    # Naive DM estimate (zepid)):
    dm = RiskDifference()
    dm.fit(
        target_trial_population.to_pandas(),
        COLNAME_INTERVENTION_STATUS,
        outcome_name,
    )
    estimate_difference_in_mean = {
        RESULT_ATE: dm.results.RiskDifference[1],
        RESULT_ATE_LB: dm.results.RiskDifference[1],
        RESULT_ATE_UB: dm.results.RiskDifference[1],
        "event_aggregations": str(None),
        "estimation_methods": "Difference in mean",
        "treatment_model": str(None),
        "outcome_model": str(None),
    }
    log_estimate(estimate_difference_in_mean, cohort_folder / "estimates")

    logger.info(
        f"{len(runs_to_be_launch)} configs to run :\n {runs_to_be_launch}\n------"
    )
    for run_config in runs_to_be_launch:
        logger.info(f"Running {run_config}")
        # Variable aggregation
        # TODO: should rewrite make_count with polars.
        patient_features_last = event_features.sort(
            [COLNAME_PATIENT_ID, COLNAME_START]
        ).pivot(
            index=STAY_KEYS,
            columns=COLNAME_CODE,
            values="value",
            aggregate_function=run_config["event_aggregations"],
        )
        event_features_names = list(
            set(patient_features_last.columns).difference(set(STAY_KEYS))
        )
        X = patient_features_last.join(
            target_trial_population,
            on=STAY_KEYS,
            how="inner",
        )[
            [
                *event_features_names,
                *static_features,
                COLNAME_INTERVENTION_STATUS,
                outcome_name,
            ]
        ].to_pandas()

        X[feature_types.binary_features] = X[
            feature_types.binary_features
        ].fillna(value=0)

        # 3 - Identification and estimation
        column_transformer = make_column_tranformer(
            numerical_features=feature_types.numerical_features,
            categorical_features=feature_types.categorical_features,
        )
        treatment_estimator = run_config["estimators"].get(
            "treatment_estimator", None
        )
        treatment_hp_kwargs = run_config["estimators"].get(
            "treatment_estimator_kwargs", {}
        )
        if treatment_estimator is not None:
            treatment_pipeline = RandomizedSearchCV(
                Pipeline(
                    [
                        ("preprocessor", column_transformer),
                        ("treatment_estimator", treatment_estimator),
                    ]
                ),
                param_distributions=treatment_hp_kwargs,
                n_iter=10,
                n_jobs=-1,
            )
        # TODO: allow outcome models
        outcome_estimator = run_config["estimators"].get(
            "outcome_estimator", None
        )
        if outcome_estimator is not None:
            outcome_pipeline = make_pipeline(
                *[column_transformer, outcome_estimator]
            )
        else:
            outcome_pipeline = None
        # 4 - Estimation
        ## TODO: dowhy machinery, not very useful for ATE IMHO, favor causalml api
        model = CausalModel(
            data=X,
            treatment=COLNAME_INTERVENTION_STATUS,
            outcome=outcome_name,
            common_causes=[*event_features_names, *static_features],
        )
        identified_estimand = model.identify_effect(
            optimize_backdoor=True, proceed_when_unidentifiable=True
        )
        estimate = model.estimate_effect(
            identified_estimand,
            method_name=run_config["estimation_methods"],
            method_params={
                "propensity_score_model": treatment_pipeline,
                "min_ps_score": 0.05,
                "max_ps_score": 0.95,
                "outcome_pipeline": outcome_pipeline,
            },
            confidence_intervals=True,
        )
        lower_bound, upper_bound = estimate.get_confidence_intervals()
        # logging
        run_logs = {}
        run_logs[RESULT_ATE] = estimate.value
        run_logs[RESULT_ATE_LB] = lower_bound
        run_logs[RESULT_ATE_UB] = upper_bound
        run_logs["event_aggregations"] = str(run_config["event_aggregations"])
        run_logs["estimation_methods"] = run_config["estimation_methods"]
        run_logs["treatment_model"] = str(treatment_estimator)
        run_logs["outcome_model"] = str(outcome_estimator)
        log_estimate(run_logs, cohort_folder / "estimates")


if __name__ == "__main__":
    run_experiment(config)
