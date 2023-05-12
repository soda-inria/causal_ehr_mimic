from copy import deepcopy
from datetime import datetime
from loguru import logger
import polars as pl
import pandas as pd
import numpy as np
from sklearn import clone
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV
from sklearn.utils import Bunch
from caumim.constants import *
from caumim.framing.albumin_for_sepsis import COHORT_CONFIG_ALBUMIN_FOR_SEPSIS
from caumim.framing.utils import create_cohort_folder
from caumim.experiments.configurations import ESTIMATOR_RIDGE, ESTIMATOR_RF
from caumim.experiments.utils import (
    InferenceWrapper,
    fit_randomized_search,
    make_column_transformer,
)
from caumim.inference.utils import make_random_search_pipeline

from caumim.variables.selection import get_albumin_events_zhou_baseline
from caumim.variables.utils import (
    feature_emergency_at_admission,
    feature_insurance_medicare,
)
from caumim.experiments.utils import log_estimate

from dowhy import CausalModel
from sklearn.pipeline import Pipeline, make_pipeline
from zepid import RiskDifference


sensitivity_config = Bunch(
    **{
        "expe_name": None,
        "cohort_folder": DIR2COHORT / "albumin_for_sepsis",
        "outcome_name": COLNAME_MORTALITY_28D,
        "experience_grid_dict": {
            "event_aggregation": [
                # [pl.min(), pl.max()], # TODO: test other forms of feature
                # aggregations.
                pl.element().first(),  # ], pl.element().first(),
                pl.element().last(),
                # pl.element().median(),
            ],
            "estimation_method": [
                "backdoor.propensity_score_weighting",
                "LinearDML",
                "TLearner",
                "LinearDRLearner",
                # "CausalForest",
            ],
            "estimator": [
                # ESTIMATOR_RIDGE,
                ESTIMATOR_RF
            ],
        },
        "fraction": 1,
        "random_state": 0,
    }
)


def run_sensitivity_experiment(config):
    cohort_folder = config.cohort_folder
    if "expe_name" not in config.keys():
        expe_name = datetime.now().strftime("%Y%m%d%H%M%S")
        log_folder = (
            DIR2EXPERIENCES
            / cohort_folder.name
            / ("estimates" + f"_{expe_name}")
        )
    else:
        log_folder = DIR2EXPERIENCES / config.expe_name
    # 1 - Framing
    target_trial_population = pl.read_parquet(
        cohort_folder / FILENAME_TARGET_POPULATION
    )
    # FOR TESTING: subsample the data
    target_trial_population = target_trial_population.sample(
        fraction=config["fraction"], shuffle=True, seed=config.random_state
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

    # event features TODO: this is the only code specific to the albumin study.
    # I could make it more generalizable by asking for a event_features function.
    event_features, feature_types = get_albumin_events_zhou_baseline(
        target_trial_population
    )
    experience_grid_dict = {
        "event_aggregation": config.experience_grid_dict["event_aggregation"],
        "estimation_method": config.experience_grid_dict["estimation_method"],
        "estimator": config.experience_grid_dict["estimator"],
    }
    runs_to_be_launch = list(ParameterGrid(experience_grid_dict))

    # Naive DM estimate (zepid)):
    t0 = datetime.now()
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
        "event_aggregation": str(None),
        "estimation_method": "Difference in mean",
        "treatment_model": str(None),
        "outcome_model": str(None),
        "compute_time": (datetime.now() - t0).total_seconds(),
        "cohort_name": cohort_folder.name,
        "outcome_name": outcome_name,
    }
    log_estimate(estimate_difference_in_mean, cohort_folder / log_folder)

    logger.info(
        f"{len(runs_to_be_launch)} configs to run :\n {runs_to_be_launch}\n------"
    )

    for run_config in runs_to_be_launch:
        t0 = datetime.now()
        logger.info(f"Running {run_config}")
        # Variable aggregation
        # TODO: should rewrite make_count with polars.
        patient_features_aggregated = event_features.sort(
            [COLNAME_PATIENT_ID, COLNAME_START]
        ).pivot(
            index=STAY_KEYS,
            columns=COLNAME_CODE,
            values="value",
            aggregate_function=run_config["event_aggregation"],
        )
        event_features_names = list(
            set(patient_features_aggregated.columns).difference(set(STAY_KEYS))
        )
        X = patient_features_aggregated.join(
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
        column_transformer = make_column_transformer(
            numerical_features=feature_types.numerical_features,
            categorical_features=feature_types.categorical_features,
        )
        # Both treatment and outcome models are the same models for now.
        estimator_config = run_config["estimator"]
        estimator_name = estimator_config["name"]
        treatment_pipeline = make_random_search_pipeline(
            estimator=clone(estimator_config["treatment_estimator"]),
            column_transformer=column_transformer,
            param_distributions=estimator_config[
                "treatment_param_distributions"
            ],
        )
        outcome_pipeline = make_random_search_pipeline(
            estimator=clone(estimator_config["outcome_estimator"]),
            column_transformer=column_transformer,
            param_distributions=estimator_config["outcome_param_distributions"],
        )

        # First, find appropriate hyperparameters for the pipelines:
        logger.info("Fitting randomsearch for hyperparameter optimization")
        a = X[COLNAME_INTERVENTION_STATUS]
        y = X[outcome_name]
        treatment_best_pipeline, outcome_best_pipeline = fit_randomized_search(
            X=X.drop(columns=[COLNAME_INTERVENTION_STATUS, outcome_name]),
            a=a,
            y=y,
            treatment_pipeline=treatment_pipeline,
            outcome_pipeline=outcome_pipeline,
        )
        # Then apply transformer since econml does not support nan input:
        X_transformed = column_transformer.fit_transform(
            X.drop([COLNAME_INTERVENTION_STATUS, outcome_name], axis=1)
        )
        X_transformed = pd.DataFrame(
            X_transformed, columns=column_transformer.get_feature_names_out()
        )
        X_a = pd.concat([X_transformed, a], axis=1)
        # 4 - Estimation
        inference_wrapper = InferenceWrapper(
            treatment_pipeline=treatment_best_pipeline,
            outcome_pipeline=outcome_best_pipeline,
            estimation_method=run_config["estimation_method"],
            outcome_name=outcome_name,
            treatment_name=COLNAME_INTERVENTION_STATUS,
        )
        inference_wrapper.fit(X_a, y)
        results = inference_wrapper.predict(X=X_a)

        results["event_aggregation"] = str(run_config["event_aggregation"])
        results["estimation_method"] = run_config["estimation_method"]
        results["treatment_model"] = estimator_name
        results["outcome_model"] = estimator_name
        results["compute_time"] = (datetime.now() - t0).total_seconds()
        results["cohort_name"] = cohort_folder.name
        results["outcome_name"] = outcome_name
        log_estimate(results, log_folder)


if __name__ == "__main__":
    run_sensitivity_experiment(sensitivity_config)
