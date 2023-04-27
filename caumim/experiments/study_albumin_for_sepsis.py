import polars as pl
import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
from caumim.constants import *
from caumim.framing.albumin_for_sepsis import COHORT_CONFIG_ALBUMIN_FOR_SEPSIS
from caumim.framing.utils import create_cohort_folder
from caumim.inference.estimation import make_column_tranformer

from caumim.variables.selection import get_albumin_events_zhou_baseline
from caumim.variables.utils import (
    feature_emergency_at_admission,
    feature_insurance_medicare,
)

from dowhy import CausalModel
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from zepid import RiskDifference


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
    outcome_name = COLNAME_MORTALITY_28D
    static_features = [
        "admission_age",
        "Female",
        COLNAME_EMERGENCY_ADMISSION,
        COLNAME_INSURANCE_MEDICARE,
    ]

    # event features
    event_features, feature_types = get_albumin_events_zhou_baseline(
        target_trial_population
    )
    experience_grid_dict = {
        "event_aggregations": config.experience_grid_dict["event_aggregations"],
        "estimation_methods": config.experience_grid_dict["estimation_methods"],
        "estimators": config.experience_grid_dict["estimators"],
    }
    run_to_be_launch = list(ParameterGrid(experience_grid_dict))

    outcome_name = config.outcome_name

    # Naive DM estimate:

    dm = RiskDifference()
    dm.fit(X, COLNAME_INTERVENTION_STATUS, outcome_name)
    # dm.summary()
    dm_results = {
        RESULT_ATE: dm.results.RiskDifference[1],
        RESULT_ATE_LB: dm.results.RiskDifference[1],
        RESULT_ATE_UB: dm.results.RiskDifference[1],
    }

    run_logs = 
    for run_config in run_to_be_launch:
        # TODO: should rewrite make_count with polars.
        patient_features_last = event_features.sort(
            [COLNAME_PATIENT_ID, COLNAME_START]
        ).pivot(
            index=STAY_KEYS,
            columns=COLNAME_CODE,
            values="value",
            aggregate_function=pl.element().first(),
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

        # 3 - Identification
        model = CausalModel(
            data=X,
            treatment=COLNAME_INTERVENTION_STATUS,
            outcome=outcome_name,
            common_causes=[*event_features_names, *static_features],
        )
        # model.view_model(size=(15, 15))
        # from IPython.display import Image, display
        # display(Image(filename=cohort_folder / "causal_model.png"))

        identified_estimand = model.identify_effect(
            optimize_backdoor=True, proceed_when_unidentifiable=True
        )
        print(identified_estimand)
        # note: long to run on 22 variables if not forcing optimize_backdoor.
        column_transformer = make_column_tranformer(
            numerical_features=feature_types.numerical_features,
            categorical_features=feature_types.categorical_features,
        )
        treatment_estimator = run_config.estimators.get(
            "treatment_estimator", None
        )
        outcome_estimator = run_config.estimators.get("outcome_estimator", None)
        if treatment_estimator is not None:
            treatment_pipeline = make_pipeline(
                *[column_transformer, treatment_estimator]
            )
        if outcome_estimator is not None:
            outcome_pipeline = make_pipeline(
                *[column_transformer, outcome_estimator]
            )
        # 4 - Estimation
        estimate = model.estimate_effect(
            identified_estimand,
            method_name=run_config.estimation_methods,
            method_params={
                "propensity_score_model": treatment_pipeline,
                "min_ps_score": 0.05,
                "max_ps_score": 0.95,
            },
            # TODO: adapt with outcome_pipeline
            outcome_pipeline=outcome_pipeline,
            confidence_intervals=True,
        )
        lower_bound, upper_bound = estimate.get_confidence_intervals()
        results = {}
        results[RESULT_ATE] = estimate.value
        results[RESULT_ATE_LB] = lower_bound
        results[RESULT_ATE_UB] = upper_bound
