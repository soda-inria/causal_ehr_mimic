from copy import deepcopy
from datetime import datetime
from loguru import logger
from matplotlib import pyplot as plt
import seaborn as sns

import polars as pl
import pandas as pd
import numpy as np
from sklearn import clone
from sklearn.calibration import cross_val_predict
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV
from sklearn.utils import Bunch
from caumim.constants import *
from caumim.experiments.configurations import ESTIMATOR_RIDGE, ESTIMATOR_RF
from caumim.experiments.utils import (
    InferenceWrapper,
    fit_randomized_search,
    make_column_transformer,
)
from caumim.inference.scores import normalized_total_variation
from caumim.inference.utils import make_random_search_pipeline

from caumim.variables.selection import get_event_covariates_albumin_zhou
from caumim.variables.utils import (
    feature_emergency_at_admission,
    feature_insurance_medicare,
)
from caumim.experiments.utils import log_estimate

from zepid import RiskDifference


sensitivity_config = Bunch(
    **{
        "expe_name": None,
        "cohort_folder": DIR2COHORT / "albumin_for_sepsis__obs_1d",
        "outcome_name": COLNAME_MORTALITY_28D,
        "experience_grid_dict": {
            "event_aggregations": [
                {
                    "first": pl.col(COLNAME_VALUE).first(),
                    "last": pl.col(COLNAME_VALUE).last(),
                },
                {"first": pl.col(COLNAME_VALUE).first()},
                # {"last": pl.col(COLNAME_VALUE).last()},
            ],
            "estimation_method": [
                "DML",
                "backdoor.propensity_score_matching",
                "backdoor.propensity_score_weighting",
                "TLearner",
                "DRLearner",
                # "CausalForest",
            ],
            "estimator": [ESTIMATOR_RIDGE],  # ESTIMATOR_RF
        },
        "fraction": 1,
        "random_state": 0,
        "bootstrap_num_samples": 50,
    }
)


def run_sensitivity_experiment(config):
    cohort_folder = config.cohort_folder
    if "expe_name" not in config.keys():
        config["expe_name"] = None
    if config.expe_name is None:
        expe_name = datetime.now().strftime("%Y%m%d%H%M%S")
        dir_folder = (
            DIR2EXPERIENCES
            / cohort_folder.name
            / ("estimates" + f"_{expe_name}")
        )
    else:
        dir_folder = DIR2EXPERIENCES / config.expe_name
    log_folder = dir_folder / "logs"
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
    event_features, feature_types = get_event_covariates_albumin_zhou(
        target_trial_population
    )
    experience_grid_dict = {
        "event_aggregations": config.experience_grid_dict["event_aggregations"],
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
        "ntv": -1.0,
        "event_aggregations": str(None),
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
        aggregate_functions = {
            k: v.alias(k) for k, v in run_config["event_aggregations"].items()
        }
        aggregation_names = list(aggregate_functions.keys())
        patient_features_aggregated = (
            event_features.sort([COLNAME_PATIENT_ID, COLNAME_START])
            .groupby(STAY_KEYS + [COLNAME_CODE])
            .agg(list(aggregate_functions.values()))
        ).pivot(
            index=STAY_KEYS,
            columns=COLNAME_CODE,
            values=aggregation_names,
            aggregate_function=None,
        )
        # particular case when there is only one aggregation function
        if len(aggregate_functions) == 1:
            patient_features_aggregated.columns = [
                f"{aggregation_names[0]}_code_{col}"
                if col not in STAY_KEYS
                else col
                for col in patient_features_aggregated.columns
            ]

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
        colnames_binary_features = [
            f"{agg_name_}_code_{col}"
            for agg_name_ in aggregation_names
            for col in feature_types.binary_features
        ]

        colnames_numerical_features = [
            f"{agg_name_}_code_{col}"
            for agg_name_ in aggregation_names
            for col in feature_types.numerical_features
        ]
        colnames_categorical_features = [
            f"{agg_name_}_code_{col}"
            for agg_name_ in aggregation_names
            for col in feature_types.categorical_features
        ]
        X[colnames_binary_features] = X[colnames_binary_features].fillna(
            value=0
        )
        # 3 - Identification and estimation
        column_transformer = make_column_transformer(
            numerical_features=colnames_numerical_features,
            categorical_features=colnames_categorical_features,
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
        ## If the causal model is not a doubly robust, wrap the treatment or outcome models into a
        inference_wrapper = InferenceWrapper(
            treatment_pipeline=treatment_best_pipeline,
            outcome_pipeline=outcome_best_pipeline,
            estimation_method=run_config["estimation_method"],
            outcome_name=outcome_name,
            treatment_name=COLNAME_INTERVENTION_STATUS,
            bootstrap_num_samples=config.bootstrap_num_samples,
        )
        inference_wrapper.fit(X_a, y)
        results = inference_wrapper.predict(X=X_a)
        # Evaluate the statistical assumptions:
        ### Overlap
        hat_e = cross_val_predict(
            estimator=treatment_best_pipeline,
            X=X.drop(
                columns=[COLNAME_INTERVENTION_STATUS, outcome_name],
                errors="ignore",
            ),
            y=a,
            n_jobs=-1,
            method="predict_proba",
        )[:, 1]
        #### With NTV (low dimensional score)
        ntv = normalized_total_variation(hat_e, a.mean())
        #### Save estimated ps distributions (graphically)
        hat_analysis_df = pd.DataFrame(
            np.vstack([hat_e, a.values]).T,
            columns=[LABEL_PS, LABEL_TREATMENT],
        )
        hat_analysis_df[LABEL_TREATMENT] = hat_analysis_df.apply(
            lambda x: TREATMENT_LABELS[int(x[LABEL_TREATMENT])], axis=1
        )
        fix, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(
            ax=ax,
            data=hat_analysis_df,
            x=LABEL_PS,
            hue=LABEL_TREATMENT,
            stat="probability",
            common_norm=False,
            palette=TREATMENT_PALETTE,
        )
        ps_plot_name = (
            f"ps_distribution__{str(aggregation_names)}__{estimator_name}"
        )
        ps_folder = dir_folder / "ps_distributions"
        ps_folder.mkdir(exist_ok=True)
        plt.savefig(ps_folder / f"{ps_plot_name}.pdf", bbox_inches="tight")
        plt.savefig(ps_folder / f"{ps_plot_name}.png", bbox_inches="tight")

        #### With [standardized difference](https://onlinelibrary.wiley.com/doi/full/10.1002/sim.6607) TODO:

        ### Well-specified models
        # TODO: log the scores of the estimators
        # (after refitting through the inference wrapper: need to get them back)
        results["ntv"] = ntv
        results["event_aggregations"] = str(aggregation_names)
        results["estimation_method"] = run_config["estimation_method"]
        results["treatment_model"] = estimator_name
        results["outcome_model"] = estimator_name
        results["compute_time"] = (datetime.now() - t0).total_seconds()
        results["cohort_name"] = cohort_folder.name
        results["outcome_name"] = outcome_name

        log_estimate(results, log_folder)


if __name__ == "__main__":
    run_sensitivity_experiment(sensitivity_config)
