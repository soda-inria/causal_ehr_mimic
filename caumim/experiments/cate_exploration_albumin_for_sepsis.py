from copy import deepcopy
from datetime import datetime
from typing import Dict
from loguru import logger
from matplotlib import pyplot as plt
import seaborn as sns

import polars as pl
import pandas as pd
import numpy as np
from sklearn import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    ParameterGrid,
    cross_val_predict,
    RandomizedSearchCV,
    train_test_split,
)
from caumim.constants import *

from caumim.experiments.configurations import ESTIMATOR_RIDGE, ESTIMATOR_RF
from caumim.experiments.utils import (
    InferenceWrapper,
    fit_randomized_search,
    make_column_transformer,
)
from caumim.inference.scores import normalized_total_variation
from caumim.inference.utils import make_random_search_pipeline

from caumim.variables.selection import (
    get_event_covariates_albumin_zhou,
    get_septic_shock_from_features,
)
from caumim.variables.utils import (
    feature_emergency_at_admission,
    feature_insurance_medicare,
)
from caumim.experiments.utils import log_estimate, CateConfig

from dowhy import CausalModel
from sklearn.pipeline import Pipeline, make_pipeline
from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression
from sklearn.linear_model import Ridge


cate_config = CateConfig(
    expe_name=None,
    cohort_folder=DIR2COHORT / "albumin_for_sepsis__obs_1d",
    outcome_name=COLNAME_MORTALITY_28D,
    experience_grid_dict={
        "event_aggregations": [
            {
                "first": pl.col(COLNAME_VALUE).first(),
                "last": pl.col(COLNAME_VALUE).last(),
            },
            # {"first": pl.col(COLNAME_VALUE).first()},
        ],
        "estimation_method": [
            "DML"
            # "DRLearner",
            # "CausalForest",
        ],
        "estimator": [ESTIMATOR_RF],
        "model_final": [
            StatsModelsLinearRegression(fit_intercept=False),
            Ridge(fit_intercept=False, random_state=0),
            RandomForestRegressor(random_state=0),
        ],
    },
    fraction=1,
    random_state=0,
    train_test_random_state=0,
    bootstrap_num_samples=10,
)


def run_cate_experiment(config: CateConfig):
    cohort_folder = config.cohort_folder
    if config.expe_name is None:
        expe_name = datetime.now().strftime("%Y%m%d%H%M%S")
        dir_folder = (
            DIR2EXPERIENCES
            / cohort_folder.name
            / ("cate_estimates" + f"_{expe_name}")
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
        fraction=config.fraction, shuffle=True, seed=config.random_state
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
    static_confounder_names = [
        "admission_age",
        "Female",
        COLNAME_EMERGENCY_ADMISSION,
        COLNAME_INSURANCE_MEDICARE,
        "White",
    ]
    # TODO: might be better in the config
    cate_feature_names = ["admission_age", "Female", "White", "septic_shock"]

    # Static confounders are static features minus CATE features
    if cate_feature_names is not None:
        static_confounder_names = set(static_confounder_names).difference(
            set(cate_feature_names)
        )
    outcome_name = config.outcome_name

    event_features, feature_types = get_event_covariates_albumin_zhou(
        target_trial_population
    )
    # adding septic shock
    target_trial_population = get_septic_shock_from_features(
        target_trial_population
    )

    experience_grid_dict = {
        "event_aggregations": config.experience_grid_dict["event_aggregations"],
        "estimation_method": config.experience_grid_dict["estimation_method"],
        "estimator": config.experience_grid_dict["estimator"],
        "model_final": config.experience_grid_dict["model_final"],
    }
    runs_to_be_launch = list(ParameterGrid(experience_grid_dict))

    # Naive DM estimate (zepid)):
    t0 = datetime.now()

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
        X_all = patient_features_aggregated.join(
            target_trial_population,
            on=STAY_KEYS,
            how="inner",
        )
        X_df = X_all[
            [
                *event_features_names,
                *static_confounder_names,
                *cate_feature_names,
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
        X_df[colnames_binary_features] = X_df[colnames_binary_features].fillna(
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
        # split data
        (
            train_X,
            test_X,
            train_a,
            test_a,
            train_y,
            test_y,
            train_X_cate,
            test_X_cate,
        ) = train_test_split(
            X_df.drop(
                columns=[
                    outcome_name,
                    COLNAME_INTERVENTION_STATUS,
                    *cate_feature_names,
                ]
            ),
            X_df[COLNAME_INTERVENTION_STATUS],
            X_df[outcome_name],
            X_df[cate_feature_names],
            test_size=config.test_size,
            random_state=config.train_test_random_state,
            stratify=X_df[COLNAME_INTERVENTION_STATUS],
        )

        # First, find appropriate hyperparameters for the pipelines:
        logger.info("Fitting randomsearch for hyperparameter optimization")
        treatment_best_pipeline, outcome_best_pipeline = fit_randomized_search(
            X=train_X,
            a=train_a,
            y=train_y,
            treatment_pipeline=treatment_pipeline,
            outcome_pipeline=outcome_pipeline,
        )
        # Then apply transformer since econml does not support nan input:
        train_X_transformed = column_transformer.fit_transform(train_X)
        train_X_transformed_df = pd.DataFrame(
            train_X_transformed,
            index=train_X.index,
            columns=column_transformer.get_feature_names_out(),
        )
        test_X_transformed = pd.DataFrame(
            column_transformer.transform(test_X),
            index=test_X.index,
            columns=column_transformer.get_feature_names_out(),
        )
        train_X_a = pd.concat(
            [train_X_transformed_df, train_a],
            axis=1,
        )
        test_X_a = pd.concat(
            [test_X_transformed, test_a],
            axis=1,
        )

        # 4 - Estimation
        inference_wrapper = InferenceWrapper(
            treatment_pipeline=treatment_best_pipeline,
            outcome_pipeline=outcome_best_pipeline,
            estimation_method=run_config["estimation_method"],
            outcome_name=outcome_name,
            treatment_name=COLNAME_INTERVENTION_STATUS,
            bootstrap_num_samples=config.bootstrap_num_samples,
            model_final=run_config["model_final"],
        )

        inference_wrapper.fit(train_X_a, train_y, X_cate=train_X_cate)

        train_cate = inference_wrapper.predict_cate(
            X_cate=train_X_cate, alpha=0.05
        )
        test_cate = inference_wrapper.predict_cate(
            X_cate=test_X_cate, alpha=0.05
        )

        results = {
            **{f"train_{k}": v for k, v in train_cate.items()},
            **test_cate,
        }
        results["test_y_mse"] = inference_wrapper.inference_estimator_.score(
            Y=test_y,
            T=test_X_a[COLNAME_INTERVENTION_STATUS],
            X=test_X_cate,
            W=test_X_a.drop(columns=[COLNAME_INTERVENTION_STATUS]),
        )

        # Evaluate the statistical assumptions:
        ### Overlap
        hat_e = cross_val_predict(
            estimator=treatment_best_pipeline,
            X=train_X,
            y=train_a,
            n_jobs=-1,
            method="predict_proba",
        )[:, 1]
        #### With NTV (low dimensional score)
        ntv = normalized_total_variation(hat_e, train_a.mean())
        #### Save estimated ps distributions (graphically)
        hat_analysis_df = pd.DataFrame(
            np.vstack([hat_e, train_a.values]).T,
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
        ps_folder.mkdir(exist_ok=True, parents=True)
        plt.savefig(ps_folder / f"{ps_plot_name}.pdf", bbox_inches="tight")
        plt.savefig(ps_folder / f"{ps_plot_name}.png", bbox_inches="tight")
        #### TODO: With [standardized difference](https://onlinelibrary.wiley.com/doi/full/10.1002/sim.6607) TODO:
        ### Well-specified models
        #### treatment
        results["train_brier_score"] = brier_score_loss(
            y_true=train_a,
            y_prob=hat_e,
        )
        results["train_roc_auc"] = roc_auc_score(
            y_true=train_a,
            y_score=hat_e,
        )
        results["train_pr_auc"] = average_precision_score(
            y_true=train_a,
            y_score=hat_e,
        )
        #### outcome
        ### neeed to refit the outcome model: not that good...
        y_hat = cross_val_predict(
            estimator=outcome_best_pipeline,
            X=train_X,
            y=train_y,
            n_jobs=-1,
        )
        results["train_y_r2"] = r2_score(
            y_true=train_y,
            y_pred=y_hat,
        )
        results["train_y_mse"] = mean_squared_error(
            y_true=train_y,
            y_pred=y_hat,
        )
        results["ntv"] = ntv
        results["event_aggregations"] = str(aggregation_names)
        results["estimation_method"] = run_config["estimation_method"]
        results["treatment_model"] = estimator_name
        results["outcome_model"] = estimator_name
        results["compute_time"] = (datetime.now() - t0).total_seconds()
        results["cohort_name"] = cohort_folder.name
        results["outcome_name"] = outcome_name
        results["model_final"] = str(run_config["model_final"]).split("(")[0]
        log_estimate(results, log_folder)


if __name__ == "__main__":
    run_cate_experiment(cate_config)
