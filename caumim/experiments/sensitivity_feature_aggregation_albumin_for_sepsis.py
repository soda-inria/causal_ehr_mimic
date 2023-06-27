from sklearn.utils import Bunch
import polars as pl

from caumim.constants import COLNAME_MORTALITY_28D, COLNAME_VALUE, DIR2COHORT
from caumim.experiments.configurations import ESTIMATOR_RF, ESTIMATOR_RIDGE
from caumim.experiments.sensitivity_albumin_for_sepsis import (
    run_sensitivity_experiment,
)

immortal_time_bias_config = Bunch(
    **{
        "outcome_name": COLNAME_MORTALITY_28D,
        "experience_grid_dict": {
            "event_aggregations": [
                {"first": pl.col(COLNAME_VALUE).first()},
                {"last": pl.col(COLNAME_VALUE).last()},
                {"median": pl.col(COLNAME_VALUE).median()},
                {
                    "first": pl.col(COLNAME_VALUE).first(),
                    "last": pl.col(COLNAME_VALUE).last(),
                    "median": pl.col(COLNAME_VALUE).median(),
                },
            ],
            "estimation_method": [
                "LinearDML",
                "LinearDRLearner",
                "backdoor.propensity_score_weighting",
            ],
            "estimator": [ESTIMATOR_RIDGE, ESTIMATOR_RF],
        },
        "fraction": 1,
        "random_state": 0,
    }
)

cohort_names = [
    "albumin_for_sepsis__obs_1d",
]

if __name__ == "__main__":
    for cohort_name_ in cohort_names:
        cohort_config = Bunch(**immortal_time_bias_config.copy())
        cohort_config["cohort_folder"] = DIR2COHORT / cohort_name_
        cohort_config[
            "expe_name"
        ] = "sensitivity_feature_aggregation_albumin_for_sepsis"
        run_sensitivity_experiment(cohort_config)
