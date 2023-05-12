from sklearn.utils import Bunch
import polars as pl

from caumim.constants import COLNAME_MORTALITY_28D, DIR2COHORT
from caumim.experiments.configurations import ESTIMATOR_RF, ESTIMATOR_RIDGE
from caumim.experiments.sensitivity_albumin_for_sepsis import (
    run_sensitivity_experiment,
)

immortal_time_bias_config = Bunch(
    **{
        "outcome_name": COLNAME_MORTALITY_28D,
        "experience_grid_dict": {
            "event_aggregation": [
                pl.element().first(),
            ],
            "estimation_method": [
                "LinearDML",
                "LinearDRLearner",
            ],
            "estimator": [ESTIMATOR_RF],
        },
        "fraction": 1,
        "random_state": 0,
    }
)

cohort_names = [
    # "albumin_for_sepsis__obs_0f25d",
    "albumin_for_sepsis__obs_1d",
    "albumin_for_sepsis__obs_3d",
]

if __name__ == "__main__":
    for cohort_name_ in cohort_names:
        cohort_config = Bunch(**immortal_time_bias_config.copy())
        cohort_config["cohort_folder"] = DIR2COHORT / cohort_name_
        cohort_config[
            "expe_name"
        ] = "immortal_time_bias_double_robust_forest_agg_last"
        run_sensitivity_experiment(cohort_config)
